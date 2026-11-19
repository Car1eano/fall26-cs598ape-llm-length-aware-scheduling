"""Run L40S scheduler smoke tests, or FCFS pilots with --pilot-rates 2,5,30."""

import json
import os
import platform
import random
import signal
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/benchmark")
VOLUME_NAME = "cs598ape-scheduler-smoke"
OUTPUT = Path("/runs")
GPU = "L40S"
MAX_WORKERS = 5
IMAGE_TAG = "vllm/vllm-openai:v0.26.0"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
POLICIES = {
    "native_fcfs": None,
    "queue_only": "gpu_experiments.length_aware_queue_only_scheduler.LengthAwareQueueOnlyScheduler",
    "aging_alpha05": "gpu_experiments.adaptive_alpha05_scheduler.AdaptiveAlpha05Scheduler",
}
EXTRA_POLICIES = {
    "native_priority": None,
    **{
        name: f"gpu_experiments.scheduler_variants.{cls}"
        for name, cls in {
            "prompt_only": "PromptOnlyScheduler",
            "noisy_length": "NoisyLengthScheduler",
            "aging_alpha1": "AgingAlpha1",
            "aging_alpha2": "AgingAlpha2",
            "aging_alpha5": "AgingAlpha5",
            "aging_alpha10": "AgingAlpha10",
            "wait_promotion": "WaitPromotionScheduler",
            "hrrn": "HRRNScheduler",
        }.items()
    },
}
PROFILE_POLICIES = {
    name: f"gpu_experiments.profiled_schedulers.{cls}"
    for name, cls in {
        "profiled_fcfs": "ProfiledFCFS",
        "profiled_queue": "ProfiledQueueOnly",
        "profiled_aging": "ProfiledAging",
        "profiled_optimized_aging": "ProfiledOptimizedAging",
    }.items()
}
ALL_POLICIES = {**POLICIES, **EXTRA_POLICIES, **PROFILE_POLICIES}


def configure_requests(policy, requests, seed):
    if policy == "noisy_length":
        rng = random.Random(seed)
        return [dict(row, vllm_xargs={"length_noise_scale": rng.uniform(0.7, 1.3)})
                for row in requests]
    if policy == "native_priority":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        prompts = dict.fromkeys(row["prompt"] for row in requests)
        lengths = {prompt: len(tokenizer.encode(prompt, add_special_tokens=True))
                   for prompt in prompts}
        return [dict(row, priority=lengths[row["prompt"]] + row["max_tokens"])
                for row in requests]
    return requests


app = modal.App("cs598ape-scheduler-smoke")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry(IMAGE_TAG)
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python", "python -m pip --version")
    .workdir(str(REMOTE_ROOT))
)
for package in ("benchmarks", "workload", "gpu_experiments"):
    for source in sorted((ROOT / package).glob("*.py")):
        image = image.add_local_file(source, str(REMOTE_ROOT / package / source.name))


def wait_for_server(server, health_url):
    deadline = time.monotonic() + 360
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited with status {server.returncode}")
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("vLLM failed to become healthy within 360 seconds")


def stop_server(server):
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


@app.function(image=image, gpu=GPU, cpu=2, memory=8192, timeout=1200,
              max_containers=MAX_WORKERS, volumes={str(OUTPUT): volume})
def smoke(policy: str, requests: list, source: dict, batch_id: str) -> dict:
    sys.path.insert(0, str(REMOTE_ROOT))
    from benchmarks.audit import audit_run
    from workload.generator import write_jsonl

    relative_path = f"{batch_id}/{policy}-{uuid4().hex[:8]}"
    run_dir = OUTPUT / relative_path
    run_dir.mkdir(parents=True)
    workload = run_dir / "workload.jsonl"
    write_jsonl(requests, workload)
    base_url = "http://127.0.0.1:8000"
    command = [
        "vllm", "serve", MODEL, "--revision", REVISION,
        "--host", "127.0.0.1", "--port", "8000",
        "--dtype", "float16", "--scheduling-policy",
        "priority" if policy == "native_priority" else "fcfs", "--async-scheduling",
        "--no-enable-prefix-caching", "--max-num-seqs", str(source.get("max_num_seqs", 4)),
        "--max-model-len", "2048", "--gpu-memory-utilization", "0.30",
        "--enforce-eager",
    ]
    scheduler = ALL_POLICIES[policy]
    if scheduler:
        command.extend(["--scheduler-cls", scheduler])

    config = {
        **source, "policy": policy, "image": IMAGE_TAG, "gpu_requested": GPU,
        "model": MODEL, "model_revision": REVISION, "server_command": command,
        "workload_requests": len(requests),
    }
    config_file = run_dir / "config.json"
    started = time.monotonic()
    status = {"ok": False, "policy": policy, "path": relative_path,
              "started_at_unix": time.time()}
    log_path = run_dir / "server.log"
    try:
        config["python"] = platform.python_version()
        config["packages"] = {name: version(name) for name in ("vllm", "torch", "aiohttp")}
        if config["packages"]["vllm"] != "0.26.0":
            raise RuntimeError(f"Unexpected vLLM version: {config['packages']['vllm']}")
        config["gpu_actual"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip()
        config_file.write_text(json.dumps(config, indent=2) + "\n")
        env = {**os.environ, "PYTHONPATH": str(REMOTE_ROOT)}
        profile_path = run_dir / "scheduler_profile.jsonl"
        if policy in PROFILE_POLICIES:
            env["CS598_SCHED_PROFILE_PATH"] = str(profile_path)
            if "target_rate" not in source:
                env["CS598_SCHED_PROFILE_INTERVAL"] = "8"
        with ExitStack() as stack:
            log = stack.enter_context(log_path.open("w"))
            finish_traces = None
            if source.get("trace_queue"):
                from gpu_experiments.queue_traces import queue_receiver

                endpoint, finish_traces = stack.enter_context(
                    queue_receiver(run_dir / "queue_times.jsonl", {row["id"] for row in requests})
                )
                command.extend(["--otlp-traces-endpoint", endpoint])
                env["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"
                env["OTEL_BSP_SCHEDULE_DELAY"] = "500"
            server = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            telemetry = None
            try:
                wait_for_server(server, f"{base_url}/health")
                if scheduler and f"Using custom scheduler class {scheduler}" not in log_path.read_text(
                    errors="replace"
                ):
                    raise RuntimeError(f"Server log does not confirm scheduler {scheduler}")
                requests = configure_requests(policy, requests, source.get("trace_seed", 598))
                if finish_traces:
                    requests = [dict(row, request_id=f"cs598-measure-{row['id']}") for row in requests]
                if finish_traces or policy in ("native_priority", "noisy_length"):
                    write_jsonl(requests, run_dir / "submitted_workload.jsonl")
                    workload = run_dir / "submitted_workload.jsonl"
                if "target_rate" in source:
                    from benchmarks.client import run_workload
                    import asyncio
                    warmup = [dict(row, arrival_time_s=0, request_id=f"cs598-warmup-{row['id']}") for row in
                              [next(r for r in requests if r["bucket"] == bucket)
                               for bucket in ("short", "long")]]
                    warmed = asyncio.run(run_workload(warmup, f"{base_url}/v1/completions", MODEL))
                    if any(r["error"] or r["output_tokens"] != w["max_tokens"]
                           for r, w in zip(warmed, warmup)):
                        raise RuntimeError("Warm-up failed")
                    with urlopen(f"{base_url}/metrics", timeout=10) as response:
                        (run_dir / "metrics_before.txt").write_bytes(response.read())
                if policy in PROFILE_POLICIES:
                    gpu_log = stack.enter_context((run_dir / "gpu_utilization.csv").open("w"))
                    telemetry = subprocess.Popen(
                        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory",
                         "--format=csv,nounits", "-lms", "1000"],
                        stdout=gpu_log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                with (run_dir / "benchmark.log").open("w") as benchmark_log:
                    subprocess.run(
                        [
                            sys.executable, "-m", "benchmarks.run_benchmark",
                            "--workload", str(workload), "--model", MODEL,
                            "--url", f"{base_url}/v1/completions", "--out-dir", str(run_dir),
                        ],
                        env=env, stdout=benchmark_log, stderr=subprocess.STDOUT,
                        check=True, timeout=720 if "target_rate" in source else 90,
                    )
                if policy in PROFILE_POLICIES:
                    stop_server(telemetry)
                    telemetry = None
                if finish_traces:
                    finish_traces()
                if "target_rate" in source:
                    time.sleep(2)
                    with urlopen(f"{base_url}/metrics", timeout=10) as response:
                        (run_dir / "metrics_after.txt").write_bytes(response.read())
            finally:
                if telemetry is not None:
                    stop_server(telemetry)
                stop_server(server)

        audit = audit_run(run_dir, {row["id"]: row for row in requests})
        if policy in PROFILE_POLICIES and not (
            profile_path.is_file() and profile_path.stat().st_size
        ):
            raise RuntimeError("Missing scheduler profile snapshots")
        keys = ("ok", "rows", "failed", "token_mismatches", "bad_timestamps", "bad_ids", "summary_ok")
        status.update(dict(zip(keys, audit, strict=True)))
    except Exception:
        status["error"] = traceback.format_exc()
    finally:
        config_file.write_text(json.dumps(config, indent=2) + "\n")
        status["elapsed_s"] = time.monotonic() - started
        status["finished_at_unix"] = time.time()
        (run_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        volume.commit()
    return status


@app.local_entrypoint()
def main(download_dir: str = "", pilot_rates: str = "", formal: bool = False,
         max_num_seqs: int = 4, paired_plan: str = "", mix: str = "", rates: str = "",
         policies: str = "", calibrate: bool = False, trace_queue: bool = False):
    from workload.generator import read_jsonl

    selected_policies = policies.split(",") if policies else list(POLICIES)
    if len(set(selected_policies)) != len(selected_policies) or any(
        policy not in ALL_POLICIES for policy in selected_policies
    ):
        raise ValueError(f"Expected distinct policies from {list(ALL_POLICIES)}")

    if formal:
        from gpu_experiments.modal_formal import run_formal

        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        run_formal(download_dir, max_num_seqs, paired_plan, mix, rates,
                   selected_policies if policies else None, trace_queue)
        return

    if pilot_rates:
        pilot(pilot_rates, download_dir)
        return

    workload = Path("gpu_experiments/results/fairness_900_rate30/workload.jsonl")
    rows = read_jsonl(ROOT / workload)
    if calibrate:
        isolated_service(rows, download_dir, max_num_seqs)
        return

    # Submit a mixed burst larger than the smoke-test concurrency cap.
    short = [row for row in rows if row["bucket"] == "short"][:4]
    long = [row for row in rows if row["bucket"] == "long"][:4]
    requests = [{**row, "arrival_time_s": 0.0} for row in long + short]
    git = ["git", "-C", str(ROOT)]
    source = {
        "repo_commit": subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip(),
        "repo_status": subprocess.check_output([*git, "status", "--porcelain"], text=True).splitlines(),
        "workload_source": str(workload),
        "workload_seed": None,
        "trace_queue": trace_queue,
        "workload_selection": "First four long and first four short, sent simultaneously",
    }
    batch_id = f"smoke-{uuid4().hex}"
    destination = Path(download_dir) if download_dir else ROOT / "results" / "modal_smoke"
    batch_dir = destination / batch_id
    batch_dir.mkdir(parents=True)
    print(f"Volume path: {batch_id}", flush=True)
    worker = smoke
    if trace_queue and set(selected_policies) == set(EXTRA_POLICIES):
        # Include the native FCFS control in the traced policy validation batch.
        selected_policies.insert(0, "native_fcfs")
        worker = smoke.with_options(max_containers=10)
        print("Testing up to 10 concurrent single-L40S workers", flush=True)
    calls = [worker.spawn(policy, requests, source, batch_id) for policy in selected_policies]
    outcomes = []
    for call in calls:
        outcome = call.get()
        outcomes.append(outcome)
        print(json.dumps(outcome, indent=2), flush=True)

    peak = max(
        sum(item["started_at_unix"] <= point["started_at_unix"] < item["finished_at_unix"]
            for item in outcomes)
        for point in outcomes
    )
    print(f"Peak concurrent L40S runs: {peak}", flush=True)
    failed = [outcome["policy"] for outcome in outcomes if not outcome["ok"]]

    download_batch(batch_id, destination)
    for policy in selected_policies:
        if not list(batch_dir.glob(f"{policy}-*/status.json")):
            raise RuntimeError(f"Could not retrieve status for {policy} from {batch_id}")
    print(f"Downloaded smoke-test artifacts to {batch_dir}", flush=True)
    if failed:
        raise SystemExit(f"Smoke tests failed: {', '.join(failed)}")


def download_batch(batch_id, destination):
    for entry in volume.listdir(batch_id, recursive=True):
        if entry.type.name != "FILE":
            continue
        remote_path = Path(entry.path)
        path = destination / remote_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            for chunk in volume.read_file(remote_path.as_posix()):
                output.write(chunk)


def isolated_service(rows, download_dir, max_num_seqs):
    """Measure isolated end-to-end service times on the benchmark server."""
    if max_num_seqs < 1:
        raise ValueError("max_num_seqs must be positive")
    requests = [
        dict(row, id=index, arrival_time_s=2.0 * index)
        for index, row in enumerate(
            [next(row for row in rows if row["bucket"] == bucket)
             for bucket in ("short", "long")] * 12
        )
    ]
    batch_id = f"calibration-{uuid4().hex}"
    destination = Path(download_dir) if download_dir else ROOT / "results/modal_calibration"
    source = {"target_rate": 0.5, "max_num_seqs": max_num_seqs, "isolated_service": True}
    print(f"Volume path: {batch_id}", flush=True)
    result = smoke.remote("native_fcfs", requests, source, batch_id)
    download_batch(batch_id, destination)
    print(json.dumps(result, indent=2), flush=True)
    if not result["ok"]:
        raise SystemExit("Isolated service run failed")


def pilot(rates: str, download_dir: str):
    from workload.generator import read_jsonl

    selected = [float(rate) for rate in rates.split(",")]
    if not all(0 < rate <= 1000 for rate in selected):
        raise ValueError("Expected positive arrival rates up to 1000 req/s")
    batch_id = f"pilot-{uuid4().hex}"
    destination = Path(download_dir) if download_dir else ROOT / "results" / "modal_pilot"
    calls = []
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for mix in ("fairness_900_rate30", "balanced_900_rate30_seed123"):
        path = Path("gpu_experiments/results") / mix / "workload.jsonl"
        original = read_jsonl(ROOT / path)
        for rate in selected:
            rows = [dict(row, arrival_time_s=row["arrival_time_s"] * 30 / rate) for row in original]
            source = {"mix": mix, "target_rate": rate, "workload_source": str(path),
                      "repo_commit": commit, "warmup": "one short and one long, concurrent"}
            calls.append(smoke.spawn("native_fcfs", rows, source, batch_id))
    print(f"Volume path: {batch_id}", flush=True)
    outcomes = [call.get() for call in calls]
    download_batch(batch_id, destination)
    print(json.dumps(outcomes, indent=2))
    if not all(result["ok"] for result in outcomes):
        raise SystemExit("Pilot audit failed")
