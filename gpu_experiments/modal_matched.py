"""Run capacity-calibrated single-container scheduler comparisons on Modal."""

import asyncio
import json
import os
import random
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from uuid import uuid4
from zoneinfo import ZoneInfo

import modal

from gpu_experiments.modal_smoke import GPU, OUTPUT, POLICIES, ROOT, image, run_gpu_benchmark, volume
from workload.generator import read_jsonl
from workload.modal_followups import p4_trace

app = modal.App("cs598ape-matched-blocks", tags={"study": "cs598ape-matched"})
BALANCED = "gpu_experiments/results/balanced_900_rate30_seed123/workload.jsonl"
VARIANTS = (*POLICIES, "aging_alpha01", "aging_alpha1", "aging_alpha2", "aging_alpha5",
            "aging_alpha10", "native_priority", "prompt_only", "noisy_length",
            "wait_promotion", "hrrn", "hybrid")
STRESS = (*POLICIES, "wait_promotion", "hrrn", "wait_limit_5")
RHOS = (0.8, 0.9, 1.0, 0.5, 1.3)
GPU_COST_PER_S = 0.000542
COMPUTE_COST_PER_S = GPU_COST_PER_S + 2 * 0.0000131 + 8 * 0.00000222


def assign_arrivals(rows, seed, rate):
    rng = random.Random(seed)
    elapsed = 0.0
    result = []
    for index, row in enumerate(rows):
        elapsed += rng.expovariate(rate)
        result.append(dict(row, id=index, arrival_time_s=elapsed))
    return result


def block_plan(replicates):
    blocks = []
    for kind, rho, policies in [
        ("variants", 1.1, VARIANTS),
        *(("rho", rho, tuple(POLICIES)) for rho in RHOS[:3]),
        ("stress", None, STRESS),
        *(("rho", rho, tuple(POLICIES)) for rho in RHOS[3:]),
    ]:
        for rep in range(1, replicates + 1):
            block = f"{kind}-{rho}-r{rep}" if rho is not None else f"{kind}-r{rep}"
            rng = random.Random(f"598/{block}")
            blocks.append({"block": block, "campaign": kind, "rho": rho, "replicate": rep,
                           "trace_seed": rng.getrandbits(32),
                           "policy_order": rng.sample(policies, len(policies))})
    return blocks


@app.function(image=image, gpu=GPU, cpu=2, memory=8192, timeout=7200,
              max_containers=10, volumes={str(OUTPUT): volume})
def matched_block(block, template, batch, deadline):
    started = time.time()
    block_path = f"{batch}/{block['block']}"
    root = OUTPUT / block_path
    root.mkdir(parents=True)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,uuid",
         "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    identity = {"hostname": socket.gethostname(), "gpu_identity": gpu,
                "modal_task_id": os.getenv("MODAL_TASK_ID")}
    seed = block["trace_seed"]
    source = {**block, **identity, "max_num_seqs": 4, "trace_queue": True}
    result = {**source, "path": block_path, "runs": [], "ok": False,
              "started_at_unix": started}
    try:
        if deadline - time.time() < 360:
            raise TimeoutError("Insufficient time before GPU cutoff")
        short = next(r for r in template if r["bucket"] == "short")
        long = next(r for r in template if r["bucket"] == "long")
        calibration = [dict(row, id=i, arrival_time_s=i / 80)
                       for i, row in enumerate([short, long] * 160)]
        cal_source = {**source, "target_rate": 80, "trace_queue": False, "calibration": True}
        cal = run_gpu_benchmark("native_fcfs", calibration, cal_source, f"{block_path}/calibration")
        result["calibration"] = cal
        if not cal["ok"]:
            raise RuntimeError("Capacity calibration failed")
        summary = json.loads((OUTPUT / cal["path"] / "summary.json").read_text())
        capacity = summary["overall"]["throughput_tokens_per_s"]
        if capacity <= 0:
            raise RuntimeError("Invalid capacity estimate")
        if block["campaign"] == "stress":
            requests = p4_trace(seed)
            rate = 30
        else:
            rows = list(template)
            random.Random(seed).shuffle(rows)
            rate = block["rho"] * capacity / mean(r["max_tokens"] for r in rows)
            requests = assign_arrivals(rows, seed ^ 598, rate)
        source.update(target_rate=rate, capacity_tok_s=capacity)
        result.update(target_rate=rate, capacity_tok_s=capacity)
        result_path = root / "block.json"
        for position, policy in enumerate(block["policy_order"], 1):
            for attempt in (1, 2):
                if deadline - time.time() < 360:
                    raise TimeoutError("Insufficient time for another policy before GPU cutoff")
                metadata = {**source, "policy_position": position, "attempt": attempt}
                outcome = run_gpu_benchmark(policy, requests, metadata, block_path)
                result["runs"].append(outcome)
                result_path.write_text(json.dumps(result, indent=2) + "\n")
                volume.commit()
                print(f"{block['block']} {position}/{len(block['policy_order'])} {policy} "
                      f"passed={outcome['ok']}", flush=True)
                if outcome["ok"]:
                    break
        result["ok"] = {r["policy"] for r in result["runs"] if r["ok"]} == set(block["policy_order"])
    except Exception as error:
        result["error"] = str(error)
    finally:
        result["elapsed_s"] = time.time() - started
        (root / "block.json").write_text(json.dumps(result, indent=2) + "\n")
        volume.commit()
    return result


def download(prefix, destination):
    entries = (entry for entry in volume.listdir(prefix, recursive=True) if entry.type.name == "FILE")
    for entry in sorted(entries, key=lambda file: file.path.endswith("/block.json")):
        path = destination / entry.path
        if path.is_file() and path.stat().st_size == entry.size:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("wb") as output:
            for chunk in volume.read_file(entry.path):
                output.write(chunk)
        temporary.replace(path)


@app.local_entrypoint()
def main(replicates: int = 5, cutoff: str = "2026-11-19 20:30", workers: int = 10,
         only_blocks: str = ""):
    if not 1 <= workers <= 10 or replicates < 1:
        raise ValueError("Expected 1–10 workers and at least one replicate")
    deadline = datetime.fromisoformat(cutoff).replace(tzinfo=ZoneInfo("America/Chicago")).timestamp()
    if deadline - time.time() < 360:
        raise SystemExit("Too close to the GPU cutoff")
    plan = block_plan(replicates)
    if only_blocks:
        requested = set(only_blocks.split(","))
        plan = [block for block in plan if block["block"] in requested]
        if {block["block"] for block in plan} != requested:
            raise ValueError("Unrecognized block name")
    for label in ("variants", "rho", "stress"):
        subset = [block for block in plan if block["campaign"] == label]
        runs = sum(len(block["policy_order"]) for block in subset)
        low = (runs * 235 + len(subset) * 100) * COMPUTE_COST_PER_S
        high = (runs * 390 + len(subset) * 150) * COMPUTE_COST_PER_S
        print(f"{label}: {runs} runs + {len(subset)} calibrations; estimated "
              f"GPU/CPU/RAM cost US${low:.2f}–${high:.2f}", flush=True)
    batch = f"matched-{uuid4().hex}"
    destination = ROOT / "results/modal_matched"
    root = destination / batch
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({"workers": workers, "cutoff": cutoff,
                                               "workload_source": BALANCED, "blocks": plan}, indent=2) + "\n")
    template = read_jsonl(ROOT / BALANCED)
    print(f"Volume path: {batch} | {len(plan)} blocks | up to {workers} L40S", flush=True)

    async def drive():
        semaphore = asyncio.Semaphore(workers)
        with (root / "progress.jsonl").open("x") as journal:
            async def run(block):
                async with semaphore:
                    if deadline - time.time() < 360:
                        return
                    call = await matched_block.spawn.aio(block, template, batch, deadline)
                    try:
                        result = await call.get.aio(timeout=max(1, deadline - time.time()))
                    except (TimeoutError, modal.exception.TimeoutError):
                        await call.cancel.aio(terminate_containers=True)
                        result = {"block": block["block"], "ok": False, "error": "GPU cutoff reached"}
                    except Exception as error:
                        result = {"block": block["block"], "ok": False, "error": str(error)}
                    journal.write(json.dumps(result) + "\n")
                    journal.flush()
                    print(f"Block {block['block']} passed={result['ok']} "
                          f"capacity={result.get('capacity_tok_s', 'unavailable')}", flush=True)
                try:
                    await asyncio.to_thread(download, f"{batch}/{block['block']}", destination)
                except Exception as error:
                    print(f"Incremental download deferred: {error}", flush=True)
            await asyncio.gather(*(run(block) for block in plan))

    asyncio.run(drive())
    download(batch, destination)
    print(f"Matched results saved in {root}", flush=True)
