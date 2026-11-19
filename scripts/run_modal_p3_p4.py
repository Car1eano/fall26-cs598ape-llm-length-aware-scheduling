"""Chain the P3 workloads and P4 long-request stress test after P2."""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workload.modal_followups import write_plan

RESULTS = ROOT / "results"
MODAL = [str(ROOT / ".venv/bin/modal"), "run", "-m", "gpu_experiments.modal_smoke"]


def p2_ready():
    for stats in (RESULTS / "modal_p2").glob("formal-*/scheduler_profiles.csv"):
        marker = stats.with_name("complete.json")
        if not marker.is_file() or json.loads(marker.read_text())["passed_runs"] != 20:
            continue
        with stats.open(newline="") as f:
            rows = list(csv.DictReader(f))
        keys = {(r["block"], r["policy"]) for r in rows if r.get("throughput_tok_s")}
        if len(keys) == 20:
            return True
    return False


def run_plan(output, plan, expected):
    before = set(output.glob("formal-*"))
    subprocess.run([*MODAL, "--formal", "--paired-plan", str(plan),
                    "--trace-queue", "--download-dir", str(output)], cwd=ROOT, check=True)
    (batch,) = set(output.glob("formal-*")) - before
    complete = json.loads((batch / "complete.json").read_text())
    if complete["passed_runs"] != expected:
        raise RuntimeError(f"Expected {expected} audited runs in {batch}")
    return batch


def analyze(batches, output):
    subprocess.run([sys.executable, "-m", "report_analysis.modal_results", *map(str, batches),
                    "--out", str(output)], cwd=ROOT, check=True)


def run_phase(phase, expected):
    output = RESULTS / f"modal_{phase}"
    plan = write_plan(output / "inputs", phase)
    batch = run_plan(output, plan, expected)
    analyze([batch], batch / "analysis")
    return batch


def resume_p3(batch):
    plan = json.loads((batch / "plan.json").read_text())
    records = [json.loads(line) for line in (batch / "progress.jsonl").read_text().splitlines()]
    passed = {(r["block"], r["policy"]) for r in records if r["ok"]}
    blocks = []
    for stage in plan["stages"]:
        for block in stage:
            policies = [p for p in block["policy_order"] if (block["block"], p) not in passed]
            if policies:
                blocks.append({**block, "policy_order": policies})
    remaining = sum(len(block["policy_order"]) for block in blocks)
    if len(passed) + remaining != 75:
        raise ValueError(f"Expected 75 P3 runs in {batch}")
    batches = [batch]
    if remaining:
        repair_plan = batch / "remaining-plan.json"
        repair_plan.write_text(json.dumps({"stages": [blocks]}, indent=2) + "\n")
        print(f"Replaying only {remaining} missing P3 runs", flush=True)
        batches.append(run_plan(batch.parent, repair_plan, remaining))
    analyze(batches, batch / "analysis")
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-p3", type=Path)
    args = parser.parse_args()
    deadline = time.monotonic() + 4 * 3600
    print("Waiting for 20 audited P2 profiles and the local P2 CSV", flush=True)
    while not p2_ready():
        if time.monotonic() >= deadline:
            raise SystemExit("P2 did not finish within the wait window")
        time.sleep(30)

    if args.resume_p3:
        p3 = resume_p3(args.resume_p3.resolve(strict=True))
    else:
        print("P2 verified. Starting P3 workload sweep (75 runs).", flush=True)
        p3 = run_phase("p3", 75)
    print(f"P3 complete at {p3}. Starting P4 starvation test (15 runs).", flush=True)
    p4 = run_phase("p4", 15)
    subprocess.run([sys.executable, "-m", "report_analysis.modal_starvation", str(p4)],
                   cwd=ROOT, check=True)
    print(f"P4 GPU tests complete: {p4 / 'long_probe_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
