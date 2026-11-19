"""Run P2 diagnostics after the current P1 benchmark artifacts are complete."""

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "results/modal_formal"
P2 = ROOT / "results/modal_p2"
MODAL = [str(ROOT / ".venv/bin/modal"), "run", "-m", "gpu_experiments.modal_smoke"]
PROFILES = "profiled_fcfs,profiled_queue,profiled_aging,profiled_optimized_aging"
EXPECTED = {
    (4, (5,), 45), (16, (5,), 15), (64, (5,), 15), (256, (5,), 15),
    (4, (10, 15, 20), 90),
}


def read_json(path):
    return json.loads(path.read_text())


def load_journal(path):
    if not path.is_file():
        return []
    text = path.read_text()
    # A writer may be appending the final line while we poll.
    return [json.loads(line) for line in text[:text.rfind("\n") + 1].splitlines()]


def completed_p1(main_plan):
    complete = set()
    for plan_path in P1.glob("*/plan.json"):
        if plan_path.stat().st_mtime < main_plan.stat().st_mtime:
            continue
        plan = read_json(plan_path)
        blocks = [block for stage in plan["stages"] for block in stage]
        expected = {(b["block"], policy) for b in blocks for policy in b["policy_order"]}
        signature = (plan.get("max_num_seqs", 4), tuple(sorted({b["target_rate"] for b in blocks})),
                     len(expected))
        if signature not in EXPECTED:
            continue
        journal = load_journal(plan_path.with_name("progress.jsonl"))
        passed = {(r["block"], r["policy"]): r for r in journal if r["ok"]}
        if expected != passed.keys():
            continue
        if signature == (4, (10, 15, 20), 90) and not plan_path.with_name("complete.json").is_file():
            continue
        for record in passed.values():
            path = P1 / record["path"]
            if not all((path / f).is_file() and (path / f).stat().st_size for f in
                       ("config.json", "status.json", "summary.json", "raw_results.jsonl", "workload.jsonl")):
                break
        else:
            complete.add(signature)
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("main_plan", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    main_plan = args.main_plan.resolve(strict=True)
    if args.check:
        print(f"Completed P1 batches: {sorted(completed_p1(main_plan))}")
        return

    deadline = time.monotonic() + 5 * 3600
    last = None
    while time.monotonic() < deadline:
        complete = completed_p1(main_plan)
        if complete != last:
            print(f"P1 batch checks passed: {len(complete)}/{len(EXPECTED)}", flush=True)
            last = complete
        if complete == EXPECTED:
            break
        time.sleep(30)
    else:
        raise SystemExit("P1 did not complete all expected runs within the wait window")

    print("P1 audit and downloads verified. Starting P2 smoke tests.", flush=True)
    subprocess.run([*MODAL, "--policies", PROFILES, "--download-dir", str(P2 / "smoke")],
                   cwd=ROOT, check=True)
    previous = set(P2.glob("formal-*"))
    subprocess.run([*MODAL, "--formal", "--paired-plan", str(main_plan), "--mix", "balanced",
                    "--rates", "30", "--policies", PROFILES, "--download-dir", str(P2)],
                   cwd=ROOT, check=True)
    (batch,) = set(P2.glob("formal-*")) - previous
    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "report_analysis.modal_profile", str(batch)],
                   cwd=ROOT, check=True)
    print(f"P2 completed: {batch / 'scheduler_profiles.csv'}", flush=True)


if __name__ == "__main__":
    main()
