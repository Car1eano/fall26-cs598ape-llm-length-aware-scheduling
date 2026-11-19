"""Run the paired scheduler experiment on the Modal benchmark runner."""

import asyncio
import json
import random
import subprocess
from pathlib import Path
from uuid import uuid4

from gpu_experiments.modal_smoke import MAX_WORKERS, POLICIES, ROOT, download_batch, smoke
from workload.generator import read_jsonl

MIXES = {
    "short_heavy": "fairness_900_rate30",
    "balanced": "balanced_900_rate30_seed123",
}
STAGES = ((2, 5), (30,))
REPETITIONS = 5
SEED = 598


def make_plan(stages=STAGES):
    rng = random.Random(SEED)
    plans = []
    for rates in stages:
        blocks = []
        for mix, original in MIXES.items():
            path = Path("gpu_experiments/results") / original / "workload.jsonl"
            template = read_jsonl(ROOT / path)
            for replicate in range(1, REPETITIONS + 1):
                seed = rng.getrandbits(32)
                for rate in rates:
                    order = list(POLICIES)
                    rng.shuffle(order)
                    blocks.append({
                        "block": f"{mix}-{rate}rps-r{replicate}",
                        "mix": mix, "target_rate": rate, "replicate": replicate,
                        "trace_seed": seed, "policy_order": order,
                        "workload_source": str(path),
                        "warmup": "one short and one long, concurrent",
                    })
            assert len(template) == 900
        rng.shuffle(blocks)
        plans.append(blocks)
    return plans


def make_trace(block):
    if "trace_file" in block:
        return read_jsonl(ROOT / block["trace_file"])

    # Shuffle the original mix, preserving its exact short and long counts.
    # Use the same random draws within a block for every scheduling policy.
    rows = read_jsonl(ROOT / block["workload_source"])
    rng = random.Random(block["trace_seed"])
    rng.shuffle(rows)
    arrival = 0.0
    for index, row in enumerate(rows):
        arrival += rng.expovariate(block["target_rate"])
        row.update(id=index, arrival_time_s=arrival)
    return rows


async def run_stage(blocks, batch_id, source, journal):
    slots = asyncio.Semaphore(MAX_WORKERS)

    async def run_block(block):
        requests = make_trace(block)
        for position, policy in enumerate(block["policy_order"], start=1):
            for attempt in (1, 2):
                metadata = {**source, **block, "policy_position": position, "attempt": attempt}
                async with slots:
                    try:
                        outcome = await smoke.remote.aio(policy, requests, metadata, batch_id)
                    except Exception as error:
                        outcome = {"ok": False, "error": str(error), "policy": policy}
                record = {**outcome, "block": block["block"], "attempt": attempt}
                journal.write(json.dumps(record) + "\n")
                journal.flush()
                print(json.dumps({k: v for k, v in record.items() if k != "error"}), flush=True)
                if outcome["ok"]:
                    break

    await asyncio.gather(*(run_block(block) for block in blocks))


def select_plans(paired_plan, mix, rates, policies=None):
    selected_rates = {float(rate) for rate in rates.split(",")} if rates else set()
    if not all(0 < rate <= 1000 for rate in selected_rates):
        raise ValueError("Expected positive arrival rates up to 1000 req/s")
    if paired_plan:
        plans = json.loads(Path(paired_plan).read_text())["stages"]
    else:
        stages = (tuple(sorted(selected_rates)),) if selected_rates else STAGES
        plans = make_plan(stages)
    selected = [
        [block for block in stage if (not mix or block["mix"] == mix)
         and (not selected_rates or block["target_rate"] in selected_rates)]
        for stage in plans
    ]
    selected = [stage for stage in selected if stage]
    if not selected:
        raise ValueError("No planned runs match the selected mix and arrival rates")
    if policies is not None:
        rng = random.Random(SEED)
        for stage in selected:
            for block in stage:
                block["policy_order"] = rng.sample(policies, len(policies))
    return selected


def run_formal(download_dir, max_num_seqs=4, paired_plan="", mix="", rates="", policies=None, trace_queue=False):
    plans = select_plans(paired_plan, mix, rates, policies)
    batch = f"formal-{uuid4().hex}"
    destination = Path(download_dir) if download_dir else ROOT / "results/modal_formal"
    root = destination / batch
    root.mkdir(parents=True)
    source = {
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "plan_seed": SEED,
        "max_num_seqs": max_num_seqs,
        "trace_queue": trace_queue,
        "paired_plan": paired_plan or None,
    }
    (root / "plan.json").write_text(json.dumps({**source, "stages": plans}, indent=2) + "\n")
    with (root / "progress.jsonl").open("x") as journal:
        for index, blocks in enumerate(plans, start=1):
            stage = f"{batch}/stage{index}"
            print(f"Stage {index}: {sum(len(b['policy_order']) for b in blocks)} runs; Volume path: {stage}",
                  flush=True)
            asyncio.run(run_stage(blocks, stage, source, journal))
            download_batch(stage, destination)
            print(f"Downloaded {root / f'stage{index}'}", flush=True)

    records = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    expected = {(block["block"], policy) for blocks in plans for block in blocks
                for policy in block["policy_order"]}
    passed = {(record["block"], record["policy"]) for record in records if record["ok"]}
    if expected != passed:
        raise SystemExit(f"Missing valid runs: {sorted(expected - passed)}")
    (root / "complete.json").write_text(json.dumps({"passed_runs": len(passed)}) + "\n")
    print(f"All {len(passed)} formal runs passed. Results: {root}", flush=True)
