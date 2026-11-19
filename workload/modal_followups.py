"""Generate deterministic paired workloads for the Modal P3 and P4 experiments."""

import json
import math
import random
from pathlib import Path

from workload.generator import make_prompt, write_jsonl

N_REQUESTS = 900
REPETITIONS = 5
POLICIES = ["native_fcfs", "queue_only", "aging_alpha05"]
P3_SCENARIOS = ("short_10pct", "short_30pct", "short_70pct", "lognormal_lengths", "bursty_balanced")
P4_SCENARIO = "sustained_short_arrivals"


def poisson_arrivals(rate, rng):
    arrival = 0.0
    times = []
    for _ in range(N_REQUESTS):
        arrival += rng.expovariate(rate)
        times.append(arrival)
    return times


def two_sizes(short_count, rng):
    groups = [("short", 32, 16)] * short_count + [("long", 256, 128)] * (N_REQUESTS - short_count)
    rng.shuffle(groups)
    return groups


def continuous_lengths(rng):
    lengths = [max(16, min(512, round(rng.lognormvariate(math.log(96), 0.85))))
               for _ in range(N_REQUESTS)]
    ordered = sorted(range(N_REQUESTS), key=lengths.__getitem__)
    buckets = {}
    for index, position in enumerate(ordered):
        buckets[position] = ("short", "medium", "long")[index // (N_REQUESTS // 3)]
    return [(buckets[i], n, max(8, n // 2)) for i, n in enumerate(lengths)]


def p3_trace(scenario, seed):
    arrivals = poisson_arrivals(5, random.Random(seed))
    rng = random.Random(seed ^ 0x598)
    if scenario == "lognormal_lengths":
        sizes = continuous_lengths(rng)
    elif scenario == "bursty_balanced":
        sizes = two_sizes(455, rng)
        # Thirty bursts of thirty requests each, with within-burst spacing
        # spread across one second. Match the Poisson trace's total span.
        bursts = sorted(wave * 6 + rng.random() for wave in range(30) for _ in range(30))
        span = (arrivals[-1] - arrivals[0]) / (bursts[-1] - bursts[0])
        arrivals = [arrivals[0] + (time - bursts[0]) * span for time in bursts]
    else:
        percent = {"short_10pct": 10, "short_30pct": 30, "short_70pct": 70}[scenario]
        sizes = two_sizes(N_REQUESTS * percent // 100, rng)
    return make_rows(arrivals, sizes)


def p4_trace(seed):
    rng = random.Random(seed)
    arrivals = poisson_arrivals(30, rng)
    step = arrivals[-1] / 6
    # One long probe at each of twelve early positions while short traffic
    # continues afterwards. The other 888 requests are short.
    probes = set(range(12, 12 * 12 + 1, 12))
    sizes = [("long", 256, 128) if i in probes else ("short", 32, 16)
             for i in range(N_REQUESTS)]
    assert max(arrivals[i] for i in probes) < step * 2
    return make_rows(arrivals, sizes)


def make_rows(arrivals, sizes):
    return [{"id": i, "arrival_time_s": arrival, "bucket": bucket,
             "prompt": make_prompt(prompt), "max_tokens": output}
            for i, (arrival, (bucket, prompt, output)) in enumerate(zip(arrivals, sizes, strict=True))]


def write_plan(directory, phase):
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(3504 if phase == "p3" else 4504)
    scenarios = P3_SCENARIOS if phase == "p3" else (P4_SCENARIO,)
    blocks = []
    for replicate in range(1, REPETITIONS + 1):
        seed = rng.getrandbits(32)
        for scenario in scenarios:
            name = f"{scenario}-r{replicate}"
            path = directory / f"{name}.jsonl"
            trace = p3_trace(scenario, seed) if phase == "p3" else p4_trace(seed)
            write_jsonl(trace, path)
            blocks.append({
                "block": name, "mix": scenario, "replicate": replicate, "trace_seed": seed,
                "trace_file": str(path), "target_rate": 5 if phase == "p3" else 30,
                "policy_order": rng.sample(POLICIES, len(POLICIES)),
                "warmup": "one short and one long, concurrent",
            })
    rng.shuffle(blocks)
    plan = directory / "plan.json"
    plan.write_text(json.dumps({"phase": phase, "stages": [blocks]}, indent=2) + "\n")
    return plan
