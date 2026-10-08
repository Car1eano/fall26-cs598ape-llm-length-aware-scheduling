"""Generate Poisson-arrival workloads: python -m workload.generator --help."""
import argparse
import json
import random
from pathlib import Path
from workload.scenarios import SCENARIOS


def make_prompt(num_tokens: int) -> str:
    """Create a repeatable filler prompt; token count is approximate."""
    return " ".join(["hello"] * num_tokens)


def generate_workload(scenario: str, num_requests: int, rate: float, seed: int = 0):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if rate <= 0 or num_requests < 0:
        raise ValueError("rate must be positive and num_requests nonnegative")
    spec = SCENARIOS[scenario]
    rng = random.Random(seed)
    rows = []
    arrival = 0.0
    for i in range(num_requests):
        arrival += rng.expovariate(rate)
        _, bucket, prompt_tokens, max_tokens = rng.choices(
            spec, weights=[item[0] for item in spec], k=1
        )[0]
        rows.append({"id": i, "arrival_time_s": arrival,
                     "prompt": make_prompt(prompt_tokens),
                     "max_tokens": max_tokens, "bucket": bucket})
    return rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = generate_workload(args.scenario, args.num_requests, args.rate, args.seed)
    write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} requests to {args.out}")


if __name__ == "__main__":
    main()
