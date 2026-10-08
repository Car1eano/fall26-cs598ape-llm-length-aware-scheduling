# Length-Aware Request Scheduling for vLLM

CS598 Advanced Performance Engineering, Fall 2026

Team:
- Carleano Chang (poching2)
- Christine Liou (ycliou2)

## Project Goal

Study how request ordering affects LLM inference serving performance.
We compare vLLM's default FCFS scheduling with a planned length-aware
scheduler and a fairness mechanism based on aging or maximum wait time.

Metrics include throughput, Time to First Token (TTFT), and end-to-end
latency at P50, P95, and P99.

## Current Progress

Completed:
- Working vLLM CPU inference server.
- Synthetic workload generator with Poisson arrivals.
- Streaming benchmark client and metric collection.
- Aggregation of results across repeated runs.
- 35 passing unit tests.
- FCFS baseline experiments with uniform and mixed request lengths.
- Validation of 300 requests in the concurrency-limited baseline:
  zero failed requests and zero output-token-count mismatches.

Not yet implemented:
- Length-aware scheduling.
- Aging or maximum-wait-time fairness mechanism.
- Fairness metrics and larger workload experiments.

## Experiment Environment

| Component | Configuration |
|-----------|---------------|
| OS | Ubuntu 26.04.1 LTS |
| CPU | Intel Xeon Silver 4216 @ 2.10 GHz, 4 vCPUs |
| RAM | Approximately 15 GiB |
| GPU | None |
| Python | 3.12.15 |
| vLLM | 0.31.0+cpu |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Model dtype | float32 |
| KV-cache allocation | 4 GiB |
| CPU thread binding | Cores 0-2 |
| Scheduling policy | FCFS |

Prefix caching and chunked prefill were enabled in these experiments.
Results describe this CPU environment and should not be generalized
directly to GPU serving.

## Repository Structure

- `scripts/`: server startup and baseline experiment scripts.
- `workload/`: synthetic request generation and workload scenarios.
- `benchmarks/`: streaming client, metrics, and result aggregation.
- `tests/`: unit tests.
- `results/`: workloads, per-run summaries, and aggregate tables.

## Workloads

Requests use Poisson arrivals and workload seed 0.

| Scenario | Request mix | Approximate prompt tokens | Output tokens |
|----------|-------------|---------------------------|---------------|
| A_uniform | All medium | 64 | 64 |
| B_mixed | 80% short | 32 | 16 |
| B_mixed | 20% long | 256 | 128 |

Prompt lengths are approximate because prompts use repeated filler text.
The benchmark requests temperature 0 and ignore_eos to control output
length.

For the 50-request mixed workload used here, the realized mix was
37 short and 13 long requests.

## Setup

Use Python 3.12 and `uv` to create the project environment:

```bash
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install aiohttp numpy pytest
```

Install the vLLM CPU wheel used in these experiments:

```bash
uv pip install \
  "https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp38-abi3-manylinux_2_39_x86_64.whl" \
  --torch-backend cpu
```

This wheel requires a compatible Linux x86_64 environment with
glibc 2.39 or newer. A C++ compiler is also required for CPU startup
compilation. On the course VM, it was installed with:

```bash
sudo apt update
sudo apt install -y build-essential
```

## Run Unit Tests

From the repository root:

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

Verified result: 35 passed.

These tests cover workload generation, metric calculation, and
aggregation. They do not yet validate a custom scheduler.

## Reproduce the Concurrency-Limited FCFS Baseline

Run all commands from the repository root.

Start the server in a background tmux session:

```bash
tmux new -d -s server \
  "env MAX_NUM_SEQS=4 bash scripts/start_server.sh 2>&1 | tee server.log"
```

Wait until `server.log` contains `Application startup complete`.

Confirm that the running command includes `--max-num-seqs 4`:

```bash
ps -eo pid,args | grep '[v]llm serve'
```

Run 50 requests per scenario at an average arrival rate of 1 request
per second, with three repetitions per scenario:

```bash
tmux new -d -s bench_cap4 \
  "TAG=fcfs_cap4 bash scripts/run_baseline.sh 50 1.0 3 2>&1 | tee bench_cap4.log"
```

Monitor progress:

```bash
tail -f bench_cap4.log
```

Use a new TAG when repeating experiments to avoid overwriting previous
results. Server and benchmark log filenames should also be changed or
backed up when preserving an earlier run.

## Preliminary Results

The following values are arithmetic means of the per-run metrics
across three repetitions. Percentiles are averaged across runs,
rather than computed from pooled requests. Times are in seconds.
Throughput is measured in output tokens per second.

| Scenario | Group | TTFT P50 | TTFT P95 | TTFT P99 | Latency P50 | Latency P95 | Latency P99 | Tokens/s |
|----------|-------|----------|----------|----------|-------------|-------------|-------------|----------|
| A | overall | 21.49 | 33.57 | 34.76 | 28.79 | 40.82 | 42.05 | 31.8 |
| B | overall | 11.93 | 17.63 | 19.30 | 16.02 | 29.11 | 31.13 | 26.3 |
| B | short | 11.82 | 18.02 | 19.54 | 13.58 | 20.37 | 21.54 | — |
| B | long | 12.04 | 15.31 | 15.51 | 26.00 | 31.05 | 31.43 | — |

Validation:
- Two scenarios, three runs each, 50 requests per run.
- 300 recorded requests.
- Zero failed requests.
- Every recorded output-token count matched the requested max_tokens.
- Maximum sampled Running count: 4.
- Maximum sampled Waiting count: 18.

The sampled queue statistics confirm that waiting occurred under the
concurrency limit. TTFT includes queueing, prefill, and other overhead;
it is not a direct measurement of queue wait time.

These results establish an FCFS baseline. They do not yet demonstrate
an improvement from length-aware scheduling.

## Result Files

- `results/fcfs_*`: initial baseline at 0.5 requests/s without an
  explicitly configured concurrency limit.
- `results/load_fcfs_*`: earlier experiment at 1.0 requests/s.
  The intended concurrency limit of 4 was not applied.
- `results/fcfs_cap4_*`: validated experiment at 1.0 requests/s
  with max_num_seqs=4.
- `fcfs_cap4_artifact.tar.gz`: snapshot containing scripts, benchmark
  code, tests, cap4 results, and logs.

Raw per-request results and logs are excluded from normal Git tracking
by `.gitignore`; a snapshot is included in the artifact archive.
Download a copy of the archive before the course VM is removed.

## Limitations and Next Steps

Current experiments use a small model, synthetic filler prompts, one
workload seed, and a CPU-only VM. Repeated prompts can benefit from
prefix caching. Token-count validation does not establish equality of
generated token sequences across scheduling policies.

Next steps:
1. Inspect and document vLLM's waiting-queue and admission logic.
2. Define the request-length estimate and implement length-aware ordering.
3. Compare FCFS and length-aware scheduling under identical conditions.
4. Add aging or a maximum-wait-time mechanism.
5. Evaluate fairness and performance across more arrival rates,
   workload seeds, and length distributions.