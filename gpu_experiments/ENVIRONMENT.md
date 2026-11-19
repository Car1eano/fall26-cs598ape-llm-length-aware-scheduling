# GPU Experiment Environment

## Hardware

| Component | Configuration |
|---|---|
| Server | Shared Hydra GPU server |
| GPU | NVIDIA GeForce RTX 5090 |
| GPUs detected | 2 |
| NVIDIA driver | 575.57.08 |
| Driver-reported CUDA | 12.9 |

The main experiments used a shared GPU server.
The presence of two GPUs does not mean both were used
simultaneously for every benchmark.

## Software

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| vLLM | 0.26.0 |
| Model | Qwen/Qwen2.5-0.5B-Instruct |

## Main GPU Workloads

| Scenario | Short | Long | Total |
|---|---:|---:|---:|
| Fairness | 813 | 87 | 900 |
| Balanced | 455 | 445 | 900 |

Both use a target Poisson arrival rate of
30 requests per second.

## Benchmark Implementation

- Runner: `benchmarks/run_benchmark.py`
- HTTP client: `benchmarks/client.py`
- Raw output: `raw_results.jsonl`
- Summary output: `summary.json`

## Reproducing Statistical Analysis

Run these commands from the repository root:

    python report_analysis/analyze.py
    python report_analysis/plot_results.py
    python report_analysis/sensitivity_analysis.py

These commands reproduce the statistical analysis using
saved benchmark results without rerunning GPU inference.

## Reproducibility Limitations

Performance may vary due to shared GPU usage,
runtime settings, model configuration, and vLLM version.

The smoke-test startup scripts are not verified to match
the full 900-request benchmark configuration.

The complete original GPU server launch commands
have not yet been independently verified.

## Modal L40S pilot

The Modal pilot uses one L40S per server, vLLM 0.26.0, float16, eager
execution, disabled prefix caching, a 2,048-token context limit and
`--max-num-seqs 4`. It preserves the 900 requests and rescales their
arrival times by 30 divided by the target rate. Each independent server
receives one short and one long warm-up request before measurement.

From the repository root, using an authenticated Modal environment:

```bash
.venv/bin/modal run -m gpu_experiments.modal_smoke --pilot-rates 2,5,30
.venv/bin/python report_analysis/modal_pilot.py
```

The first command allocates GPU resources and starts one FCFS run per
mix and rate. The second reads downloaded results offline and prints CSV.
The queue-time column is the change in the server histogram sum divided
by the change in count. The analysis verifies coverage against the number
of measured requests. Each run retains both metric snapshots, the exact
workload, server command, logs, per-request results and audit status.

## Modal formal benchmark and P1 follow-ups

The formal driver records its randomized policy order and trace seeds in
`plan.json`. The three policies share each trace. The runner uses at most
five independent single-L40S containers and restarts the server per run.
The original Modal batches launched before this change were capped at four workers.
Every attempt has its own Volume folder and the local journal records
whether it passed the request audit.

From the repository root with Modal authenticated:

```bash
.venv/bin/modal run -m gpu_experiments.modal_smoke --formal
```

Replay a subset of the existing plan at a different concurrency cap:

```bash
.venv/bin/modal run -m gpu_experiments.modal_smoke --formal \
  --paired-plan results/modal_formal/formal-0be5c458493145cdb43b397d76833f3e/plan.json \
  --mix balanced --rates 5 --max-num-seqs 16
```

Without a paired plan, `--rates 10,15,20` generates new blocks at those
rates for both mixes. `--policies` accepts a comma-separated list of policy
names. Use it on its own to smoke-test those policies with eight requests,
or add it to a formal run to select the strategies in each block.

Additional policies are `native_priority`, `prompt_only`, `noisy_length`,
`aging_alpha1`, `aging_alpha2`, `aging_alpha5`, `aging_alpha10`,
`wait_promotion` and `hrrn`. The native priority baseline passes tokenized
prompt length plus maximum completion length to vLLM. Noisy Length draws a
fixed per-request multiplier from [0.7, 1.3] using the trace seed. Wait
Promotion moves requests that have waited at least ten seconds to the
front of their respective waiting queues in arrival order. Admission is
still subject to vLLM capacity and queue selection. HRRN uses total token
count as a service-cost proxy.

Aggregate runs locally with:

```bash
.venv/bin/python -m report_analysis.modal_results \
  results/modal_formal/formal-0be5c458493145cdb43b397d76833f3e \
  --out results/modal_analysis
```

The mean queue time and interval containing the run maximum come from
server-side histogram deltas. Per-request TTFT and end-to-end latency
come from the client trace. The TTFT SLO fraction is the share below one
second. Goodput is the number meeting that SLO per benchmark second.
Run `modal_smoke --calibrate` through the same Modal CLI to collect
isolated FCFS latencies. Pass the downloaded run directory with
`--calibration` to include class-specific slowdown and Jain's index on
the inverse mean slowdown of the two classes.

For the full follow-up sequence, use
`bash scripts/run_modal_p1.sh MAIN_PLAN`. The script first runs the paired
cap-16 comparison, then checks the new schedulers and trace collection,
measures isolated service times, and runs the selected policy, concurrency-cap
and arrival-rate sweeps. Supply an existing cap-16 batch directory as the
optional second argument to reuse that in-progress comparison.

`--trace-queue` enables a container-local OTLP receiver. It retains one
queue-time record per measured request in `queue_times.jsonl` and checks
request ID coverage. This provides exact maximum queue times for traced
runs. For the original 90 runs the histogram bounds remain the available
maximum-wait measurement.

## P2 scheduler profiling

```bash
.venv/bin/python scripts/run_modal_p2.py MAIN_PLAN
```

The launcher checks the audit journal and downloaded artifacts for the
five P1 follow-up batches, then runs scheduler smoke tests and replays
the five balanced, 30 req/s traces. Each trace runs under profiled FCFS,
Queue-Only, Aging and Aging with unchanged-queue rewrites skipped.
The optimized scheduler still computes the full time-dependent stable
sort on each call. Results and the summary CSV go to `results/modal_p2/`.

The scheduler records cumulative CPU timings and scheduled batch composition
in `scheduler_profile.jsonl` every 128 nonempty steps. The summary uses the last snapshot,
so it may omit a final partial block. Sampled NVIDIA GPU utilization spans
the measured workload. The scheduled prompt-token share represents batch
composition, while GPU utilization represents device-wide activity.

## P3 workload sensitivity and P4 long-request stress test

With an audited P2 batch and its scheduler profile CSV in `results/modal_p2/`,
run the following command from the repo root:

```bash
.venv/bin/python scripts/run_modal_p3_p4.py
```

The launcher writes reproducible input plans to `results/modal_p3/inputs/`
and `results/modal_p4/inputs/`. It runs 75 workload-sensitivity trials,
then 15 trials with long probes under sustained short arrivals. Each batch
downloads its audited runs and per-request queue traces before creating the
analysis CSVs. To recover a partial P3 batch while retaining successful
runs, pass its directory with `--resume-p3`.

## Final report

The final source is `report/final_report.tex`. The figures use the audited
Modal data in `results/modal_analysis/individual_runs.csv`. From the repo
root with Matplotlib and Tectonic installed:

```bash
.venv/bin/python -m report_analysis.plot_modal_results \
  results/modal_analysis/individual_runs.csv --out report/figures
mkdir -p output/pdf
.venv/bin/tectonic --untrusted --outdir output/pdf report/final_report.tex
```
