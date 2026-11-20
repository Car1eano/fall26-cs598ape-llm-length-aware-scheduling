# Length-Aware Request Scheduling for vLLM

**CS598 Advanced Performance Engineering — Fall 2026**

**Team**
- Carleano Chang (`poching2`)
- Christine Liou (`ycliou2`)


The [final report PDF](output/pdf/final_report.pdf) and its [LaTeX source](report/final_report.tex) describe the capacity-calibrated Modal study.

## Current Modal experiments

The current environment-controlled comparison uses one NVIDIA L40S per
Modal container, vLLM 0.26.0, Qwen2.5-0.5B-Instruct, and four concurrent
sequences. Each block starts with a native FCFS capacity calibration on
a fixed 320-request saturated trace. The block restarts vLLM for each
policy, randomizes policy order, and keeps GPU UUID, driver, visible
memory, hostname, and workload together. For the balanced trace, the
target rate is `rho * capacity_tok_s / mean_requested_output_tokens`.
Each workload level uses five seeded blocks. Modal may reuse an idle
container across blocks. Policy effects are computed within blocks and
the environment labels remain available for cross-block comparisons.
The completed campaign contains 35 audited blocks, 175 policy runs,
and 35 capacity calibrations.

The core comparison uses FCFS, Queue-Only, and Aging alpha=0.5. The
variant comparison at rho=1.1 adds Aging alpha=0.1, 1, 2, 5, and 10,
native priority, prompt-only length, noisy length, wait promotion,
HRRN, and Hybrid. The stress comparison adds a five-second
waiting-priority rule to FCFS, Queue-Only, Aging, ten-second promotion,
and HRRN.

The five-second rule moves overdue requests ahead of younger requests
at the next eligible admission. Native vLLM 0.26.0 schedules running
requests before admission and applies a shared running-sequence limit,
token budget, and KV allocation checks. A dedicated oldest-request
slot would require changing that admission machinery, including slot
accounting for skipped or resumed requests. This experiment retains
native admission and uses priority promotion.

### Hardware and software requirements

| Task | Requirements |
|---|---|
| Matched GPU experiments | An authenticated Modal CLI, available compute credits, and one NVIDIA L40S per concurrent block (at least 46,068 MiB visible in the recorded runs). The `vllm/vllm-openai:v0.26.0` image supplies Python 3.12, vLLM 0.26.0, and PyTorch 2.11.0+cu130 as recorded for the matched runs. |
| Offline analysis and figures | Python 3.12, NumPy, Matplotlib, and the saved results. Runs on CPU without a Modal account. Tectonic is used separately to compile the report PDF. |
| Workload, metrics, and aggregation tests | Python 3.12, pytest, and NumPy on CPU. |
| Full test suite, including scheduler tests | The pinned vLLM 0.26.0 image and pytest 9.0.2. Verified in a Modal CPU container using both scheduler test files. |

From the repository root, set up the local analysis environment with:

```bash
python3.12 -m venv .venv
.venv/bin/pip install modal numpy matplotlib pytest
```

To reproduce the experiment and run the full suite, authenticate the
Modal CLI and replace the cutoff placeholder with a future date and
time in America/Chicago. One worker needs one L40S at a time. The
recorded campaign used ten concurrent workers.

```bash
.venv/bin/modal run -m gpu_experiments.modal_matched \
  --workers 1 --cutoff "YYYY-MM-DD HH:MM"
.venv/bin/modal run -m gpu_experiments.modal_tests
```

The GPU runner records the plan and replays the repository's balanced
900-request workload template alongside seeded stress workloads. To
rebuild the tables and figures from existing results on CPU:

```bash
.venv/bin/python -m report_analysis.matched_results \
  results/modal_matched/matched-* --out results/modal_matched_analysis
.venv/bin/python -m report_analysis.matched_evidence results/modal_matched_analysis
.venv/bin/python -m report_analysis.plot_matched \
  --matched results/modal_matched_analysis/individual_runs.csv \
  --out report/figures/matched
```

Use `--cutoff "YYYY-MM-DD HH:MM"` to set a GPU deadline in America/Chicago
and `--only-blocks` to replay named incomplete blocks. Usage estimates are
printed before each group is submitted. Each block and attempt retains
its own output path. The full suite has **56 passing tests** in the
vLLM 0.26.0 Modal image, including the real scheduler imports and the
five-second promotion tests. The per-file counts are recorded in
[`results/test_report.txt`](results/test_report.txt).

The analysis directory contains the run audit, per-block FCFS differences,
the matched policy summaries in the report’s Table 5, each stress probe's TTFT, and a claim-to-run index. The
queue-depth panel uses vLLM's periodic waiting-count logs at rho=1.1.
`queue_depth_samples.csv` preserves those samples for replotting. If the
logs contain no complete comparison, the queue-depth panel is omitted.

### Historical results and the environment issue

The original RTX 5090 experiment and the first Modal sweep are preserved
as exploratory results. In the first Modal sweep, the same balanced FCFS
configuration at 5 requests/s produced short P95 TTFT of 1.19 s in its
one driver-610 run and a mean of 58.19 s across four driver-620 runs.
Prompt-only and Aging alpha=1 and 5 each ran only in the faster
environment in this comparison. Their low pooled long-P95 values describe
that assignment. Reproducing a historical fixed-rate result near 5
requests/s is sensitive to the host's capacity. The current experiment
calibrates capacity per block and compares policies within the same
container.

## 1. Project Overview

Large Language Model (LLM) inference requests have different prompt and output lengths. Under heavy load, request scheduling affects responsiveness, latency, and throughput.

This project investigates length-aware scheduling for vLLM. We compare native scheduling with custom policies that prioritize shorter estimated requests and use waiting-time-based priority adjustments.

Our objective is to understand the trade-offs between short-request responsiveness, long-request latency, and overall throughput.

The project includes scheduler implementations, synthetic workload generation, unit tests, GPU experiments, statistical analysis, and reproducible benchmark summaries.

## 2. Scheduling Policies

Five policies were evaluated in the early RTX 5090 exploration.

| Policy | Description |
|---|---|
| Native FCFS | Baseline using vLLM's native FCFS scheduling configuration |
| Queue-Only | Reorders waiting requests by estimated request length |
| Aging alpha=0.1 | Length-aware priority adjusted using waiting time, with alpha=0.1 |
| Aging alpha=0.5 | Length-aware priority adjusted using waiting time, with alpha=0.5 |
| Hybrid | Conditional length-aware and aging-based scheduling |

The Queue-Only implementation estimates request length using the prompt-token count and requested maximum output-token count.

Waiting requests are reordered before the underlying vLLM scheduler performs scheduling.

The aging-based implementation uses a priority heuristic based on estimated request length and elapsed waiting time.

These heuristics aim to improve responsiveness and mitigate long-request delays, but do not provide a formal starvation-freedom guarantee.

## 3. Repository Structure

| Path | Description |
|---|---|
| `gpu_experiments/` | Scheduler implementations, Modal runners, smoke tests, and profiling |
| `report_analysis/matched_*.py` | Current block audit, paired analysis, and claim-to-run index |
| `report_analysis/plot_matched.py` | Figures for the current Modal study |
| `report_analysis/analyze.py` | Main benchmark aggregation and audit |
| `report_analysis/plot_results.py` | Performance figure generation |
| `report_analysis/sensitivity_analysis.py` | Analysis of an anomalous Queue-Only run |
| `report_analysis/raw_results/` | Preserved GPU benchmark summary JSON files |
| `report_analysis/figures/` | Performance figures in PDF and PNG formats |
| `tests/` | Workload, metrics, aggregation, and scheduler unit tests |
| `workload/` | Synthetic workload generation |
| `benchmarks/` | Streaming benchmark client and metric collection |
| `scripts/` | CPU baseline scripts and drivers for the Modal follow-up batches |
| `results/modal_matched/` | Current Modal plans, calibration traces, and per-run artifacts |
| `results/modal_matched_analysis/` | Audited CSVs, paired results, and billing records |
| `results/test_report.txt` | Full unit-test verification and per-file counts |
| `results/` | Current and historical experiment results |
| `report/` | LaTeX source, result notes, and report figures |
| `artifacts/` | Archived experiment artifacts |

## 4. Experimental Environments

### 4.1 Preliminary CPU Experiments

The initial baseline experiments used the following environment:

| Component | Configuration |
|---|---|
| Operating system | Ubuntu 26.04.1 LTS |
| CPU | Intel Xeon Silver 4216 |
| Available CPU resources | 4 vCPUs |
| Memory | Approximately 15 GiB |
| Python | 3.12.15 |
| vLLM | 0.31.0+cpu |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| GPU | None |

These experiments validated the workload generator, benchmark client, metrics collection, and FCFS baseline.

### 4.2 Early RTX 5090 Exploration

The early scheduling comparisons were conducted on a shared GPU server.

| Component | Configuration |
|---|---|
| GPU | NVIDIA RTX 5090 |
| vLLM | 0.26.0 |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Requests per run | 900 |
| Target arrival rate | 30 requests/second |
| Workload type | Synthetic mixed-length requests |

The preliminary CPU and main GPU experiments used different hardware and vLLM versions. Their absolute performance measurements should not be directly compared.

### 4.3 GPU Smoke-Test Configuration

The repository includes:

- `gpu_experiments/start_gpu_smoke_test.sh`
- `gpu_experiments/start_length_aware_smoke_test.sh`

The scripts configure local vLLM servers with the following smoke-test settings:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Data type: `float16`
- Maximum concurrent sequences: 4
- Maximum model length: 2048
- GPU memory utilization: 0.30
- Eager execution enabled
- Localhost-only API server

The length-aware smoke-test script also disables prefix caching and specifies a custom scheduler class.

These scripts are preliminary smoke tests. Their settings should not be assumed to exactly reproduce the main 900-request GPU benchmarks.

GPU device allocation and port selection must be coordinated with the administrator of the shared server.

## 5. Workload Design

Synthetic workloads use Poisson request arrivals and two request classes.

| Request class | Approximate prompt length | Requested output length |
|---|---:|---:|
| Short | 32 tokens | 16 tokens |
| Long | 256 tokens | 128 tokens |

The early RTX 5090 experiments evaluate two request distributions.

| Scenario | Short requests | Long requests | Total |
|---|---:|---:|---:|
| `fairness_900_rate30` | 813 | 87 | 900 |
| `balanced_900_rate30_seed123` | 455 | 445 | 900 |

The early RTX 5090 runs used a target arrival rate of 30 requests per second.

## 6. Evaluation Metrics

The benchmark collects:

- Time to First Token (TTFT)
- End-to-end request latency
- P50, P95, and P99 latency percentiles
- Generated output-token throughput
- Separate short-request and long-request measurements

TTFT includes queueing and processing overhead and is not a direct measurement of queue waiting time alone.

Aggregate results are arithmetic means of per-run measurements. The analysis also reports sample standard deviations.

Percentiles are computed within each run and then averaged across runs, rather than calculated from pooled requests.

## 7. Early RTX 5090 Results

### 7.1 Fairness Workload: Approximately 90% Short Requests

| Policy | Runs | Throughput (tokens/s) | Short P95 TTFT (s) | Long P95 TTFT (s) |
|---|---:|---:|---:|---:|
| Native FCFS | 3 | 573.74 | 10.107 | 10.056 |
| Queue-Only | 5 | 561.64 | 0.590 | 19.012 |
| Aging alpha=0.1 | 3 | 553.64 | 0.644 | 19.551 |
| Aging alpha=0.5 | 5 | 573.04 | 0.680 | 18.266 |
| Hybrid | 3 | 560.01 | 0.629 | 18.880 |

Queue-Only reduces short-request P95 TTFT by approximately **94.2%** compared with Native FCFS.

However, long-request P95 TTFT increases from approximately 10.06 to 19.01 seconds.

This demonstrates a significant trade-off between short-request responsiveness and long-request latency.

### 7.2 Balanced Workload: Approximately 50% Short Requests

| Policy | Runs | Throughput (tokens/s) | Short P95 TTFT (s) | Long P95 TTFT (s) |
|---|---:|---:|---:|---:|
| Native FCFS | 3 | 577.49 | 79.310 | 77.468 |
| Queue-Only | 5 | 582.80 | 0.664 | 77.282 |
| Aging alpha=0.1 | 3 | 571.43 | 0.721 | 79.618 |
| Aging alpha=0.5 | 5 | 585.54 | 1.322 | 76.815 |
| Hybrid | 3 | 556.50 | 6.315 | 82.219 |

Queue-Only reduces short-request P95 TTFT by approximately **99.2%** compared with Native FCFS.

The effect on long-request latency and throughput depends on the workload and scheduling policy.

These results demonstrate improvements in short-request responsiveness, not universal improvements for all requests.

### 7.3 Performance Figures

The following figures are available in PDF and PNG formats:

- `report_analysis/figures/short_p95_ttft.pdf`
- `report_analysis/figures/long_p95_ttft.pdf`
- `report_analysis/figures/throughput.pdf`

The figures compare all five policies across both workload scenarios and include variability across repeated runs.

## 8. Testing

The project has **56 passing unit tests in the Modal vLLM image**.

The test suite covers:

- Synthetic workload generation
- Benchmark metric calculations
- Statistical aggregation
- Queue-Only scheduling behavior
- Aging-based scheduler behavior

Run tests from the repository root using a compatible Python environment:

```bash
python -m pytest tests/ -v
```

The main GPU experiments provide larger-scale performance evaluation using 900 requests per run.

Unit tests do not establish exhaustive correctness under every runtime configuration, nor do they prove starvation freedom.

### 8.1 Running the GPU Benchmark Client

The benchmark runner sends a JSONL workload to an already
running vLLM OpenAI-compatible completions server.

The runner accepts four arguments:

- `--workload`: input workload JSONL file
- `--model`: model identifier
- `--url`: completions API endpoint
- `--out-dir`: output directory

Example command, after starting a compatible vLLM server:

    python -m benchmarks.run_benchmark \
      --workload gpu_experiments/results/fairness_900_rate30/workload.jsonl \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --url http://127.0.0.1:8000/v1/completions \
      --out-dir /tmp/vllm-benchmark-example

Replace the workload path and URL as needed.

The benchmark generates:

- `raw_results.jsonl`: per-request measurements
- `summary.json`: aggregate performance statistics

The client uses asynchronous HTTP streaming and schedules
requests according to their specified arrival times.

### GPU Scheduler Implementations

| Policy | Implementation |
|---|---|
| Native FCFS | vLLM default scheduler |
| Queue-Only | `gpu_experiments/length_aware_queue_only_scheduler.py` |
| Aging alpha=0.1 | `gpu_experiments/adaptive_length_aware_scheduler.py` |
| Aging alpha=0.5 | Aging scheduler configured with alpha=0.5 |
| Hybrid | `gpu_experiments/hybrid_conditional_aging_scheduler.py` |

Custom schedulers reorder waiting requests before invoking
the underlying vLLM scheduling logic.

The existing GPU startup scripts are smoke-test configurations,
not verified complete reproductions of the formal experiments.

See `gpu_experiments/ENVIRONMENT.md` for environment details.

## 9. Reproducing the Statistical Analysis

The repository contains 40 preserved GPU benchmark summary files.

Of these, 38 runs are used in the primary policy comparisons. Two additional runs are retained for auditability: an original anomalous Queue-Only run and an instrumented scheduler run.

### 9.1 Install Analysis Dependencies

Use a compatible Python environment with NumPy and Matplotlib:

```bash
python -m pip install numpy matplotlib
```

The aggregation and sensitivity-analysis scripts use Python standard-library modules.

### 9.2 Generate Statistical Summaries

From the repository root:

```bash
python report_analysis/analyze.py
```

This generates:

- `report_analysis/individual_runs.csv`
- `report_analysis/summary_statistics.csv`
- `report_analysis/run_audit.csv`

### 9.3 Generate Performance Figures

```bash
python report_analysis/plot_results.py
```

This generates the short-request TTFT, long-request TTFT, and throughput comparison figures.

### 9.4 Run Sensitivity Analysis

```bash
python report_analysis/sensitivity_analysis.py
```

This generates:

- `report_analysis/sensitivity_analysis.csv`

These commands reproduce the reported statistical analysis using the preserved JSON summaries, without requiring access to the original GPU server.

They do not rerun the GPU inference experiments themselves.

## 10. Sensitivity Analysis

One original Queue-Only run in the Fairness workload showed noticeably lower throughput than the other repetitions.

Both the original run and its retest are preserved.

| Run selection | Throughput (tokens/s) | Short P95 TTFT (s) | Long P95 TTFT (s) |
|---|---:|---:|---:|
| Original five runs | 547.73 | 0.637 | 19.997 |
| Retest five runs | 561.64 | 0.590 | 19.012 |
| All six runs | 549.12 | 0.633 | 19.885 |

The primary comparison uses the retest instead of the anomalous original run.

The short-request improvement relative to FCFS remains substantial under all three selections.

The original measurement is retained rather than deleted, allowing the effect of this selection to be independently examined.

## 11. Preliminary CPU Baseline

Before implementing the GPU schedulers, we established a concurrency-limited CPU FCFS baseline.

The validated preliminary experiments used:

- Two workload scenarios
- Three repetitions per scenario
- 50 requests per repetition
- 300 recorded requests
- Maximum concurrent sequences: 4

All 300 recorded requests completed successfully, and the recorded output-token counts matched the requested counts.

The CPU baseline used vLLM 0.31.0+cpu and is preserved in the repository's historical `results/` directory.

These measurements were used for initial validation and are not directly compared with the later GPU scheduler results.

## 12. Limitations

The project has several limitations:

1. The experiments use a relatively small LLM.
2. Synthetic workloads may not represent production request distributions.
3. The early RTX 5090 comparisons use two workload distributions and limited repetitions.
4. Experiments were performed on shared GPU infrastructure, where external activity may affect measurements.
5. Prioritizing shorter requests can substantially increase long-request latency.
6. Aging-based priority heuristics do not formally guarantee starvation freedom.
7. The preserved JSON summaries support statistical reproducibility but do not include every original per-request trace or runtime log.
8. The exact launch settings for the early RTX 5090 runs were not preserved.

Results should not be generalized to arbitrary models, hardware configurations, or production workloads.

## 13. Conclusion

Length-aware scheduling can substantially improve short-request responsiveness in LLM serving.

Across the two early RTX 5090 workloads of 900 requests each, Queue-Only scheduling reduced short-request P95 TTFT by approximately 94.2% and 99.2% compared with Native FCFS.

However, these gains must be evaluated alongside long-request latency and throughput.

The experiments demonstrate that the effectiveness of a scheduling policy depends on the workload distribution and the performance objective.
