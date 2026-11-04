# Length-Aware Request Scheduling for vLLM

**CS598 Advanced Performance Engineering — Fall 2026**

**Team**
- Carleano Chang (`poching2`)
- Christine Liou (`ycliou2`)

## 1. Project Overview

Large Language Model (LLM) inference requests have different prompt and output lengths. Under heavy load, request scheduling affects responsiveness, latency, and throughput.

This project investigates length-aware scheduling for vLLM. We compare native scheduling with custom policies that prioritize shorter estimated requests and use waiting-time-based priority adjustments.

Our objective is to understand the trade-offs between short-request responsiveness, long-request latency, and overall throughput.

The project includes scheduler implementations, synthetic workload generation, unit tests, GPU experiments, statistical analysis, and reproducible benchmark summaries.

## 2. Scheduling Policies

Five policies are evaluated in the main GPU experiments.

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
| `gpu_experiments/` | GPU scheduler implementations and smoke-test scripts |
| `report_analysis/analyze.py` | Main benchmark aggregation and audit |
| `report_analysis/plot_results.py` | Performance figure generation |
| `report_analysis/sensitivity_analysis.py` | Analysis of an anomalous Queue-Only run |
| `report_analysis/raw_results/` | Preserved GPU benchmark summary JSON files |
| `report_analysis/figures/` | Performance figures in PDF and PNG formats |
| `tests/` | Workload, metrics, aggregation, and scheduler unit tests |
| `workload/` | Synthetic workload generation |
| `benchmarks/` | Streaming benchmark client and metric collection |
| `scripts/` | Preliminary CPU baseline scripts |
| `results/` | Historical CPU experiment results |
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

### 4.2 Main GPU Experiments

The main scheduling comparisons were conducted on a shared GPU server.

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

The main GPU experiments evaluate two request distributions.

| Scenario | Short requests | Long requests | Total |
|---|---:|---:|---:|
| `fairness_900_rate30` | 813 | 87 | 900 |
| `balanced_900_rate30_seed123` | 455 | 445 | 900 |

Both scenarios use a target arrival rate of 30 requests per second.

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

## 7. Main GPU Results

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

The project has **46 passing unit tests**.

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
3. The main GPU comparisons cover only two workload distributions and a limited number of repeated runs.
4. Experiments were performed on shared GPU infrastructure, where external activity may affect measurements.
5. Prioritizing shorter requests can substantially increase long-request latency.
6. Aging-based priority heuristics do not formally guarantee starvation freedom.
7. The preserved JSON summaries support statistical reproducibility but do not include every original per-request trace or runtime log.
8. The complete main GPU benchmark launch procedure has not yet been packaged into a verified one-command reproduction script.

Results should not be generalized to arbitrary models, hardware configurations, or production workloads.

## 13. Conclusion

Length-aware scheduling can substantially improve short-request responsiveness in LLM serving.

Across the two main 900-request workloads, Queue-Only scheduling reduced short-request P95 TTFT by approximately 94.2% and 99.2% compared with Native FCFS.

However, these gains must be evaluated alongside long-request latency and throughput.

The experiments demonstrate that the effectiveness of a scheduling policy depends on the workload distribution and the performance objective.
