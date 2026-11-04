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
