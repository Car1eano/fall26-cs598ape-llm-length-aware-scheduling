# Modal results for the final report

## Methods

The primary experiment ran on Modal with one NVIDIA L40S, vLLM 0.26.0,
and Qwen2.5-0.5B-Instruct per run. The design crossed three scheduling
policies, two synthetic request mixes, and target rates of 2, 5, and
30 requests/s at a cap of four concurrent sequences. Each of the 18
conditions had five repetitions. Each repetition reused the same 900
requests and arrival times across policies, with randomized policy order.
The cap-16 check replayed all five balanced traces at 5 requests/s under
the same three policies. These two batches contain 105 audited runs.

Compute percentiles within each run, then report the mean of the five
per-run percentiles in each condition. Plots show every repetition.
TTFT is measured at the client. Mean queue time is calculated from vLLM
histogram deltas with the warm-up excluded.

## Findings supported by completed data

- **Short-request response under congestion.** In balanced traffic at
  30 requests/s and cap 4, mean short P95 TTFT was 141.53 s with FCFS,
  1.81 s with Queue-Only, and 2.84 s with Aging. All five trace-paired
  comparisons improved short P95 for both policies. Matching the
  logged GPU environment preserved the direction for Queue-Only
  (four pairs) and Aging (two pairs).
- **Long requests and aging.** In balanced traffic at 5 requests/s,
  mean long P95 TTFT was 46.72 s for FCFS, 35.78 s for Queue-Only, and
  32.45 s for Aging. Aging's mean short P95 was 3.28 s versus 1.23 s
  for Queue-Only. At 30 requests/s, the long-request comparison varied
  substantially across repetitions and GPU environments. Avoid
  presenting the pooled long-request penalty as a general effect of aging.
- **Sensitivity to concurrency cap.** At balanced, 5 requests/s, raising
  the cap from 4 to 16 reduced each strategy's mean short and long P95
  to the 52–56 ms range, with effectively zero mean queueing at cap 16.
  In the four FCFS pairs with the same log-reported L40S driver and visible
  memory, short P95 fell from 46.85–64.93 s to 49–54 ms. This supports a
  strong role for the concurrency cap in this workload.

## GPU environment variation

At 30 requests/s, 11 runs listed driver 610.57.04 and 49,140 MiB visible
memory and reached 433–499 generated tokens/s. The other 19 runs listed
driver 620.70.06 and 46,068 MiB and reached 234–263 tokens/s. This association
does not identify the cause. The policy-level means mix execution
environments. Treat reported throughput as descriptive and use matched
comparisons where possible. For balanced traffic at 30 requests/s,
Queue-Only's four trace- and environment-matched pairs averaged
+3.60 tokens/s and -0.82 s of long P95 TTFT relative to FCFS. Aging's
two matched pairs averaged +5.16 tokens/s and -2.59 s of long P95 TTFT.
These subsets are small, and a matching driver and memory label is
weaker than matching the underlying host and runtime configuration.

## Files

- `figures/modal_ttft_by_load.png`
- `figures/modal_throughput_by_load.png`
- `figures/modal_cap_sensitivity.png`
- `../results/modal_analysis/` for the condition means, run-level data,
  and paired differences
