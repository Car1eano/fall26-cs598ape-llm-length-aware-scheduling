# Modal results for the final report

## Capacity-calibrated comparison

Each block runs a 320-request saturated FCFS calibration, restarts vLLM
for each policy, and replays one workload in randomized policy order
on the same Modal container and L40S. The balanced arrival rate is
`rho * capacity_tok_s / mean_requested_output_tokens`. Policy effects
come from subtracting the FCFS result within each block. Summaries show
means of run-level metrics, with five blocks for each condition.

At 5 requests/s in the first balanced sweep, the driver-610 FCFS run
produced 354.33 tokens/s with 0.31 s of mean queueing. The four
driver-620 FCFS runs averaged 259.76 tokens/s and 32.30 s of queueing.
All 15 historical Prompt-Only, Aging alpha=1, and Aging alpha=5 runs in
Table 3 used driver 610. Their 1.48–1.55 s long P95 TTFT values reflect
the environment assignment as well as the policy. Use the matched
Table 3 comparisons for policy conclusions, and treat driver and
memory as environment labels when citing the earlier sweep.

### Load sweep

Across 35 calibrated blocks, FCFS capacity ranged from 233.46 to 452.51
generated tokens/s. At rho=0.5, all three core policies had mean short
and long P95 TTFT below 0.60 s. At rho=1.3, Queue-Only lowered short P95
from 43.01 to 1.44 s, raised long P95 from 42.85 to 45.03 s, and
changed throughput from 299.77 to 298.63 generated tokens/s. Aging
alpha=0.5 recorded 4.04 s short P95 and 47.01 s long P95 at that load.
The absolute throughput curve uses different blocks at each offered
load. The per-run table includes throughput divided by calibrated
capacity for cross-load interpretation. The queue-depth panel uses
vLLM waiting-count samples at approximately ten-second intervals from
one rho=1.1 block.

### Table 3 at ρ=1.1

Fourteen policies ran in all five blocks. FCFS averaged 24.70 s for
short P95 TTFT, 24.42 s for long P95, 25.82 s for long P99, and
287.22 generated tokens/s. Queue-Only averaged 1.43 s for short P95,
19.02 s for long P95, 20.08 s for long P99, and 295.62 tokens/s.
It lowered long P95 in four of five blocks. Aging alpha=0.5 achieved a
2.29 s short P95, with long P95 increasing by 2.72 s on average.

| Policy | Historical long P95 (s) | Matched long P95 (s) | Mean matched Δ vs FCFS (s) | Blocks with lower long P95 |
| --- | ---: | ---: | ---: | ---: |
| Prompt-Only | 1.55 | 18.81 | −5.61 | 3/5 |
| Aging alpha=1 | 1.48 | 19.66 | −4.76 | 3/5 |
| Aging alpha=5 | 1.50 | 26.97 | +2.55 | 1/5 |

The historical means use a fixed 5 requests/s. The matched means use
capacity-normalized load. Compare each policy against the FCFS result
in its own block. Prompt-Only and Aging alpha=1 retained a mean long
P95 improvement, with variation across blocks. Aging alpha=5 increased
mean long P95 under the matched design. Aging strength had no monotonic
relationship with long P95. Noisy Length increased long P99 by
81.22 s on average while lowering short P95 by 23.16 s.

### Sustained-short stress test

Five same-container blocks replayed 888 short requests and 12 long
probes at 30 requests/s. The late count reports how many long probes
received their first token after the last short request was sent.
TTFT values are means of the five run-level P95s.

| Policy | Late long probes | Long P95 TTFT (s) | Short P95 TTFT (s) |
| --- | ---: | ---: | ---: |
| FCFS | 0/60 | 6.25 | 20.96 |
| Queue-Only | 59/60 | 48.03 | 17.05 |
| Aging alpha=0.5 | 59/60 | 50.45 | 19.28 |
| HRRN | 54/60 | 49.69 | 19.17 |
| 10-second promotion | 0/60 | 10.53 | 21.77 |
| 5-second promotion | 0/60 | 8.36 | 23.64 |

Five-second promotion improved long P95 over ten-second promotion
by 2.18 s and increased short P95 by 1.87 s. Actual queueing still
depends on admission capacity. Maximum observed queue time for a
long probe with five-second promotion was 14.72 s.

The [claim-to-run index](../results/modal_matched_analysis/claim_evidence.csv)
links these conclusions to their source records. Five figures are
available in [PDF and PNG](figures/matched/), and both scheduler test
files are included in the [56-test result](../results/test_report.txt).

## Historical fixed-rate comparison

### Methods

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

### Findings from the historical data

- **Short-request response under congestion.** In balanced traffic at
  30 requests/s and cap 4, mean short P95 TTFT was 141.53 s with FCFS,
  1.81 s with Queue-Only, and 2.84 s with Aging. All five trace-paired
  comparisons improved short P95 for both policies. Matching the
  logged GPU environment preserved the direction for Queue-Only
  (four pairs) and Aging (two pairs).
- **Pooled long-request latency.** In balanced traffic at 5 requests/s,
  mean long P95 TTFT was 46.72 s for FCFS, 35.78 s for Queue-Only, and
  32.45 s for Aging. Aging's mean short P95 was 3.28 s versus 1.23 s
  for Queue-Only. At 30 requests/s, the long-request comparison varied
  substantially across repetitions and GPU environments. The comparison
  pools hosts with different measured service capacity.
- **Sensitivity to concurrency cap.** At balanced, 5 requests/s, raising
  the cap from 4 to 16 reduced each strategy's mean short and long P95
  to the 52–56 ms range, with effectively zero mean queueing at cap 16.
  In the four FCFS pairs with the same log-reported L40S driver and visible
  memory, short P95 fell from 46.85–64.93 s to 49–54 ms. This supports a
  strong role for the concurrency cap in this workload.

### GPU environment variation

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

### Historical files

- `figures/modal_ttft_by_load.png`
- `figures/modal_throughput_by_load.png`
- `figures/modal_cap_sensitivity.png`
- `../results/modal_analysis/` for the condition means, run-level data,
  and paired differences
