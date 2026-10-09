# Experiment artifacts

Store one timestamped archive per completed experiment here. Keep summaries
and workloads under `results/` as before. Raw results and logs remain ignored
individually, but the archives in this directory are tracked by Git.

From the project directory, after all six benchmark runs finish:

```bash
bash scripts/package_results.sh 《實驗TAG》 《server_log檔名》 《benchmark_log檔名》
git add artifacts/
git commit -m "Archive 《實驗TAG》 raw results and logs"
git push
```

Replace each `《...》` placeholder (including the brackets) with the actual value
before running these commands. `《實驗TAG》` is the TAG used for the benchmark.
Use the actual log filenames. The third argument is optional. The script
requires A/B runs 1, 2 and 3, runs the audit on existing raw results, and
packages workloads, per-run raw results and summaries, the aggregate table,
logs, audit output, and packaging metadata. It does not run a benchmark.
A successful audit does not establish server settings or historical hardware;
review the matching server log separately. Verify the expected total requests.

List archive contents without extracting:

```bash
tar -tzf artifacts/《壓縮檔名》.tar.gz
```

To inspect without overwriting the current results, extract into a fresh folder:

```bash
mkdir -p /tmp/experiment-review
tar -xzf artifacts/《壓縮檔名》.tar.gz -C /tmp/experiment-review
```

The earlier root-level `fcfs_cap4_artifact.tar.gz` is retained for compatibility.
