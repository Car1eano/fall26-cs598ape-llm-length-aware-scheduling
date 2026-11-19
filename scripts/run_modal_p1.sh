#!/usr/bin/env bash
# Complete the P1 follow-up batches using up to five single-L40S workers per batch.
set -euo pipefail

cd "$(dirname "$0")/.."
plan=${1:?Usage: bash scripts/run_modal_p1.sh MAIN_PLAN [EXISTING_CAP16_BATCH|--resume-after-variants]}
cap16_batch=${2:-}
modal_run=(.venv/bin/modal run -m gpu_experiments.modal_smoke)
replay=(--formal --paired-plan "$plan" --mix balanced --rates 5)
variants=native_priority,prompt_only,noisy_length,aging_alpha1,aging_alpha2,aging_alpha5,aging_alpha10,wait_promotion,hrrn

if [[ "$cap16_batch" != "--resume-after-variants" ]]; then
if [[ -n "$cap16_batch" ]]; then
    # Let the running cap-16 batch release its GPUs before submitting more work.
    .venv/bin/python - "$cap16_batch" <<'PY'
import json
import sys
import time
from pathlib import Path

batch = Path(sys.argv[1])
deadline = time.monotonic() + 45 * 60
print(f"Waiting for the cap-16 batch at {batch}", flush=True)
while time.monotonic() < deadline:
    records = [json.loads(line) for line in (batch / "progress.jsonl").read_text().splitlines()
               if line.endswith("}")]
    successful = [record for record in records if record["ok"]]
    if len(successful) == 15 and all(
        (batch.parent / record["path"] / "status.json").is_file() for record in successful
    ):
        break
    time.sleep(5)
else:
    raise SystemExit("Cap-16 batch did not finish successfully within 45 minutes")
PY
else
    "${modal_run[@]}" "${replay[@]}" --max-num-seqs 16
fi

"${modal_run[@]}" --policies "$variants" --trace-queue
"${modal_run[@]}" --calibrate
"${modal_run[@]}" "${replay[@]}" --policies "$variants" --trace-queue
fi
for cap in 64 256; do
    "${modal_run[@]}" "${replay[@]}" --max-num-seqs "$cap"
done
"${modal_run[@]}" --formal --rates 10,15,20 --trace-queue
