#!/usr/bin/env bash
# The protocol, private OOF and small resumable checkpoints are retained.
set -u
out=/root/phase6b_token_budget_audit
python=/root/phase6a_titan_runtime/env/bin/python
script=/root/diffusion_phase5_20261007/scripts/phase6b_token_budget_audit.py
mkdir -p "$out"
if [ -f "$out/DONE.json" ]; then
  echo "Already completed; refusing to rerun."
  exit 0
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [ ! -f "$out/manifest.json" ]; then
  HF_MODULES_CACHE="$out/runtime_modules/prepare" "$python" -u "$script" --mode prepare --out "$out" > "$out/prepare.log" 2>&1
  status=$?
  if [ "$status" -ne 0 ]; then
    printf '%s\n' "$status" > "$out/EXIT_CODE"
    exit "$status"
  fi
fi
run_worker() {
  worker=$1
  for attempt in 1 2 3; do
    if [ -f "$out/worker_${worker}_DONE.json" ]; then
      return 0
    fi
    echo "WORKER_START id=$worker attempt=$attempt"
    HF_MODULES_CACHE="$out/runtime_modules/worker${worker}" "$python" -u "$script" --mode worker --worker-id "$worker" --workers 3 --out "$out" >> "$out/worker_${worker}.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$out/worker_${worker}_attempt_${attempt}_EXIT_CODE"
    if [ "$status" -eq 0 ]; then return 0; fi
    echo "WORKER_FAILED id=$worker attempt=$attempt exit=$status"
    sleep 5
  done
  return "$status"
}
pids=()
for worker in 0 1 2; do
  run_worker "$worker" &
  pids+=("$!")
  sleep 2
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [ "$status" -eq 0 ]; then
  "$python" -u "$script" --mode aggregate --out "$out" > "$out/aggregate.log" 2>&1
  status=$?
fi
printf '%s\n' "$status" > "$out/EXIT_CODE"
echo "COORDINATOR_EXIT=$status"
exit "$status"
