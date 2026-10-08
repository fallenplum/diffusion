#!/usr/bin/env bash
set -eu
out=${PHASE7_OUT:-/root/phase7_p0_biology_audit_v4}
repo=/root/diffusion_phase5_20261007
python=/root/phase6a_titan_runtime/env/bin/python
script=$repo/scripts/phase7_p0_biology_prompt_audit.py
mkdir -p "$out"
trap 'printf "1\n" > "$out/EXIT_CODE"' ERR
if [ -f "$out/DONE.json" ]; then echo 'Already computed; refusing rerun.'; exit 0; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ ! -f "$out/manifest.json" ]; then
  "$python" -u "$script" --mode self-check
  "$python" -u "$script" --mode prepare --out "$out" > "$out/prepare.log" 2>&1
fi
pids=()
for w in 0 1 2; do
  "$python" -u "$script" --mode worker --worker-id "$w" --workers 3 --out "$out" > "$out/worker_${w}.log" 2>&1 &
  pids+=("$!")
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
if [ "$status" -eq 0 ]; then
  "$python" -u "$script" --mode aggregate --out "$out" > "$out/aggregate.log" 2>&1 || status=1
fi
printf '%s\n' "$status" > "$out/EXIT_CODE"
exit "$status"
