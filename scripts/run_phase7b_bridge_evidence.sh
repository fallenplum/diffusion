#!/usr/bin/env bash
# Fixed-configuration audits, resumable from formal artifacts, no Git push here.
set -euo pipefail
cd /root/diffusion_phase5_20261007
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/root/phase6a_titan_runtime/env/bin/python
OUT=/root/phase7b_bridge_evidence_v2
mkdir -p "$OUT"
if [ ! -f "$OUT/manifest.json" ]; then
  "$PY" scripts/phase7b_bridge_evidence_audit.py --mode prepare --out "$OUT"
fi
if [ ! -f "$OUT/A_decision.json" ] || [ ! -f "$OUT/B_DONE.json" ]; then
  pids=()
  for w in 0 1 2; do
    "$PY" scripts/phase7b_bridge_evidence_audit.py --mode worker --out "$OUT" --worker-id "$w" > "$OUT/worker_$w.log" 2>&1 &
    pids+=($!)
  done
  if [ ! -f "$OUT/B_DONE.json" ]; then
    "$PY" scripts/phase7b_bridge_evidence_audit.py --mode alignment --out "$OUT" > "$OUT/alignment.log" 2>&1 &
    pids+=($!)
  fi
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  printf '%s\n' "$rc" > "$OUT/EXITCODE"
  test "$rc" -eq 0
  OPENBLAS_NUM_THREADS=4 "$PY" scripts/phase7b_bridge_evidence_audit.py --mode aggregate_a --out "$OUT"
fi
printf '0\n' > "$OUT/EXITCODE"
"$PY" scripts/validate_phase7b_bridge_evidence.py --out "$OUT"
"$PY" scripts/report_phase7b_bridge_evidence.py
