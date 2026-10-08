#!/usr/bin/env bash
# Formal, resumable sequence. Inputs and private caches remain on the server.
set -euo pipefail
cd /root/diffusion_phase5_20261007
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/root/phase6a_titan_runtime/env/bin/python
P0R=/root/phase7_p0r_robustness_v1
SURV=/root/phase7a_foundation_headroom_v1
mkdir -p "$P0R" "$SURV"
for source in upstream_rna.txt upstream_source.json sample_quality.tsv cdr_endpoint_reference.csv; do
  test -s "$P0R/$source" || { echo "Required source missing: $P0R/$source" >&2; exit 1; }
done

run_workers() {
  local script=$1 out=$2 workers=$3
  local pids=() rc=0
  for ((w=0;w<workers;w++)); do
    "$PY" "scripts/$script" --mode worker --out "$out" --worker-id "$w" > "$out/worker_$w.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  test "$rc" -eq 0 || { printf '%s\n' "$rc" > "$out/EXITCODE"; return "$rc"; }
}

if [ ! -f "$P0R/manifest.json" ]; then
  "$PY" scripts/phase7_p0r_robustness.py --mode prepare --out "$P0R"
fi
if [ ! -f "$P0R/DONE.json" ]; then
  run_workers phase7_p0r_robustness.py "$P0R" 3
  "$PY" scripts/phase7_p0r_robustness.py --mode aggregate --out "$P0R"
fi
"$PY" scripts/validate_phase7_audits.py --phase p0r --out "$P0R"
printf '0\n' > "$P0R/EXITCODE"
"$PY" scripts/audit_phase7_source_provenance.py

"$PY" scripts/phase7a_foundation_survival_headroom.py --mode cache --out "$SURV"
if [ ! -f "$SURV/manifest.json" ]; then
  "$PY" scripts/phase7a_foundation_survival_headroom.py --mode prepare --out "$SURV"
fi
if [ ! -f "$SURV/DONE.json" ]; then
  run_workers phase7a_foundation_survival_headroom.py "$SURV" 6
  OPENBLAS_NUM_THREADS=4 "$PY" scripts/phase7a_foundation_survival_headroom.py --mode aggregate --out "$SURV"
fi
"$PY" scripts/validate_phase7_audits.py --phase survival --out "$SURV"
printf '0\n' > "$SURV/EXITCODE"
"$PY" scripts/report_phase7_decisions.py
