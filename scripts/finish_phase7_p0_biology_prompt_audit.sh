#!/usr/bin/env bash
# This finalizer validates and prepares reviewable public artifacts; pushing is a separate verified step.
set -eu
out=${PHASE7_OUT:-/root/phase7_p0_biology_audit_v4}
repo=/root/diffusion_phase5_20261007
python=/root/phase6a_titan_runtime/env/bin/python
dest=$repo/reports/phase7_p0_biology_prompt_audit
test "$(cat "$out/EXIT_CODE")" -eq 0
test -f "$out/DONE.json"
trap 'printf "1\n" > "$out/FINISH_EXIT_CODE"' ERR
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" "$repo/scripts/validate_phase7_p0_biology_prompt_audit.py" --out "$out" > "$out/independent_validation.log" 2>&1
"$python" "$repo/scripts/report_phase7_p0_biology_prompt_audit.py" --out "$out"
"$python" - "$out" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1]); done=json.loads((root/'DONE.json').read_text())
done['status']='PASS'
done['independent_validation']='independent_validation.json'
(root/'DONE.json').write_text(json.dumps(done,indent=2)+'\n')
PY
mkdir -p "$dest"
for f in protocol.json manifest.json encoder_audit.json DONE.json validation.json independent_validation.json CONCLUSIONS_ZH.md metrics.csv paired_comparisons.csv selection_summary.csv aggregate_metrics.csv; do
  cp "$out/$f" "$dest/$f"
done
"$python" - "$dest" <<'PY'
from pathlib import Path
import re,sys
root=Path(sys.argv[1])
for p in root.iterdir():
    assert p.suffix in {'.json','.csv','.md'},p.name
    assert p.stat().st_size <= 2*1024*1024,'oversized public artifact: '+p.name
    s=p.read_text(encoding='utf-8')
    assert not re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}',s),'patient id in '+p.name
    assert not re.search(r'hf_[A-Za-z0-9]{20,}',s),'credential in '+p.name
print('PUBLICATION_BOUNDARY_PASS')
PY
cd "$repo"
"$python" scripts/check_protected_data.py > "$out/protected_data_check.log"
printf '0\n' > "$out/FINISH_EXIT_CODE"
echo 'VALIDATED_AND_PUBLIC_ARTIFACTS_PREPARED'
