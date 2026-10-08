#!/usr/bin/env bash
# Finish only a fully completed formal matrix; never publish private predictions.
set -u
out=/root/phase6b_token_budget_audit
repo=/root/diffusion_phase5_20261007
python=/root/phase6a_titan_runtime/env/bin/python
destination="$repo/reports/phase6b_token_budget_audit"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
while [ ! -f "$out/EXIT_CODE" ]; do
  sleep 15
done
if [ "$(cat "$out/EXIT_CODE")" -ne 0 ] || [ ! -f "$out/DONE.json" ]; then
  echo "Formal matrix failed; refusing to publish."
  printf '1\n' > "$out/FINISH_EXIT_CODE"
  exit 1
fi
set -e
trap 'printf "1\n" > "$out/FINISH_EXIT_CODE"' ERR
"$python" "$repo/scripts/validate_phase6b_token_budget_audit.py" --out "$out" > "$out/independent_validation.log" 2>&1
"$python" "$repo/scripts/report_phase6b_token_budget_audit.py" --out "$out" > "$out/conclusion_report.log" 2>&1
mkdir -p "$destination"
for file in protocol.json manifest.json encoder_audit.json selection_amendment.json DONE.json validation.json independent_validation.json decision_audit.json RESULTS.md CONCLUSIONS_ZH.md metrics.csv paired_comparisons_by_seed.csv selection_summary.csv inner_learning_curves.csv mechanism_diagnostics.csv aggregate_metrics.csv aggregate_comparisons.csv budget_summary.csv convergence_diagnostics.csv aggregate_convergence.csv; do
  cp "$out/$file" "$destination/$file"
done
"$python" - "$destination" <<'PY'
from pathlib import Path
import re, sys
root=Path(sys.argv[1])
for p in root.iterdir():
    assert p.suffix in {'.json','.csv','.md'}, p.name
    assert p.stat().st_size <= 10 * 1024 * 1024, 'oversized public artifact: ' + p.name
    content=p.read_text(encoding='utf-8')
    assert not re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}',content), 'patient identifier in ' + p.name
    assert not re.search(r'hf_[A-Za-z0-9]{20,}',content), 'credential in ' + p.name
print('PUBLICATION_BOUNDARY_PASS')
PY
cd "$repo"
"$python" scripts/check_protected_data.py > "$out/protected_data_check.log"
git add scripts/validate_phase6b_token_budget_audit.py scripts/report_phase6b_token_budget_audit.py scripts/finish_phase6b_token_budget_audit.sh reports/phase6b_token_budget_audit
"$python" scripts/check_protected_data.py > "$out/protected_data_check_staged.log"
git diff --cached --check
git commit -m 'Complete nested TITAN token prompt budget and mechanism audit'
sha=$(git rev-parse HEAD)
pushed=0
for attempt in 1 2 3; do
  if GIT_SSH_COMMAND=/tmp/github_ssh_wrapper.sh git push github HEAD:codex/phase5-endpoint-supervision-budget-audit; then
    remote=$(GIT_SSH_COMMAND=/tmp/github_ssh_wrapper.sh git ls-remote github refs/heads/codex/phase5-endpoint-supervision-budget-audit | cut -f1)
    if [ "$remote" = "$sha" ]; then
      pushed=1
      break
    fi
  fi
  sleep 10
done
if [ "$pushed" -ne 1 ]; then exit 1; fi
printf '%s\n' "$sha" > "$out/PUSH_VERIFIED_SHA"
printf '0\n' > "$out/FINISH_EXIT_CODE"
echo "COMPLETE_AND_REMOTE_VERIFIED=$sha"
