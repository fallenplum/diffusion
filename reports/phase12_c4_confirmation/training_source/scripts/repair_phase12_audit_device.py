"""Preserve a CPU/GPU audit-comparison failure and resume post-training audits.

This patch changes only the device used for exact checkpoint equality checking.
It does not alter any trained parameters, predictions, selection or statistic.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path('/root/phase12_c4_private')
archive = PRIVATE / 'audit_failure_device_comparison'
archive.mkdir(exist_ok=True)
target = ROOT / 'scripts/audit_phase12_test_sensitivity.py'
old = 'torch.equal(value,before[key])'
new = 'torch.equal(value.detach().cpu(),before[key])'
source = target.read_text()
if old not in source:
    raise RuntimeError('Expected original audit expression before repair')
shutil.copy2(target, archive / target.name)
for name in ['final_audit.log', 'FINAL_AUDIT_FAILURE.json']:
    path = PRIVATE / name
    if path.exists():
        shutil.copy2(path, archive / name)
target.write_text(source.replace(old, new), encoding='utf8')
source_hash = json.loads((ROOT / 'reports/phase12_c4_confirmation/HARD_GATES.json').read_text())['source_hash']
journal_path = PRIVATE / 'BUG_JOURNAL.json'
journal = json.loads(journal_path.read_text())
journal.append({
    'severity': 'P1_AUDIT_IMPLEMENTATION',
    'stage': 'fixed_model_test_sensitivity_checkpoint_equality',
    'bug': 'torch.equal compared CUDA current state against CPU before-state',
    'affected': 'audit stopped before first completed fold; no training or OOF affected',
    'repair': 'compare detached CPU current state to CPU before-state exactly; rerun all 15 checkpoints and 20 locked contexts',
    'post_hoc': False,
    'source_hash': source_hash,
})
journal_path.write_text(json.dumps(journal, indent=2))
failure = PRIVATE / 'FINAL_AUDIT_FAILURE.json'
if failure.exists():
    failure.unlink()  # Original failure is retained in the archive above.
env = os.environ.copy()
env.update({
    'PYTHONPATH': '/root/phase10_trimodal_eval_deps',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
})
code = """
import json, subprocess, sys
from pathlib import Path
private = Path('/root/phase12_c4_private')
for name in ['audit_phase12_test_sensitivity.py', 'report_phase12.py']:
    print('RUN', name, flush=True)
    result = subprocess.call([sys.executable, 'scripts/' + name])
    if result:
        (private / 'FINAL_AUDIT_FAILURE.json').write_text(json.dumps({'script': name, 'exit': result}))
        raise SystemExit(result)
print('AUDITS_AND_REPORT_COMPLETE_READY_FOR_INTERPRETATION_REVIEW', flush=True)
"""
with (PRIVATE / 'final_audit_resume.log').open('w') as output:
    child = subprocess.Popen([sys.executable, '-u', '-c', code], cwd=ROOT, env=env,
                             stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps({'status': 'AUDIT_REPAIR_STARTED', 'pid': child.pid,
                  'numeric_training_source_hash_unchanged': source_hash,
                  'original_failure_preserved': str(archive)}))
