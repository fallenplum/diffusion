"""Resume the locked R5 scheduler with a reporting-only schema adapter.

No training script, receipt, selector, metric or original report is modified.
The original failure and source hashes remain auditable.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_r5_schedule as original


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    arguments = sys.argv[1:]
    if '--private' not in arguments or '--resume' not in arguments:
        raise ValueError('EXPLICIT_PRIVATE_AND_RESUME_REQUIRED')
    private = Path(arguments[arguments.index('--private') + 1])
    lock = json.loads((private / 'LOCK.json').read_text())
    error = private / 'SCHEDULE_ERROR.json'
    if not error.exists():
        raise ValueError('ORIGINAL_SCHEDULE_FAILURE_REQUIRED')
    before = {str(p.relative_to(ROOT)): digest(p) for p in
              (ROOT/'scripts/bgb_en_r5_schedule.py', ROOT/'scripts/bgb_en_r5_report.py')}
    evidence = dict(status='REPORTING_SCHEMA_ONLY_RECOVERY',
                    original_failure_sha256=digest(error),
                    original_source_sha256=before,
                    contract_sha256=lock['contract_sha256'],
                    started_unix_seconds=time.time(),
                    new_training_protocol=False, original_receipts_edited=False,
                    metrics_or_selection_rule_changed=False)
    saved_error = private / 'SCHEDULE_ERROR_INITIAL_REPORT_SCHEMA.private.json'
    if saved_error.exists():
        if digest(saved_error) != digest(error):
            raise ValueError('ORIGINAL_FAILURE_EVIDENCE_CHANGED')
    else:
        saved_error.write_bytes(error.read_bytes())
    certificate = private / 'REPORTING_RECOVERY.private.json'
    if certificate.exists():
        previous = json.loads(certificate.read_text())
        if previous['contract_sha256'] != evidence['contract_sha256']:
            raise ValueError('RECOVERY_CONTRACT_CHANGED')
    else:
        certificate.write_text(json.dumps(evidence, indent=2)+'\n')
    run = original.subprocess.run

    def routed(command, *args, **kwargs):
        if isinstance(command, list) and len(command) > 1 and Path(command[1]).resolve() == (ROOT/'scripts/bgb_en_r5_report.py').resolve():
            command = list(command)
            command[1] = str(ROOT/'scripts/bgb_en_r5_report_recovery.py')
        return run(command, *args, **kwargs)

    original.subprocess.run = routed
    try:
        original.main()
    finally:
        original.subprocess.run = run
        after = {name: digest(ROOT/name) for name in before}
        if after != before:
            raise ValueError('LOCKED_R5_REPORT_OR_SCHEDULER_MODIFIED')


if __name__ == '__main__':
    main()
