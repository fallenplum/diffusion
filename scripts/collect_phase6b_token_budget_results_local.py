"""Collect only audited public reports after the GPU job and verified Git push."""
from pathlib import Path
import datetime
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE = 'root@connect.westb.seetacloud.com'
KEY = str(Path.home() / '.ssh/codex_m2_transfer_ed25519')
SSH = ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '42577', REMOTE]
STATUS = ROOT / 'outputs/local_auto_collect/phase6b_token_budget_audit.json'
STATUS.parent.mkdir(parents=True, exist_ok=True)


def state(payload):
    payload['time'] = datetime.datetime.now().isoformat()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(STATUS)


state({'status': 'WAITING_FOR_COMPLETE_MATRIX_AND_VERIFIED_PUSH'})
deadline = time.time() + 24 * 3600
while time.time() < deadline:
    try:
        check = subprocess.run(SSH + ['cat /root/phase6b_token_budget_audit/FINISH_EXIT_CODE 2>/dev/null'],
                               capture_output=True, text=True, timeout=35)
        code = check.stdout.strip()
        if code == '1':
            state({'status': 'REMOTE_FINISH_FAILED', 'log': '/root/phase6b_token_budget_audit/finish.log'})
            raise SystemExit(1)
        if code == '0':
            copied = subprocess.run(['scp', '-i', KEY, '-o', 'BatchMode=yes', '-P', '42577', '-r',
                                     REMOTE + ':/root/diffusion_phase5_20261007/reports/phase6b_token_budget_audit',
                                     str(ROOT / 'reports')], capture_output=True, text=True, timeout=180)
            if copied.returncode == 0:
                dest = ROOT / 'reports/phase6b_token_budget_audit'
                audit = json.loads((dest / 'independent_validation.json').read_text())
                assert audit['status'] == 'PASS' and audit['fitted_cells'] == 240 and audit['complete_oof_files'] == 300
                remote_sha = subprocess.check_output(SSH + ['cat /root/phase6b_token_budget_audit/PUSH_VERIFIED_SHA'],
                                                     text=True, timeout=35).strip()
                state({'status': 'COMPLETE_COLLECTED', 'remote_verified_sha': remote_sha, 'report_directory': str(dest)})
                print('COMPLETE_COLLECTED', remote_sha, flush=True)
                raise SystemExit(0)
    except (subprocess.TimeoutExpired, OSError):
        pass
    time.sleep(60)
state({'status': 'COLLECTION_TIMEOUT', 'gpu_job_not_stopped': True})
