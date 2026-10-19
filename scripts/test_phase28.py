"""Run only synthetic Phase28 tests and save an anonymous aggregate receipt."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase28_reread import runtime


def main():
    temporary = runtime.PRIVATE/'synthetic_test_tmp_final'
    if temporary.exists():
        raise RuntimeError('synthetic base temp already exists; choose a fresh run explicitly, never delete it implicitly')
    paths = sorted(ROOT.glob('tests/test_phase28*.py'))
    junit = runtime.PRIVATE/'SYNTHETIC_TESTS.xml'
    cmd = [sys.executable, '-c', "import torch,pytest,sys; torch.set_num_threads(1); sys.exit(pytest.main(sys.argv[1:]))",
           *[str(p) for p in paths], '-q', '-p', 'no:cacheprovider', '--basetemp='+str(temporary), '--junitxml='+str(junit)]
    started = time.monotonic()
    process = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    (runtime.PRIVATE/'SYNTHETIC_TEST_LOG.txt').write_text(process.stdout+'\n'+process.stderr)
    suites = ET.parse(junit).getroot().findall('testsuite')
    counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    result = {'status': 'PASS' if process.returncode == 0 else 'FAIL', 'counts': counts,
              'wall_seconds': time.monotonic()-started, 'source': runtime.source(),
              'script_sha256': runtime.sha(__file__), 'script_path': 'scripts/test_phase28.py',
              'test_files_sha256': {p.relative_to(ROOT).as_posix(): runtime.sha(p) for p in paths},
              'synthetic_only': True, 'no_real_patient_training_or_endpoint_evaluation': True,
              'weights_or_raw_arrays_exported': False}
    runtime.atomic_json(runtime.REPORT/'SYNTHETIC_TESTS.json', result)
    print(json.dumps(result))
    if process.returncode:
        print(process.stdout[-14000:]); print(process.stderr[-3000:])
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
