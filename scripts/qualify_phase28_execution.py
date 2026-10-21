"""Synthetic post-fit/delivery toolchain tests; preserve all original P0 files."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase28_reread import runtime as rt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', choices=('FINAL', 'WIDE', 'POSTFIT', 'PUBLICATION'))
    args = parser.parse_args()
    paths = ['tests/test_phase28_budget_readout.py', 'tests/test_phase28_formal_report.py',
             'tests/test_phase28_formal_publish.py', 'scripts/test_phase28_diagnostics.py',
             'scripts/test_phase28_concurrency.py']
    if args.revision:
        paths.append('tests/test_phase28_mechanism_report.py')
    if args.revision in ('WIDE', 'POSTFIT', 'PUBLICATION'): paths.append('tests/test_phase28_wide_concurrency.py')
    suffix = '_'+args.revision if args.revision else ''
    base = rt.PRIVATE/('execution_toolchain_synthetic_tmp'+suffix)
    destination = rt.REPORT/('FORMAL_TOOLCHAIN_QUALIFICATION'+suffix+'.json')
    if base.exists() or destination.exists(): raise RuntimeError('explicit fresh synthetic qualification required')
    identity = rt.source(); start = time.monotonic()
    if identity != '0ccd4897d30e654ebcb3f55d4891a3b76312f5e6a28dcc30070520000fc5372b':
        raise RuntimeError('auxiliary qualification cannot approve a different training source')
    tool_paths = ['scripts/'+name+'.py' for name in ('coordinate_phase28', 'monitor_phase28',
        'audit_phase28_budget_readout', 'diagnose_phase28', 'report_phase28_formal',
        'publish_phase28_formal', 'tune_phase28_concurrency', 'qualify_phase28_execution')]
    if args.revision: tool_paths.append('scripts/report_phase28_mechanism.py')
    if args.revision in ('WIDE', 'POSTFIT', 'PUBLICATION'): tool_paths.append('scripts/tune_phase28_wide_concurrency.py')
    before_tools = {name: rt.sha(ROOT/name) for name in tool_paths}
    before_tests = {name: rt.sha(ROOT/name) for name in paths}
    p0_names = ('INPUT_AUDIT.json', 'NUMERICAL_QUALIFICATION.json', 'DEPENDENCY_QUALIFICATION.json')
    before_p0 = {name: rt.sha(rt.REPORT/name) for name in p0_names}
    before_evaluation = ({name: rt.sha(rt.REPORT/name) for name in
        ('INNER_RESULTS.json', 'INNER_GATE.json', 'decision.json', 'REPLAY_AUDIT.json')}
        if args.revision in ('POSTFIT', 'PUBLICATION') else None)
    before_completed_reports = ({name: rt.sha(rt.REPORT/name) for name in
        ('BUDGET_READOUT_AUDIT.json', 'FORMAL_DELIVERY.json', 'RESULTS_FORMAL.md',
         'MECHANISM_SUMMARY.json', 'MECHANISM_RESULTS.md')}
        if args.revision == 'PUBLICATION' else None)
    junit = rt.PRIVATE/('EXECUTION_TOOLCHAIN_SYNTHETIC'+suffix+'.xml')
    completed = subprocess.run([sys.executable, '-c',
        'import torch,pytest,sys; torch.set_num_threads(1); sys.exit(pytest.main(sys.argv[1:]))',
        *paths, '-q', '-p', 'no:cacheprovider', '--basetemp='+str(base), '--junitxml='+str(junit)],
        cwd=ROOT, text=True, capture_output=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    (rt.PRIVATE/('EXECUTION_TOOLCHAIN_SYNTHETIC'+suffix+'.log')).write_text(completed.stdout+'\n'+completed.stderr)
    suites = ET.parse(junit).getroot().findall('testsuite')
    counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    result = {'status': 'PASS' if completed.returncode == 0 else 'FAIL', 'source': identity,
              'script_sha256': rt.sha(__file__), 'script_path': 'scripts/qualify_phase28_execution.py',
              'counts': counts, 'wall_seconds': time.monotonic()-start,
              'synthetic_only': True, 'patient_or_validation_endpoint_accesses': 0,
              'new_patient_training_fits': 0,
              'revision': args.revision or 'INITIAL_AUXILIARY',
              'test_files_sha256': before_tests,
              'tool_files_sha256': before_tools,
              'P0_receipt_sha256_unchanged': before_p0,
              'initial_auxiliary_receipt_sha256': (rt.sha(rt.REPORT/'FORMAL_TOOLCHAIN_QUALIFICATION.json')
                  if args.revision else None),
              'initial_auxiliary_receipt_retained_with_its_original_test_hashes': bool(args.revision),
              'previous_FINAL_receipt_sha256': (rt.sha(rt.REPORT/'FORMAL_TOOLCHAIN_QUALIFICATION_FINAL.json')
                  if args.revision in ('WIDE', 'POSTFIT', 'PUBLICATION') else None),
              'previous_WIDE_receipt_sha256': (rt.sha(rt.REPORT/'FORMAL_TOOLCHAIN_QUALIFICATION_WIDE.json')
                  if args.revision in ('POSTFIT', 'PUBLICATION') else None),
              'previous_POSTFIT_receipt_sha256': (rt.sha(rt.REPORT/'FORMAL_TOOLCHAIN_QUALIFICATION_POSTFIT.json')
                  if args.revision == 'PUBLICATION' else None),
              'preserved_evaluation_report_sha256': before_evaluation,
              'preserved_completed_report_sha256': before_completed_reports,
              'python_version': sys.version,
              'original_P0_qualification_reports_unchanged': True}
    if rt.source() != identity: raise RuntimeError('locked training source changed during auxiliary qualification')
    if before_tests != {name: rt.sha(ROOT/name) for name in paths} or before_tools != {name: rt.sha(ROOT/name) for name in tool_paths}:
        raise RuntimeError('auxiliary tools changed during qualification')
    if before_p0 != {name: rt.sha(rt.REPORT/name) for name in p0_names}:
        raise RuntimeError('original P0 qualification bytes changed during auxiliary qualification')
    for group in (before_evaluation, before_completed_reports):
        if group is not None and group != {name: rt.sha(rt.REPORT/name) for name in group}:
            raise RuntimeError('completed scientific reports changed during publication-only qualification')
    rt.atomic_json(destination, result)
    print(json.dumps(result))
    if completed.returncode:
        print(completed.stdout[-14000:]); print(completed.stderr[-3000:]); raise SystemExit(completed.returncode)


if __name__ == '__main__': main()
