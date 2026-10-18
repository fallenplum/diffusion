"""Record actual synthetic qualification or audit its immutable source hashes.

Never reads patient datasets, fits a cohort model, contacts a server, stages
Git changes or publishes. Synthetic numerical Cox fits stay in process memory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/phase23_cdei'
CONFIG = ROOT / 'configs/phase23_cdei_g0.json'
PARENT = '0c851828894c5ee6a2ed04985bbf1aa8f1be7899'
EXTERNAL_SOURCES = [
    'src/phase19_attention/attention.py', 'src/phase19_attention/model.py',
    'src/phase20_selective/model.py', 'src/phase20_selective/operators.py',
    'src/phase10_trimodal/math.py', 'src/phase17_ssd/losses.py']
REPORT_NAMES = {'PROTOCOL.md', 'RESULTS.md', 'LITERATURE_AUDIT.md',
                'QUALIFICATION.json', 'decision.json', 'ARTIFACT_MANIFEST.json',
                'REAL_DATA_QUALIFICATION_PUBLIC.json', 'INNER_RESULTS.json',
                'INNER_GATE.json', 'OUTER_RESULTS.json', 'RUN_AUDIT.json',
                'MECHANISM_AUDIT.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    target = Path(path)
    if target.resolve() != (REPORT/'QUALIFICATION.json').resolve():
        raise ValueError('numerical qualification may update only QUALIFICATION.json')
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.writing')
    try:
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_files():
    paths = list((ROOT/'src/phase23_cdei').glob('*.py'))
    paths += list((ROOT/'tests').glob('test_phase23_cdei_*.py'))
    paths += [CONFIG, Path(__file__).resolve()]
    paths += [ROOT/p for p in EXTERNAL_SOURCES]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def parent_copy_audit():
    manifest = ROOT/'reports/phase22_spatial_bridge/ARTIFACT_MANIFEST.json'
    d = json.loads(manifest.read_text(encoding='utf-8'))
    if d['status'] != 'PASS' or d['total_files'] != 47:
        raise ValueError('unexpected parent anonymous publication manifest')
    for row in d['public_files']:
        p = (ROOT/row['path']).resolve()
        if not p.is_relative_to(ROOT.resolve()) or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('parent publication bytes changed')
    return {'status': 'PASS_LOCAL_PARENT_PUBLIC_FILE_HASHES',
            'parent_commit': PARENT, 'manifest_files_verified': len(d['public_files']),
            'manifest_sha256': sha(manifest), 'current_remote_full_tree_audited': False}


def public_files():
    paths = list((ROOT/'src/phase23_cdei').glob('*.py'))
    paths += list((ROOT/'tests').glob('test_phase23_cdei_*.py'))
    paths += [CONFIG, Path(__file__).resolve()]
    for p in REPORT.rglob('*'):
        if p.is_file():
            if p.parent != REPORT or p.name not in REPORT_NAMES:
                raise ValueError('unreviewed report in public scope')
            paths.append(p)
    secret = re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|\bhf_[A-Za-z0-9]{20,}|\bgh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    out = []
    for p in sorted(set(paths)):
        if p.name == 'ARTIFACT_MANIFEST.json':
            continue
        raw = p.read_bytes()
        if p.is_symlink() or len(raw) > 2*1024**2 or secret.search(raw):
            raise ValueError('unsafe public source or report')
        raw.decode('utf-8-sig')
        out.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': sha(p)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tests', action='store_true')
    args = parser.parse_args()
    parent = parent_copy_audit()
    source = source_files()
    qualification = REPORT/'QUALIFICATION.json'
    if args.run_tests:
        tests = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'tests').glob('test_phase23_cdei_*.py'))
        temp = 'tmp/phase23_qualification_'+uuid.uuid4().hex
        cmd = [sys.executable, '-m', 'pytest', *tests, '-q', '-p', 'no:cacheprovider', '--basetemp', temp]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        found = re.search(r'(\d+) passed in ([\d.]+)s', result.stdout)
        if result.returncode or not found or re.search(r'\b(?:skipped|xfailed|xpassed)\b', result.stdout):
            print(result.stdout[-6000:] + result.stderr[-3000:])
            raise RuntimeError('actual numerical qualification failed')
        if source_files() != source:
            raise RuntimeError('source changed during qualification')
        import numpy, scipy, sklearn, torch
        write(qualification, {
            'schema': 'phase23_cdei_numerical_qualification_v3', 'status': 'PASS_NUMERICAL_ONLY',
            'source_hashes': source, 'parent': parent,
            'tests': {'status': 'PASS', 'count': int(found[1]), 'seconds': float(found[2]),
                      'command': 'python -m pytest tests/test_phase23_cdei_*.py -q -p no:cacheprovider --basetemp <fresh-workspace-directory>',
                      'files': tests, 'skipped': 0, 'expected_failures': 0},
            'versions': {'python': sys.version.split()[0], 'torch': torch.__version__,
                         'numpy': numpy.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
            'scope': 'synthetic_arithmetic_reference_data_warmup_full_pipeline_and_evaluator_only',
            'formal_patient_model_fits': 0, 'DSS_or_PFI_patient_labels_read': False,
            'optimizer_call_count_not_inferred_from_passed_tests': True,
            'synthetic_weights_persisted': False,
            'patient_reference_arrays_or_predictions_persisted': False,
            'remote_asset_qualification': 'SEPARATE_REAL_DATA_QUALIFICATION_RECEIPT',
            'scientific_mechanism_result': 'NOT_TESTED', 'novelty': 'UNKNOWN_NOT_ESTABLISHED'})

    else:
        d = json.loads(qualification.read_text(encoding='utf-8'))
        if d['status'] != 'PASS_NUMERICAL_ONLY' or d['source_hashes'] != source:
            raise ValueError('numerical qualification source identity drift')
    # This inventory is read-only. Final publication and scientific decisions
    # belong to their dedicated scripts, regardless of later test reruns.
    inventory = public_files()
    q = json.loads(qualification.read_text(encoding='utf-8'))
    print(json.dumps({'status': q['status'], 'tests': q['tests']['count'],
                      'patient_fits_in_this_numerical_invocation': 0,
                      'public_files_in_read_only_inventory': len(inventory),
                      'scientific_decision_modified': False,
                      'publication_manifest_modified': False,
                      'remote_asset_qualification': q.get('remote_asset_qualification',
                          'SEPARATE_REAL_DATA_QUALIFICATION_RECEIPT')}))


if __name__ == '__main__':
    main()
