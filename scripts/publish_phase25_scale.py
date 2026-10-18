"""Audit only the appended, anonymous frozen-scale diagnostic artifacts."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase25_local import runtime

PARENT = '1a367f74cc764c2b460a12e56e96d7fecc005985'
REPORT = ROOT/'reports/phase25_b_scale_diagnostic'
NAMES = {'PROTOCOL.md', 'RESULTS.md', 'DIAGNOSTIC.json', 'ARTIFACT_MANIFEST.json'}
FILES = {'configs/phase25_b_scale_diagnostic.json', 'scripts/audit_phase25_b_scale.py',
         'scripts/publish_phase25_scale.py', 'src/phase25_scale_diagnostic.py',
         'tests/test_phase25_scale_diagnostic.py'}


def build():
    spec = importlib.util.spec_from_file_location('_phase25_scale_guard', ROOT/'scripts/publish_phase23_cdei.py')
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    guard.PARENT, guard.PARENT_FILES = PARENT, 2669
    guard.AGGREGATE_RECORD_LISTS |= {'two_seed_means', 'fit_distribution_rows', 'teacher_rows', 'residual_rows'}
    guard.REPORT = REPORT.relative_to(ROOT).as_posix()
    parent = guard.verify_parent(runtime.PRIVATE/'scale_diagnostic_PARENT_BYTES_PRIVATE.json')
    parent.pop('parent_all2556verified', None)
    parent['parent_all2669verified'] = True
    present = {p.name for p in REPORT.iterdir()}
    guard.require(present <= NAMES and NAMES-{'ARTIFACT_MANIFEST.json'} <= present,
                  'missing or unreviewed diagnostic report')
    paths = FILES | {guard.REPORT+'/'+n for n in present}
    guard.allowed_path = lambda name: name in paths
    rows = [guard.audit_public_file(p, root=ROOT) for p in sorted(paths)
            if p != guard.REPORT+'/ARTIFACT_MANIFEST.json']
    diagnostic = json.loads((REPORT/'DIAGNOSTIC.json').read_text())
    guard.require(diagnostic['status'] == 'COMPLETE_POSTHOC_DIAGNOSTIC_ONLY' and
                  not diagnostic['alpha_selected'] and not diagnostic['outer_started'] and
                  diagnostic['original_STOP_unchanged'] and diagnostic['new_training_fits'] == 0,
                  'diagnostic scope drift')
    return {'schema': 'phase25_frozen_scale_anonymous_manifest_v1', 'status': 'PASS',
            'training_source': runtime.source(), 'parent_commit': PARENT,
            'parent_verification': parent, 'public_files': rows,
            'total_files': len(rows), 'total_bytes': sum(r['bytes'] for r in rows),
            'manifest_self_excluded': True, 'alpha_selected': False,
            'new_training_fits': 0, 'independent_performance_evidence': False,
            'scope': 'only_appended_frozen_scale_diagnostic;not_a_new_model_or_gate'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--write', action='store_true')
    args = parser.parse_args(); value = build(); path = REPORT/'ARTIFACT_MANIFEST.json'
    if args.write: runtime.atomic_json(path, value)
    elif json.loads(path.read_text()) != value: raise ValueError('manifest bytes or evidence drift')
    print(json.dumps({'status': 'PASS', 'mode': 'write' if args.write else 'audit',
                      'files_excluding_manifest': value['total_files'], 'historical_files_verified': 2669}))


if __name__ == '__main__': main()
