"""Audit explicit Phase23 public files and optionally write a hash manifest.

Default execution is read-only, including the Git index. --write updates only
the anonymous manifest. This script never stages, commits, pushes, fits models,
loads private checkpoints or calls a qualification script. Partial/negative
results remain publishable when accurately identified by their own reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = '0c851828894c5ee6a2ed04985bbf1aa8f1be7899'
PARENT_FILES = 2556
MAX_BYTES = 2 * 1024 ** 2
REPORT = 'reports/phase23_cdei'
MANIFEST = REPORT + '/ARTIFACT_MANIFEST.json'
CONFIG = 'configs/phase23_cdei_g0.json'
MARKDOWN_REPORTS = {'PROTOCOL.md', 'RESULTS.md', 'LITERATURE_AUDIT.md'}
JSON_REPORTS = {'QUALIFICATION.json', 'REAL_DATA_QUALIFICATION_PUBLIC.json',
                'decision.json', 'INNER_RESULTS.json', 'INNER_GATE.json',
                'OUTER_RESULTS.json', 'RUN_AUDIT.json', 'MECHANISM_AUDIT.json',
                'ARTIFACT_MANIFEST.json'}
SENSITIVE_TEXT = re.compile(
    rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|'
    rb'\bhf_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}|'
    rb'\bgh[pousr]_[A-Za-z0-9]{20,}|'
    rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----')
SECRET_ASSIGNMENT = re.compile(
    rb'\b(?:password|passwd|access_token|api_key|ssh_password)\s*[:=]\s*["\'][^"\'\r\n]+["\']',
    re.I)
DENIED_REPORT_FIELDS = {
    'patient_id', 'patient_ids', 'patient_index', 'patient_indices',
    'sample_id', 'sample_ids', 'slide_id', 'slide_ids', 'donor_id', 'donor_ids',
    'donor_indices', 'barcodes', 'barcode', 'fit_indices', 'valid_indices',
    'query_indices', 'owner', 'coords', 'coordinates', 'coords_um', 'coords_px',
    'risk', 'risks', 'raw_risk', 'raw_risks', 'anchor_risk', 'anchor_fit',
    'anchor_valid', 'fit_risk', 'matched_risk', 'prediction', 'predictions',
    'time', 'times', 'event', 'events', 'dss_time', 'dss_event', 'pfi_time',
    'pfi_event', 'survival_time', 'survival_event', 'selection_labels', 'fit_labels',
    'encoded_tokens', '_encoded_tokens', 'tokens', 'raw_tokens', 'token_values',
    '_fit_indices', 'model_state', 'state_dict', 'weights', 'scaler',
    'features', 'fit_features', 'valid_features', 'reference_state',
    'gene_names', 'gene_symbols', 'selected_genes', 'selected_gene_symbols',
    'pca_components', 'pca_state', 'provenance_json', 'password', 'passwd',
    'access_token', 'api_key', 'credentials', 'private_key', 'signed_url',
}
AGGREGATE_RECORD_LISTS = {
    'files', 'public_files', 'rows', 'fold_rows', 'seed_rows', 'three_seed_means',
    'mechanism_tests', 'anchor_noninferiority_screen', 'paired_comparisons',
    'repair_damage', 'seed_repair_damage', 'intervals', 'runs', 'training_curve',
    'cells', 'fits', 'attempts', 'stages', 'checks', 'comparisons', 'metrics',
    'shared_patient_count_matrices', 'selections', 'gradient_groups', 'failed_checks',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative_name(path):
    p = PurePosixPath(str(path).replace('\\', '/'))
    require(not p.is_absolute() and '..' not in p.parts and '\n' not in str(p)
            and '\r' not in str(p), 'unsafe relative artifact path')
    return p.as_posix()


def allowed_path(relative):
    p = PurePosixPath(relative_name(relative))
    return bool(
        (p.parent.as_posix() == 'src/phase23_cdei' and p.suffix == '.py')
        or (p.parent.as_posix() == 'tests' and p.name.startswith('test_phase23_cdei_') and p.suffix == '.py')
        or (p.parent.as_posix() == 'scripts' and 'phase23_cdei' in p.name and p.suffix == '.py')
        or p.as_posix() == CONFIG
        or (p.parent.as_posix() == REPORT and p.name in MARKDOWN_REPORTS | JSON_REPORTS))


def publication_paths(root=ROOT):
    root = Path(root)
    candidates = list((root / 'src/phase23_cdei').glob('*.py'))
    candidates += list((root / 'tests').glob('test_phase23_cdei_*.py'))
    candidates += list((root / 'scripts').glob('*phase23_cdei*.py'))
    candidates += [root / CONFIG]
    for entry in (root / REPORT).iterdir():
        require(entry.is_file() and entry.name in MARKDOWN_REPORTS | JSON_REPORTS,
                'unreviewed artifact in Phase23 report directory')
        candidates.append(entry)
    result = []
    for path in sorted(set(candidates)):
        relative = path.relative_to(root).as_posix()
        require(allowed_path(relative), 'artifact outside explicit Phase23 whitelist')
        require(path.is_file() and not path.is_symlink(), 'missing or linked public artifact')
        require(path.resolve().is_relative_to(root.resolve()), 'artifact resolves outside publication root')
        result.append(relative)
    require(any(p.startswith('src/phase23_cdei/') for p in result), 'Phase23 source missing')
    require(CONFIG in result, 'Phase23 configuration missing')
    return result


def audit_aggregate_json(value, location='root'):
    """Reject patient vectors/state and require explicit aggregate record lists."""
    if isinstance(value, dict):
        for key, child in value.items():
            require(isinstance(key, str), 'report JSON key must be text')
            require(key.lower() not in DENIED_REPORT_FIELDS,
                    'protected raw field in aggregate report: ' + key)
            require(not key.isdigit(), 'numeric-indexed records are not anonymous aggregates')
            audit_aggregate_json(child, key)
    elif isinstance(value, list):
        if not value:
            return
        numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
        require(not numeric or len(value) <= 16,
                'long numeric vector forbidden in aggregate report')
        if any(isinstance(x, dict) for x in value):
            require(location in AGGREGATE_RECORD_LISTS,
                    'unreviewed record list in aggregate report: ' + location)
        if any(isinstance(x, list) for x in value):
            raise ValueError('nested numeric/record arrays forbidden in aggregate report')
        for child in value:
            audit_aggregate_json(child, location)
    elif isinstance(value, float):
        require(math.isfinite(value), 'nonfinite report number')
    else:
        require(value is None or isinstance(value, (str, int, bool)), 'unsupported report JSON value')


def audit_public_file(relative, *, root=ROOT):
    relative = relative_name(relative)
    require(allowed_path(relative), 'file outside explicit public whitelist')
    path = Path(root) / relative
    require(path.is_file() and not path.is_symlink(), 'public file is missing or a symlink')
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_BYTES, 'public file exceeds finite 2 MiB text scope')
    require(not SENSITIVE_TEXT.search(raw) and not SECRET_ASSIGNMENT.search(raw),
            'credential or patient identifier detected in public candidate')
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        raise ValueError('public artifact must be UTF-8 text') from error
    require('\x00' not in text, 'binary content in public text artifact')
    if relative.startswith(REPORT + '/'):
        if path.suffix == '.json':
            value = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
            require(isinstance(value, dict), 'report root must be an aggregate object')
            audit_aggregate_json(value)
        else:
            # Markdown is allowed prose/summary tables, never a raw numeric dump.
            require(not re.search(r'\[(?:\s*-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,){16,}', text),
                    'long numeric vector embedded in report prose')
            require(not re.search(r'["\'](?:risk|risks|patient_ids|fit_indices|state_dict)["\']\s*:\s*[\[{]', text, re.I),
                    'protected patient/state payload embedded in report prose')
    elif path.suffix == '.json':
        require(isinstance(json.loads(text), dict), 'configuration root must be an object')
    return {'path': relative, 'bytes': len(raw), 'sha256': sha_bytes(raw)}


def git(root, *arguments):
    return subprocess.check_output(['git', *arguments], cwd=root, stderr=subprocess.PIPE)


def verify_parent(snapshot_path, *, root=ROOT):
    """Check every inherited worktree byte and independently check parent blobs."""
    root = Path(root)
    snapshot_path = Path(snapshot_path)
    require(snapshot_path.is_file(), 'private parent byte snapshot is missing')
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8-sig'))
    require(snapshot.get('parent_commit') == PARENT, 'unexpected parent commit in private snapshot')
    files = snapshot.get('files_sha256')
    require(isinstance(files, dict) and len(files) == PARENT_FILES,
            'private parent snapshot must contain exactly 2556 files')
    if 'tracked_file_count' in snapshot:
        require(snapshot['tracked_file_count'] == PARENT_FILES, 'snapshot tracked count changed')
    tree = git(root, 'ls-tree', '-rz', '--name-only', PARENT)
    names = [relative_name(name.decode('utf-8')) for name in tree.split(b'\0') if name]
    require(len(names) == PARENT_FILES and set(names) == set(files),
            'snapshot paths differ from the exact 2556-file Git parent')
    subprocess.run(['git', 'merge-base', '--is-ancestor', PARENT, 'HEAD'], cwd=root,
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for name in names:
        require(re.fullmatch('[0-9a-f]{64}', str(files[name])) is not None,
                'invalid parent snapshot checksum')
        path = root / name
        require(path.is_file() and sha_file(path) == files[name],
                'historical working-tree bytes changed: ' + name)
    # A snapshot by itself is not authoritative: compare it with Git's actual
    # parent blobs through one bounded streaming cat-file process.
    proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=root,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    verified = 0
    try:
        for name in names:
            proc.stdin.write((PARENT + ':' + name + '\n').encode('utf-8'))
            proc.stdin.flush()
            header = proc.stdout.readline().strip().split()
            require(len(header) == 3 and header[1] == b'blob', 'Git parent path is not a blob')
            remaining = int(header[2])
            h = hashlib.sha256()
            while remaining:
                block = proc.stdout.read(min(1024 * 1024, remaining))
                require(bool(block), 'truncated Git parent object stream')
                h.update(block); remaining -= len(block)
            require(proc.stdout.read(1) == b'\n', 'invalid Git object delimiter')
            require(h.hexdigest() == files[name], 'snapshot differs from Git parent bytes: ' + name)
            verified += 1
        proc.stdin.close()
        require(proc.wait(timeout=30) == 0, 'Git parent object verification failed')
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
    return {'status': 'PASS', 'parent_commit': PARENT,
            'parent_snapshot_sha256': sha_file(snapshot_path),
            'inherited_tracked_files': PARENT_FILES,
            'working_tree_files_verified': PARENT_FILES,
            'git_parent_blobs_verified': verified, 'parent_all2556verified': True}


def build_manifest(*, root=ROOT, private_root=None):
    from src.phase23_cdei import runtime
    from src.phase23_cdei.evaluation import evaluator_identity
    root = Path(root)
    paths = publication_paths(root)
    rows = [audit_public_file(p, root=root) for p in paths if p != MANIFEST]
    identity = runtime.source()
    require(runtime.cfg().get('parent_commit') == PARENT, 'configuration parent differs from locked ancestor')
    private_root = runtime.PRIVATE if private_root is None else Path(private_root)
    parent = verify_parent(Path(private_root) / 'PARENT_TRACKED_BYTES_PRIVATE.json', root=root)
    evaluator_files = ('src/phase23_cdei/evaluation.py', 'scripts/evaluate_phase23_cdei.py',
                       'src/phase10_trimodal/reference.py', 'src/phase21_relative/evaluation.py',
                       'src/phase13_rna/data.py')
    bound_evaluation = {name: sha_file(root/name) for name in evaluator_files}
    evaluation_hash = sha_bytes(json.dumps(bound_evaluation, sort_keys=True,
                                         separators=(',', ':')).encode())
    require(evaluation_hash == evaluator_identity(), 'evaluator identity does not bind current source bytes')
    require(runtime.source() == identity, 'training bytes changed during publication audit')
    return {'schema': 'phase23_cdei_anonymous_manifest_v1', 'status': 'PASS',
            'scope': 'explicit_new_Phase23_text_files_only_no_whole_repository_privacy_claim',
            'training_source': identity, 'configuration_sha256': sha_file(root/CONFIG),
            'evaluator_sha256': evaluation_hash, 'evaluator_files': bound_evaluation,
            'publisher_sha256': sha_file(Path(__file__)), 'parent_commit': PARENT,
            'parent_verification': parent, 'parent_all2556verified': True,
            'public_files': rows, 'total_files': len(rows),
            'total_bytes': sum(row['bytes'] for row in rows),
            'manifest_self_excluded': True, 'git_index_mutated': False,
            'scientific_completion_claimed_by_publication_audit': False,
            'patient_level_arrays_or_weights_public': False,
            'mechanism_conclusion': 'read_RESULTS_and_decision;technical_PASS_is_not_mechanism_support'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='write only the anonymous manifest after all audits pass')
    args = parser.parse_args()
    expected = build_manifest()
    target = ROOT / MANIFEST
    if args.write:
        from src.phase23_cdei.runtime import atomic_json
        atomic_json(target, expected)
    else:
        require(target.is_file(), 'manifest missing; use --write after reviewing public files')
        actual = json.loads(target.read_text(encoding='utf-8-sig'))
        require(actual == expected, 'manifest differs from current audited artifacts; review before --write')
    audit_public_file(MANIFEST)
    print(json.dumps({'status': 'PASS', 'mode': 'WRITE_MANIFEST_ONLY' if args.write else 'READ_ONLY_AUDIT',
                      'training_source': expected['training_source'],
                      'evaluator_sha256': expected['evaluator_sha256'],
                      'parent_all2556verified': True,
                      'public_files_excluding_manifest': expected['total_files'],
                      'public_bytes_excluding_manifest': expected['total_bytes'],
                      'git_staged_or_pushed': False}, sort_keys=True))


if __name__ == '__main__':
    main()
