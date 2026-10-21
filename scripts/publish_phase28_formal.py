"""Fail-closed publication of completed, anonymous Phase28 formal artifacts.

``--check`` performs read-only qualification and manifest planning. Publication
requires an explicit ``--publish``; a failed network push can be continued only
with ``--push-committed`` after the exact single new commit is revalidated.
The frozen P0 publisher and all existing qualification files remain unchanged.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path('/root/autodl-tmp/phase28_reread_private')
REPORT_REL = 'reports/phase28_bidir_reread/'
BRANCH = 'research/phase28-bidirectional-evidence-reread'
ORIGIN = 'git@github.com:fallenplum/diffusion.git'
ANCESTOR = 'd8f837bb97cdccea8960b0a4b1648a1a56b55a14'
P0 = 'c12494606af9e95b704925c03760b5b2c767db1b'
SOURCE = '0ccd4897d30e654ebcb3f55d4891a3b76312f5e6a28dcc30070520000fc5372b'
MANIFEST_REL = REPORT_REL+'FORMAL_PUBLIC_MANIFEST.json'
DECISION_REL = REPORT_REL+'decision.json'
CELLS = {(s, f, a) for s in (31, 47) for f in (1, 2, 3)
         for a in ('C00', 'C10', 'C01', 'C11', 'DIRECT')}
REQUIRED_NEW = (
    'scripts/coordinate_phase28.py', 'scripts/monitor_phase28.py',
    'scripts/audit_phase28_budget_readout.py', 'scripts/diagnose_phase28.py',
    'scripts/report_phase28_formal.py', 'scripts/report_phase28_mechanism.py',
    'scripts/publish_phase28_formal.py', 'tests/test_phase28_formal_publish.py',
    'tests/test_phase28_mechanism_report.py',
    *(REPORT_REL+n for n in ('P0_DECISION.json', 'REPLAY_AUDIT.json', 'INNER_RESULTS.json',
                             'INNER_GATE.json', 'BUDGET_READOUT_AUDIT.json',
                             'FORMAL_DELIVERY.json', 'RESULTS_FORMAL.md',
                             'MECHANISM_SUMMARY.json', 'MECHANISM_RESULTS.md',
                             'FORMAL_TOOLCHAIN_QUALIFICATION_FINAL.json')),
)
FINAL_TEST_FILES = frozenset((
    'tests/test_phase28_budget_readout.py', 'tests/test_phase28_formal_report.py',
    'tests/test_phase28_formal_publish.py', 'scripts/test_phase28_diagnostics.py',
    'scripts/test_phase28_concurrency.py', 'tests/test_phase28_mechanism_report.py'))
FINAL_TOOL_FILES = frozenset('scripts/'+name+'.py' for name in (
    'coordinate_phase28', 'monitor_phase28', 'audit_phase28_budget_readout', 'diagnose_phase28',
    'report_phase28_formal', 'publish_phase28_formal', 'tune_phase28_concurrency',
    'qualify_phase28_execution', 'report_phase28_mechanism'))
WIDE_TEST_FILES = FINAL_TEST_FILES|{'tests/test_phase28_wide_concurrency.py'}
WIDE_TOOL_FILES = FINAL_TOOL_FILES|{'scripts/tune_phase28_wide_concurrency.py'}
PROTECTED_PATTERN = re.compile(
    r'TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4}'
    r'|\bhf_[A-Za-z0-9]{20,}'
    r'|\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}'
    r'|-----BEGIN [A-Z ]*PRIVATE KEY-----', re.I)
PROTECTED_ARRAY_KEYS = frozenset((
    'patient_ids', 'patient_id', 'sample_ids', 'sample_id', 'gene_symbols', 'gene_ids',
    'coordinates', 'coords', 'selected_indices', 'patient_indices', 'fit_indices',
    'valid_indices', 'wrong_context_mapping', 'donor_mapping', 'model_state',
    'optimizer_state', 'raw_rna', 'raw_cna', 'rna_values', 'cna_values', 'measurement_values',
    'patch_features', 'gene_evidence', 'gene_set_membership', 'pathway_membership',
    'hallmark_membership', 'gmt_contents',
))
PROTECTED_RISK_KEYS = frozenset(('risk', 'risks', 'fit_risk', 'valid_risk', 'valid_risks',
                                'fit_components', 'valid_components', 'weights', 'raw_arrays'))
SUMMARY_KEYS = frozenset(('count', 'mean', 'SD', 'RMS', 'min', 'max', 'minimum', 'maximum',
                         'median', 'abs_q95', 'abs_p95', 'p05', 'p25', 'p75', 'p95', 'p99',
                         'L1', 'L2', 'nonzero_count', 'finite_count'))


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args])


def git_text(*args, root=ROOT):
    return git(*args, root=root).decode('utf-8').strip()


def load_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON number: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def canonical_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')


def require_safe_relative(relative):
    require(isinstance(relative, str) and relative and '\\' not in relative and ':' not in relative,
            'public candidate must use plain repository-relative POSIX path')
    p = PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts and '.' not in p.parts
            and p.as_posix() == relative, 'public candidate path is not canonical')
    require(p.suffix in ('.py', '.json', '.md'), 'binary, array or unsupported artifact extension')
    allowed = (re.fullmatch(r'scripts/[a-z][a-z0-9_]*phase28[a-z0-9_]*\.py', relative)
               or re.fullmatch(r'tests/test_phase28[a-z0-9_]*\.py', relative)
               or re.fullmatch(REPORT_REL+r'[A-Za-z0-9_]+\.(?:json|md)', relative)
               or re.fullmatch(REPORT_REL+r'diagnostics/seed(?:31|47)_fold[123]_'
                               r'(?:C00|C10|C01|C11|DIRECT)_(?:history|mechanism)\.json', relative))
    require(allowed is not None, 'candidate outside explicit Phase28 publication namespace')
    return relative


def safe_file(root, relative):
    require_safe_relative(relative)
    root = Path(root).resolve(strict=True)
    path = root/relative
    current = root
    for part in PurePosixPath(relative).parts:
        current = current/part
        require(not current.is_symlink(), 'public candidate path contains a symlink')
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), 'public candidate missing/outside repository')
    require(0 < resolved.stat().st_size <= 8*1024**2, 'public artifact empty or larger than 8 MiB')
    return resolved


def _summary(value):
    return (isinstance(value, dict) and bool(value) and set(value) <= SUMMARY_KEYS
            and all(type(v) in (int, float) and math.isfinite(v) for v in value.values()))


def _gene_module_gradient_summary(value, trail):
    """An exact module-name exception, never permission for a gene tensor."""
    def nonnegative(x): return type(x) in (int, float) and math.isfinite(x) and x >= 0
    if trail in (('artifact_audit', 'rows', '[]', 'training', 'last_data_gradient_L2_by_module'),
                 ('artifact_audit', 'rows', '[]', 'training', 'last_total_gradient_L2_by_module')):
        return nonnegative(value)
    if not isinstance(value, dict): return False
    if trail == ('rows', '[]', 'training', 'data_vs_total_gradient_by_module'):
        fields = {'first_data_L2', 'last_data_L2', 'max_data_L2',
                  'epochs_nonzero_data_gradient', 'last_total_L2'}
        return (set(value) == fields and all(nonnegative(x) for x in value.values())
                and type(value['epochs_nonzero_data_gradient']) is int
                and 0 <= value['epochs_nonzero_data_gradient'] <= 80)
    if trail == ('history', 'data_versus_total_gradient_by_module'):
        fields = {'data_L2_first10_mean', 'data_L2_last10_mean', 'data_L2_maximum',
                  'epochs_with_nonzero_data_gradient', 'total_including_L2_last10_mean', 'data_gradient_L2_curve'}
        if set(value) != fields: return False
        if not all(nonnegative(value[k]) for k in fields-{'data_gradient_L2_curve'}): return False
        if type(value['epochs_with_nonzero_data_gradient']) is not int or not 0 <= value['epochs_with_nonzero_data_gradient'] <= 80: return False
        curve = value['data_gradient_L2_curve']
        return (isinstance(curve, list) and len(curve) == 80 and all(
            isinstance(row, dict) and set(row) == {'epoch', 'L2'}
            and type(row['epoch']) is int and row['epoch'] == index+1 and nonnegative(row['L2'])
            for index, row in enumerate(curve)))
    if trail == ('arms', '[]', 'last10_parameter_gradient_norms'):
        if set(value) != {'data_gradient_L2', 'total_including_L2_gradient_L2', 'norm_difference_is_not_regularizer_gradient_norm'}: return False
        if value['norm_difference_is_not_regularizer_gradient_norm'] is not True: return False
        fields = {'mean_equal_cells', 'SD_between_cells', 'minimum_cell', 'maximum_cell', 'available_cells', 'expected_cells'}
        for key in ('data_gradient_L2', 'total_including_L2_gradient_L2'):
            item = value[key]
            if not isinstance(item, dict) or set(item) != fields or not all(nonnegative(x) for x in item.values()): return False
            if type(item['available_cells']) is not int or type(item['expected_cells']) is not int or item['available_cells'] != 6 or item['expected_cells'] != 6: return False
        return True
    return False


def privacy_json(value, trail=()):
    """Anonymous aggregate trees only; even short numeric arrays are rejected."""
    if isinstance(value, dict):
        for key, item in value.items():
            require(isinstance(key, str) and not PROTECTED_PATTERN.search(key), 'protected JSON key')
            lower = key.lower()
            if lower in PROTECTED_ARRAY_KEYS:
                counter = (lower == 'patient_ids' and trail in (
                               ('central_feature_array_read_counts',),
                               ('mechanism', 'central_feature_array_read_counts'))
                           and type(item) is int and item >= 0)
                module_summary = lower == 'gene_evidence' and _gene_module_gradient_summary(item, trail)
                require(counter or module_summary, 'protected identity, measurement, index or tensor JSON field: '+key)
            if lower in PROTECTED_RISK_KEYS:
                require(_summary(item), 'raw risk/weight field must be an explicit aggregate summary: '+key)
            privacy_json(item, trail+(key,))
    elif isinstance(value, list):
        require(not (value and all(type(v) in (int, float, bool) for v in value)),
                'numeric/boolean JSON vector is not a public aggregate')
        for item in value:
            privacy_json(item, trail+('[]',))
    elif isinstance(value, str):
        require(not PROTECTED_PATTERN.search(value), 'protected identifier or credential in JSON string')
    elif type(value) in (int, float):
        require(math.isfinite(value), 'nonfinite public scalar')
    else:
        require(value is None or type(value) is bool, 'unsupported public JSON value')


def privacy_file(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    require(not PROTECTED_PATTERN.search(text), 'protected identifier or credential pattern in public text')
    if Path(path).suffix == '.json':
        privacy_json(load_json(path))
    elif Path(path).suffix == '.py':
        for node in ast.walk(ast.parse(text, filename=Path(path).name)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                require(not PROTECTED_PATTERN.search(node.value),
                        'protected identifier or credential in decoded Python string')
    elif Path(path).suffix == '.md':
        require(not re.search(r'\[\s*[-+]?\d+(?:\.\d+)?\s*(?:,\s*[-+]?\d+(?:\.\d+)?\s*){1,}\]', text),
                'raw numeric array literal in public Markdown')


def p0_tree_audit(*, root=ROOT, private=PRIVATE, committed=False):
    """All P0 bytes survive, except decision's documented byte-preserving move."""
    require(git_text('branch', '--show-current', root=root) == BRANCH, 'wrong publication branch')
    require(git_text('remote', 'get-url', 'origin', root=root) == ORIGIN, 'wrong Git remote')
    head = git_text('rev-parse', 'HEAD', root=root)
    require(git_text('rev-parse', 'HEAD^', root=root) == P0 if committed else head == P0,
            'formal publication must be exactly one new child of the P0 commit')
    require(not git_text('diff', '--cached', '--name-only', root=root), 'pre-existing staged changes are forbidden')
    subprocess.check_call(['git', '-C', str(root), 'merge-base', '--is-ancestor', ANCESTOR, P0])
    original = load_json(Path(private)/'PARENT_FILES_PRIVATE.json')
    require(isinstance(original, dict) and len(original) == 2738, '2738-file ancestor snapshot required')
    ancestor = git('ls-tree', '-r', '--name-only', '-z', ANCESTOR, root=root).decode().split('\0')
    require(set(filter(None, ancestor)) == set(original), 'ancestor snapshot inventory differs from Git tree')
    for relative, digest in original.items():
        path = Path(root)/relative
        require(path.is_file() and not path.is_symlink() and sha(path) == digest
                and digest_bytes(git('show', ANCESTOR+':'+relative, root=root)) == digest,
                'historical ancestor file changed: '+relative)
    paths = list(filter(None, git('ls-tree', '-r', '--name-only', '-z', P0, root=root).decode().split('\0')))
    for relative in paths:
        path = Path(root)/relative
        require(path.is_file() and not path.is_symlink(), 'P0 tracked file deleted or replaced: '+relative)
        if relative != DECISION_REL:
            require(sha(path) == digest_bytes(git('show', P0+':'+relative, root=root)),
                    'P0 source, qualification or report changed: '+relative)
    old_decision = git('show', P0+':'+DECISION_REL, root=root)
    require((Path(root)/(REPORT_REL+'P0_DECISION.json')).read_bytes() == old_decision,
            'P0 decision archive must preserve exact original Git bytes')
    # No tracked P0 file may be deleted; the only permitted modification is decision.
    dirty = set(filter(None, git('diff', '--name-only', '-z', P0, root=root).decode().split('\0')))
    if committed:
        require(not git_text('diff', '--name-only', root=root), 'dirty tracked tree after formal commit')
    else:
        require(dirty <= {DECISION_REL}, 'tracked change outside documented formal decision replacement')
    return {'P0_tracked_files_verified': len(paths), 'ancestor_files_verified': len(original),
            'P0_paths': set(paths), 'HEAD': head}


def verify_mechanism_summary(bindings, diagnostic_paths, *, root=ROOT, source=SOURCE):
    """Bind the interpretation report to all sixty actually published inputs."""
    directory = Path(root)/REPORT_REL
    report = load_json(directory/'MECHANISM_SUMMARY.json')
    require(report.get('schema') == 'phase28_anonymous_mechanism_summary_v1'
            and report.get('status') == 'COMPLETE_SIXTY_POSTFIT_DIAGNOSTICS'
            and report.get('source') == source and report.get('stage') == 'prototype'
            and report.get('models') == 30 and report.get('history_reports') == 30
            and report.get('mechanism_reports') == 30,
            'mechanism summary source/schema/coverage differs')
    require(report.get('diagnostic_script_path') == 'scripts/diagnose_phase28.py'
            and report.get('diagnostic_script_sha256') == sha(Path(root)/'scripts/diagnose_phase28.py')
            and report.get('script_path') == 'scripts/report_phase28_mechanism.py'
            and report.get('script_sha256') == sha(Path(root)/'scripts/report_phase28_mechanism.py'),
            'mechanism summary or diagnostic executable SHA differs')
    inputs = {'INNER_RESULTS.json': sha(directory/'INNER_RESULTS.json')}
    require(len(diagnostic_paths) == 60 and len(set(diagnostic_paths)) == 60,
            'sixty unique diagnostic inputs required for mechanism summary')
    for relative in diagnostic_paths:
        require(relative.startswith(REPORT_REL+'diagnostics/'), 'mechanism diagnostic namespace differs')
        within_report = relative[len(REPORT_REL):]
        require_safe_relative(relative)
        inputs[within_report] = sha(Path(root)/relative)
    require(report.get('input_report_sha256') == inputs and len(inputs) == 61,
            'mechanism summary must bind exact current sixty diagnostics and formal results')
    required = {(v['seed'], v['fold'], v['arm']) for v in bindings}
    require(len(bindings) == 30 and required == CELLS, 'mechanism needs thirty unique locked checkpoint bindings')
    expected = sorted(({k: v[k] for k in ('seed', 'fold', 'arm', 'checkpoint_sha256')} for v in bindings),
                      key=lambda v: (v['seed'], v['fold'], v['arm']))
    require(report.get('checkpoint_bindings') == expected, 'mechanism/current checkpoint bindings differ')
    require(report.get('MECHANISM_RESULTS_md_sha256') == sha(directory/'MECHANISM_RESULTS.md'),
            'mechanism Markdown differs from bound summary')
    for name in ('new_forward_calls', 'new_training_steps', 'endpoint_loaders_called'):
        require(type(report.get(name)) is int and report[name] == 0, 'mechanism summary performed predictor/endpoint work')
    for name in ('new_scientific_gate', 'effectiveness_or_histology_or_causality_claimed',
                 'norm_difference_used_as_regularizer_gradient_norm', 'original_FORMAL_DELIVERY_contract_modified',
                 'patient_arrays_or_checkpoints_published'):
        require(report.get(name) is False, 'mechanism summary exceeded descriptive scope')
    return {'summary_sha256': sha(directory/'MECHANISM_SUMMARY.json'),
            'Markdown_sha256': sha(directory/'MECHANISM_RESULTS.md'),
            'bound_inputs': len(inputs), 'checkpoint_bindings': len(expected)}


def verify_final_toolchain(*, root=ROOT, source=SOURCE):
    directory = Path(root)/REPORT_REL
    postfit = (directory/'FORMAL_TOOLCHAIN_QUALIFICATION_POSTFIT.json').exists()
    wide = (directory/'FORMAL_TOOLCHAIN_QUALIFICATION_WIDE.json').exists()
    publication = (directory/'FORMAL_TOOLCHAIN_QUALIFICATION_PUBLICATION.json').exists()
    revision = 'PUBLICATION' if publication else ('POSTFIT' if postfit else ('WIDE' if wide else 'FINAL'))
    path = directory/('FORMAL_TOOLCHAIN_QUALIFICATION_'+revision+'.json')
    expected_tests = WIDE_TEST_FILES if wide else FINAL_TEST_FILES
    expected_tools = WIDE_TOOL_FILES if wide else FINAL_TOOL_FILES
    report = load_json(path)
    require(report.get('status') == 'PASS' and report.get('source') == source
            and report.get('revision') == revision and report.get('synthetic_only') is True
            and report.get('script_path') == 'scripts/qualify_phase28_execution.py'
            and report.get('script_sha256') == sha(Path(root)/'scripts/qualify_phase28_execution.py'),
            'final auxiliary toolchain qualification source/executable differs')
    counts = report.get('counts')
    require(isinstance(counts, dict) and set(counts) == {'tests', 'failures', 'errors', 'skipped'}
            and all(type(v) is int for v in counts.values()) and counts['tests'] > 0
            and all(counts[v] == 0 for v in ('failures', 'errors', 'skipped')),
            'final auxiliary tests did not all pass without skips')
    for field, expected in (('test_files_sha256', expected_tests), ('tool_files_sha256', expected_tools)):
        rows = report.get(field)
        require(isinstance(rows, dict) and set(rows) == expected
                and rows == {name: sha(Path(root)/name) for name in expected},
                'final auxiliary qualification has stale test/tool files: '+field)
    p0_names = ('INPUT_AUDIT.json', 'NUMERICAL_QUALIFICATION.json', 'DEPENDENCY_QUALIFICATION.json')
    require(report.get('P0_receipt_sha256_unchanged') == {name: sha(directory/name) for name in p0_names},
            'final auxiliary qualification changed original P0 receipts')
    require(report.get('initial_auxiliary_receipt_sha256') == sha(directory/'FORMAL_TOOLCHAIN_QUALIFICATION.json')
            and report.get('initial_auxiliary_receipt_retained_with_its_original_test_hashes') is True
            and report.get('original_P0_qualification_reports_unchanged') is True,
            'initial auxiliary receipt is not preserved and bound')
    for name in ('patient_or_validation_endpoint_accesses', 'new_patient_training_fits'):
        require(type(report.get(name)) is int and report[name] == 0, 'auxiliary tests used patient endpoints or fits')
    if wide:
        require(report.get('previous_FINAL_receipt_sha256') == sha(directory/'FORMAL_TOOLCHAIN_QUALIFICATION_FINAL.json'),
                'user-authorized wide qualification must preserve the previous FINAL receipt')
    if postfit:
        require(wide and report.get('previous_WIDE_receipt_sha256') == sha(directory/'FORMAL_TOOLCHAIN_QUALIFICATION_WIDE.json'),
                'post-fit numeric repair must preserve the previous WIDE receipt')
        require(report.get('preserved_evaluation_report_sha256') == {name: sha(directory/name) for name in
                    ('INNER_RESULTS.json', 'INNER_GATE.json', 'decision.json', 'REPLAY_AUDIT.json')},
                'post-fit numeric repair changed existing once-only evaluation artifacts')
    if publication:
        require(postfit and report.get('previous_POSTFIT_receipt_sha256') == sha(directory/'FORMAL_TOOLCHAIN_QUALIFICATION_POSTFIT.json'),
                'publication-only repair must preserve previous POSTFIT qualification')
        require(report.get('preserved_completed_report_sha256') == {name: sha(directory/name) for name in
                ('BUDGET_READOUT_AUDIT.json', 'FORMAL_DELIVERY.json', 'RESULTS_FORMAL.md',
                 'MECHANISM_SUMMARY.json', 'MECHANISM_RESULTS.md')},
                'publication-only privacy repair changed completed scientific reports')
    return {'report_sha256': sha(path), 'tests': counts['tests'], 'failures': 0, 'errors': 0, 'skipped': 0,
            'bound_test_files': len(expected_tests), 'bound_tool_files': len(expected_tools)}


def validate_final_evidence(*, root=ROOT, private=PRIVATE):
    """Verify saved outputs without requesting any endpoint or performing a fit."""
    sys.path.insert(0, str(root))
    from src.phase28_reread import runtime as rt
    from src.phase28_reread import evaluation as ev
    require(Path(root).resolve() == rt.ROOT.resolve() and Path(private) == rt.PRIVATE,
            'formal publisher root/private runtime differs')
    require(rt.source() == SOURCE, 'formal training source drift')
    rt.verify_qualifications(SOURCE)
    qualification = ev.verify_qualifications(rt.REPORT, SOURCE, rt.source_files(),
                                            project_root=root, private_root=private)
    _, artifact = ev.verify_stage(private, 'prototype', SOURCE)
    replay = ev.verify_replay(rt.REPORT/'REPLAY_AUDIT.json', SOURCE, artifact, qualification,
                              rt.source_files(), project_root=root)
    result, gate, decision, budget, delivery = (load_json(rt.REPORT/name) for name in (
        'INNER_RESULTS.json', 'INNER_GATE.json', 'decision.json', 'BUDGET_READOUT_AUDIT.json', 'FORMAL_DELIVERY.json'))
    require(result.get('status') == 'COMPLETE_BIDIRECTIONAL_REREAD_INNER_PROTOTYPE'
            and result.get('source') == SOURCE and result.get('stage') == 'prototype'
            and result.get('artifact_audit') == artifact and result.get('valid_endpoint_callback_calls') == 1
            and result.get('evaluator_sha256') == ev.evaluator_identity()
            and result.get('independent_replay') == replay,
            'formal result source, completed models, evaluator or replay binding differs')
    expected_gate = ev.prototype_gate(result['seed_rows'], source=SOURCE,
        receipt_fingerprint=artifact['receipt_fingerprint'], evaluator_sha256=ev.evaluator_identity())
    expected_decision = {'status': gate['classification'], 'decision': gate['decision'],
        'source': SOURCE, 'stage': 'prototype', 'candidate': 'C11', 'outer_admitted': False,
        'outer_started': False, 'confirmation_claimed': False, 'separate_new_protocol_required': True}
    require(gate == expected_gate and decision == expected_decision,
            'predeclared gate or no-outer decision differs')
    attempt = load_json(Path(private)/'EVALUATION_STARTED_PRIVATE.json')
    require(attempt.get('source') == SOURCE and attempt.get('stage') == 'prototype'
            and attempt.get('evaluator_sha256') == ev.evaluator_identity()
            and attempt.get('replay_report_sha256') == replay['report_sha256']
            and attempt.get('receipt_fingerprint') == artifact['receipt_fingerprint'],
            'once-only evaluation attempt lineage differs')
    require(budget.get('schema') == 'phase28_checkpoint_budget_readout_v1'
            and budget.get('status') == 'PASS_OUTCOME_BLIND_CHECKPOINT_AGGREGATION'
            and budget.get('source') == SOURCE and budget.get('neural_models') == 30
            and budget.get('script_sha256') == sha(Path(root)/'scripts/audit_phase28_budget_readout.py')
            and budget.get('replay_report_sha256') == replay['report_sha256']
            and budget.get('receipt_fingerprint') == artifact['receipt_fingerprint'], 'budget source/binding differs')
    require(len(budget.get('rows', [])) == 30 and
            {(v.get('seed'), v.get('fold'), v.get('arm')) for v in budget['rows']} == CELLS,
            'budget rows must cover all thirty distinct cells')
    for field in ('new_predictor_fits', 'new_optimizer_steps', 'new_forward_calls', 'endpoint_array_requests'):
        require(type(budget.get(field)) is int and budget[field] == 0, 'budget audit performed a forbidden operation')
    require(budget.get('rescaling_coefficients_searched') is False
            and budget.get('admission_gate_changed') is False
            and budget.get('patient_risks_indices_or_model_tensors_public') is False,
            'budget audit modified the locked experiment or disclosed patient-level artifacts')
    require(delivery.get('schema') == 'phase28_formal_delivery_v1'
            and delivery.get('status') == 'COMPLETE_INNER_FORMAL_REPORTS'
            and delivery.get('source') == SOURCE and delivery.get('stage') == 'prototype'
            and delivery.get('neural_models') == 30
            and delivery.get('script_path') == 'scripts/report_phase28_formal.py'
            and delivery.get('script_sha256') == sha(Path(root)/'scripts/report_phase28_formal.py')
            and delivery.get('receipt_fingerprint') == artifact['receipt_fingerprint']
            and delivery.get('independent_replay_report_sha256') == replay['report_sha256']
            and delivery.get('scientific_decision') == gate['decision']
            and delivery.get('outer_started') is False and delivery.get('confirmation_claimed') is False
            and delivery.get('patient_level_artifacts_published') is False,
            'formal report delivery schema/source/privacy/binding differs')
    expected_reports = ('INNER_RESULTS.json', 'INNER_GATE.json', 'decision.json',
                        'BUDGET_READOUT_AUDIT.json', 'RESULTS_FORMAL.md')
    require(delivery.get('report_sha256') == {n: sha(rt.REPORT/n) for n in expected_reports},
            'formal delivery does not bind all actual report bytes')
    diag_rows = []
    script_digest = sha(Path(root)/'scripts/diagnose_phase28.py')
    for binding in artifact['model_bindings']:
        for mode in ('history', 'mechanism'):
            relative = REPORT_REL+f"diagnostics/seed{binding['seed']}_fold{binding['fold']}_{binding['arm']}_{mode}.json"
            d = load_json(Path(root)/relative)
            require(d.get('version') == 'phase28_postfit_diagnostics_v1'
                    and d.get('status') == 'PASS_POSTFIT_'+mode.upper()+'_DIAGNOSTICS'
                    and d.get('mode') == mode and d.get('source') == SOURCE
                    and all(d.get(k) == binding[k] for k in ('seed', 'fold', 'arm', 'checkpoint_sha256'))
                    and d.get('script_path') == 'scripts/diagnose_phase28.py'
                    and d.get('script_sha256') == script_digest
                    and d.get('valid_endpoint_reads') == 0 and d.get('new_optimizer_steps') == 0
                    and d.get('hyperparameters_or_checkpoint_changed') is False,
                    'missing/stale postfit diagnostic: '+relative)
            diag_rows.append(relative)
    mechanism = verify_mechanism_summary(artifact['model_bindings'], diag_rows, root=root)
    toolchain = verify_final_toolchain(root=root)
    return {'neural_models': 30, 'epochs_each': 80, 'complete_optimizer_steps': 2400,
            'receipt_fingerprint': artifact['receipt_fingerprint'], 'replay_report_sha256': replay['report_sha256'],
            'independent_replay_risk_vectors': 300, 'independent_reference_vectors': 90,
            'scientific_decision': gate['decision'], 'diagnostics': diag_rows,
            'mechanism_summary': mechanism, 'final_auxiliary_toolchain': toolchain}


def plan_publication(extra=(), *, root=ROOT, private=PRIVATE, committed=False):
    parent = p0_tree_audit(root=root, private=private, committed=committed)
    evidence = validate_final_evidence(root=root, private=private)
    wide_files = ()
    if (Path(root)/(REPORT_REL+'FORMAL_TOOLCHAIN_QUALIFICATION_WIDE.json')).exists():
        wide_files = ('scripts/tune_phase28_wide_concurrency.py', 'tests/test_phase28_wide_concurrency.py',
            REPORT_REL+'FORMAL_TOOLCHAIN_QUALIFICATION_WIDE.json', REPORT_REL+'CONCURRENCY_WIDE_AUDIT.json',
            REPORT_REL+'WIDE_CONCURRENCY_AUTHORIZATION.json', REPORT_REL+'WIDE_CONCURRENCY_PLAN.md')
    if (Path(root)/(REPORT_REL+'FORMAL_TOOLCHAIN_QUALIFICATION_POSTFIT.json')).exists():
        wide_files += (REPORT_REL+'FORMAL_TOOLCHAIN_QUALIFICATION_POSTFIT.json', REPORT_REL+'POSTFIT_REPAIR_AUDIT.json')
    if (Path(root)/(REPORT_REL+'FORMAL_TOOLCHAIN_QUALIFICATION_PUBLICATION.json')).exists():
        wide_files += (REPORT_REL+'FORMAL_TOOLCHAIN_QUALIFICATION_PUBLICATION.json', REPORT_REL+'PUBLICATION_PRIVACY_REPAIR.md')
    paths = sorted(set((*REQUIRED_NEW, DECISION_REL, *evidence['diagnostics'], *extra, *wide_files)))
    require(MANIFEST_REL not in paths, 'manifest self-inclusion is forbidden')
    rows = []
    for relative in paths:
        require(relative not in parent['P0_paths'] or relative == DECISION_REL,
                'P0 artifacts are immutable and must not be restaged: '+relative)
        path = safe_file(root, relative)
        privacy_file(path)
        rows.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path),
                     'change': 'modified_documented_decision' if relative == DECISION_REL else 'new'})
    manifest = {'schema': 'phase28_formal_public_manifest_v1', 'status': 'PASS', 'training_source': SOURCE,
        'branch': BRANCH, 'origin': ORIGIN, 'formal_parent_commit': P0, 'historical_ancestor_commit': ANCESTOR,
        'P0_tracked_files_verified': parent['P0_tracked_files_verified'],
        'historical_ancestor_files_verified': parent['ancestor_files_verified'],
        'qualification_and_training_source_bytes_unchanged': True, 'P0_files_deleted': 0,
        'P0_decision_preserved_byte_for_byte': True, 'P0_results_and_public_manifest_unchanged': True,
        'patient_arrays_weights_identifiers_or_credentials_published': False,
        'licensed_gene_set_contents_added': False, 'outer_started': False,
        'evidence': {k: v for k, v in evidence.items() if k != 'diagnostics'},
        'diagnostic_reports': len(evidence['diagnostics']), 'files': rows}
    privacy_json(manifest)
    if committed:
        require((Path(root)/MANIFEST_REL).read_bytes() == canonical_bytes(manifest),
                'committed formal manifest differs from revalidated publication plan')
        changed = set(filter(None, git('diff', '--name-only', '-z', P0, 'HEAD', root=root).decode().split('\0')))
        require(changed == set(paths)|{MANIFEST_REL}, 'committed paths differ from exact manifest')
    return manifest


def remote_tip(*, root=ROOT):
    rows = git_text('ls-remote', '--exit-code', 'origin', 'refs/heads/'+BRANCH, root=root).splitlines()
    require(len(rows) == 1 and rows[0].split()[1] == 'refs/heads/'+BRANCH, 'remote branch lookup differs')
    return rows[0].split()[0]


def push_verified(commit, *, root=ROOT):
    require(remote_tip(root=root) in (P0, commit), 'remote advanced; refusing force, pull or overwrite')
    subprocess.check_call(['git', '-C', str(root), 'push', 'origin', 'HEAD:refs/heads/'+BRANCH])
    require(remote_tip(root=root) == commit, 'remote SHA differs after publication')


def publish(manifest, *, root=ROOT):
    require(remote_tip(root=root) == P0, 'remote branch is not the expected P0 parent')
    paths = [row['path'] for row in manifest['files']]
    target = Path(root)/MANIFEST_REL
    content = canonical_bytes(manifest)
    require(not target.exists() or target.read_bytes() == content, 'existing formal manifest differs')
    temporary = target.with_name(target.name+'.partial')
    temporary.write_bytes(content); temporary.replace(target)
    paths.append(MANIFEST_REL)
    for row in manifest['files']:
        require(sha(Path(root)/row['path']) == row['sha256'], 'public candidate changed after planning')
    subprocess.check_call(['git', '-C', str(root), 'add', '--', *paths])
    staged = set(filter(None, git('diff', '--cached', '--name-only', '-z', root=root).decode().split('\0')))
    require(staged == set(paths), 'staged files differ from explicit formal manifest')
    expected = {row['path']: row['sha256'] for row in manifest['files']}
    expected[MANIFEST_REL] = digest_bytes(content)
    require(all(digest_bytes(git('show', ':'+relative, root=root)) == digest for relative, digest in expected.items()),
            'staged bytes differ from inspected anonymous files')
    require(git_text('rev-parse', 'HEAD', root=root) == P0, 'HEAD changed before commit')
    subprocess.check_call(['git', '-C', str(root), 'commit', '-m',
                           'Complete locked Phase28 bidirectional evidence reread inner audit'])
    commit = git_text('rev-parse', 'HEAD', root=root)
    require(git_text('rev-parse', 'HEAD^', root=root) == P0, 'formal commit parent differs')
    push_verified(commit, root=root)
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--publish', action='store_true')
    action.add_argument('--push-committed', action='store_true')
    parser.add_argument('--include-new', action='append', default=[], metavar='REPO_RELATIVE_PATH',
                        help='Explicit additional new Phase28 source/test/anonymous report; never a glob')
    args = parser.parse_args()
    manifest = plan_publication(args.include_new, committed=args.push_committed)
    if args.check:
        print(json.dumps({'status': 'READY_TO_PUBLISH_NO_GIT_OR_REPORT_WRITES', 'source': SOURCE,
                          'planned_files_including_manifest': len(manifest['files'])+1,
                          'neural_models': 30, 'diagnostic_reports': 60})); return
    if args.push_committed:
        commit = git_text('rev-parse', 'HEAD'); push_verified(commit)
    else:
        commit = publish(manifest)
    print(json.dumps({'status': 'FORMAL_PUSH_VERIFIED', 'commit': commit, 'branch': BRANCH,
                      'historical_files_unchanged': 2738, 'new_or_replaced_public_files': len(manifest['files'])+1}))


if __name__ == '__main__':
    main()
