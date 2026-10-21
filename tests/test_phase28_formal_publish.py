"""Synthetic privacy and publication boundaries; no Git/network/patient access."""
from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1]/'scripts/publish_phase28_formal.py'
SPEC = importlib.util.spec_from_file_location('phase28_formal_publish', MODULE)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def test_anonymous_statistics_and_named_model_rows_are_allowed():
    publisher.privacy_json({'source': 'a'*64, 'models': [
        {'arm': 'C11', 'seed': 31, 'Harrell': .7, 'Uno60': .65,
         'fit_risk': {'count': 100, 'mean': 0., 'SD': 1., 'RMS': 1.}},
        {'arm': 'C00', 'seed': 47, 'Harrell': .69, 'Uno60': .64}],
        'central_feature_array_read_counts': {'patient_ids': 1}, 'patient_arrays_public': False})


@pytest.mark.parametrize('value', [[.1], [.1, .2], [[.1, .2]], [True, False], [1, False, .2]])
def test_even_short_or_nested_numeric_measurement_vectors_are_rejected(value):
    with pytest.raises(ValueError, match='vector'):
        publisher.privacy_json({'anything': value})


@pytest.mark.parametrize('key', ['patient_ids', 'gene_symbols', 'coordinates', 'fit_indices',
                                'wrong_context_mapping', 'model_state', 'raw_cna', 'rna_values'])
def test_sensitive_arrays_cannot_hide_as_string_lists_or_empty_lists(key):
    with pytest.raises(ValueError, match='protected identity'):
        publisher.privacy_json({key: []})


def test_only_exact_audited_identity_read_counter_is_accepted():
    publisher.privacy_json({'mechanism': {'central_feature_array_read_counts': {'patient_ids': 1}}})
    with pytest.raises(ValueError):
        publisher.privacy_json({'patient_ids': 512})
    with pytest.raises(ValueError):
        publisher.privacy_json({'other': {'patient_ids': 1}})
    with pytest.raises(ValueError):
        publisher.privacy_json({'central_feature_array_read_counts': {'patient_ids': True}})
    with pytest.raises(ValueError):
        publisher.privacy_json({'mechanism': {'central_feature_array_read_counts': {'patient_ids': [1, 2]}}})


@pytest.mark.parametrize('key', ['risk', 'fit_risk', 'valid_risks', 'weights', 'raw_arrays'])
def test_protected_risk_fields_require_explicit_statistics(key):
    with pytest.raises(ValueError, match='aggregate summary'):
        publisher.privacy_json({key: .123})
    with pytest.raises(ValueError, match='aggregate summary'):
        publisher.privacy_json({key: {'0': .1, '1': .2}})
    publisher.privacy_json({key: {'count': 50, 'RMS': .05, 'mean': .01}})


@pytest.mark.parametrize('separator', ['-', '_'])
def test_identifier_pattern_catches_rna_cna_style_sample_names(separator):
    identifier = separator.join(('TCGA', 'A1', 'B2C3'))
    with pytest.raises(ValueError, match='identifier'):
        publisher.privacy_json({'description': identifier+separator+'01A'})


def test_escaped_identifier_in_json_string_is_also_rejected(tmp_path):
    identifier = '-'.join(('TCGA', 'A1', 'B2C3'))
    path = tmp_path/'example.json'
    path.write_text(json.dumps({'note': identifier}).replace('-', '\\u002d'))
    with pytest.raises(ValueError, match='identifier'):
        publisher.privacy_file(path)


def test_access_credentials_are_rejected_without_real_credentials_in_fixture():
    for prefix in ('hf_', 'ghp_'):
        with pytest.raises(ValueError, match='credential'):
            publisher.privacy_json({'note': prefix+'a'*30})


def test_escaped_identifier_in_python_literal_is_rejected(tmp_path):
    identifier = '-'.join(('TCGA', 'A1', 'B2C3'))
    path = tmp_path/'example.py'
    path.write_text('value = '+repr(identifier).replace('-', '\\u002d'))
    with pytest.raises(ValueError, match='decoded Python string'):
        publisher.privacy_file(path)


@pytest.mark.parametrize('path', ['../outside.json', '/tmp/raw.json', 'C:/raw.json',
    'reports/phase28_bidir_reread/raw.npz', 'reports/phase28_bidir_reread/weights.pt',
    'reports/phase28_bidir_reread/slide.h5', 'scripts/unrelated.py',
    'reports/phase28_bidir_reread/diagnostics/seed99_fold1_C11_history.json'])
def test_public_candidate_paths_fail_closed(path):
    with pytest.raises(ValueError):
        publisher.require_safe_relative(path)


def test_allowed_candidates_are_narrow_phase28_files():
    for path in ('scripts/coordinate_phase28.py', 'tests/test_phase28_formal_publish.py',
                 'reports/phase28_bidir_reread/RESULTS_FORMAL.md',
                 'reports/phase28_bidir_reread/diagnostics/seed31_fold1_C11_mechanism.json'):
        assert publisher.require_safe_relative(path) == path


def test_nonfinite_json_and_scalar_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='nonfinite'):
        publisher.privacy_json({'RMS': float('nan')})
    path = tmp_path/'aggregate.json'; path.write_text('{"mean":NaN}')
    with pytest.raises(ValueError, match='nonfinite'):
        publisher.privacy_file(path)


def test_markdown_cannot_embed_raw_numeric_array(tmp_path):
    path = tmp_path/'report.md'; path.write_text('risk = [0.1, 0.2]')
    with pytest.raises(ValueError, match='array literal'):
        publisher.privacy_file(path)


def test_candidate_contents_remain_within_size_and_real_file_boundaries(tmp_path):
    relative = 'reports/phase28_bidir_reread/AGGREGATES.json'
    path = tmp_path/relative; path.parent.mkdir(parents=True); path.write_text('{"mean":0.1}')
    assert publisher.safe_file(tmp_path, relative) == path.resolve()
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='empty'):
        publisher.safe_file(tmp_path, relative)


def test_check_plan_does_not_stage_or_write_manifest(monkeypatch, tmp_path):
    report = tmp_path/'reports/phase28_bidir_reread'
    report.mkdir(parents=True)
    relative = 'reports/phase28_bidir_reread/AGGREGATES.json'
    (tmp_path/relative).write_text('{"mean":0.1}')
    monkeypatch.setattr(publisher, 'REQUIRED_NEW', (relative,))
    monkeypatch.setattr(publisher, 'DECISION_REL', relative)
    monkeypatch.setattr(publisher, 'p0_tree_audit', lambda **kwargs: {
        'P0_paths': set(), 'P0_tracked_files_verified': 1, 'ancestor_files_verified': 2738})
    monkeypatch.setattr(publisher, 'validate_final_evidence', lambda **kwargs: {
        'neural_models': 30, 'diagnostics': [], 'scientific_decision': 'STOP'})
    monkeypatch.setattr(publisher, 'git', lambda *a, **k: (_ for _ in ()).throw(AssertionError('no Git writes')))
    manifest = publisher.plan_publication(root=tmp_path, private=tmp_path)
    assert manifest['status'] == 'PASS' and len(manifest['files']) == 1
    assert not (tmp_path/publisher.MANIFEST_REL).exists()


def test_p0_source_cannot_be_added_as_new_candidate(monkeypatch, tmp_path):
    path = 'scripts/run_phase28.py'
    decision = tmp_path/publisher.DECISION_REL
    decision.parent.mkdir(parents=True); decision.write_text('{"decision":"STOP"}')
    monkeypatch.setattr(publisher, 'REQUIRED_NEW', (path,))
    monkeypatch.setattr(publisher, 'p0_tree_audit', lambda **kwargs: {
        'P0_paths': {path}, 'P0_tracked_files_verified': 1, 'ancestor_files_verified': 2738})
    monkeypatch.setattr(publisher, 'validate_final_evidence', lambda **kwargs: {'diagnostics': []})
    with pytest.raises(ValueError, match='immutable'):
        publisher.plan_publication(root=tmp_path, private=tmp_path)


def test_full_actual_evaluator_statistics_tree_is_privacy_compatible():
    """Exercise the actual statistical output schema, not a hand-picked subset."""
    import numpy as np
    from src.phase28_reread import evaluation as ev
    n = 18
    time = np.arange(1., n+1)
    event = np.ones(n); event[[2, 3, 7, 10, 13, 16, 17]] = 0
    cells = {}
    for seed in ev.SEEDS:
        for fold, valid in zip(ev.FOLDS, (np.arange(4), np.arange(4, 9), np.arange(9, 15))):
            fit = np.setdiff1d(np.arange(n), valid)
            risks = {name: -time[valid] for name in ev.RISK_NAMES}
            risks['C11'] = time[valid] if fold == 2 else -time[valid]
            cells[seed, fold] = {'fit': fit, 'valid': valid, 'risks': risks}
    endpoints = {'time': time, 'event': event, 'valid': np.ones(n, bool)}
    point = ev.point_results(cells, endpoints, cohort_size=n)
    paired = ev.pair_results(cells, endpoints, point, cohort_size=n)
    gate = ev.prototype_gate(point['seed_rows'], source=publisher.SOURCE,
                            receipt_fingerprint='b'*64, evaluator_sha256='c'*64)
    # JSON serialization reproduces the actual persisted public value types.
    public = json.loads(json.dumps({'status': 'synthetic', 'source': publisher.SOURCE,
                                   **point, **paired, 'gate': gate}))
    publisher.privacy_json(public)


def bound_mechanism(root):
    """Minimal algebraic provenance fixture; every input file is synthetic."""
    directory = root/publisher.REPORT_REL
    directory.mkdir(parents=True)
    (root/'scripts').mkdir()
    for script in ('diagnose_phase28.py', 'report_phase28_mechanism.py'):
        (root/'scripts'/script).write_text('synthetic = True\n')
    (directory/'INNER_RESULTS.json').write_text('{"synthetic":true}')
    (directory/'MECHANISM_RESULTS.md').write_text('# Synthetic anonymous mechanism\n')
    bindings, paths = [], []
    for seed, fold, arm in sorted(publisher.CELLS):
        bindings.append({'seed': seed, 'fold': fold, 'arm': arm,
                         'checkpoint_sha256': publisher.digest_bytes(f'synthetic-{seed}-{fold}-{arm}'.encode())})
        for mode in ('history', 'mechanism'):
            relative = publisher.REPORT_REL+f'diagnostics/seed{seed}_fold{fold}_{arm}_{mode}.json'
            path = root/relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"synthetic":true}')
            paths.append(relative)
    inputs = {'INNER_RESULTS.json': publisher.sha(directory/'INNER_RESULTS.json')}
    inputs.update({p[len(publisher.REPORT_REL):]: publisher.sha(root/p) for p in paths})
    payload = {'schema': 'phase28_anonymous_mechanism_summary_v1', 'status': 'COMPLETE_SIXTY_POSTFIT_DIAGNOSTICS',
        'source': 'a'*64, 'stage': 'prototype', 'models': 30, 'history_reports': 30, 'mechanism_reports': 30,
        'diagnostic_script_path': 'scripts/diagnose_phase28.py',
        'diagnostic_script_sha256': publisher.sha(root/'scripts/diagnose_phase28.py'),
        'script_path': 'scripts/report_phase28_mechanism.py',
        'script_sha256': publisher.sha(root/'scripts/report_phase28_mechanism.py'),
        'input_report_sha256': inputs, 'checkpoint_bindings': copy.deepcopy(bindings),
        'MECHANISM_RESULTS_md_sha256': publisher.sha(directory/'MECHANISM_RESULTS.md'),
        'new_forward_calls': 0, 'new_training_steps': 0, 'endpoint_loaders_called': 0,
        'new_scientific_gate': False, 'effectiveness_or_histology_or_causality_claimed': False,
        'norm_difference_used_as_regularizer_gradient_norm': False,
        'original_FORMAL_DELIVERY_contract_modified': False, 'patient_arrays_or_checkpoints_published': False}
    (directory/'MECHANISM_SUMMARY.json').write_text(json.dumps(payload))
    return bindings, paths, payload


@pytest.mark.parametrize('tamper', ['source', 'status', 'models', 'history', 'mechanism', 'script',
    'diagnostic_script', 'input_missing', 'input_digest', 'checkpoint', 'Markdown', 'endpoint', 'gate'])
def test_mechanism_summary_rejects_stale_or_incomplete_binding(tmp_path, tamper):
    bindings, paths, payload = bound_mechanism(tmp_path)
    if tamper == 'source': payload['source'] = 'f'*64
    if tamper == 'status': payload['status'] = 'PARTIAL'
    if tamper == 'models': payload['models'] = 29
    if tamper == 'history': payload['history_reports'] = 29
    if tamper == 'mechanism': payload['mechanism_reports'] = 29
    if tamper == 'script': payload['script_sha256'] = 'f'*64
    if tamper == 'diagnostic_script': payload['diagnostic_script_sha256'] = 'f'*64
    if tamper == 'input_missing': payload['input_report_sha256'].pop('INNER_RESULTS.json')
    if tamper == 'input_digest': payload['input_report_sha256']['INNER_RESULTS.json'] = 'f'*64
    if tamper == 'checkpoint': payload['checkpoint_bindings'][0]['checkpoint_sha256'] = 'f'*64
    if tamper == 'Markdown': payload['MECHANISM_RESULTS_md_sha256'] = 'f'*64
    if tamper == 'endpoint': payload['endpoint_loaders_called'] = 1
    if tamper == 'gate': payload['new_scientific_gate'] = True
    (tmp_path/publisher.REPORT_REL/'MECHANISM_SUMMARY.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        publisher.verify_mechanism_summary(bindings, paths, root=tmp_path, source='a'*64)


def test_mechanism_summary_binds_exactly_61_inputs_and_30_checkpoints(tmp_path):
    bindings, paths, payload = bound_mechanism(tmp_path)
    verified = publisher.verify_mechanism_summary(bindings, paths, root=tmp_path, source='a'*64)
    assert verified['bound_inputs'] == 61 and verified['checkpoint_bindings'] == 30
    publisher.privacy_json(payload)
    assert 'scripts/report_phase28_mechanism.py' in publisher.REQUIRED_NEW
    assert 'tests/test_phase28_mechanism_report.py' in publisher.REQUIRED_NEW
    assert publisher.REPORT_REL+'MECHANISM_SUMMARY.json' in publisher.REQUIRED_NEW
    assert publisher.REPORT_REL+'MECHANISM_RESULTS.md' in publisher.REQUIRED_NEW


def test_actual_mechanism_generate_outputs_and_all_diagnostics_pass_privacy(tmp_path):
    fixture_file = Path(__file__).with_name('test_phase28_mechanism_report.py')
    spec = importlib.util.spec_from_file_location('mechanism_publisher_fixture', fixture_file)
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    directory = tmp_path/publisher.REPORT_REL; directory.mkdir(parents=True)
    scripts = tmp_path/'scripts'; scripts.mkdir()
    for name in ('diagnose_phase28.py', 'report_phase28_mechanism.py'):
        (scripts/name).write_bytes((MODULE.parent/name).read_bytes())
    results, _ = fixture.fixture_reports(directory)
    payload = fixture.report.generate(directory, fixture.SOURCE,
        diagnostic_script_sha256=publisher.sha(scripts/'diagnose_phase28.py'),
        summarizer_script_sha256=publisher.sha(scripts/'report_phase28_mechanism.py'))
    paths = sorted((p.relative_to(tmp_path).as_posix() for p in (directory/'diagnostics').glob('*.json')))
    publisher.verify_mechanism_summary(results['artifact_audit']['model_bindings'], paths,
                                       root=tmp_path, source=fixture.SOURCE)
    for path in (*directory.glob('*.json'), *directory.glob('*.md'), *(directory/'diagnostics').glob('*.json')):
        publisher.privacy_file(path)
    assert payload['models'] == 30


def bound_final_toolchain(root):
    directory = root/publisher.REPORT_REL; directory.mkdir(parents=True)
    all_files = publisher.FINAL_TEST_FILES|publisher.FINAL_TOOL_FILES
    for name in all_files:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('synthetic = True\n')
    p0 = {}
    for name in ('INPUT_AUDIT.json', 'NUMERICAL_QUALIFICATION.json', 'DEPENDENCY_QUALIFICATION.json'):
        (directory/name).write_text('{"synthetic":true}'); p0[name] = publisher.sha(directory/name)
    initial = directory/'FORMAL_TOOLCHAIN_QUALIFICATION.json'; initial.write_text('{"synthetic":true}')
    report = {'status': 'PASS', 'source': 'a'*64, 'revision': 'FINAL', 'synthetic_only': True,
        'script_path': 'scripts/qualify_phase28_execution.py',
        'script_sha256': publisher.sha(root/'scripts/qualify_phase28_execution.py'),
        'counts': {'tests': 50, 'failures': 0, 'errors': 0, 'skipped': 0},
        'test_files_sha256': {name: publisher.sha(root/name) for name in publisher.FINAL_TEST_FILES},
        'tool_files_sha256': {name: publisher.sha(root/name) for name in publisher.FINAL_TOOL_FILES},
        'P0_receipt_sha256_unchanged': p0, 'initial_auxiliary_receipt_sha256': publisher.sha(initial),
        'initial_auxiliary_receipt_retained_with_its_original_test_hashes': True,
        'original_P0_qualification_reports_unchanged': True,
        'patient_or_validation_endpoint_accesses': 0, 'new_patient_training_fits': 0}
    (directory/'FORMAL_TOOLCHAIN_QUALIFICATION_FINAL.json').write_text(json.dumps(report))
    return report


@pytest.mark.parametrize('tamper', ['source', 'status', 'revision', 'failure', 'error', 'skip',
    'test_missing', 'tool_missing', 'test_SHA', 'tool_SHA', 'P0_SHA', 'initial_SHA', 'endpoint'])
def test_final_toolchain_requires_current_all_pass_tests_tools_and_preserved_receipts(tmp_path, tamper):
    report = bound_final_toolchain(tmp_path)
    if tamper == 'source': report['source'] = 'f'*64
    if tamper == 'status': report['status'] = 'FAIL'
    if tamper == 'revision': report['revision'] = 'INITIAL_AUXILIARY'
    if tamper == 'failure': report['counts']['failures'] = 1
    if tamper == 'error': report['counts']['errors'] = 1
    if tamper == 'skip': report['counts']['skipped'] = 1
    if tamper == 'test_missing': report['test_files_sha256'].pop(next(iter(publisher.FINAL_TEST_FILES)))
    if tamper == 'tool_missing': report['tool_files_sha256'].pop(next(iter(publisher.FINAL_TOOL_FILES)))
    if tamper == 'test_SHA': report['test_files_sha256'][next(iter(publisher.FINAL_TEST_FILES))] = 'f'*64
    if tamper == 'tool_SHA': report['tool_files_sha256'][next(iter(publisher.FINAL_TOOL_FILES))] = 'f'*64
    if tamper == 'P0_SHA': report['P0_receipt_sha256_unchanged']['INPUT_AUDIT.json'] = 'f'*64
    if tamper == 'initial_SHA': report['initial_auxiliary_receipt_sha256'] = 'f'*64
    if tamper == 'endpoint': report['patient_or_validation_endpoint_accesses'] = 1
    (tmp_path/publisher.REPORT_REL/'FORMAL_TOOLCHAIN_QUALIFICATION_FINAL.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        publisher.verify_final_toolchain(root=tmp_path, source='a'*64)


def test_final_toolchain_receipt_is_anonymous_and_binds_six_tests_nine_tools(tmp_path):
    report = bound_final_toolchain(tmp_path)
    verified = publisher.verify_final_toolchain(root=tmp_path, source='a'*64)
    assert verified['bound_test_files'] == 6 and verified['bound_tool_files'] == 9
    publisher.privacy_json(report)
    publisher.require_safe_relative(publisher.REPORT_REL+'FORMAL_TOOLCHAIN_QUALIFICATION_FINAL.json')


def module_gradient_fixture(kind):
    if kind == 'budget':
        return ('rows', '[]', 'training', 'data_vs_total_gradient_by_module'), {
            'first_data_L2': 0., 'last_data_L2': .1, 'max_data_L2': .2,
            'epochs_nonzero_data_gradient': 79, 'last_total_L2': .11}
    if kind == 'history':
        return ('history', 'data_versus_total_gradient_by_module'), {
            'data_L2_first10_mean': .01, 'data_L2_last10_mean': .1, 'data_L2_maximum': .2,
            'epochs_with_nonzero_data_gradient': 79, 'total_including_L2_last10_mean': .11,
            'data_gradient_L2_curve': [{'epoch': i+1, 'L2': .1} for i in range(80)]}
    stats = {'mean_equal_cells': .1, 'SD_between_cells': .01, 'minimum_cell': .08,
             'maximum_cell': .12, 'available_cells': 6, 'expected_cells': 6}
    return ('arms', '[]', 'last10_parameter_gradient_norms'), {
        'data_gradient_L2': copy.deepcopy(stats), 'total_including_L2_gradient_L2': copy.deepcopy(stats),
        'norm_difference_is_not_regularizer_gradient_norm': True}


def wrap_module_gradient(trail, body):
    result = {'gene_evidence': body}
    for key in reversed(trail): result = [result] if key == '[]' else {key: result}
    return result


@pytest.mark.parametrize('kind', ['budget', 'history', 'summary'])
def test_exact_gene_module_gradient_statistics_are_public_aggregates(kind):
    trail, body = module_gradient_fixture(kind)
    publisher.privacy_json(wrap_module_gradient(trail, body))
    with pytest.raises(ValueError): publisher.privacy_json({'gene_evidence': body})


@pytest.mark.parametrize('kind', ['budget', 'history', 'summary'])
@pytest.mark.parametrize('tamper', ['raw_vector', 'extra_tensor'])
def test_gradient_module_exception_does_not_admit_raw_gene_arrays(kind, tamper):
    trail, body = module_gradient_fixture(kind)
    if tamper == 'raw_vector': body = [1., 2., 3.]
    else: body['raw_gene_measurement'] = [1., 2., 3.]
    with pytest.raises(ValueError): publisher.privacy_json(wrap_module_gradient(trail, body))


@pytest.mark.parametrize('tamper', ['short_curve', 'duplicate_epoch', 'negative_norm', 'non_scalar'])
def test_gene_module_history_requires_exact_eighty_scalar_epoch_norms(tamper):
    trail, body = module_gradient_fixture('history')
    if tamper == 'short_curve': body['data_gradient_L2_curve'].pop()
    if tamper == 'duplicate_epoch': body['data_gradient_L2_curve'][3]['epoch'] = 3
    if tamper == 'negative_norm': body['data_gradient_L2_curve'][3]['L2'] = -.1
    if tamper == 'non_scalar': body['data_gradient_L2_curve'][3]['L2'] = [.1, .2]
    with pytest.raises(ValueError): publisher.privacy_json(wrap_module_gradient(trail, body))


@pytest.mark.parametrize('kind', ['last_data_gradient_L2_by_module', 'last_total_gradient_L2_by_module'])
def test_formal_artifact_gene_module_norm_is_a_single_nonnegative_scalar(kind):
    trail = ('artifact_audit', 'rows', '[]', 'training', kind)
    publisher.privacy_json(wrap_module_gradient(trail, .1))
    for value in ([.1,.2], {'raw': [.1,.2]}, -.1, 'tensor'):
        with pytest.raises(ValueError): publisher.privacy_json(wrap_module_gradient(trail, value))
