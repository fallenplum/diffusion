"""Pure synthetic qualification receipts; no CUDA, data or training required."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase28_reread import runtime as mod
from src.phase28_reread import data


@pytest.fixture
def certificates(tmp_path, monkeypatch):
    root, private = tmp_path/'repo', tmp_path/'private'
    report, environment = root/'reports/phase28_bidir_reread', private/'env'
    report.mkdir(parents=True); environment.mkdir(parents=True)
    monkeypatch.setattr(mod, 'ROOT', root); monkeypatch.setattr(mod, 'PRIVATE', private)
    monkeypatch.setattr(mod, 'REPORT', report); monkeypatch.setattr(mod, 'ENV_PYTHON', environment/'bin/python')
    monkeypatch.setattr(mod.sys, 'prefix', str(environment))
    identity = 'a'*64
    monkeypatch.setattr(mod, 'source', lambda: identity)
    source_names = ('data', 'spatial', 'runtime')
    for name in source_names:
        p = root/f'src/phase28_reread/{name}.py'; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(name)
    monkeypatch.setattr(mod, 'source_files', lambda: {
        f'src/phase28_reread/{name}.py': mod.sha(root/f'src/phase28_reread/{name}.py') for name in source_names})
    for name in ('audit_phase28_inputs.py', 'supplement_phase28_geometry.py',
                 'qualify_phase28.py', 'qualify_phase28_official_mamba.py'):
        p = root/'scripts'/name; p.parent.mkdir(exist_ok=True); p.write_text('synthetic '+name)
    slide_root = private/'native_slides'; slide_folder = slide_root/'cohort'; slide_folder.mkdir(parents=True)
    h5_rows = []
    for index in range(520):
        p = slide_folder/f'synthetic_slide_{index}.h5'; p.write_bytes(b'synthetic H5 metadata only')
        stat = p.stat()
        h5_rows.append({'index': index, 'owner': index % 512, 'path': str(p),
                        'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    monkeypatch.setattr(data, 'SLIDE_ROOT', slide_root)
    canonical, gmt, prepared_root = private/'features.npz', private/'sets.gmt', private/'prepared'
    canonical.write_bytes(b'synthetic canonical features'); gmt.write_bytes(b'synthetic licensed GMT')
    prepared_root.mkdir()
    prepared = {}
    for fold in (1, 2, 3):
        p = prepared_root/f'prototype_fold{fold}.pt'; p.write_bytes(b'synthetic frozen state '+str(fold).encode())
        prepared[str(fold)] = mod.sha(p)
    for name, value in {'COHORT_NPZ': canonical, 'COHORT_SHA256': mod.sha(canonical), 'GMT_PATH': gmt,
                        'GMT_SHA256': mod.sha(gmt), 'PREPARED_ROOT': prepared_root}.items():
        monkeypatch.setattr(data, name, value)
    registry = private/'H5_SOURCES_PRIVATE.json'
    module_paths = {}
    for name in ('mamba_ssm', 'causal_conv1d', 'selective_scan_cuda', 'causal_conv1d_cuda',
                 'mamba_ssm.modules.mamba_simple', 'mamba_ssm.ops.selective_scan_interface',
                 'causal_conv1d.causal_conv1d_interface'):
        filename = '__init__.py' if name in ('mamba_ssm', 'causal_conv1d') else name+'.so'
        path = environment/'lib'/name/filename; path.parent.mkdir(parents=True); path.write_text('official '+name)
        module_paths[name] = path
    monkeypatch.setattr(mod.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(module_paths[name]))
                        if name in module_paths else None)
    common = {'source': identity, 'source_files': mod.source_files(), 'script_path': 'scripts/qualify_phase28.py',
              'script_sha256': mod.sha(root/'scripts/qualify_phase28.py'),
              'new_optimizer_steps': 0, 'patient_data_or_endpoint_reads': 0}
    inputs = {'status': 'PASS', 'stage': 'prototype_input_qualification_only',
        'input_code_sha256': {'data': mod.source_files()['src/phase28_reread/data.py'],
            'spatial': mod.source_files()['src/phase28_reread/spatial.py'],
            'audit_script': mod.sha(root/'scripts/audit_phase28_inputs.py')},
        'audit_script_sha256': mod.sha(root/'scripts/audit_phase28_inputs.py'),
        'geometry_risks': {'supplement_script_sha256': mod.sha(root/'scripts/supplement_phase28_geometry.py')},
        'canonical_NPZ_sha256': data.COHORT_SHA256, 'official_Hallmark_2024_1_sha256': data.GMT_SHA256,
        'geometry': {'slides': 520}, 'folds': [{'fold': int(k), 'prepared_sha256': v} for k, v in prepared.items()],
        'new_neural_training_steps': 0, 'new_predictor_fits': 0,
        'optimizer_constructions': 0, 'neural_forward_calls': 0, 'valid_endpoint_array_reads': 0,
        'forbidden_API_call_counts': {'forbidden': 0}}
    dependency = {**deepcopy(common), 'status': 'PASS_OFFICIAL_CUDA_SEQUENCE_AND_FULL_EFRON_GRADCACHE',
        'official_script_path': 'scripts/qualify_phase28_official_mamba.py',
        'official_script_sha256': mod.sha(root/'scripts/qualify_phase28_official_mamba.py'),
        'source_kind': 'official_release_wheels', 'normal_fresh_process_import': True,
        'custom_S6_substitute': False, 'last_state_used_for_objective': False,
        'official_modules': {name: {'filename': p.name, 'sha256': mod.sha(p)} for name, p in module_paths.items()}}
    numerical = {**deepcopy(common), 'status': 'PASS_SYNTHETIC_COMPLETE_MODEL_CUDA_AND_FULL_RISK_GRADIENT_CACHE'}
    reports = {'INPUT_AUDIT.json': inputs, 'NUMERICAL_QUALIFICATION.json': numerical,
               'DEPENDENCY_QUALIFICATION.json': dependency}
    registry_payload = {'status': 'PASS_PRIVATE_INPUT_SOURCE_REGISTRY', 'version': 'phase28_input_sources_v1',
        'input_code_sha256': deepcopy(inputs['input_code_sha256']), 'canonical_NPZ_sha256': data.COHORT_SHA256,
        'official_Hallmark_2024_1_sha256': data.GMT_SHA256, 'prepared_sha256_by_fold': prepared,
        'H5_files': h5_rows, 'full_original_H5_and_coords_audited': True}
    registry.write_text(json.dumps(registry_payload))
    inputs['private_registry_sha256'] = mod.sha(registry)
    def save(*, bind_dependency=True):
        (report/'DEPENDENCY_QUALIFICATION.json').write_text(json.dumps(dependency, sort_keys=True))
        if bind_dependency:
            numerical['dependency_receipt_sha256'] = mod.sha(report/'DEPENDENCY_QUALIFICATION.json')
        for name in ('INPUT_AUDIT.json', 'NUMERICAL_QUALIFICATION.json'):
            (report/name).write_text(json.dumps(reports[name], sort_keys=True))
    save()
    return SimpleNamespace(root=root, private=private, report=report, environment=environment,
                           identity=identity, reports=reports, module_paths=module_paths, save=save,
                           registry_payload=registry_payload, registry_path=registry, slide_root=slide_root)


def test_current_bound_receipts_return_anonymous_replayable_hash_record(certificates):
    c = certificates
    result = mod.verify_qualifications(c.identity)
    assert result['source_identity'] == result['source'] == c.identity
    assert result['status'] == 'PASS_CURRENT_SOURCE_QUALIFICATIONS'
    assert result['source_files'] == mod.source_files() and result['official_module_files_verified'] == 7
    assert result['native_H5_metadata_checked_count'] == 520 and result['native_H5_patient_coverage'] == 512
    assert result['native_H5_full_SHA_recomputed_here'] is False
    assert result['formal_environment_prefix_verified'] is True
    for name in c.reports:
        assert result['qualification_receipt_sha256'][name] == mod.sha(c.report/name)
    assert str(c.root) not in json.dumps(result) and str(c.environment) not in json.dumps(result)


@pytest.mark.parametrize('target,field,value,reason', [
    ('INPUT_AUDIT.json', 'status', 'PASS_BUT_NOT_QUALIFIED', 'status'),
    ('NUMERICAL_QUALIFICATION.json', 'status', 'PASS_BUT_NOT_QUALIFIED', 'source/schema'),
    ('DEPENDENCY_QUALIFICATION.json', 'status', 'PASS_BUT_NOT_QUALIFIED', 'source/schema'),
    ('NUMERICAL_QUALIFICATION.json', 'source', 'b'*64, 'source/schema'),
    ('DEPENDENCY_QUALIFICATION.json', 'source_files', {}, 'source/schema'),
    ('NUMERICAL_QUALIFICATION.json', 'script_path', '../scripts/qualify_phase28.py', 'path not approved'),
    ('DEPENDENCY_QUALIFICATION.json', 'script_path', '/scripts/qualify_phase28.py', 'path not approved'),
    ('NUMERICAL_QUALIFICATION.json', 'script_sha256', '0'*64, 'script SHA'),
    ('DEPENDENCY_QUALIFICATION.json', 'official_script_path', 'scripts/other.py', 'path not approved'),
    ('DEPENDENCY_QUALIFICATION.json', 'official_script_sha256', '0'*64, 'script SHA'),
    ('DEPENDENCY_QUALIFICATION.json', 'custom_S6_substitute', True, 'provenance'),
    ('NUMERICAL_QUALIFICATION.json', 'patient_data_or_endpoint_reads', 1, 'isolation'),
    ('INPUT_AUDIT.json', 'valid_endpoint_array_reads', 1, 'isolation'),
])
def test_qualification_receipt_source_schema_and_entrypoints_fail_closed(certificates, target, field, value, reason):
    c = certificates; c.reports[target][field] = value; c.save()
    with pytest.raises(ValueError, match=reason): mod.verify_qualifications(c.identity)


@pytest.mark.parametrize('field', ['data', 'spatial', 'audit_script', 'supplement_script', 'registry'])
def test_original_input_code_supplement_and_private_registry_remain_bound(certificates, field):
    c = certificates
    paths = {'data': c.root/'src/phase28_reread/data.py', 'spatial': c.root/'src/phase28_reread/spatial.py',
             'audit_script': c.root/'scripts/audit_phase28_inputs.py',
             'supplement_script': c.root/'scripts/supplement_phase28_geometry.py',
             'registry': c.private/'H5_SOURCES_PRIVATE.json'}
    paths[field].write_text('changed')
    with pytest.raises(ValueError): mod.verify_qualifications(c.identity)


def test_formal_environment_must_be_exact_qualified_prefix(certificates, monkeypatch):
    c = certificates; monkeypatch.setattr(mod.sys, 'prefix', str(c.environment.parent))
    with pytest.raises(ValueError, match='isolated Python'): mod.verify_qualifications(c.identity)


@pytest.mark.parametrize('mutation,reason', [('changed_file', 'file SHA'), ('missing', 'coverage'),
    ('filename_path', 'basename'), ('outside_environment', 'outside qualified'), ('missing_spec', 'real file')])
def test_actual_official_module_bytes_and_origin_are_qualified(certificates, monkeypatch, mutation, reason):
    c = certificates; dep = c.reports['DEPENDENCY_QUALIFICATION.json']
    if mutation == 'changed_file': c.module_paths['selective_scan_cuda'].write_text('mutated extension')
    elif mutation == 'missing': del dep['official_modules']['causal_conv1d_cuda']
    elif mutation == 'filename_path': dep['official_modules']['mamba_ssm']['filename'] = '../__init__.py'
    elif mutation == 'outside_environment':
        outside = c.root/'selective_scan_cuda.so'; outside.write_text('official selective_scan_cuda')
        c.module_paths['selective_scan_cuda'] = outside
    else: monkeypatch.setattr(mod.importlib.util, 'find_spec', lambda name: None)
    c.save()
    with pytest.raises(ValueError, match=reason): mod.verify_qualifications(c.identity)


def test_numerical_receipt_must_reference_exact_current_dependency_receipt(certificates):
    c = certificates
    c.reports['NUMERICAL_QUALIFICATION.json']['dependency_receipt_sha256'] = '0'*64
    c.save(bind_dependency=False)
    with pytest.raises(ValueError, match='different dependency'): mod.verify_qualifications(c.identity)


def test_source_drift_during_module_check_fails_closed(certificates, monkeypatch):
    c = certificates
    def resolve(name):
        monkeypatch.setattr(mod, 'source', lambda: 'b'*64)
        return SimpleNamespace(origin=str(c.module_paths[name]))
    monkeypatch.setattr(mod.importlib.util, 'find_spec', resolve)
    with pytest.raises(ValueError, match='source changed during'): mod.verify_qualifications(c.identity)


def test_guard_precedes_Data_and_saves_cell_receipt_without_training(certificates, monkeypatch):
    from src.phase28_reread import train
    c = certificates; cell = c.private/'cell'
    monkeypatch.setattr(mod, 'cfg', lambda: {'training': {'learning_rate': .001}})
    monkeypatch.setattr(mod, 'cell_dir', lambda *args: cell)
    def stop_after_guard(*args, **kwargs): raise RuntimeError('synthetic stop before any Data or model')
    monkeypatch.setattr(train, 'Data', stop_after_guard)
    with pytest.raises(RuntimeError, match='synthetic stop'):
        train.train_cell(31, 1, 'C00')
    result = json.loads((cell/'QUALIFICATIONS_PRIVATE.json').read_text())
    assert result['source_identity'] == c.identity and result['official_module_files_verified'] == 7
    assert not (cell/'START_PRIVATE.json').exists()


def test_changed_cell_qualification_receipt_cannot_be_silently_replaced(certificates, monkeypatch):
    from src.phase28_reread import train
    c = certificates; cell = c.private/'cell'; cell.mkdir()
    monkeypatch.setattr(mod, 'cfg', lambda: {'training': {'learning_rate': .001}})
    monkeypatch.setattr(mod, 'cell_dir', lambda *args: cell)
    stored = cell/'QUALIFICATIONS_PRIVATE.json'; stored.write_text('{"source_identity":"old"}')
    before = stored.read_bytes()
    with pytest.raises(ValueError, match='receipts changed'):
        train.train_cell(31, 1, 'C00')
    assert stored.read_bytes() == before


@pytest.mark.parametrize('mutation,reason', [('bytes', 'size/mtime'), ('added_H5', 'catalogue'),
    ('outside_root', 'outside native'), ('missing_owner', 'patient coverage'),
    ('canonical', 'fingerprint'), ('GMT', 'fingerprint'), ('prepared', 'prepared file')])
def test_audited_original_sources_cannot_silently_change(certificates, mutation, reason):
    c = certificates
    if mutation == 'bytes': Path(c.registry_payload['H5_files'][0]['path']).write_bytes(b'changed')
    elif mutation == 'added_H5': (c.slide_root/'cohort/extra.h5').write_bytes(b'new')
    elif mutation in ('outside_root', 'missing_owner'):
        if mutation == 'outside_root':
            p = c.private/'outside.h5'; p.write_bytes(b'synthetic H5 metadata only')
            s = p.stat(); row = c.registry_payload['H5_files'][0]
            row.update(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)
        else:
            for row in c.registry_payload['H5_files']:
                if row['owner'] == 511: row['owner'] = 510
        c.registry_path.write_text(json.dumps(c.registry_payload))
        c.reports['INPUT_AUDIT.json']['private_registry_sha256'] = mod.sha(c.registry_path); c.save()
    elif mutation == 'canonical': data.COHORT_NPZ.write_bytes(b'changed')
    elif mutation == 'GMT': data.GMT_PATH.write_bytes(b'changed')
    else: (data.PREPARED_ROOT/'prototype_fold1.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match=reason): mod.verify_qualifications(c.identity)
