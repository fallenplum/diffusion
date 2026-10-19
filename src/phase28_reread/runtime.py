"""New Phase28 namespace; historical experiment source files stay immutable."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = Path('/root/autodl-tmp/phase28_reread_private')
REPORT = ROOT/'reports/phase28_bidir_reread'
CONFIG = ROOT/'configs/phase28_bidir_reread.json'
PARENT = 'd8f837bb97cdccea8960b0a4b1648a1a56b55a14'
LEGACY_SOURCE = '71a26be63273a0aa6a5f3c29aedbe98ea64b52d1b4dd3c45fbff1a2059c4f548'
ENV_PYTHON = PRIVATE/'env/bin/python'
ARMS = ('C00', 'C10', 'C01', 'C11', 'DIRECT')
CONTROLS = ('MATCHED', 'OFF', 'GENE_MEAN', 'GENE_WRONG', 'PATH_MEAN', 'PATH_WRONG',
            'BOTH_WRONG', 'CONSTANT_VALUES', 'WHOLE_OMICS_WRONG')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def cfg():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def source_files():
    paths = ['src/phase28_reread/'+name+'.py' for name in
             ('__init__', 'runtime', 'data', 'spatial', 'model', 'training_math', 'train')]
    paths += ['configs/phase28_bidir_reread.json', 'scripts/run_phase28.py']
    return {p: sha(ROOT/p) for p in paths}


def source():
    from src.phase27_readout.runtime import source as inherited
    if inherited() != LEGACY_SOURCE or cfg()['parent_commit'] != PARENT:
        raise ValueError('Phase28 inherited source/parent drift')
    return hashlib.sha256(json.dumps({'ancestor_source': LEGACY_SOURCE, 'files': source_files()},
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _require(value, message):
    if not bool(value):
        raise ValueError(message)


def _qualification_script(evidence, path_key, sha_key, expected):
    """Only the literal approved repository-relative entrypoint is accepted."""
    _require(evidence.get(path_key) == expected, 'qualification script path not approved: '+path_key)
    path = ROOT/expected
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError('qualification script resolves outside repository') from None
    _require(relative == expected and resolved.is_file(), 'qualification script path alias/symlink rejected')
    actual = sha(resolved)
    _require(evidence.get(sha_key) == actual, 'qualification script SHA mismatch: '+sha_key)
    return actual


def _official_module_files(evidence, environment):
    modules = evidence.get('official_modules')
    required = {'mamba_ssm', 'causal_conv1d', 'selective_scan_cuda', 'causal_conv1d_cuda',
                'mamba_ssm.modules.mamba_simple', 'mamba_ssm.ops.selective_scan_interface',
                'causal_conv1d.causal_conv1d_interface'}
    _require(isinstance(modules, dict) and required <= set(modules),
             'official dependency module coverage incomplete')
    checked = {}
    for name, record in sorted(modules.items()):
        _require(isinstance(name, str) and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', name)
                 and isinstance(record, dict) and set(record) == {'filename', 'sha256'},
                 'invalid official module receipt schema')
        filename = record['filename']
        _require(isinstance(filename, str) and filename and Path(filename).name == filename
                 and '/' not in filename and '\\' not in filename,
                 'official module filename must be basename only')
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            raise ValueError('cannot resolve qualified official module: '+name) from None
        _require(spec is not None and isinstance(spec.origin, str)
                 and spec.origin not in ('built-in', 'frozen'), 'official module lacks a real file: '+name)
        path = Path(spec.origin).resolve(strict=True)
        try:
            path.relative_to(environment)
        except ValueError:
            raise ValueError('official module is outside qualified environment: '+name) from None
        _require(path.is_file() and path.name == filename and sha(path) == record['sha256'],
                 'official module basename/file SHA differs: '+name)
        checked[name] = {'filename': filename, 'sha256': record['sha256']}
    return checked


def _input_source_files(inputs, code):
    """Bind audited original files without rereading H5 payloads or endpoints."""
    from . import data
    registry_path = PRIVATE/'H5_SOURCES_PRIVATE.json'
    before = sha(registry_path)
    _require(inputs.get('private_registry_sha256') == before, 'input private source registry SHA differs')
    registry = json.loads(registry_path.read_text(encoding='utf-8-sig'))
    _require(registry.get('status') == 'PASS_PRIVATE_INPUT_SOURCE_REGISTRY'
             and registry.get('version') == 'phase28_input_sources_v1'
             and registry.get('input_code_sha256') == code
             and registry.get('full_original_H5_and_coords_audited') is True,
             'input private source registry schema/code differs')
    root = data.SLIDE_ROOT.resolve(strict=True)
    rows = registry.get('H5_files')
    _require(isinstance(rows, list) and len(rows) == 520, 'input H5 registry coverage differs')
    paths, owners, catalog = set(), set(), []
    for index, row in enumerate(rows):
        _require(isinstance(row, dict) and row.get('index') == index
                 and type(row.get('owner')) is int and 0 <= row['owner'] < 512
                 and type(row.get('bytes')) is int and row['bytes'] > 0
                 and type(row.get('mtime_ns')) is int, 'input H5 registry metadata schema differs')
        path = Path(row['path'])
        _require(not path.is_symlink(), 'input H5 symlink source rejected')
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            raise ValueError('audited H5 path is outside native slide root') from None
        _require(resolved.is_file() and resolved.suffix == '.h5' and resolved not in paths,
                 'input H5 source is missing/duplicated or not an H5 file')
        metadata = resolved.stat()
        _require((metadata.st_size, metadata.st_mtime_ns) == (row['bytes'], row['mtime_ns']),
                 'audited H5 size/mtime metadata changed')
        paths.add(resolved); owners.add(row['owner'])
        catalog.append({'relative_path': relative, 'owner': row['owner'], 'index': index,
                        'bytes': metadata.st_size, 'mtime_ns': metadata.st_mtime_ns})
    _require(owners == set(range(512)), 'input H5 patient coverage differs')
    current_paths = {p.resolve(strict=True) for p in data.SLIDE_ROOT.glob('*/*.h5')}
    _require(current_paths == paths, 'current native H5 catalogue differs from audited catalogue')
    _require(inputs.get('geometry', {}).get('slides') == len(rows), 'input H5 audit count differs')
    fingerprints = {}
    for key, path, expected in (
            ('canonical_NPZ_sha256', data.COHORT_NPZ, data.COHORT_SHA256),
            ('official_Hallmark_2024_1_sha256', data.GMT_PATH, data.GMT_SHA256)):
        actual = sha(path)
        _require(registry.get(key) == inputs.get(key) == expected == actual,
                 'audited canonical NPZ/Hallmark source fingerprint differs')
        fingerprints[key] = actual
    prepared = registry.get('prepared_sha256_by_fold')
    _require(isinstance(prepared, dict) and set(prepared) == {'1', '2', '3'},
             'audited prepared fold coverage differs')
    fold_rows = inputs.get('folds')
    _require(isinstance(fold_rows, list) and len(fold_rows) == 3
             and {row.get('fold') for row in fold_rows} == {1, 2, 3}, 'input fold source rows differ')
    reported = {str(row['fold']): row.get('prepared_sha256') for row in fold_rows}
    for fold, digest in prepared.items():
        actual = sha(data.PREPARED_ROOT/('prototype_fold'+fold+'.pt'))
        _require(actual == digest == reported[fold], 'audited frozen prepared file fingerprint differs')
    _require(sha(registry_path) == before, 'input private registry changed during metadata verification')
    return {'input_private_registry_sha256': before,
            'native_H5_metadata_checked_count': len(rows), 'native_H5_patient_coverage': len(owners),
            'native_H5_catalog_metadata_sha256': hashlib.sha256(json.dumps(catalog, sort_keys=True,
                            separators=(',', ':')).encode()).hexdigest(),
            'native_H5_full_SHA_recomputed_here': False,
            'native_H5_full_SHA_bound_by_original_INPUT': True,
            'input_source_fingerprints': fingerprints, 'prepared_sha256_by_fold': prepared}


def verify_qualifications(identity):
    """Fail closed on stale reports, code, dependencies or interpreter prefix.

    The INPUT certificate predates the final training source, so its original
    data/spatial/audit and geometry-supplement script hashes are checked directly.
    CUDA numerical/dependency certificates must bind the complete current source
    map. No model/data/endpoint is loaded by this guard.
    """
    _require(identity == source(), 'qualification request uses a stale training identity')
    active_files = source_files()
    environment = ENV_PYTHON.parent.parent.resolve()
    _require(Path(sys.prefix).resolve() == environment,
             'formal training must use the qualified isolated Python environment')
    reports, report_hashes = {}, {}
    for name in ('INPUT_AUDIT.json', 'NUMERICAL_QUALIFICATION.json', 'DEPENDENCY_QUALIFICATION.json'):
        path = REPORT/name
        _require(path.is_file(), 'formal training requires qualification: '+name)
        digest = sha(path)
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        _require(isinstance(payload, dict) and sha(path) == digest, 'qualification changed while being read')
        reports[name], report_hashes[name] = payload, digest
    inputs = reports['INPUT_AUDIT.json']
    _require(inputs.get('status') == 'PASS'
             and inputs.get('stage') == 'prototype_input_qualification_only', 'input qualification status/stage differs')
    code = inputs.get('input_code_sha256')
    _require(isinstance(code, dict) and set(code) == {'data', 'spatial', 'audit_script'},
             'input original code SHA schema differs')
    for key in ('data', 'spatial'):
        _require(code[key] == active_files.get('src/phase28_reread/'+key+'.py'),
                 'input qualification uses stale '+key+' implementation')
    input_script = sha(ROOT/'scripts/audit_phase28_inputs.py')
    _require(code['audit_script'] == inputs.get('audit_script_sha256') == input_script,
             'input audit script SHA differs')
    geometry = inputs.get('geometry_risks')
    supplement_script = sha(ROOT/'scripts/supplement_phase28_geometry.py')
    _require(isinstance(geometry, dict) and geometry.get('supplement_script_sha256') == supplement_script,
             'input geometry supplement script SHA differs')
    input_sources = _input_source_files(inputs, code)
    registry_sha = input_sources['input_private_registry_sha256']
    for key in ('new_neural_training_steps', 'new_predictor_fits', 'optimizer_constructions',
                'neural_forward_calls', 'valid_endpoint_array_reads'):
        _require(type(inputs.get(key)) is int and inputs[key] == 0, 'input isolation counter differs: '+key)
    forbidden = inputs.get('forbidden_API_call_counts')
    _require(isinstance(forbidden, dict) and bool(forbidden)
             and all(type(value) is int and value == 0 for value in forbidden.values()),
             'input audit forbidden API was used')
    statuses = {'NUMERICAL_QUALIFICATION.json': 'PASS_SYNTHETIC_COMPLETE_MODEL_CUDA_AND_FULL_RISK_GRADIENT_CACHE',
                'DEPENDENCY_QUALIFICATION.json': 'PASS_OFFICIAL_CUDA_SEQUENCE_AND_FULL_EFRON_GRADCACHE'}
    qualifying_script = None
    for name, status in statuses.items():
        evidence = reports[name]
        _require(evidence.get('status') == status and evidence.get('source') == identity
                 and evidence.get('source_files') == active_files, 'qualification source/schema stale: '+name)
        qualifying_script = _qualification_script(evidence, 'script_path', 'script_sha256', 'scripts/qualify_phase28.py')
        for key in ('new_optimizer_steps', 'patient_data_or_endpoint_reads'):
            _require(type(evidence.get(key)) is int and evidence[key] == 0, 'qualification isolation counter differs')
    numerical, dependency = reports['NUMERICAL_QUALIFICATION.json'], reports['DEPENDENCY_QUALIFICATION.json']
    _require(numerical.get('dependency_receipt_sha256') == report_hashes['DEPENDENCY_QUALIFICATION.json'],
             'numerical certificate references a different dependency receipt')
    official_script = _qualification_script(dependency, 'official_script_path', 'official_script_sha256',
                                              'scripts/qualify_phase28_official_mamba.py')
    _require(dependency.get('source_kind') == 'official_release_wheels'
             and dependency.get('normal_fresh_process_import') is True
             and dependency.get('custom_S6_substitute') is False
             and dependency.get('last_state_used_for_objective') is False,
             'official dependency provenance/sequence semantics differ')
    modules = _official_module_files(dependency, environment)
    _require(source() == identity and source_files() == active_files,
             'training source changed during qualification verification')
    _require(all(sha(REPORT/name) == digest for name, digest in report_hashes.items())
             and sha(PRIVATE/'H5_SOURCES_PRIVATE.json') == registry_sha,
             'qualification receipt/registry changed during verification')
    _require(sha(ROOT/'scripts/audit_phase28_inputs.py') == input_script
             and sha(ROOT/'scripts/supplement_phase28_geometry.py') == supplement_script
             and sha(ROOT/'scripts/qualify_phase28.py') == qualifying_script
             and sha(ROOT/'scripts/qualify_phase28_official_mamba.py') == official_script,
             'qualification entrypoint changed during verification')
    return {'status': 'PASS_CURRENT_SOURCE_QUALIFICATIONS', 'source': identity, 'source_identity': identity,
            'source_files': active_files, 'qualification_receipt_sha256': report_hashes,
            'input_audit_script_sha256': input_script, 'geometry_supplement_script_sha256': supplement_script,
            **input_sources, 'qualifying_script_sha256': qualifying_script,
            'official_script_sha256': official_script, 'official_modules': modules,
            'formal_environment_prefix_verified': True, 'official_module_files_verified': len(modules)}


def cell_dir(seed, fold, arm):
    if seed not in cfg()['seeds'] or fold not in cfg()['folds'] or arm not in ARMS:
        raise ValueError('cell outside the fixed 30-fit inner matrix')
    return PRIVATE/'prototype'/f'seed{seed}_fold{fold}'/arm


def verify_done(path, identity, seed, fold, arm):
    path = Path(path); d = json.loads((path/'DONE_PRIVATE.json').read_text())
    expected = {'status': 'COMPLETE', 'source': identity, 'stage': 'prototype',
                'seed': seed, 'fold': fold, 'arm': arm, 'epochs': 80, 'optimizer_steps': 80}
    if any(d.get(k) != v for k, v in expected.items()) or sha(path/'checkpoint.pt') != d.get('checkpoint_sha256'):
        raise ValueError('formal Phase28 completion identity/SHA mismatch')
    return d
