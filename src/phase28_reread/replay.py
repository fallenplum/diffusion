"""Independent saved-model inference and risk replay; no predictor training.

The prediction loop below does not call the training prediction/encoding/
context helpers. It invokes the locked model operator on original Data inputs.
Existing prepared files contain fit-label bytes; the replay removes those
attributes and blocks label getters and endpoint files before any inference.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .evaluation import (ARMS, CONTROLS, FOLDS, SEEDS, REPLAY_SCHEMA, REPLAY_FORBIDDEN,
                         require, sha256_file, verify_qualifications, verify_stage)


ATOL = 1e-6
FEATURE_KEYS = frozenset(('patient_ids', 'folds', 'rna_genes', 'rna_gene_symbols',
                          'cna_genes', 'cna_gene_symbols', 'cancer_type'))


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def compare(actual, expected, label, *, exact=False):
    if isinstance(actual, torch.Tensor): actual = actual.detach().cpu().numpy()
    if isinstance(expected, torch.Tensor): expected = expected.detach().cpu().numpy()
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all(),
            'invalid replay dimensions/values: '+label)
    error = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
    require(np.array_equal(a, b) if exact else error <= ATOL, 'saved risk/reference replay differs: '+label)
    return error


@contextlib.contextmanager
def inference_guards(data_module, train_module):
    """Block training, supervision getters and direct endpoint-file access."""
    from src.phase13_rna import data as parent
    counts = dict.fromkeys(REPLAY_FORBIDDEN, 0)
    original_load = np.load

    def forbidden(label):
        def fail(*args, **kwargs):
            counts[label] += 1
            raise RuntimeError('Phase28 inference-only replay forbids '+label)
        return fail

    class FeatureView:
        def __init__(self, wrapped): self.wrapped = wrapped
        def __enter__(self): return self
        def __exit__(self, *args): self.wrapped.close()
        def __getitem__(self, key):
            if key not in FEATURE_KEYS:
                return forbidden('central.forbidden_key')()
            return self.wrapped[key]

    def load(file, *args, **kwargs):
        try: path = Path(file).resolve()
        except TypeError: path = None
        if path is not None and path.name == 'endpoints.npz':
            return forbidden('np.load_endpoint')()
        value = original_load(file, *args, **kwargs)
        return FeatureView(value) if path == data_module.COHORT_NPZ.resolve() else value

    targets = ((data_module.Data, 'fit_labels', 'Data.fit_labels'),
               (train_module, 'train_cell', 'train.train_cell'), (train_module, 'predict', 'train.predict'),
               (train_module, 'forward_patients', 'train.forward_patients'),
               (train_module, 'risk_gradient_cache', 'train.risk_gradient_cache'),
               (train_module, 'efron_loss', 'train.efron_loss'),
               (torch.optim.Optimizer, '__init__', 'Optimizer.__init__'),
               (torch.Tensor, 'backward', 'Tensor.backward'), (torch.autograd, 'grad', 'autograd.grad'),
               (parent, 'endpoints', 'parent.endpoints'))
    with contextlib.ExitStack() as stack:
        for owner, name, label in targets:
            stack.enter_context(patch.object(owner, name, forbidden(label)))
        stack.enter_context(patch.object(np, 'load', load))
        yield counts


def _bank(output, encoded):
    answer = []
    for state in output['states']:
        answer.append({'gene': state['gene'].detach().cpu().clone(),
            'gene_valid': encoded['gene_valid'].detach().cpu().clone(),
            'path': [value.detach().cpu().clone() for value in state['path']],
            'path_valid': [value.detach().cpu().clone() for value in encoded['path_valid']]})
    return answer


def mean_bank(banks):
    """Independent equal-patient/equal-WSI/equal-region matched fit mean."""
    require(bool(banks) and all(len(bank) == 2 for bank in banks), 'two-round nonempty fit bank required')
    answer = []
    for layer in range(2):
        states = [bank[layer] for bank in banks]
        masks = torch.stack([state['gene_valid'] for state in states])
        genes = torch.stack([state['gene'] for state in states])
        mean_gene = (genes*masks[..., None]).sum(0)/masks.sum(0).clamp_min(1)[:, None]
        per_patient = []
        for state in states:
            per_slide = torch.stack([value.reshape(-1, 2, 128).mean(0) for value in state['path']])
            per_patient.append(per_slide.mean(0))
        mean_path = torch.stack(per_patient).mean(0)
        answer.append({'gene': mean_gene, 'gene_valid': masks.any(0), 'path': [mean_path],
                       'path_valid': [torch.ones(2, dtype=torch.bool)]})
    return answer


def foreign_context(bank, encoded, device, *, gene, path):
    """Frozen bank replacement, preserving target own original evidence."""
    result = []
    require(len(bank) == 2, 'two-round context required')
    for state in bank:
        row = {}
        if gene:
            row.update(gene=state['gene'].to(device), gene_valid=state['gene_valid'].to(device))
        if path:
            donor = state['path']; masks = state['path_valid']
            require(bool(donor) and len(donor) == len(masks), 'real donor WSI context required')
            same = len(donor) == len(encoded['path0']) and all(len(a) == len(b) for a, b in zip(donor, encoded['path0']))
            pooled = torch.cat(donor).reshape(-1, 2, 128)
            pooled_mask = torch.cat(masks).reshape(-1, 2)
            values, valid = [], []
            for slide, original in enumerate(encoded['path0']):
                if same:
                    values.append(donor[slide].to(device)); valid.append(masks[slide].to(device))
                else:
                    count = len(original)//2
                    # Exact rational midpoint quantile, avoiding a divide-
                    # then-multiply rounding change at integer boundaries.
                    index = ((2*torch.arange(count)+1)*len(pooled)//(2*count)).long().clamp_max(len(pooled)-1)
                    values.append(pooled[index].reshape(-1, 128).to(device))
                    valid.append(pooled_mask[index].reshape(-1).to(device))
            row.update(path=values, path_valid=valid)
        result.append(row)
    return result


def _encode(model, ds, patient, *, molecular_patient=None):
    device = next(model.parameters()).device
    source = int(patient if molecular_patient is None else molecular_patient)
    return model.encode(ds.molecular[source].to(device), ds.molecular_mask[source].to(device),
                        ds.gene_ids.to(device), ds.load_patient_slides(int(patient), device=device))


def independent_wrong_map(ds):
    from src.phase23_cdei.data import partition_derangement, raw_rna_missing_patterns
    from .data import read_feature_npz
    raw = read_feature_npz()
    patterns = raw_rna_missing_patterns(raw['rna_genes'])
    mapping, _ = partition_derangement(ds.cancer_type, patterns, np.asarray(ds.valid), 280000+ds.seed*10+ds.fold)
    mapping = np.asarray(mapping, np.int64)
    if mapping.shape == (len(ds.patient_ids),): mapping = mapping[ds.valid]
    require(mapping.shape == ds.valid.shape and np.isin(mapping, ds.valid).all()
            and not np.any(mapping == ds.valid), 'independent donor map is not partition-local zero-self')
    return mapping


@torch.no_grad()
def replay_record(record, ds, model, *, wrong_map=None):
    """Independently recompute a saved model's fit risk and nine valid vectors.

    A synthetic harness may provide a deterministic map; formal run_replay uses
    the outcome-blind map independently derived from original feature missingness.
    """
    model.eval(); device = next(model.parameters()).device
    require(not any(hasattr(ds, name) for name in ('_fit_time', '_fit_event', 'valid_time', 'valid_event', 'endpoints')),
            'replay Data view retained supervision arrays')
    compare(record['fit'], ds.fit, 'fit order', exact=True)
    compare(record['valid'], ds.valid, 'valid order', exact=True)
    require(record['prepared_sha256'] == ds.prepared_sha256, 'replay prepared SHA differs')
    for name in ('CP_valid', 'RNA_valid', 'M0_valid'):
        compare(record[name], getattr(ds, name), 'exact '+name, exact=True)
    compare(ds.M0_valid, ds.CP_valid+ds.RNA_valid, 'external unit sum', exact=True)
    before = state_hash(model.state_dict())
    fit_values, banks = [], []
    for patient in ds.fit:
        encoded = _encode(model, ds, int(patient))
        output = model.read_encoded(encoded, ds.base[int(patient)])
        fit_values.append(float(output['risk'])); banks.append(_bank(output, encoded))
    maximum = compare(np.asarray(fit_values), record['fit_risk'], 'fit risk')
    means = mean_bank(banks); del banks
    valid_banks = {}
    for patient in ds.valid:
        encoded = _encode(model, ds, int(patient))
        output = model.read_encoded(encoded, ds.base[int(patient)])
        valid_banks[int(patient)] = _bank(output, encoded)
    mapping = independent_wrong_map(ds) if wrong_map is None else np.asarray(wrong_map, np.int64)
    require(mapping.shape == ds.valid.shape and np.isin(mapping, ds.valid).all()
            and not np.any(mapping == ds.valid), 'invalid replay donor map')
    compare(mapping, record['wrong_context_mapping'], 'saved independent donor map', exact=True)
    lookup = {int(patient): position for position, patient in enumerate(ds.valid)}
    risks = {name: [] for name in CONTROLS}
    for patient, donor in zip(ds.valid, mapping):
        patient, donor = int(patient), int(donor)
        encoded = _encode(model, ds, patient)
        own = model.read_encoded(encoded, ds.base[patient])
        off = model.read_encoded(encoded, ds.base[patient], interaction_off=True)
        require(torch.equal(off['risk'], own['base']), 'OFF not exact same-checkpoint learned main')
        outputs = {'MATCHED': own['risk'], 'OFF': off['risk'],
                   'CONSTANT_VALUES': model.read_encoded(encoded, ds.base[patient],
                       constant_path_value=True, constant_gene_value=True)['risk']}
        for name, bank, use_gene, use_path in (
                ('GENE_MEAN', means, True, False), ('GENE_WRONG', valid_banks[donor], True, False),
                ('PATH_MEAN', means, False, True), ('PATH_WRONG', valid_banks[donor], False, True),
                ('BOTH_WRONG', valid_banks[donor], True, True)):
            context = foreign_context(bank, encoded, device, gene=use_gene, path=use_path)
            outputs[name] = model.read_encoded(encoded, ds.base[patient], context=context)['risk']
        swapped = _encode(model, ds, patient, molecular_patient=donor)
        offset = float(ds.CP_valid[lookup[patient]]+ds.RNA_valid[lookup[donor]])
        outputs['WHOLE_OMICS_WRONG'] = model.read_encoded(swapped, offset)['risk']
        for name in CONTROLS: risks[name].append(float(outputs[name]))
    for name, values in risks.items():
        maximum = max(maximum, compare(np.asarray(values), record['valid_risks'][name], name))
    require(state_hash(model.state_dict()) == before, 'inference changed saved model state')
    return {'risk_vectors_replayed': 10, 'reference_vectors_checked': 3,
            'maximum_absolute_risk_error': maximum, 'model_state_unchanged': True,
            'OFF_exact_same_arm_learned_main': True, 'external_references_elementwise_exact': True}


def verify_raw_h5_registry(path):
    """Rehash each original H5 once; no raw array is published or cached."""
    saved = json.loads(Path(path).read_text()); rows = saved.get('H5_files', [])
    require(saved.get('status') == 'PASS_PRIVATE_INPUT_SOURCE_REGISTRY' and len(rows) == 520,
            'full audited private H5 registry missing')
    total = 0
    for row in rows:
        source = Path(row['path']); before = source.stat()
        require(before.st_size == row['bytes'] and before.st_mtime_ns == row['mtime_ns']
                and sha256_file(source) == row['H5_sha256'], 'original H5 source fingerprint changed')
        after = source.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'H5 changed during replay hash pass')
        total += before.st_size
    return {'original_H5_files_rehashed': len(rows), 'original_H5_bytes_rehashed': total,
            'private_source_paths_or_identifiers_public': False}


def run_replay(identity, *, device='cuda'):
    """Formal inference only; caller must explicitly authorize after all 30 fits."""
    from . import runtime, data, train
    from .model import BidirectionalReread
    from .spatial import OfficialSpatialEncoder
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'deterministic CUBLAS environment must precede torch import')
    torch.use_deterministic_algorithms(True)
    require(device == 'cuda' and torch.cuda.is_available(), 'official replay requires CUDA; no synthetic fallback')
    require(runtime.source() == identity, 'requested training source changed')
    runtime.verify_qualifications(identity)
    files = runtime.source_files()
    qualification = verify_qualifications(runtime.REPORT, identity, files, project_root=runtime.ROOT, private_root=runtime.PRIVATE)
    _, audit = verify_stage(runtime.PRIVATE, 'prototype', identity)
    target = runtime.REPORT/'REPLAY_AUDIT.json'
    lock = runtime.PRIVATE/'REPLAY_STARTED_PRIVATE.json'
    require(not target.exists() and not lock.exists(), 'existing replay/attempt requires explicit review; refusing repeat')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'source': identity, 'stage': 'prototype', 'receipt_fingerprint': audit['receipt_fingerprint']}, stream)
    script_sha = sha256_file(runtime.ROOT/'scripts/replay_phase28.py')
    module_sha = sha256_file(Path(__file__))
    rows = []; maximum = 0.
    bindings = {(row['seed'], row['fold'], row['arm']): row for row in audit['model_bindings']}
    raw_input = verify_raw_h5_registry(runtime.PRIVATE/'H5_SOURCES_PRIVATE.json')
    with inference_guards(data, train) as calls:
        for seed in SEEDS:
            for fold in FOLDS:
                ds = data.Data(fold, seed, device='cpu', cache_slides=4)
                for name in ('_fit_time', '_fit_event'):
                    require(hasattr(ds, name), 'prepared fit-label boundary changed')
                    delattr(ds, name)
                for arm in ARMS:
                    directory = runtime.PRIVATE/'prototype'/f'seed{seed}_fold{fold}'/arm
                    record = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
                    torch.set_num_threads(1); torch.manual_seed(seed+1000*fold); torch.cuda.manual_seed_all(seed+1000*fold)
                    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
                    spatial = OfficialSpatialEncoder()
                    require(not spatial.test_only, 'synthetic scan prohibited in formal replay')
                    model = BidirectionalReread(ds.molecular.shape[1], ds.slot_membership.cpu(), spatial, arm=arm, rank=4).to(device)
                    require(state_hash(model.state_dict()) == record['initial_state_sha256'], 'saved initial model hash changed')
                    model.load_state_dict(record['model_state'], strict=True)
                    result = replay_record(record, ds, model)
                    row = {**bindings[seed, fold, arm], **result}
                    rows.append(row); maximum = max(maximum, result['maximum_absolute_risk_error'])
                    del model, record; gc.collect(); torch.cuda.empty_cache()
                del ds; gc.collect()
        require(not any(calls.values()), 'inference called a forbidden training/endpoint API')
    require(runtime.source() == identity and runtime.source_files() == files, 'training closure changed during replay')
    require(sha256_file(runtime.ROOT/'scripts/replay_phase28.py') == script_sha
            and sha256_file(Path(__file__)) == module_sha, 'replay executable changed during run')
    for binding in audit['model_bindings']:
        directory = runtime.PRIVATE/'prototype'/f"seed{binding['seed']}_fold{binding['fold']}"/binding['arm']
        require(sha256_file(directory/'checkpoint.pt') == binding['checkpoint_sha256']
                and sha256_file(directory/'DONE_PRIVATE.json') == binding['DONE_sha256'], 'private saved artifact changed')
    again = verify_qualifications(runtime.REPORT, identity, files, project_root=runtime.ROOT, private_root=runtime.PRIVATE)
    require(again == qualification, 'qualification artifacts changed during replay')
    report = {'schema': REPLAY_SCHEMA, 'status': 'PASS', 'source': identity, 'stage': 'prototype', 'source_files': files,
        'neural_models': 30, 'seed_partition_cells': 6, 'risk_vectors_replayed': 300, 'reference_vectors_checked': 90,
        'maximum_absolute_risk_error': maximum, 'absolute_tolerance': ATOL, 'rows': rows,
        'new_predictor_fits': 0, 'optimizer_steps': 0, 'endpoint_API_reads': 0, 'forbidden_API_call_counts': calls,
        'private_checkpoints_and_receipts_unchanged': True, 'training_source_bytes_unchanged': True,
        'model_state_unchanged': True, 'external_references_elementwise_exact': True, 'no_valid_endpoints_loaded': True,
        'fit_label_bytes_deserialized_in_prepared': True, 'fit_label_getter_calls': 0,
        'replay_script_sha256': script_sha, 'replay_module_sha256': module_sha,
        'qualification_report_hashes': {key: value['report_sha256'] for key, value in qualification.items()},
        'prediction_loop_independent_of_training_helpers': True, 'raw_input_recheck': raw_input,
        'patient_arrays_weights_or_IDs_public': False}
    require(len(rows) == 30, 'replay did not cover all saved models')
    runtime.atomic_json(target, report)
    return report
