"""One geometry-selected real-fit input qualification, without supervision.

This independently bound probe does not change the locked training source.
Only min/median/max selected-Patch-size fit patients enter the CUDA model.
Public output contains three-patient aggregate measurements, never risk vectors.
"""
from __future__ import annotations

from collections import Counter
import contextlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import h5py
import numpy as np
import torch

from src.phase28_reread import data, runtime, train, training_math


REPORT = runtime.REPORT/'REAL_INPUT_QUALIFICATION.json'
FEATURE_KEYS = frozenset(('patient_ids', 'folds', 'rna_genes', 'rna_gene_symbols',
                         'cna_genes', 'cna_gene_symbols', 'cancer_type'))


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def aggregate(values):
    x = np.asarray(values, np.float64)
    require(x.ndim == 1 and len(x) == 3 and np.isfinite(x).all(), 'invalid anonymous aggregate')
    return {'count': 3, 'minimum': float(x.min()), 'mean': float(x.mean()), 'maximum': float(x.max())}


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), 'nonfinite model tensor')
    elif isinstance(value, dict):
        for item in value.values(): finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value: finite_tree(item)


def rms(value):
    return float(value.detach().double().square().mean().sqrt())


@contextlib.contextmanager
def no_supervision_or_training():
    counters = Counter({name: 0 for name in ('Data.fit_labels', 'Optimizer.__init__',
        'forbidden_central_NPZ_array', 'training_entrypoint', 'array_or_weight_save')})
    reads = Counter()
    original_load = np.load

    class FeaturesOnly:
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *args): self.value.close()
        def __getitem__(self, name):
            if name not in FEATURE_KEYS:
                counters['forbidden_central_NPZ_array'] += 1
                raise RuntimeError('real-input probe rejects non-feature NPZ keys')
            reads[name] += 1
            return self.value[name]

    def load(file, *args, **kwargs):
        try: central = Path(file).resolve() == data.COHORT_NPZ.resolve()
        except TypeError: central = False
        result = original_load(file, *args, **kwargs)
        return FeaturesOnly(result) if central else result

    def blocked(name):
        def reject(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError('real-input qualification forbids '+name)
        return reject

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(np, 'load', load))
        stack.enter_context(patch.object(data.Data, 'fit_labels', blocked('Data.fit_labels')))
        stack.enter_context(patch.object(torch.optim.Optimizer, '__init__', blocked('Optimizer.__init__')))
        stack.enter_context(patch.object(torch, 'save', blocked('array_or_weight_save')))
        for name in ('save', 'savez', 'savez_compressed', 'savetxt'):
            stack.enter_context(patch.object(np, name, blocked('array_or_weight_save')))
        for module, names in ((train, ('train_cell', 'efron_loss', 'risk_gradient_cache')),
                              (training_math, ('efron_loss', 'risk_gradient_cache'))):
            for name in names:
                stack.enter_context(patch.object(module, name, blocked('training_entrypoint')))
        yield counters, reads


def geometry_selection(ds):
    """Only original coordinates of fit patients, before looking at any risk."""
    rows = []
    for patient in np.asarray(ds.fit, np.int64):
        selected = 0
        for index in ds.patient_slides[int(patient)]:
            row = ds.slide_catalog[index]
            before = Path(row['path']).stat()
            require((before.st_size, before.st_mtime_ns) == (row['size'], row['mtime_ns']),
                    'native input metadata changed before geometry selection')
            with h5py.File(row['path'], 'r') as handle:
                coords = np.asarray(handle['coords']).reshape(-1, 2)
                plan = ds._plan(index, coords)
            selected += len(plan['selected_indices'])
        rows.append((selected, int(patient)))
    rows.sort()  # Ties use the immutable canonical index, never an outcome.
    selected = [rows[0], rows[(len(rows)-1)//2], rows[-1]]
    require(len({row[1] for row in selected}) == 3, 'geometry extrema/median are not three distinct fit patients')
    return selected


def diagnose_one(model, ds, patient):
    require(int(patient) in set(map(int, ds.fit)), 'non-fit model input rejected')
    raw = ds.molecular[patient].to('cuda').detach().requires_grad_()
    mask = ds.molecular_mask[patient].to('cuda')
    slides = ds.load_patient_slides(patient, device='cuda')
    encoded = model.encode(raw, mask, ds.gene_ids.to('cuda'), slides)
    output = model.read_encoded(encoded, ds.base[patient], details=True)
    finite_tree(encoded); finite_tree(output)
    # These derivatives test the actual cross-modal graph. Zero or tiny finite
    # derivatives are reported honestly; no minimum effect size is enforced.
    rna_gradient = torch.autograd.grad(output['path_interaction'], raw, retain_graph=True)[0]
    ep_gradient = torch.autograd.grad(output['gene_interaction'], encoded['path_evidence'], retain_graph=True)
    finite_tree(rna_gradient); finite_tree(ep_gradient)
    with torch.no_grad():
        off = model.read_encoded(encoded, ds.base[patient], interaction_off=True)
        require(torch.equal(off['risk'], off['base']), 'OFF does not exactly recover the same own main effects')
        require(torch.equal(off['risk'], ds.base[patient].to('cuda')),
                'formal zero-initialized main heads do not recover the frozen offset')
    changes_p, changes_g, tv_p, tv_g = [], [], [], []
    for layer in output['trace']:
        for entry in layer['path']:
            changes_p.append(entry['change'])
            tv_p.append(.5*(entry['conditional_weight']-entry['base_weight']).abs().sum(-1))
        changes_g.append(layer['gene_change'])
        tv_g.append(.5*(layer['gene_conditional_weight']-layer['gene_base_weight']).abs().sum(-1))
    cp, cg = torch.cat(changes_p), torch.cat(changes_g)
    ptv, gtv = torch.cat(tv_p), torch.cat(tv_g)
    maximum = float(torch.maximum(cp.detach().abs().max(), cg.detach().abs().max()))
    require(maximum <= model.eta+1e-6, 'bounded latent update exceeded eta')
    result = {'selected_Patches': sum(len(slide['features']) for slide in slides), 'WSIs': len(slides),
        'path_tokens': sum(len(value) for value in encoded['path0']),
        'valid_gene_tokens': int(encoded['gene_valid'].sum()),
        'path_reread_attention_TV_mean': float(ptv.detach().double().mean()),
        'path_reread_attention_TV_maximum': float(ptv.detach().max()),
        'gene_reread_attention_TV_mean': float(gtv.detach().double().mean()),
        'gene_reread_attention_TV_maximum': float(gtv.detach().max()),
        'path_token_update_RMS': rms(cp), 'gene_token_update_RMS': rms(cg),
        'latent_update_maximum_absolute': maximum,
        'RNA_to_path_interaction_gradient_L2': float(rna_gradient[:, 0].double().norm()),
        'all_molecular_to_path_interaction_gradient_L2': float(rna_gradient.double().norm()),
        'Ep_to_gene_interaction_gradient_L2': float(torch.stack([v.double().square().sum() for v in ep_gradient]).sum().sqrt()),
        'initial_interaction_squared': float(output['interaction'].detach().double().square()),
        'initial_path_correction_squared': float(output['path_interaction'].detach().double().square()),
        'initial_gene_correction_squared': float(output['gene_interaction'].detach().double().square())}
    require(all(np.isfinite(value) for value in result.values()), 'nonfinite aggregate scalar')
    del encoded, output, off, raw, mask, slides, rna_gradient, ep_gradient
    return result


def main():
    began = time.monotonic()
    require(not REPORT.exists(), 'existing real-input probe receipt: refusing a silent repeat')
    identity, script_sha = runtime.source(), runtime.sha(__file__)
    runtime.atomic_json(REPORT, {'status': 'RUNNING_REAL_FIT_INPUT_QUALIFICATION', 'source': identity,
                                'script_path': 'scripts/probe_phase28_real_inputs.py', 'script_sha256': script_sha})
    stage = 'qualifications'
    try:
        qualifications = runtime.verify_qualifications(identity)
        require(torch.cuda.is_available(), 'official real-input probe requires CUDA')
        with no_supervision_or_training() as (calls, reads):
            stage = 'fit_only_Data_construction'
            ds = data.Data(1, 31, device='cpu', cache_slides=0)
            require(not any(hasattr(ds, name) for name in ('valid_time', 'valid_event', '_valid_time', '_valid_event')),
                    'held-out endpoint attribute present')
            del ds._fit_time, ds._fit_event
            stage = 'outcome_blind_geometry_selection'
            choices = geometry_selection(ds)
            stage = 'official_C11_formal_initialization'
            torch.cuda.reset_peak_memory_stats()
            model = train.make_model(ds, 'C11', 31, 1)
            model.eval()
            original = train.model_hash(model.state_dict())
            rows = []
            stage = 'three_real_fit_inputs_and_cross_gradients'
            compute_began = time.monotonic()
            for selected_count, patient in choices:
                torch.cuda.synchronize()
                row = diagnose_one(model, ds, patient)
                require(row['selected_Patches'] == selected_count, 'selected original-row accounting changed')
                rows.append(row)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            compute_seconds = time.monotonic()-compute_began
            require(train.model_hash(model.state_dict()) == original, 'real probe changed model weights')
            require(not any(calls.values()) and set(reads) == FEATURE_KEYS and all(v == 1 for v in reads.values()),
                    'forbidden supervision/optimizer/save or unexpected NPZ feature access')
            require(runtime.source() == identity and runtime.sha(__file__) == script_sha,
                    'source or probe script changed during execution')
            statistics = {name: aggregate([row[name] for row in rows]) for name in rows[0]
                          if not name.endswith('_squared')}
            for name in ('interaction', 'path_correction', 'gene_correction'):
                statistics['initial_'+name+'_RMS_across_three_patients'] = float(np.sqrt(
                    np.mean([row['initial_'+name+'_squared'] for row in rows])))
            result = {'status': 'PASS_THREE_REAL_FIT_INPUTS_FORWARD_AND_CROSS_GRADIENTS',
                'stage': 'input_qualification_only', 'source': identity, 'source_files': runtime.source_files(),
                'script_path': 'scripts/probe_phase28_real_inputs.py', 'script_sha256': script_sha,
                'qualification_receipt_sha256': qualifications['qualification_receipt_sha256'],
                'prepared_sha256': ds.prepared_sha256, 'fold': 1, 'seed': 31, 'arm': 'C11',
                'patients': 3, 'selection': 'fit_only_selected_original_Patch_count_min_lower_median_max;canonical_index_tiebreak',
                'aggregate_weighting': 'equal_patient_weight_for_three_geometry_selected_examples',
                'statistics': statistics, 'formal_initialization_unchanged': True, 'model_weights_unchanged': True,
                'finite_encoded_and_detail_tensors': True, 'finite_both_cross_direction_autograd': True,
                'cross_gradient_minimum_effect_threshold_applied': False, 'OFF_exact_same_main_and_frozen_offset': True,
                'latent_per_coordinate_update_bound': .1, 'bound_absolute_tolerance': 1e-6,
                'new_optimizer_steps': 0, 'new_predictor_fits': 0, 'valid_patient_forward_calls': 0,
                'endpoint_array_reads': 0, 'fit_label_getter_calls': 0,
                'fit_label_bytes_deserialized_by_existing_constructor_then_deleted': True,
                'canonical_feature_arrays_constructor_scope': 'existing_Data_loads512_feature_rows;only_fit_used_for_selection_and_forward',
                'forbidden_API_call_counts': dict(calls), 'central_feature_array_read_counts': dict(reads),
                'new_weights_raw_arrays_or_patient_IDs_saved': False, 'H5_payload_full_SHA_repeated': False,
                'CUDA_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                'CUDA_peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                'CUDA_compute_seconds': compute_seconds, 'wall_seconds': time.monotonic()-began,
                'not_DSS_increment_or_histology_validation': True}
            runtime.atomic_json(REPORT, result)
        print(json.dumps({'status': result['status'], 'source': identity, 'patients': 3,
                          'wall_seconds': result['wall_seconds']}), flush=True)
    except BaseException as error:
        runtime.atomic_json(REPORT, {'status': 'FAILED_REAL_INPUT_QUALIFICATION_NO_RETRY', 'source': identity,
            'script_path': 'scripts/probe_phase28_real_inputs.py', 'script_sha256': script_sha,
            'failed_stage': stage, 'exception_type': type(error).__name__,
            'no_silent_repeat': True, 'not_DSS_or_histology_evidence': True})
        print(json.dumps({'status': 'FAILED_REAL_INPUT_QUALIFICATION_NO_RETRY', 'stage': stage,
                          'exception_type': type(error).__name__}), flush=True)
        raise


if __name__ == '__main__':
    main()
