"""Nested, resumable training-budget audit for the repaired TITAN token prompts.

Selection has access only to one stratified inner split of outer-train.
Outer-test predictions are made once per selected model after a fresh refit.
Previous fixed-three-epoch OOF are read only by the final aggregate stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from phase6b_rna_conditioned_titan_concepts import write_json
from phase6b_token_prompt_titan import (
    ENDPOINTS, TOKEN_METHODS, FrozenTextProjector, PromptNet, audit_projector,
    bootstrap_delta, class_weights, fit_numeric, load_rna_only, load_titan,
    metrics, normalize_label, official_text_prototypes, sha256, text_tokens, unit,
)

REPO = Path('/root/diffusion_phase5_20261007')
MODEL = Path('/root/phase6a_titan_runtime/model')
CACHE = Path('/root/phase6a_titan_runtime/cache_v1')
RNA = Path('/root/autodl-tmp/diffusion_data/data/data_8/coadread_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt')
FOLDS = REPO / 'outputs/phase_m1r/fixed_patient_folds.csv'
TARGETS = CACHE / 'clinical_targets.csv'
CONCEPTS = REPO / 'configs/phase6a_titan_crc_morphology_concepts.json'
BASELINE = Path('/root/phase6b_token_prompt_titan_repair')
CLASSES = {'T': ['T1', 'T2', 'T3', 'T4'], 'N': ['N0', 'N1', 'N2'], 'M': ['M0', 'M1'], 'Stage': ['I', 'II', 'III', 'IV']}
S4 = 'S4_wrong_patient_rna_token_prompt'


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def atomic_torch(path, obj):
    import torch
    temporary = path.with_suffix('.tmp')
    torch.save(obj, temporary)
    temporary.replace(path)


def cpu_state(net):
    return {name: value.detach().cpu().clone() for name, value in net.net.state_dict().items()}


def prepare(out, protocol_file):
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    protocol = json.loads(protocol_file.read_text())
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise RuntimeError('existing protocol differs; cannot silently resume')
    write_json(out / 'protocol.json', protocol)
    baseline_audit = json.loads((BASELINE / 'input_audit.json').read_text())
    manifest = {'protocol_sha256': sha256(out / 'protocol.json'), 'inputs': {},
                'source_sha256': {name: sha256(REPO / 'scripts' / name) for name in
                                  ('phase6b_token_budget_audit.py', 'phase6b_token_prompt_titan.py', 'phase6b_rna_conditioned_titan_concepts.py')},
                'private_outputs_published': False, 'survival_columns_read': []}
    for name, path in [('weight', MODEL / 'model.safetensors'), ('representations', CACHE / 'patient_representations.npz'),
                       ('rna', RNA), ('folds', FOLDS), ('targets', TARGETS), ('concepts', CONCEPTS)]:
        manifest['inputs'][name] = {'sha256': sha256(path), 'bytes': path.stat().st_size, 'path': str(path)}
    checks = {'weight': baseline_audit['model_weight_sha256'], 'representations': baseline_audit['representation_cache_sha256'],
              'rna': baseline_audit['rna']['input_sha256'], 'folds': baseline_audit['folds_sha256'],
              'targets': baseline_audit['targets_sha256'], 'concepts': baseline_audit['concepts_sha256']}
    if any(manifest['inputs'][key]['sha256'] != value for key, value in checks.items()):
        raise RuntimeError('input differs from repaired three-epoch baseline')
    rep = np.load(CACHE / 'patient_representations.npz', allow_pickle=False)
    ids = rep['patient_ids'].astype(str)
    folds = pd.read_csv(FOLDS).set_index('patient_id').loc[ids, 'fold'].to_numpy()
    targets = pd.read_csv(TARGETS, dtype=str, usecols=['patient_id', *ENDPOINTS]).set_index('patient_id').loc[ids]
    raw, rna_audit = load_rna_only(RNA, list(ids))
    contents = {'patient_ids': ids, 'image': unit(rep['semantic']), 'static': rep['concept_cosine'].astype(np.float32),
                'raw_rna': raw, 'folds': folds, 'text_prototypes': unit(rep['text_prototypes'])}
    for endpoint in ENDPOINTS:
        normalized = [normalize_label(v, endpoint) for v in targets[endpoint]]
        mapping = {value: index for index, value in enumerate(CLASSES[endpoint])}
        contents['y_' + endpoint] = np.asarray([mapping[v] if v is not None else -1 for v in normalized])
    np.savez_compressed(out / 'private_dataset.npz', **contents)
    manifest['private_dataset_sha256'] = sha256(out / 'private_dataset.npz')
    manifest['rna_audit'] = rna_audit
    manifest['baseline_oof_sha256'] = {p.name: sha256(p) for p in sorted((BASELINE / 'oof').glob('*.npz'))}
    assert len(manifest['baseline_oof_sha256']) == 300
    manifest['target_class_counts'] = {}
    for endpoint in ENDPOINTS:
        y = contents['y_' + endpoint]
        manifest['target_class_counts'][endpoint] = {label: int(np.sum(y == k)) for k, label in enumerate(CLASSES[endpoint])}
        for fold in range(1, 6):
            tr = np.flatnonzero((folds != fold) & (y >= 0))
            assert np.bincount(y[tr], minlength=len(CLASSES[endpoint])).min() >= 2
    model, _ = load_titan(MODEL, 'cuda')
    tokens, templates, vocab = text_tokens(model, CONCEPTS)
    official = official_text_prototypes(model, vocab, 'cuda')
    assert float(np.max(np.abs(official - contents['text_prototypes']))) <= 2e-4
    projector = FrozenTextProjector(model, tokens, templates, 4, 'cuda')
    write_json(out / 'encoder_audit.json', audit_projector(model, tokens, templates, official, projector, 'cuda'))
    write_json(out / 'manifest.json', manifest)
    (out / 'cells').mkdir(exist_ok=True)
    (out / 'private_checkpoints').mkdir(exist_ok=True)
    (out / 'oof').mkdir(exist_ok=True)
    print('PREPARED', flush=True)


def device_inputs(data, rna, idx):
    import torch
    return (torch.as_tensor(data['image'][idx], device='cuda'),
            torch.as_tensor(data['static'][idx], device='cuda'),
            torch.as_tensor(rna[idx], device='cuda'))


def inference_logits(net, inputs, projector):
    import torch
    image, static, rna = inputs
    net.net.eval()
    with torch.no_grad():
        if net.method == 'S1_static_token_prompt':
            text = projector.encode(net.net.prompt(rna[:1]))[0]
            scores = image @ text.T
            return net.net.head(scores).float()
        if net.method in {'S0_static_concepts', 'S3_static_concepts_plus_rna'}:
            return net.net(static, rna, projector).float()
        output = []
        for start in range(0, len(image), 12):
            sl = slice(start, start + 12)
            projector.image_batch = image[sl]
            output.append(net.net(static[sl], rna[sl], projector).float())
        return torch.cat(output)


def probabilities(net, inputs, projector):
    import torch
    return torch.softmax(inference_logits(net, inputs, projector), dim=1)


def assess(net, inputs, labels, weights, projector):
    import torch
    logits = inference_logits(net, inputs, projector)
    weighted_ce = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
    if not bool(torch.isfinite(weighted_ce)):
        raise RuntimeError('nonfinite evaluation loss')
    ba = balanced_accuracy_score(labels.cpu().numpy(), logits.argmax(1).cpu().numpy())
    return float(weighted_ce.cpu()), float(ba)


def train_one_epoch(net, optimizer, inputs, labels, weights, projector, rng):
    import torch
    import torch.nn.functional as F
    image, static, rna = inputs
    net.net.train()
    order = rng.permutation(len(image))
    grad_norms = []
    for start in range(0, len(order), 12):
        idx = torch.as_tensor(order[start:start + 12], device='cuda')
        projector.image_batch = image[idx]
        optimizer.zero_grad(set_to_none=True)
        logits = net.net(static[idx], rna[idx], projector)
        loss = F.cross_entropy(logits, labels[idx], weight=weights)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.net.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        grad_norms.append(float(grad.detach().cpu()))
    return float(np.mean(grad_norms))


def plateau(history):
    if len(history) < 20:
        return False
    recent = np.asarray([h['train_ce'] for h in history[-10:]])
    previous = np.asarray([h['train_ce'] for h in history[-20:-10]])
    return bool(abs(recent.mean() - previous.mean()) <= max(0.001, 0.005 * previous.mean())
                and recent.std() <= max(0.002, 0.01 * recent.mean()))


def inner_select(method, data, y, tr, va, n_classes, seed, projector, protocol, directory):
    import torch
    rna, transform = fit_numeric(data['raw_rna'], tr)
    inputs = device_inputs(data, rna, tr)
    validation = device_inputs(data, rna, va)
    labels = torch.as_tensor(y[tr], dtype=torch.long, device='cuda')
    val_labels = torch.as_tensor(y[va], dtype=torch.long, device='cuda')
    weights = class_weights(y[tr], n_classes, 'cuda')
    net = PromptNet(method, n_classes, 768, 4, 4, seed + 1009, 'cuda')
    optimizer = torch.optim.AdamW(net.net.parameters(), lr=0.005, weight_decay=0.001)
    rng = np.random.default_rng(20261007)
    checkpoint = directory / 'inner.pt'
    history, best_state = [], None
    best_value, best_epoch = float('inf'), 0
    anchor_ce, anchor_ba = float('inf'), -1.0
    ce_changed, ba_changed = 0, 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
        net.net.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['rng']
        history = saved['history']
        best_state, best_value, best_epoch = saved['best_state'], saved['best_value'], saved['best_epoch']
        anchor_ce, anchor_ba = saved['anchor_ce'], saved['anchor_ba']
        ce_changed, ba_changed = saved['ce_changed'], saved['ba_changed']
    budget = protocol['budget']
    saturated = False
    for epoch in range(len(history) + 1, budget['extended_ceiling'] + 1):
        before = time.time()
        grad = train_one_epoch(net, optimizer, inputs, labels, weights, projector, rng)
        train_ce, train_ba = assess(net, inputs, labels, weights, projector)
        val_ce, val_ba = assess(net, validation, val_labels, weights, projector)
        history.append({'epoch': epoch, 'train_ce': train_ce, 'inner_val_ce': val_ce,
                        'train_ba': train_ba, 'inner_val_ba': val_ba, 'mean_grad_norm_before_clip': grad,
                        'seconds': time.time() - before})
        width = budget['smooth_window']
        smooth_ce = float(np.mean([r['inner_val_ce'] for r in history[-width:]]))
        smooth_ba = float(np.mean([r['inner_val_ba'] for r in history[-width:]]))
        history[-1].update({'smooth_val_ce': smooth_ce, 'smooth_val_ba': smooth_ba})
        if epoch >= width and smooth_ce < best_value - 1e-10:
            best_value, best_epoch, best_state = smooth_ce, epoch, cpu_state(net)
        if smooth_ce < anchor_ce - budget['meaningful_ce_improvement']:
            anchor_ce, ce_changed = smooth_ce, epoch
        if smooth_ba > anchor_ba + budget['meaningful_ba_improvement']:
            anchor_ba, ba_changed = smooth_ba, epoch
        saturated = epoch >= budget['min_epochs'] and min(epoch - ce_changed, epoch - ba_changed) >= budget['patience_epochs']
        atomic_torch(checkpoint, {'model': cpu_state(net), 'optimizer': optimizer.state_dict(), 'rng': rng.bit_generator.state,
                                 'history': history, 'best_state': best_state, 'best_value': best_value, 'best_epoch': best_epoch,
                                 'anchor_ce': anchor_ce, 'anchor_ba': anchor_ba, 'ce_changed': ce_changed, 'ba_changed': ba_changed})
        if epoch % 10 == 0 or saturated:
            print(f'INNER epoch={epoch} method={method} train_ce={train_ce:.4f} val_ce={val_ce:.4f} best={best_epoch} saturated={saturated}', flush=True)
        if saturated:
            break
        if epoch == budget['initial_ceiling']:
            print(f'EXTEND_TO_{budget["extended_ceiling"]} method={method}', flush=True)
    result = {'selected_epoch': int(best_epoch), 'inner_epochs_run': len(history), 'best_smooth_inner_val_ce': best_value,
              'validation_saturated': bool(saturated), 'numerical_training_plateau': plateau(history),
              'inner_fit': transform, 'inner_val_indices_sha256': index_hash(va), 'history': history,
              'validation_labels_used_for_parameter_updates': False}
    write_json(directory / 'inner_selection.json', result)
    del net, optimizer, inputs, validation, labels, val_labels
    torch.cuda.empty_cache()
    return result


def refit(method, data, y, tr, n_classes, seed, epochs, projector, directory):
    import torch
    rna, transform = fit_numeric(data['raw_rna'], tr)
    inputs = device_inputs(data, rna, tr)
    labels = torch.as_tensor(y[tr], dtype=torch.long, device='cuda')
    weights = class_weights(y[tr], n_classes, 'cuda')
    net = PromptNet(method, n_classes, 768, 4, 4, seed, 'cuda')
    optimizer = torch.optim.AdamW(net.net.parameters(), lr=0.005, weight_decay=0.001)
    rng = np.random.default_rng(20261007)
    checkpoint = directory / 'refit.pt'
    history = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
        net.net.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['rng']
        history = saved['history']
    for epoch in range(len(history) + 1, epochs + 1):
        grad = train_one_epoch(net, optimizer, inputs, labels, weights, projector, rng)
        ce, ba = assess(net, inputs, labels, weights, projector)
        history.append({'epoch': epoch, 'train_ce': ce, 'train_ba': ba, 'mean_grad_norm_before_clip': grad})
        atomic_torch(checkpoint, {'model': cpu_state(net), 'optimizer': optimizer.state_dict(), 'rng': rng.bit_generator.state, 'history': history})
    del optimizer, inputs, labels
    return net, rna, {'outer_fit': transform, 'history': history, 'numerical_training_plateau': plateau(history)}


def context_diagnostics(net, inputs, projector, static_prototypes):
    import torch
    if net.method != 'S2_rna_token_prompt':
        return {}
    image, static, rna = (v[:64] for v in inputs)
    with torch.no_grad():
        h = net.net.adapter(rna)
        alpha = torch.softmax(net.net.gate(h), dim=1)
        entropy = (-(alpha.clamp_min(1e-12).log() * alpha).sum(1) / np.log(4)).mean()
        scores, semantic_cosines = [], []
        original = torch.as_tensor(static_prototypes, device='cuda')
        for start in range(0, len(rna), 12):
            text = projector.encode(net.net.prompt(rna[start:start + 12]))
            scores.append(torch.einsum('bd,bkd->bk', image[start:start + 12], text))
            semantic_cosines.append((text * original[None]).sum(-1).flatten())
        features = torch.cat(scores).cpu().numpy()
        sv = np.linalg.svd(features - features.mean(0), compute_uv=False)
        energy = sv ** 2
        normalized = energy / max(energy.sum(), 1e-12)
        rank = float(np.exp(-np.sum(normalized * np.log(normalized + 1e-12))))
        return {'gate_normalized_entropy': float(entropy.cpu()),
                'gate_patient_mean_std': float(alpha.std(0, unbiased=False).mean().cpu()),
                'gate_dominant_mean_weight': float(alpha.mean(0).max().cpu()),
                'mean_concept_cosine_to_static': float(torch.cat(semantic_cosines).mean().cpu()),
                'concept_score_effective_rank': rank, 'sample_n': len(rna), 'label_columns_used': []}


def worker(out, worker_id, workers):
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    protocol = json.loads((out / 'protocol.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    assert workers == protocol['workers'] and 0 <= worker_id < workers
    assert sha256(out / 'protocol.json') == manifest['protocol_sha256']
    assert sha256(out / 'private_dataset.npz') == manifest['private_dataset_sha256']
    for name, digest in manifest['source_sha256'].items():
        assert sha256(REPO / 'scripts' / name) == digest, f'code changed after protocol lock: {name}'
    data = dict(np.load(out / 'private_dataset.npz', allow_pickle=False))
    model, _ = load_titan(MODEL, 'cuda')
    assert sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad) == 0
    tokens, templates, _ = text_tokens(model, CONCEPTS)
    projector = FrozenTextProjector(model, tokens, templates, 4, 'cuda')
    blocks = [(seed, endpoint, fold) for seed in protocol['seeds'] for endpoint in ENDPOINTS for fold in range(1, 6)]
    for block_id, (seed, endpoint, fold) in enumerate(blocks):
        if block_id % workers != worker_id:
            continue
        y = data['y_' + endpoint]
        valid = y >= 0
        tr = np.flatnonzero((data['folds'] != fold) & valid)
        test = np.flatnonzero((data['folds'] == fold) & valid)
        endpoint_seed = int.from_bytes(hashlib.sha256(endpoint.encode()).digest()[:4], 'little') % 100000
        split_seed = seed + fold * 10000 + endpoint_seed
        rel_train, rel_val = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=split_seed).split(tr, y[tr]))
        inner_train, inner_val = tr[rel_train], tr[rel_val]
        assert set(inner_train).isdisjoint(inner_val) and set(tr).isdisjoint(test)
        # S0/S3 are cheap; train every control under its own inner-only budget.
        for method in (TOKEN_METHODS[0], TOKEN_METHODS[3], TOKEN_METHODS[1], TOKEN_METHODS[2]):
            tag = f'{endpoint}__{method}__seed{seed}__fold{fold}'
            done = out / 'cells' / (tag + '.json')
            output = out / 'oof' / (tag + '.npz')
            wrong_output = out / 'oof' / (tag.replace(method, S4) + '.npz')
            if done.exists() and output.exists() and (method != TOKEN_METHODS[2] or wrong_output.exists()):
                assert json.loads(done.read_text())['protocol_sha256'] == manifest['protocol_sha256']
                print('RESUME_DONE ' + tag, flush=True)
                continue
            directory = out / 'private_checkpoints' / tag
            directory.mkdir(exist_ok=True)
            print('CELL_START ' + tag, flush=True)
            if (directory / 'inner_selection.json').exists():
                selection = json.loads((directory / 'inner_selection.json').read_text())
            else:
                selection = inner_select(method, data, y, inner_train, inner_val, len(CLASSES[endpoint]), seed + fold * 10000,
                                         projector, protocol, directory)
            net, rna, final = refit(method, data, y, tr, len(CLASSES[endpoint]), seed + fold * 10000,
                                   selection['selected_epoch'], projector, directory)
            test_inputs = device_inputs(data, rna, test)
            matched = probabilities(net, test_inputs, projector).cpu().numpy()
            assert np.isfinite(matched).all() and np.allclose(matched.sum(1), 1, atol=1e-5)
            np.savez_compressed(output, patient_ids=data['patient_ids'][test], y_true=y[test], probabilities=matched,
                                classes=np.asarray(CLASSES[endpoint]), selected_epoch=selection['selected_epoch'])
            diagnostics = context_diagnostics(net, test_inputs, projector, data['text_prototypes'])
            if method == TOKEN_METHODS[2]:
                shift = 1 + (seed + fold * 10000 + int(test[0])) % (len(test) - 1)
                permutation = np.roll(np.arange(len(test)), shift)
                assert not np.any(permutation == np.arange(len(test)))
                image, static, context = test_inputs
                wrong = probabilities(net, (image, static, context[permutation]), projector).cpu().numpy()
                np.savez_compressed(out / 'oof' / (tag.replace(method, S4) + '.npz'),
                                    patient_ids=data['patient_ids'][test], y_true=y[test], probabilities=wrong,
                                    classes=np.asarray(CLASSES[endpoint]), selected_epoch=selection['selected_epoch'])
            record = {'status': 'PASS', 'protocol_sha256': manifest['protocol_sha256'], 'seed': seed, 'endpoint': endpoint,
                      'fold': fold, 'method': method, 'parameters': net.count(), 'selection': selection, 'refit': final,
                      'outer_train_indices_sha256': index_hash(tr), 'outer_test_indices_sha256': index_hash(test),
                      'inner_train_indices_sha256': index_hash(inner_train), 'inner_val_indices_sha256': index_hash(inner_val),
                      'inner_split_seed': split_seed, 'outer_test_used_for_selection': False,
                      'inner_val_used_for_upstream_parameter_updates': False, 'diagnostics': diagnostics,
                      'oof_sha256': sha256(output), 'gpu': torch.cuda.get_device_name(0)}
            write_json(done, record)
            print('CELL_DONE ' + tag, flush=True)
            del net, test_inputs
            torch.cuda.empty_cache()
    write_json(out / f'worker_{worker_id}_DONE.json', {'status': 'PASS', 'worker': worker_id})


def aggregate(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    assert all((out / f'worker_{k}_DONE.json').exists() for k in range(protocol['workers']))
    assert sha256(out / 'protocol.json') == manifest['protocol_sha256']
    data = dict(np.load(out / 'private_dataset.npz', allow_pickle=False))
    rows, pairs, selections, curves, diagnostic_rows = [], [], [], [], []
    max_error = 0.0
    checked = 0
    for seed in protocol['seeds']:
        for endpoint in ENDPOINTS:
            y = data['y_' + endpoint]
            valid = y >= 0
            predictions, baseline = {}, {}
            for method in (*TOKEN_METHODS, S4):
                p = np.full((len(y), len(CLASSES[endpoint])), np.nan)
                b = p.copy()
                for fold in range(1, 6):
                    tag = f'{endpoint}__{method}__seed{seed}__fold{fold}'
                    test = np.flatnonzero((data['folds'] == fold) & valid)
                    for root, dest in [(out, p), (BASELINE, b)]:
                        if root == BASELINE:
                            assert sha256(root / 'oof' / (tag + '.npz')) == manifest['baseline_oof_sha256'][tag + '.npz']
                        with np.load(root / 'oof' / (tag + '.npz'), allow_pickle=False) as cell:
                            assert np.array_equal(cell['patient_ids'].astype(str), data['patient_ids'][test])
                            assert np.array_equal(cell['y_true'], y[test])
                            assert np.array_equal(cell['classes'], CLASSES[endpoint])
                            assert np.isfinite(cell['probabilities']).all()
                            dest[test] = cell['probabilities']
                    checked += 1
                    if method != S4:
                        cell = json.loads((out / 'cells' / (tag + '.json')).read_text())
                        assert sha256(out / 'oof' / (tag + '.npz')) == cell['oof_sha256']
                        assert cell['status'] == 'PASS' and not cell['outer_test_used_for_selection']
                        assert not cell['inner_val_used_for_upstream_parameter_updates']
                        assert cell['protocol_sha256'] == manifest['protocol_sha256']
                        tr = np.flatnonzero((data['folds'] != fold) & valid)
                        rel_tr, rel_va = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                                 random_state=cell['inner_split_seed']).split(tr, y[tr]))
                        assert cell['selection']['inner_fit']['fit_indices_sha256'] == index_hash(tr[rel_tr])
                        assert cell['refit']['outer_fit']['fit_indices_sha256'] == index_hash(tr)
                        assert cell['selection']['inner_fit']['selected_dim'] == 512
                        assert cell['refit']['outer_fit']['selected_dim'] == 512
                        select = cell['selection']
                        hist = select['history']
                        eligible = [h for h in hist if h['epoch'] >= protocol['budget']['smooth_window']]
                        selected = min(eligible, key=lambda h: (h['smooth_val_ce'], h['epoch']))['epoch']
                        assert selected == select['selected_epoch']
                        assert len(cell['refit']['history']) == selected
                        selections.append({'seed': seed, 'endpoint': endpoint, 'fold': fold, 'method': method,
                                           'selected_epoch': select['selected_epoch'], 'inner_epochs_run': select['inner_epochs_run'],
                                           'validation_saturated': select['validation_saturated'],
                                           'inner_numerical_plateau': select['numerical_training_plateau'],
                                           'refit_numerical_plateau': cell['refit']['numerical_training_plateau'],
                                           'parameters': cell['parameters']})
                        for h in select['history']:
                            curves.append({'seed': seed, 'endpoint': endpoint, 'fold': fold, 'method': method, **h})
                        if cell['diagnostics']:
                            diagnostic_rows.append({'seed': seed, 'endpoint': endpoint, 'fold': fold, **cell['diagnostics']})
                assert np.isfinite(p[valid]).all() and np.isfinite(b[valid]).all()
                assert np.allclose(p[valid].sum(1), 1, atol=1e-5)
                predictions[method], baseline[method] = p[valid], b[valid]
                rows.append({'seed': seed, 'endpoint': endpoint, 'method': method, 'n': int(valid.sum()),
                             **metrics(y[valid], p[valid], len(CLASSES[endpoint]))})
            comparisons = [('S2-S1', TOKEN_METHODS[2], TOKEN_METHODS[1]), ('S2-S3', TOKEN_METHODS[2], TOKEN_METHODS[3]), ('S2-S4', TOKEN_METHODS[2], S4)]
            for name, left, right in comparisons:
                stable = int.from_bytes(hashlib.sha256((endpoint + name).encode()).digest()[:4], 'little') % 100000
                pairs.append({'seed': seed, 'endpoint': endpoint, 'comparison': name,
                              **bootstrap_delta(y[valid], predictions[left], predictions[right], 2000, seed + stable)})
            for method in (*TOKEN_METHODS, S4):
                stable = int.from_bytes(hashlib.sha256((endpoint + method).encode()).digest()[:4], 'little') % 100000
                pairs.append({'seed': seed, 'endpoint': endpoint, 'comparison': 'selected_budget-minus-three_epochs/' + method,
                              **bootstrap_delta(y[valid], predictions[method], baseline[method], 2000, seed + stable)})
    assert checked == 300 and len(rows) == 60 and len(selections) == 240
    frames = {'metrics.csv': rows, 'paired_comparisons_by_seed.csv': pairs, 'selection_summary.csv': selections,
              'inner_learning_curves.csv': curves, 'mechanism_diagnostics.csv': diagnostic_rows}
    for name, frame in frames.items():
        pd.DataFrame(frame).to_csv(out / name, index=False)
    methods = pd.DataFrame(rows).groupby(['endpoint', 'method'], as_index=False).agg(
        mean_oof_ba=('balanced_accuracy', 'mean'), sd_oof_ba=('balanced_accuracy', 'std'),
        mean_oof_macro_f1=('macro_f1', 'mean'), mean_oof_auroc=('macro_ovr_auroc', 'mean'))
    methods.to_csv(out / 'aggregate_metrics.csv', index=False)
    paired = pd.DataFrame(pairs).groupby(['endpoint', 'comparison'], as_index=False).agg(
        mean_delta_ba=('delta_balanced_accuracy', 'mean'), sd_delta_ba=('delta_balanced_accuracy', 'std'),
        all_seed_ci_positive=('ci98_75_low', lambda x: bool((x > 0).all())))
    paired.to_csv(out / 'aggregate_comparisons.csv', index=False)
    budget = pd.DataFrame(selections).groupby(['endpoint', 'method'], as_index=False).agg(
        selected_epoch_median=('selected_epoch', 'median'), selected_epoch_min=('selected_epoch', 'min'),
        selected_epoch_max=('selected_epoch', 'max'), inner_epochs_median=('inner_epochs_run', 'median'),
        validation_saturated_cells=('validation_saturated', 'sum'), inner_numerical_plateau_cells=('inner_numerical_plateau', 'sum'))
    budget.to_csv(out / 'budget_summary.csv', index=False)
    validation = {'status': 'PASS', 'oof_files': checked, 'metric_rows': len(rows), 'cells': len(selections),
                  'fold_local_preprocessing_and_inner_split_audits': 240,
                  'outer_test_selection_access': False, 'survival_columns_read': [],
                  'validation_saturated_cells': int(sum(x['validation_saturated'] for x in selections)),
                  'inner_numerical_plateau_cells': int(sum(x['inner_numerical_plateau'] for x in selections)),
                  'published_patient_data': False}
    write_json(out / 'validation.json', validation)
    lines = ['# Phase6B token-level training-budget audit', '',
             '本轮保持 S0–S4、模型容量、学习率、batch size、词表和输入不变，仅增加训练预算并在 outer-train 内部选择 epoch。TITAN 完全冻结，T/N/M/Stage 是唯一目标。', '',
             '## Protocol', '',
             '- 3 seeds × 5 outer folds × 4 endpoints；每个 outer-train 有一个固定 stratified 80/20 inner split。RNA 预处理和所有 trainable parameters 仅拟合 inner-train；inner-val 只用于 epoch selection。',
             '- 至少 30 epochs；120 上限时仍未停止则扩展至 240。5-epoch 平滑的 balanced CE 最小值决定 epoch，CE 与 BA 均 20 epochs 无实质改善才停止。',
             '- epoch 锁定后，从头在完整 outer-train 重训；相同 optimizer/batch schedule，outer-test 每个模型只预测一次。历史 3-epoch OOF 只在全部训练完成后用于预算配对比较。',
             '- 本 cohort 的旧 outer-test 结果已经被研究过程看过，本轮是内部预算审计，不是全新独立确认。区间是逐比较描述性 bootstrap，不提供全家族错误率保证。', '',
             '## Computational validation', '', '```json', json.dumps(validation, indent=2), '```', '',
             '## OOF balanced accuracy', '', '| Endpoint | Method | 3-seed mean OOF BA | Seed SD |', '|---|---|---:|---:|']
    for row in methods.itertuples(index=False):
        lines.append(f'| {row.endpoint} | {row.method} | {row.mean_oof_ba:.4f} | {row.sd_oof_ba:.4f} |')
    lines += ['', '## Paired comparisons', '', '| Endpoint | Comparison | Mean delta BA | Every seed CI > 0 |', '|---|---|---:|---:|']
    for row in paired.itertuples(index=False):
        lines.append(f'| {row.endpoint} | {row.comparison} | {row.mean_delta_ba:.4f} | {row.all_seed_ci_positive} |')
    lines += ['', '## Budget and convergence', '', '| Endpoint | Method | Selected epoch median (range) | Inner epochs median | Validation saturated /15 | Numerical train plateau /15 |', '|---|---|---:|---:|---:|---:|']
    for row in budget.itertuples(index=False):
        lines.append(f'| {row.endpoint} | {row.method} | {row.selected_epoch_median:g} ({row.selected_epoch_min:g}–{row.selected_epoch_max:g}) | {row.inner_epochs_median:g} | {row.validation_saturated_cells} | {row.inner_numerical_plateau_cells} |')
    lines += ['', '## Interpretation boundaries', '',
              '训练 loss 的数值平台、内部验证的收益饱和、完整 OOF 的预算改善和 patient-specific mechanism 分别判读。验证饱和不等于数值优化已达全局最优；增加 epoch 后模型仍弱也不能直接否定医学先验。',
              '只有 S2 同时优于 S1、S3、S4 才形成必要的机制证据。gate entropy、患者差异和概念漂移诊断没有医学形态标注，不能单独证明语义正确。', '',
              'nested selection 的原则参考 [Cawley and Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)。', '',
              'TITAN 权重、患者 OOF 和私有 checkpoints 保留于服务器；只发布源码、protocol、aggregate metrics、learning curves 和审计结果。', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(out / 'DONE.json', {'status': 'PASS', 'validation': validation, 'protocol_sha256': sha256(out / 'protocol.json')})
    print(json.dumps(validation), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare', 'worker', 'aggregate'], required=True)
    parser.add_argument('--out', type=Path, default=Path('/root/phase6b_token_budget_audit'))
    parser.add_argument('--protocol', type=Path, default=REPO / 'configs/phase6b_token_budget_audit.json')
    parser.add_argument('--worker-id', type=int, default=0)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    try:
        if args.mode == 'prepare':
            prepare(args.out, args.protocol)
        elif args.mode == 'worker':
            worker(args.out, args.worker_id, args.workers)
        else:
            aggregate(args.out)
    except Exception:
        write_json(args.out / f'{args.mode}_{args.worker_id}_FAIL.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
