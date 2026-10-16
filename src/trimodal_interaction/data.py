"""Private, fit-local inputs for the clinical-conditioned association audit.

This module never prepares or changes a historical artifact.  Patient-indexed
arrays, PCA states and partitions remain under ``P``; ``Data.audit`` contains
only aggregate counts and hashes.  No CUDA allocation is made here.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.phase10_trimodal.data import digest
from src.phase10_trimodal.features import ahash, scale
from src.phase13_rna import data as parent
from src.phase17_ssd import data as old


ROOT = Path(__file__).resolve().parents[2]
P = Path('/root/autodl-tmp/trimodal_interaction_v1_private')
R = ROOT / 'reports/trimodal_interaction_v1'
CONFIG = ROOT / 'configs/trimodal_interaction_v1.json'
OLD_VISUAL = old.P / 'visual'
OLD_LOCAL = Path('/root/phase16r_light_rna_private')


def cfg():
    return json.loads(CONFIG.read_text(encoding='utf8'))


def write(path, value):
    """Atomically write JSON; callers decide whether its destination is private."""
    def cast(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=cast, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def _dependency_hashes():
    # The Phase17 identity by itself does not cover every imported ancestor.
    # Include the actual immutable preprocessing/math dependencies as well.
    paths = set()
    for directory in ['phase10_trimodal', 'phase11_semantic', 'phase12_c4',
                      'phase13_rna', 'phase15_complement', 'phase16_routing',
                      'phase16_semantic', 'phase16r_light', 'phase17_ssd']:
        paths.update((ROOT / 'src' / directory).glob('*.py'))
    for name in ['phase10_molecular_groups.json', 'phase11_semantic_anchored.json',
                 'phase12_c4_confirmation.json', 'phase13_rna_information_audit.json',
                 'phase17_ssd_fusion.json']:
        paths.add(ROOT / 'configs' / name)
    paths.add(ROOT / 'scripts/run_phase17.py')
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def source():
    """Training identity; evaluator/report corrections do not change this hash."""
    config = cfg()
    ancestor = old.source()
    if ancestor != config['parent_training_source']:
        raise ValueError('parent Phase17 training source changed')
    paths = [CONFIG] + sorted((ROOT / 'src/trimodal_interaction').glob('*.py'))
    paths += [ROOT / 'scripts/run_trimodal_interaction.py']
    own = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    payload = {'parent_training_source': ancestor,
               'dependencies': _dependency_hashes(), 'training_files': own}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@contextlib.contextmanager
def _cache_lock(path):
    # Formal execution is on Linux.  Serialize the shared fold cache across
    # independent seed jobs; a failed producer never leaves a valid receipt.
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _visual_registry():
    receipt = OLD_VISUAL / 'READY.json'
    ready = json.loads(receipt.read_text())
    if ready.get('status') != 'PASS' or ready.get('patients') != 512:
        raise ValueError('unverified historical visual cache')
    for key in ['conch', 'mask', 'owner', 'coords', 'local_row']:
        if digest(OLD_VISUAL / (key + '.npy')) != ready['hashes'][key]:
            raise ValueError('historical visual hash changed: ' + key)
    if digest(parent.DATA) != ready['dataset_sha256']:
        raise ValueError('visual and molecular cohorts differ')
    path = OLD_LOCAL / 'VISUAL_SOURCES.npz'
    metadata = OLD_LOCAL / 'VISUAL_SOURCES.json'
    local = json.loads(metadata.read_text())
    if digest(path) != local['sha256']:
        raise ValueError('historical LOCAL source changed')
    return {'visual_receipt_sha256': digest(receipt),
            'visual_hashes': {key: ready['hashes'][key] for key in
                              ['conch', 'mask', 'owner', 'coords', 'local_row']},
            'LOCAL_npz_sha256': local['sha256'],
            'LOCAL_metadata_sha256': digest(metadata),
            'dataset_sha256': ready['dataset_sha256']}


def _uniform(n, limit):
    """Outcome-blind, deterministic, uniformly spaced positions in a sequence."""
    size = min(int(n), int(limit))
    if size < 1:
        raise ValueError('empty patient visual evidence')
    out = np.floor((np.arange(size) + .5) * n / size).astype(np.int64)
    if len(np.unique(out)) != size or np.any(out < 0) or np.any(out >= n):
        raise ValueError('invalid geometry-only uniform selection')
    return out


def _patient_patches(active, budget):
    """Read only active patients from the frozen geometry-selected parent ROIs.

    Multi-slide ROIs are concatenated by immutable anonymous slide order; each
    slide is ordered by (x,y).  The single patient budget is sampled uniformly
    over that sequence, so it is not multiplied by a patient's slide count.
    The historical norm is retained exactly, including float16 rounding.
    """
    owner = np.load(OLD_VISUAL / 'owner.npy')
    mask = np.load(OLD_VISUAL / 'mask.npy')
    coords = np.load(OLD_VISUAL / 'coords.npy', mmap_mode='r')
    local_row = np.load(OLD_VISUAL / 'local_row.npy', mmap_mode='r')
    raw = np.load(OLD_VISUAL / 'conch.npy', mmap_mode='r')
    if raw.shape[:2] != mask.shape or raw.shape[-1] != 768 or owner.shape != (len(mask),):
        raise ValueError('historical CONCH array dimensions changed')
    if not np.array_equal(np.unique(owner), np.arange(512)):
        raise ValueError('historical CONCH patient coverage changed')
    values = np.zeros((512, budget, 768), np.float32)
    valid = np.zeros((512, budget), bool)
    # These arrays are private provenance, never part of the public audit.
    selected = np.full((512, budget, 2), -1, np.int64)
    selected_rows = np.full((512, budget), -1, np.int64)
    selected_coords = np.full((512, budget, 2), -1, np.int64)
    for pi in active:
        locations = []
        for slide in np.flatnonzero(owner == pi):
            rows = np.flatnonzero(mask[slide])
            xy = np.asarray(coords[slide, rows])
            order = np.lexsort((xy[:, 1], xy[:, 0]))
            locations.extend((int(slide), int(row)) for row in rows[order])
        if len(locations) > 512:
            raise ValueError('parent patient patch budget exceeded')
        locations = np.asarray(locations, np.int64)[_uniform(len(locations), budget)]
        ss, rr = locations.T
        n = len(locations)
        values[pi, :n] = np.asarray(raw[ss, rr], np.float32)
        valid[pi, :n] = True
        selected[pi, :n] = locations
        selected_rows[pi, :n] = local_row[ss, rr]
        selected_coords[pi, :n] = coords[ss, rr]
    if not np.isfinite(values).all() or not valid[active].any(1).all():
        raise ValueError('nonfinite/empty active visual evidence')
    return values, valid, {'slide_and_parent_row': selected,
                           'original_CONCH_row': selected_rows,
                           'coords': selected_coords}


def _feature_bank(d, fold, role, fit, valid, active, config, provenance):
    f_all, molecular_audit = old.molecular_features(d, fold, role)
    # The inherited function is stateless; discard every non-active row before
    # visual transforms or caching.  No held-out label is used by that function.
    f = {}
    for key in ['C', 'S', 'R', 'RM']:
        f[key] = np.zeros_like(f_all[key], dtype=np.float32)
        f[key][active] = f_all[key][active]
    meta, _ = parent.parent_features(fold, role)
    gene_indices = np.asarray(meta['phase12']['enriched_indices'], np.int64)
    if f['R'].shape != (512, 512) or gene_indices.shape != (512,):
        raise ValueError('Hallmark512 identity/dimensions changed')
    if meta['fit_indices'] != fit.tolist():
        raise ValueError('inherited RNA preprocessing fit mismatch')

    budget = int(config.get('patch_budget', 128))
    dim = int(config.get('hidden_dim', 64))
    if budget != 128 or dim != 64:
        raise ValueError('this locked first audit requires 128 patches and PCA64')
    raw, patch_mask, locations = _patient_patches(active, budget)
    pca_samples = []
    sample_locations = []
    for pi in fit:
        rows = np.flatnonzero(patch_mask[pi])
        take = rows[_uniform(len(rows), 16)]
        pca_samples.append(raw[pi, take])
        sample_locations.extend((int(pi), int(row)) for row in take)
    sample = np.concatenate(pca_samples)
    if len(sample) <= dim or sample.shape[1] != 768:
        raise ValueError('insufficient fit-only patch PCA samples')
    patch_pca = PCA(dim, whiten=False, svd_solver='randomized',
                    random_state=int(config.get('preprocessing_seed', 17321))).fit(sample)
    patch_features = np.zeros((512, budget, dim), np.float32)
    for pi in active:
        rows = np.flatnonzero(patch_mask[pi])
        patch_features[pi, rows] = patch_pca.transform(raw[pi, rows]).astype(np.float32)
    f['P'] = patch_features
    f['PM'] = patch_mask

    # This read-only source is the historical LOCAL mean.  Do not call
    # phase16r_light.visual_sources(), whose cache-miss path writes old reports.
    with np.load(OLD_LOCAL / 'VISUAL_SOURCES.npz') as archive:
        local_all = np.asarray(archive['LOCAL'], dtype=np.float64)
    if local_all.shape != (512, 768) or not np.isfinite(local_all).all():
        raise ValueError('historical LOCAL mean dimensions/finite gate failed')
    local = np.zeros_like(local_all)
    local[active] = local_all[active]
    local, before_scale = scale(local, fit)
    local_pca = PCA(32, whiten=False, svd_solver='full').fit(local[fit])
    local_pc = np.zeros((512, 32), np.float64)
    local_pc[active] = local_pca.transform(local[active])
    local_pc, after_scale = scale(local_pc, fit)
    local_pc[np.setdiff1d(np.arange(512), active)] = 0
    f['LOCAL32'] = local_pc.astype(np.float32)
    widths = [f[key].shape[1] for key in ['C', 'S', 'LOCAL32', 'R']]
    edges = np.r_[0, np.cumsum(widths)]
    blocks = {key: (int(edges[i]), int(edges[i + 1])) for i, key in
              enumerate(['C', 'S', 'LOCAL32', 'R'])}
    f['X'] = np.concatenate([f[key] for key in ['C', 'S', 'LOCAL32', 'R']], axis=1)
    for key, values in f.items():
        if not np.isfinite(values).all():
            raise ValueError('nonfinite fit-local feature: ' + key)
    references = {key: f[key][fit].mean(0).astype(np.float32) for key in ['C', 'R', 'RM']}
    state = {'patch_pca': patch_pca, 'patch_PCA_fit_rows': np.asarray(sample_locations, np.int64),
             'LOCAL_before_scale': before_scale, 'LOCAL_pca': local_pca,
             'LOCAL_after_scale': after_scale, 'references': references,
             'gene_indices': gene_indices, 'selected_patches': locations}
    audit = {'status': 'PASS', 'fold': int(fold), 'role': role,
             'source': provenance['source'], 'dataset_sha256': provenance['dataset_sha256'],
             'fit_n': len(fit), 'validation_or_test_n': len(valid),
             'active_n': len(active), 'fit_hash': ahash(fit), 'valid_hash': ahash(valid),
             'feature_shapes': {key: list(value.shape) for key, value in f.items()},
             'fit_feature_hashes': {key: ahash(value[fit]) for key, value in f.items()},
             'valid_feature_hashes': {key: ahash(value[valid]) for key, value in f.items()},
             'RNA_gene_identity_hash': ahash(gene_indices),
             'RNA_selection_hash': molecular_audit['RNA_selection'],
             'RNA_preprocessing_sha256': molecular_audit['upstream_npz'],
             'RNA_missing_fit': int((f['RM'][fit] == 0).sum()),
             'RNA_missing_validation_or_test': int((f['RM'][valid] == 0).sum()),
             'CNA_included': False, 'patch_backbone': 'frozen CONCH v1.5 only',
             'patch_budget_per_patient': budget,
             'patches_per_active_patient_min': int(patch_mask[active].sum(1).min()),
             'patches_per_active_patient_max': int(patch_mask[active].sum(1).max()),
             'patch_PCA_dimensions': dim, 'patch_PCA_fit_samples': len(sample),
             'patch_PCA_components_hash': ahash(patch_pca.components_),
             'patch_PCA_mean_hash': ahash(patch_pca.mean_),
             'patch_PCA_explained_variance_ratio_sum': float(patch_pca.explained_variance_ratio_.sum()),
             'LOCAL_PCA_components_hash': ahash(local_pca.components_),
             'reference_hashes': {key: ahash(value) for key, value in references.items()},
             'geometry_selection_hash': ahash(locations['slide_and_parent_row'][active]),
             'reference_definition': 'mean of this exact fit partition; no patient counterfactual claim',
             'LOCAL_definition': 'historical LOCAL mean; fit scale, fit full PCA32, fit scale',
             'clinical_definition': 'six legal parent fields plus explicit age-missing indicator',
             'clinical_difference_from_historical_LOCAL': 'historical LOCAL omitted extra age-missing indicator',
             'inner_outer_test_feature_transform': False if role == 'inner' else None,
             'geometry_QA': 'candidate-pool limited; no raw WSI or histological validation',
             'visual_sources': provenance['visual_sources']}
    return {'provenance': provenance, 'fit': fit, 'valid': valid, 'active': active,
            'features': f, 'state': state, 'blocks': blocks, 'audit': audit}


class Data:
    """One fit/validation role, with reproducible and partition-local controls.

    ``f`` are NumPy arrays; ``ft`` contains matching CPU tensors.  ``P`` is the
    PCA64 patch sequence, ``PM`` its Boolean mask, and ``X`` the additive design
    ``[C, S32, LOCAL_CONCH_PCA32, RNA512]``.  Non-active rows are all zero and
    must not be predicted.  ``wrong`` may only index either fit or valid rows.
    """
    def __init__(self, fold, role, seed):
        if role not in ['inner', 'outer'] or int(fold) not in range(1, 6):
            raise ValueError('invalid preprocessing role/fold')
        self.fold, self.role, self.seed = int(fold), role, int(seed)
        config = cfg()
        training_source = source()
        self.d = parent.load()
        self.part = parent.splits(self.d, 'DSS', self.fold)
        self.parts = self.part
        self.fit = self.part['inner_train' if role == 'inner' else 'outer_train']
        self.valid = self.part['inner_val' if role == 'inner' else 'outer_test']
        self.active = np.sort(np.r_[self.fit, self.valid])
        if len(np.unique(self.active)) != len(self.active):
            raise ValueError('fit/validation overlap')
        if role == 'inner' and np.intersect1d(self.active, self.part['outer_test']).size:
            raise ValueError('inner role reaches outer test')
        visual_sources = _visual_registry()
        feature_files = parent.PARENT / 'preprocessing' / f'fold{fold}_{role}'
        manifest = parent.PARENT / 'formal/manifests' / f'R0_128_seed31_fold{fold}.json'
        provenance = {'source': training_source,
                      'dataset_sha256': digest(parent.DATA),
                      'endpoint_npz_sha256': digest(parent.PRIVATE / 'endpoints.npz'),
                      'partition_manifest_sha256': digest(manifest),
                      'parent_feature_json_sha256': digest(feature_files.with_suffix('.json')),
                      'parent_feature_npz_sha256': digest(feature_files.with_suffix('.npz')),
                      'fit_hash': ahash(self.fit), 'valid_hash': ahash(self.valid),
                      'visual_sources': visual_sources}
        self.cache_path = P / 'features' / f'fold{fold}_{role}.pt'
        receipt_path = self.cache_path.with_suffix('.json')
        with _cache_lock(self.cache_path.with_suffix('.lock')):
            if self.cache_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if digest(self.cache_path) != receipt['sha256']:
                    raise ValueError('private preprocessing cache checksum changed')
                bank = torch.load(self.cache_path, map_location='cpu', weights_only=False)
                if bank['provenance'] != provenance:
                    raise ValueError('private preprocessing cache source/provenance changed')
                if not np.array_equal(bank['fit'], self.fit) or not np.array_equal(bank['valid'], self.valid):
                    raise ValueError('private preprocessing cache partition changed')
            else:
                if receipt_path.exists():
                    raise ValueError('orphan preprocessing receipt without cache')
                bank = _feature_bank(self.d, self.fold, role, self.fit, self.valid,
                                     self.active, config, provenance)
                tmp = self.cache_path.with_name(self.cache_path.name + f'.{os.getpid()}.tmp')
                torch.save(bank, tmp)
                tmp.replace(self.cache_path)
                write(receipt_path, {'status': 'PASS', 'sha256': digest(self.cache_path),
                                     'source': training_source, 'provenance': provenance})
        self.f = bank['features']
        self.audit = dict(bank['audit'])
        self.audit['cache_sha256'] = digest(self.cache_path)
        self.blocks = bank['blocks']
        self.state = bank['state']
        self.references = bank['state']['references']
        self.C0 = self.references['C']
        self.R0 = self.references['R']
        self.RM0 = self.references['RM']
        self.gene_indices = bank['state']['gene_indices']
        self.ft = {key: torch.as_tensor(value) for key, value in self.f.items()}
        self.reference_tensors = {key: torch.as_tensor(value) for key, value in self.references.items()}
        self.wrong = np.arange(512)
        for indices in [self.fit, self.valid]:
            self.wrong[indices] = old.wrong_map(self.d, indices, seed + 1000 + fold)[indices]
        self.audit['wrong_fit_hash'] = ahash(self.wrong[self.fit])
        self.audit['wrong_valid_hash'] = ahash(self.wrong[self.valid])
        self.audit['wrong_control'] = 'partition-local cancer/stage-stratified derangement; own RNA main effect retained'
        # Labels exposed by this wrapper are limited to this role's active rows.
        ev = parent.endpoints(self.d)['DSS']
        self.ev = {'time': np.full(512, np.nan), 'event': np.full(512, np.nan),
                   'valid': np.zeros(512, bool)}
        for key in ['time', 'event', 'valid']:
            self.ev[key][self.active] = ev[key][self.active]
        self.ep = self.ev
        self.t, self.e = self.ev['time'], self.ev['event']
        self.budget = int(config.get('patch_budget', 128))

    def check_indices(self, indices):
        indices = np.asarray(indices, np.int64)
        if not np.isin(indices, self.active).all():
            raise ValueError('requested prediction outside active role partition')
        return indices
