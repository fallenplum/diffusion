"""Fit-local Phase19 inputs, immutable ancestry and partition-local controls.

Patient arrays, PCA estimators and selected genes stay in the private cache.
Public audits contain shapes/counts/hashes, never patient or gene identities.
"""
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.phase17_ssd import data as old
from src.phase13_rna import data as parent
from src.phase10_trimodal.data import digest
from src.phase10_trimodal.features import ahash


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/phase19_clinical_trilinear_attention.json'
P = Path('/root/autodl-tmp/phase19_attention_private')
R = ROOT / 'reports/phase19_clinical_trilinear_attention'
VISUAL = Path('/root/autodl-tmp/phase17_ssd_private/visual')
TRAINING_TESTS = ['attention', 'data', 'model', 'training', 'baselines']
ANCESTOR_CONFIGS = [
    'phase17_ssd_fusion.json', 'phase17r_conditional_repair.json',
    'phase13_rna_information_audit.json', 'phase12_c4_confirmation.json',
    'phase11_semantic_anchored.json', 'phase10_molecular_groups.json',
]


def cfg():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def write(path, value):
    def cast(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=cast) + '\n', encoding='utf-8')
    temp.replace(path)


def _module_path(name):
    if not (name == 'src' or name.startswith('src.')):
        return None
    path = ROOT.joinpath(*name.split('.'))
    if path.with_suffix('.py').is_file():
        return path.with_suffix('.py')
    if (path / '__init__.py').is_file():
        return path / '__init__.py'
    return None


def _import_closure(initial):
    """Hash declared project imports recursively without executing imports."""
    pending, result = list(initial), set()
    while pending:
        path = Path(pending.pop())
        if path in result:
            continue
        if not path.is_file():
            raise ValueError('required training source missing: ' + path.relative_to(ROOT).as_posix())
        result.add(path)
        if path.suffix != '.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        package = list(path.relative_to(ROOT).with_suffix('').parts[:-1])
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = package[:len(package) - node.level + 1]
                    base = '.'.join(prefix + (node.module.split('.') if node.module else []))
                else:
                    base = node.module or ''
                names = [base] + [base + '.' + alias.name for alias in node.names if alias.name != '*']
            for name in names:
                dependency = _module_path(name)
                if dependency is not None and dependency not in result:
                    pending.append(dependency)
    return sorted(result, key=lambda x: x.relative_to(ROOT).as_posix())


def source_files():
    # Evaluation scripts/tests are deliberately outside training identity.
    # The five specific training tests must exist before preparation begins.
    initial = sorted((ROOT / 'src/phase19_attention').glob('*.py'))
    initial += [CONFIG, ROOT / 'scripts/run_phase19.py']
    initial += [ROOT / 'tests' / f'test_phase19_{name}.py' for name in TRAINING_TESTS]
    initial += [ROOT / 'configs' / name for name in ANCESTOR_CONFIGS]
    initial += [ROOT / 'configs/phase19_baseline_sources.json']
    result = {path.relative_to(ROOT).as_posix(): digest(path) for path in _import_closure(initial)}
    _verify_external_sources()
    return result


def _verify_external_sources():
    # The static manifest contains canonical repo-relative paths and content
    # hashes. Verify actual private source files, but do not hash their absolute
    # host paths or execute the third-party cores while computing identity.
    from .baselines import verify_baseline_sources
    return verify_baseline_sources()


def source():
    # Canonical POSIX relative names, independent of Windows/Linux separators.
    return hashlib.sha256(json.dumps(source_files(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@contextlib.contextmanager
def _cache_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if os.name == 'nt':
            # Only the synthetic cache tests use Windows; formal cache writers
            # use Linux flock below.  Lock one persistent byte on Windows.
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _cached_bank(path, provenance, producer):
    path = Path(path)
    receipt_path = path.with_suffix('.json')
    with _cache_lock(path.with_suffix('.lock')):
        if path.exists():
            if not receipt_path.is_file():
                raise ValueError('preprocessing cache lacks hash receipt')
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            if digest(path) != receipt.get('sha256'):
                raise ValueError('preprocessing cache hash changed')
            if receipt.get('provenance') != provenance:
                raise ValueError('preprocessing cache provenance changed')
            bank = torch.load(path, map_location='cpu', weights_only=False)
            if bank.get('provenance') != provenance:
                raise ValueError('serialized preprocessing provenance changed')
        else:
            if receipt_path.exists():
                raise ValueError('orphan preprocessing receipt')
            bank = producer()
            if bank.get('provenance') != provenance:
                raise ValueError('producer returned a different provenance')
            temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
            torch.save(bank, temp)
            temp.replace(path)
            write(receipt_path, {'status': 'PASS', 'sha256': digest(path), 'provenance': provenance})
    return bank, digest(path)


def measured_program_mean(values, measured, membership):
    """Fixed measured-gene mean, not a learned/program-label transformation."""
    values = np.asarray(values, np.float64)
    measured = np.asarray(measured, bool)
    membership = np.asarray(membership, bool)
    if values.ndim != 2 or measured.shape != values.shape or membership.ndim != 2 or membership.shape[1] != values.shape[1]:
        raise ValueError('program mean feature/mask/membership shapes disagree')
    if not np.isfinite(values[measured]).all():
        raise ValueError('measured program values must be finite')
    safe = np.where(measured, values, 0.)
    numerator = safe @ membership.T.astype(np.float64)
    denominator = measured.astype(np.float64) @ membership.T.astype(np.float64)
    out = numerator / np.maximum(denominator, 1.)
    return out.astype(np.float32), denominator > 0


def fit_scale_active(values, fit, active):
    values = np.asarray(values, np.float64)
    fit, active = np.asarray(fit, int), np.asarray(active, int)
    if values.ndim != 2 or not np.isfinite(values[active]).all() or not np.isin(fit, active).all():
        raise ValueError('invalid fit-local scaling input')
    mean, sd = values[fit].mean(0), values[fit].std(0)
    sd = np.where(sd < 1e-8, 1., sd)
    result = np.zeros_like(values)
    result[active] = (values[active] - mean) / sd
    return result.astype(np.float32), {'mean': mean, 'sd': sd}


def fit_pca_active(values, fit, active, components=32):
    """Fit scale/PCA/scale only on fit, transform active, zero non-active rows."""
    before, before_state = fit_scale_active(values, fit, active)
    if components > min(len(fit), before.shape[1]):
        raise ValueError('insufficient train samples/features for fixed PCA dimensions')
    pca = PCA(int(components), svd_solver='full', whiten=False).fit(before[fit])
    scores = np.zeros((len(before), components), np.float64)
    scores[active] = pca.transform(before[active])
    after, after_state = fit_scale_active(scores, fit, active)
    state = {'before_scale': before_state, 'pca': pca, 'after_scale': after_state}
    audit = {'components': int(components), 'fit_n': len(fit),
             'before_mean_hash': ahash(before_state['mean']), 'before_sd_hash': ahash(before_state['sd']),
             'PCA_components_hash': ahash(pca.components_), 'PCA_mean_hash': ahash(pca.mean_),
             'after_mean_hash': ahash(after_state['mean']), 'after_sd_hash': ahash(after_state['sd']),
             'explained_variance_ratio_sum': float(np.nan_to_num(pca.explained_variance_ratio_).sum())}
    return after, state, audit


def _visual_registry():
    receipt = VISUAL / 'READY.json'
    ready = json.loads(receipt.read_text(encoding='utf-8'))
    if ready.get('status') != 'PASS' or ready.get('patients') != 512:
        raise ValueError('historical visual cache is not verified')
    relevant = ['u', 'mask', 'owner', 'coords', 'local_row']
    for name in relevant:
        if digest(VISUAL / (name + '.npy')) != ready['hashes'][name]:
            raise ValueError('historical visual hash changed: ' + name)
    if ready['dataset_sha256'] != digest(parent.DATA):
        raise ValueError('visual/molecular patient cohort mismatch')
    return {'receipt_sha256': digest(receipt),
            'hashes': {name: ready['hashes'][name] for name in relevant},
            'dataset_sha256': ready['dataset_sha256'],
            'validation': 'geometry-only; no raw WSI/histological validation'}


def active_visuals(active, patients=512):
    active = np.asarray(active, int)
    all_owner = np.load(VISUAL / 'owner.npy')
    all_mask = np.load(VISUAL / 'mask.npy')
    raw = np.load(VISUAL / 'u.npy', mmap_mode='r')
    if raw.shape != (*all_mask.shape, 1536) or all_owner.shape != (len(raw),):
        raise ValueError('visual/owner dimensions changed')
    if not np.array_equal(np.unique(all_owner), np.arange(patients)):
        raise ValueError('visual patient coverage changed')
    if all_mask.dtype != bool or raw.shape[1] != 512:
        raise ValueError('visual mask or patch budget changed')
    if np.bincount(all_owner, weights=all_mask.sum(1), minlength=patients).max() > 512:
        raise ValueError('patient patch budget exceeded')
    slides = np.flatnonzero(np.isin(all_owner, active))
    owner = all_owner[slides].astype(np.int64)
    mask = all_mask[slides]
    visual = np.asarray(raw[slides], np.float32)
    if not mask.any(1).all() or not np.isfinite(visual).all():
        raise ValueError('nonfinite or empty active slide')
    means = np.zeros((patients, 1536), np.float64)
    slide_mean = (visual * mask[:, :, None]).sum(1, dtype=np.float64) / mask.sum(1)[:, None]
    for patient in active:
        if not np.any(owner == patient):
            raise ValueError('active patient has no slide')
        means[patient] = slide_mean[owner == patient].mean(0)
    return visual, mask, owner, slides, means


def cancer_derangement(d, indices, seed):
    """Partition-local cancer-only cyclic mismatch; no severity/age/sex input."""
    ix = np.asarray(indices, int)
    n = len(d['cancer_type'])
    if len(ix) < 2 or len(np.unique(ix)) != len(ix):
        raise ValueError('derangement requires at least two distinct partition members')
    out = np.arange(n)
    rng = np.random.default_rng(seed)
    cancer = np.asarray(d['cancer_type']).astype(str)
    singleton = []
    for kind in np.unique(cancer[ix]):
        block = ix[cancer[ix] == kind]
        if len(block) > 1:
            block = rng.permutation(block)
            out[block] = np.roll(block, 1)
        else:
            singleton.extend(block)
    if len(singleton) > 1:
        block = rng.permutation(singleton)
        out[block] = np.roll(block, 1)
    elif singleton:
        one = singleton[0]
        partner = int(rng.choice(ix[ix != one]))
        destination = out[partner]
        out[partner], out[one] = one, destination
    if (out[ix] == ix).any() or set(out[ix]) != set(ix):
        raise ValueError('invalid partition-local cancer derangement')
    return out


def mismatch_audit(d, indices, mapping):
    ix = np.asarray(indices, int)
    cancer = np.asarray(d['cancer_type']).astype(str)
    if set(mapping[ix]) != set(ix) or (mapping[ix] == ix).any():
        raise ValueError('wrong-patient mapping crossed partition or contains identity')
    return {'n': len(ix), 'map_hash': ahash(mapping[ix]),
            'identity_count': int((mapping[ix] == ix).sum()),
            'cancer_mismatch_count': int((cancer[ix] != cancer[mapping[ix]]).sum())}


def _feature_bank(d, fold, role, fit, valid, active, visual_mean, provenance):
    original, molecular_audit = old.molecular_features(d, fold, role)
    f = {}
    for key in ['C', 'S', 'R', 'G', 'L', 'RM', 'CM']:
        f[key] = np.zeros_like(original[key], dtype=np.float32)
        f[key][active] = original[key][active]
    f['membership'] = np.asarray(original['membership'], np.float32)
    f['CS'] = np.c_[f['C'], f['S']].astype(np.float32)
    if f['R'].shape != (512, 512) or f['membership'].shape != (8, 512):
        raise ValueError('locked RNA512 / eight-program dimensions changed')
    state, transforms = {}, {}
    for name, gene_key in [('GN8', 'G'), ('LN8', 'L')]:
        mean, measured = measured_program_mean(f[gene_key], f['CM'] > 0, f['membership'] > 0)
        f[name], state[name] = fit_scale_active(mean, fit, active)
        f[name + '_MASK'] = measured.astype(np.float32)
    for name, values in [('UNI32', visual_mean), ('RPCA32', f['R']),
                         ('CNAPCA32', np.c_[f['G'], f['L']])]:
        f[name], state[name], transforms[name] = fit_pca_active(values, fit, active, 32)
    for name, mask_key in [('RCOV8', 'RM'), ('CCOV8', 'CM')]:
        coverage = (f[mask_key] @ f['membership'].T
                    / f['membership'].sum(1)[None, :])
        f[name], state[name] = fit_scale_active(coverage, fit, active)
    keys = ['C', 'S', 'R', 'GN8', 'LN8', 'UNI32', 'RCOV8', 'CCOV8']
    f['X'] = np.concatenate([f[key] for key in keys], axis=1).astype(np.float32)
    edges = np.r_[0, np.cumsum([f[key].shape[1] for key in keys])]
    blocks = {key: (int(edges[i]), int(edges[i + 1])) for i, key in enumerate(keys)}
    if any(not np.isfinite(v).all() for v in f.values()):
        raise ValueError('nonfinite cached features')
    audit = {'status': 'PASS', 'fold': int(fold), 'role': role,
             'source': provenance['source'], 'fit_n': len(fit), 'valid_n': len(valid), 'active_n': len(active),
             'fit_hash': ahash(fit), 'valid_hash': ahash(valid), 'active_hash': ahash(active),
             'feature_shapes': {key: list(value.shape) for key, value in f.items()},
             'fit_feature_hashes': {key: ahash(value[fit]) for key, value in f.items() if key != 'membership'},
             'valid_feature_hashes': {key: ahash(value[valid]) for key, value in f.items() if key != 'membership'},
             'membership_hash': ahash(f['membership']), 'membership_sizes': (f['membership'] > 0).sum(1).tolist(),
             'molecular': molecular_audit, 'PCA': transforms,
             'X_order': keys, 'X_blocks': blocks,
             'CNA_program_pool': 'measured-gene mean of separate gain/loss channels; fit-local scale',
             'coverage_main_effects': '8 fixed RNA and 8 fixed CNA program measured fractions, fit-local scale; per-gene masks remain available to token encoder',
             'UNI_mean': 'valid patch mean within WSI, equal WSI mean within patient',
             'visual_representation': 'frozen UNI2-h area-weighted physical CONCH footprint; original coordinate order',
             'geometry_QA': 'geometry-only; no raw WSI/histological validation',
             'inner_transforms_outer_test': False if role == 'inner' else None}
    return {'provenance': provenance, 'fit': fit, 'valid': valid, 'active': active,
            'features': f, 'state': state, 'blocks': blocks, 'audit': audit}


class Data:
    def __init__(self, fold, role, seed, device='cuda'):
        if role not in ['inner', 'outer'] or int(fold) not in range(1, 6):
            raise ValueError('invalid fold/preprocessing role')
        self.fold, self.role, self.seed = int(fold), role, int(seed)
        self.d = parent.load()
        self.part = parent.splits(self.d, 'DSS', self.fold)
        self.parts = self.part
        self.fit = self.part['inner_train' if role == 'inner' else 'outer_train']
        self.valid = self.part['inner_val' if role == 'inner' else 'outer_test']
        self.active = np.sort(np.r_[self.fit, self.valid])
        if len(np.unique(self.active)) != len(self.active):
            raise ValueError('fit/validation overlap')
        if role == 'inner' and np.intersect1d(self.active, self.part['outer_test']).size:
            raise ValueError('inner role includes outer-test patients')
        registry = _visual_registry()
        visual, mask, self.owner, self.slides, visual_mean = active_visuals(self.active)
        self.global_wsi_indices = self.slides
        feature = parent.PARENT / 'preprocessing' / f'fold{fold}_{role}'
        provenance = {'source': source(), 'dataset_sha256': digest(parent.DATA),
                      'endpoint_npz_sha256': digest(parent.PRIVATE / 'endpoints.npz'),
                      'partition_manifest_sha256': digest(parent.PARENT / 'formal/manifests' / f'R0_128_seed31_fold{fold}.json'),
                      'parent_feature_json_sha256': digest(feature.with_suffix('.json')),
                      'parent_feature_npz_sha256': digest(feature.with_suffix('.npz')),
                      'fit_hash': ahash(self.fit), 'valid_hash': ahash(self.valid), 'visual': registry}
        self.cache_path = P / 'preprocessing' / f'fold{fold}_{role}.pt'
        bank, cache_hash = _cached_bank(self.cache_path, provenance,
            lambda: _feature_bank(self.d, self.fold, role, self.fit, self.valid,
                                  self.active, visual_mean, provenance))
        if not np.array_equal(bank['fit'], self.fit) or not np.array_equal(bank['valid'], self.valid) or not np.array_equal(bank['active'], self.active):
            raise ValueError('cached partitions changed')
        self.f, self.state, self.blocks = bank['features'], bank['state'], bank['blocks']
        self.reference_c_trainmean = self.f['C'][self.fit].mean(0).astype(np.float32)
        self.audit = dict(bank['audit'])
        self.audit.update(cache_sha256=cache_hash, visual_source=registry,
                          active_slides=len(self.owner), patch_budget_per_patient=512,
                          active_patient_patch_min=int(np.bincount(self.owner, weights=mask.sum(1), minlength=512)[self.active].min()),
                          active_patient_patch_max=int(np.bincount(self.owner, weights=mask.sum(1), minlength=512)[self.active].max()))
        self.visual = torch.as_tensor(visual, device=device)
        self.mask = torch.as_tensor(mask, device=device)
        self.ft = {key: torch.as_tensor(value, device=device) for key, value in self.f.items()}
        self.wrong_c = np.arange(512)
        self.wrong_o = np.arange(512)
        control = {}
        for name, ix in [('fit', self.fit), ('valid', self.valid)]:
            self.wrong_c[ix] = cancer_derangement(self.d, ix, self.seed + 19000 + self.fold)[ix]
            self.wrong_o[ix] = old.wrong_map(self.d, ix, self.seed + 1000 + self.fold)[ix]
            control[name] = {'clinical': mismatch_audit(self.d, ix, self.wrong_c),
                             'omics': mismatch_audit(self.d, ix, self.wrong_o)}
            stage = self.d['clinical_AJCC_PATHOLOGIC_TUMOR_STAGE'].astype(str)
            control[name]['omics']['Stage_mismatch_count'] = int((stage[ix] != stage[self.wrong_o[ix]]).sum())
        self.audit['wrong_controls'] = control
        self.audit['wrong_clinical_rule'] = 'partition-local cancer-only derangement; singleton fallback reported; no Stage/Age/Sex'
        self.audit['wrong_omics_rule'] = 'parent partition-local cancer/Stage derangement; own main effects unchanged'
        endpoint = parent.endpoints(self.d)['DSS']
        self.ep = {'time': np.full(512, np.nan), 'event': np.full(512, np.nan), 'valid': np.zeros(512, bool)}
        for key in self.ep:
            self.ep[key][self.active] = endpoint[key][self.active]
        self.ev = self.ep
        self.t, self.e = self.ep['time'], self.ep['event']
        self.budget = 512

    def check_indices(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('patient indices must be a one-dimensional integer array')
        if not np.isin(indices, self.active).all():
            raise ValueError('requested patient outside active preprocessing role')
        return indices.astype(np.int64)
