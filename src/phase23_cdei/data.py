"""Read-only inherited inputs with endpoint-free feature views for CDEI G0.

The legacy loader verifies existing fit-local caches and frozen Cox anchors.
Its temporary object contains protected endpoints; this wrapper retains only
fit labels and (for an inner role only) explicitly accessible selection labels.
No outer-test label vector or raw cohort dictionary is retained here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from src.phase20_selective import data as prior
from src.phase10_trimodal.data import digest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/phase23_cdei_g0.json'
P = Path('/root/autodl-tmp/phase23_cdei_private')
R = ROOT / 'reports/phase23_cdei'
write = prior.write


def cfg():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def source_files():
    """Hash new source and inherited AST closure without requiring new fields."""
    initial = sorted((ROOT / 'src/phase23_cdei').glob('*.py'))
    initial += [CONFIG]
    initial += sorted((ROOT / 'tests').glob('test_phase23_cdei_*.py'))
    initial += sorted((ROOT / 'scripts').glob('*phase23_cdei*.py'))
    files = dict(prior.source_files())
    files.update({p.relative_to(ROOT).as_posix(): digest(p)
                  for p in prior.prior._import_closure(initial)})
    return files


def source():
    return hashlib.sha256(json.dumps(source_files(), sort_keys=True,
                          separators=(',', ':')).encode()).hexdigest()


def _hash(values):
    values = np.ascontiguousarray(values)
    description = json.dumps({'shape': list(values.shape), 'dtype': str(values.dtype)},
                             sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(description + b'\n' + values.tobytes()).hexdigest()


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _indices(values, size, name, *, unique=True):
    result = np.asarray(values)
    if (result.ndim != 1 or not np.issubdtype(result.dtype, np.integer)
            or len(result) < 1 or (result < 0).any() or (result >= size).any()
            or (unique and len(np.unique(result)) != len(result))):
        raise ValueError(name + ' must be nonempty in-bounds integer patient indices')
    return result.astype(np.int64, copy=True)


def validated_binary_mask(values, name='availability'):
    """Never silently bool-cast nonbinary masks or mutate the legacy cache."""
    value = np.asarray(values)
    if value.ndim != 2 or not np.isfinite(value).all() or not np.isin(value, [0, 1]).all():
        raise ValueError(name + ' must be finite binary availability')
    return np.array(value, dtype=bool, copy=True)


def raw_rna_missing_patterns(raw_rna):
    """Private per-patient signatures, platform proxy rather than platform truth."""
    values = np.asarray(raw_rna)
    if values.ndim != 2 or min(values.shape) < 1 or not np.issubdtype(values.dtype, np.number):
        raise ValueError('raw RNA must be a numeric patient-by-gene matrix')
    measured = np.isfinite(values)
    prefix = str(values.shape[1]).encode() + b':'
    packed = np.packbits(measured, axis=1)
    return np.array([hashlib.sha256(prefix + row.tobytes()).hexdigest()
                     for row in packed])


class PairingUnavailableError(ValueError):
    def __init__(self, audit):
        self.audit = copy.deepcopy(audit)
        super().__init__('RNA missing-pattern singleton prevents zero-self patient pairing')


def partition_derangement(cancer, missing_patterns, indices, seed):
    """Zero-self permutation, exact missing-pattern conservation, no outcomes.

    Prefer cancer x raw-RNA-missing-pattern strata. Cancer may be relaxed only
    inside the same missing pattern to resolve singleton strata. A pattern
    with one patient blocks the cell; no patient is dropped or paired to self.
    """
    cancer = np.asarray(cancer).astype(str)
    patterns = np.asarray(missing_patterns).astype(str)
    if cancer.ndim != 1 or patterns.shape != cancer.shape:
        raise ValueError('cancer and missing-pattern vectors must align')
    ix = _indices(indices, len(cancer), 'partition')
    if any(v in ('', 'None', 'nan') for v in cancer[ix]):
        raise ValueError('cancer stratum is missing')
    rng = np.random.default_rng(int(seed))
    mapping = np.arange(len(cancer), dtype=np.int64)
    pattern_groups = [ix[patterns[ix] == p] for p in np.unique(patterns[ix])]
    primary_sizes = [int(np.count_nonzero((cancer[group] == ca)))
                     for group in pattern_groups for ca in np.unique(cancer[group])]
    audit = {'status': 'AVAILABLE', 'patients': len(ix),
             'primary_cancer_pattern_strata': len(primary_sizes),
             'primary_singleton_patients': sum(v == 1 for v in primary_sizes),
             'missing_pattern_strata': len(pattern_groups),
             'pattern_singleton_patients': sum(len(v) == 1 for v in pattern_groups),
             'RNA_pattern_scope': 'full_raw_RNA_isfinite_platform_proxy_not_verified_platform',
             'no_DSS_Stage_Age_Sex_matching': True,
             'partition_sha256': _hash(ix)}
    if audit['pattern_singleton_patients']:
        audit['status'] = 'UNAVAILABLE_PATTERN_SINGLETON'
        raise PairingUnavailableError(audit)
    for group in pattern_groups:
        singletons = []
        for ca in np.unique(cancer[group]):
            block = group[cancer[group] == ca]
            if len(block) == 1:
                singletons.append(int(block[0]))
            else:
                cycle = rng.permutation(block)
                mapping[cycle] = np.roll(cycle, 1)
        if len(singletons) > 1:
            cycle = rng.permutation(singletons)
            mapping[cycle] = np.roll(cycle, 1)
        elif singletons:
            one = singletons[0]
            other = int(rng.choice(group[group != one]))
            destination = mapping[other]
            mapping[other], mapping[one] = one, destination
    if ((mapping[ix] == ix).any() or set(mapping[ix]) != set(ix)
            or (patterns[ix] != patterns[mapping[ix]]).any()):
        raise ValueError('partition or missing-pattern invariant failed')
    audit.update(identity_count=0, missing_pattern_mismatch_count=0,
                 relaxed_cancer_match_count=int((cancer[ix] != cancer[mapping[ix]]).sum()),
                 mapping_sha256=_hash(mapping[ix]))
    return mapping, audit


def pair_mask_summary(rm, cm, membership, indices, mapping):
    """Anonymous counts describe any measured-support difference after pairing."""
    rm = validated_binary_mask(rm, 'RNA mask')
    cm = validated_binary_mask(cm, 'CNA mask')
    member = validated_binary_mask(membership, 'program membership')
    if cm.shape != rm.shape or member.shape != (8, rm.shape[1]) or not member.any(1).all():
        raise ValueError('program mask shapes changed')
    ix = np.asarray(indices, np.int64)
    RNA = rm.astype(np.int64) @ member.astype(np.int64).T > 0
    CNA = cm.astype(np.int64) @ member.astype(np.int64).T > 0
    tokens = np.concatenate([RNA, CNA, CNA], axis=1)
    own, other = tokens[ix], tokens[np.asarray(mapping)[ix]]
    shared = own & other
    counts = shared.sum(1)
    return {'patients': len(ix), 'token_slots': 24,
            'own_valid_tokens_total': int(own.sum()),
            'donor_valid_tokens_total': int(other.sum()),
            'shared_valid_tokens_total': int(shared.sum()),
            'shared_valid_tokens_patient_min': int(counts.min()),
            'shared_valid_tokens_patient_max': int(counts.max()),
            'shared_valid_tokens_patient_mean': float(counts.mean()),
            'support_changed_patients': int(np.any(own != other, axis=1).sum()),
            'shared_mask_sha256': _hash(shared)}


@dataclass(frozen=True)
class FeatureView:
    """Feature-only interface: no endpoints, source dictionary or label methods."""
    fold: int
    role: str
    seed: int
    fit: np.ndarray
    valid: np.ndarray
    active: np.ndarray
    owner: np.ndarray
    visual: torch.Tensor
    mask: torch.Tensor
    f: object
    ft: object
    wrong_c: np.ndarray
    wrong_o: np.ndarray

    def feat(self, names=None):
        keys = list(self.ft) if names is None else ([names] if isinstance(names, str) else list(names))
        if any(key not in self.ft for key in keys):
            raise ValueError('requested feature is outside the endpoint-free view')
        return {key: self.ft[key] for key in keys}

    def check_indices(self, indices):
        index = _indices(indices, len(self.f['C']), 'query', unique=False)
        if not np.isin(index, self.active).all():
            raise ValueError('query is outside the active preprocessing role')
        return index


def _label_subset(endpoint, indices):
    result = {key: np.array(endpoint[key][indices], copy=True) for key in ('time', 'event')}
    if (not np.isfinite(result['time']).all() or (result['time'] < 0).any()
            or not np.isin(result['event'], [0, 1]).all()):
        raise ValueError('invalid labels in the explicitly authorized partition')
    return result


class Data:
    """Qualified Phase20 inputs, with all inherited endpoints removed after load."""
    def __init__(self, fold, role, seed, device='cuda'):
        if role not in ('inner', 'outer'):
            raise ValueError('role must be inner or outer')
        inherited = prior.Data(fold, role, seed, device=device)
        self.fold, self.role, self.seed = int(fold), role, int(seed)
        fit = _indices(inherited.fit, len(inherited.f['C']), 'fit')
        valid = _indices(inherited.valid, len(inherited.f['C']), 'valid')
        if np.intersect1d(fit, valid).size:
            raise ValueError('fit and validation partitions overlap')
        active = np.sort(np.r_[fit, valid])
        # No parent object, ep, ev, t/e, or raw d is retained by this wrapper.
        self._fit_supervision = _label_subset(inherited.ep, fit)
        self._selection_supervision = _label_subset(inherited.ep, valid) if role == 'inner' else None
        raw_patterns = raw_rna_missing_patterns(inherited.d['rna_genes'])
        cancer = np.asarray(inherited.d['cancer_type'])
        wrong_c = np.arange(len(cancer), dtype=np.int64)
        wrong_o = np.arange(len(cancer), dtype=np.int64)
        pair_audits = {}
        for name, ix in (('fit', fit), ('valid', valid)):
            map_c, ac = partition_derangement(cancer, raw_patterns, ix, self.seed + 23000 + self.fold)
            map_o, ao = partition_derangement(cancer, raw_patterns, ix, self.seed + 24000 + self.fold)
            wrong_c[ix], wrong_o[ix] = map_c[ix], map_o[ix]
            ao['token_support'] = pair_mask_summary(inherited.f['RM'], inherited.f['CM'],
                                                    inherited.f['membership'], ix, map_o)
            pair_audits[name] = {'clinical': ac, 'omics': ao}
        f = {key: np.array(value, copy=True) for key, value in inherited.f.items()}
        for key in ('RM', 'CM'):
            f[key] = validated_binary_mask(f[key], key)
        f['membership'] = validated_binary_mask(f['membership'], 'membership')
        for value in f.values():
            value.setflags(write=False)
        ft = {key: torch.as_tensor(np.array(value, copy=True), device=inherited.visual.device)
              for key, value in f.items()}
        view = FeatureView(self.fold, self.role, self.seed, _readonly(fit), _readonly(valid),
                           _readonly(active), _readonly(inherited.owner, np.int64),
                           inherited.visual, inherited.mask, MappingProxyType(f),
                           MappingProxyType(ft), _readonly(wrong_c), _readonly(wrong_o))
        self.feature_view = view
        for name in ('fit', 'valid', 'active', 'owner', 'visual', 'mask', 'f', 'ft', 'wrong_c', 'wrong_o'):
            setattr(self, name, getattr(view, name))
        self.anchor_sd_fit = float(inherited.anchor_sd_fit)
        self.anchor_w = _readonly(inherited.anchor_w)
        self.anchor_reference = copy.deepcopy(inherited.anchor_reference)
        self.reference_c_trainmean = np.asarray(f['C'][fit].mean(0), np.float32)
        self.inherited_simple = {key: copy.deepcopy(inherited.inherited_simple[key])
                                 for key in ('source', 'fit', 'valid', 'preprocess_audit')
                                 if key in inherited.inherited_simple}
        self.inherited_simple['models'] = {
            name: {key: np.array(inherited.inherited_simple['models'][name][key], copy=True)
                   for key in ('w', 'risk')} for name in ('J0', 'J_CS')}
        self.audit = {'schema': 'phase23_endpoint_separated_data_v1', 'role': role,
                      'fold': self.fold, 'fit_n': len(fit), 'valid_n': len(valid),
                      'fit_sha256': _hash(fit), 'valid_sha256': _hash(valid),
                      'feature_cache_sha256': inherited.upstream_audit.get('cache_sha256'),
                      'anchor_reference': copy.deepcopy(self.anchor_reference),
                      'feature_hashes': {key: _hash(value) for key, value in f.items()},
                      'wrong_controls': pair_audits,
                      'outer_test_labels_retained': False,
                      'legacy_loader_temporarily_loads_protected_endpoints': True,
                      'new_feature_or_convex_fits': 0}

    def feat(self, names=None):
        return self.feature_view.feat(names)

    def check_indices(self, indices):
        return self.feature_view.check_indices(indices)

    def fit_labels(self):
        return {key: value.copy() for key, value in self._fit_supervision.items()}

    def selection_labels(self):
        if self.role != 'inner':
            raise ValueError('outer-test labels are sealed; independent evaluation only')
        return {key: value.copy() for key, value in self._selection_supervision.items()}
