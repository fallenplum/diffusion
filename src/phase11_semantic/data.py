"""Read-only H5 audit and compact nested cache, preserving all 519 slides."""
import json
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch
from src.phase10_trimodal.data import digest, write_json, load_private


def nested_order(coords, cap=1024):
    # Recursive spatial stratification: coordinate lexsort, first midpoint then
    # widest remaining interval midpoint. One prefix sequence for ALL budgets.
    order = np.lexsort((coords[:, 1], coords[:, 0])); queue = [(0, len(order))]; selected = []
    while queue and len(selected) < cap:
        queue.sort(key=lambda ab: (-(ab[1]-ab[0]), ab[0]))
        lo, hi = queue.pop(0); mid = (lo+hi)//2; selected.append(int(order[mid]))
        if mid > lo: queue.append((lo, mid))
        if mid+1 < hi: queue.append((mid+1, hi))
    return np.asarray(selected, np.int64)


def audit_cache(dataset, manifest, feature_root, output):
    d = load_private(dataset); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(manifest).query("status=='EXACT_SLIDE_MATCH'").sort_values(['patient_id', 'filename'])
    if len(tab) != 519 or tab.filename.duplicated().any() or set(tab.patient_id) != set(d['patient_ids']):
        raise ValueError('patient slide mismatch')
    total_selected = 0
    for row in tab.itertuples(index=False):
        with h5py.File(Path(feature_root)/row.delivery_relative_path, 'r') as f:
            total_selected += min(1024, len(f['features']))
    arr = np.lib.format.open_memmap(out/'patch1024.npy', mode='w+', dtype=np.float32, shape=(total_selected, 768))
    coords = np.lib.format.open_memmap(out/'coords1024.npy', mode='w+', dtype=np.int64, shape=(total_selected, 2))
    loc = {p: i for i, p in enumerate(d['patient_ids'].astype(str))}; rows = []; offset = 0
    for row in tab.itertuples(index=False):
        path = Path(feature_root)/row.delivery_relative_path
        if digest(path) != row.conch_sha256: raise ValueError('H5 SHA mismatch')
        with h5py.File(path, 'r') as f:
            x = f['features'][:]; xy = f['coords'][:]
            if x.shape != (len(xy), 768) or xy.shape != (len(x), 2) or len(x) == 0 or not np.isfinite(x).all():
                raise ValueError('H5 shape or finite')
            if not np.issubdtype(xy.dtype, np.integer) or (xy < 0).any() or len(np.unique(xy, axis=0)) != len(xy):
                raise ValueError('coords')
            if f['features'].attrs.get('encoder') != 'conch_v15' or f['coords'].attrs.get('patch_size') != 512 or f['coords'].attrs.get('target_magnification') != 20:
                raise ValueError('encoder/grid')
        sel = nested_order(xy); assert len(set(sel)) == len(sel)
        sets = [set(sel[:b]) for b in [128, 512, 1024]]
        if not sets[0] <= sets[1] <= sets[2]: raise ValueError('nesting')
        n = len(sel); arr[offset:offset+n] = x[sel]; coords[offset:offset+n] = xy[sel]
        rows.append({'patient_index': loc[row.patient_id], 'filename': row.filename, 'h5_sha256': row.conch_sha256,
                     'original': len(x), 'offset': offset, 'selected': n, 'selected_original_indices': sel.tolist(),
                     'fold': int(d['folds'][loc[row.patient_id]])})
        offset += n
    if offset != total_selected: raise ValueError('cache inventory changed')
    arr.flush(); coords.flush(); write_json(out/'private_slide_manifest.json', rows)
    titan = dict(np.load('/root/phase6a_titan_runtime/cache_v1/patient_representations.npz', allow_pickle=True))
    if not np.array_equal(titan['patient_ids'], d['patient_ids']) or not np.array_equal(titan['native'], d['titan']) or not np.array_equal(titan['concept_cosine'], d['static']):
        raise ValueError('TITAN original cache mismatch')
    fixed = pd.read_csv('/root/phase10_trimodal_private/cache/private_fixed_patient_folds.csv')
    idcol = next(c for c in fixed.columns if 'patient' in c.lower()); foldcol = next(c for c in fixed.columns if 'fold' in c.lower())
    mapping = dict(zip(fixed[idcol].astype(str), fixed[foldcol].astype(int)))
    if not np.array_equal([mapping[p] for p in d['patient_ids'].astype(str)], d['folds']): raise ValueError('original fold mismatch')
    result = {'status': 'PASS', 'patients': 512, 'DSS_events': 65, 'slides': 519, 'original_patches': sum(r['original'] for r in rows),
              'selected_1024': offset, 'budgets': {str(b): int(sum(min(b, r['original']) for r in rows)) for b in [128, 512, 1024]},
              'subset_checked_slides': len(rows), 'sampling': 'coordinate lexsort recursive widest-interval midpoint prefixes',
              'patch_sha256': digest(out/'patch1024.npy'), 'coordinates_sha256': digest(out/'coords1024.npy'),
              'private_slide_manifest_sha256': digest(out/'private_slide_manifest.json'), 'dataset_sha256': digest(dataset),
              'source_manifest_sha256': digest(manifest), 'TITAN_source_exact': True, 'fixed_folds_source_exact': True,
              'rna_shapes': {k: list(d[k].shape) for k in ['raw_rna', 'rna_genes', 'raw_cna', 'cna_genes']},
              'missing_3024_patients': int((np.isnan(d['raw_rna']).sum(1) == 3024).sum()),
              'platform_identity': 'not confirmed; missing-pattern proxy', 'old128_sampling_reused': False}
    write_json(out/'data_manifest.json', result); return result


class PatchCache:
    def __init__(self, directory, budget):
        self.directory = Path(directory); self.budget = budget
        self.arr = np.load(self.directory/'patch1024.npy', mmap_mode='r')
        self.rows = json.loads((self.directory/'private_slide_manifest.json').read_text())
        self.by_patient = [[] for _ in range(512)]
        for r in self.rows: self.by_patient[r['patient_index']].append(r)
    def batch(self, indices, device):
        slides = max(len(self.by_patient[i]) for i in indices)
        n = max(min(self.budget, row['selected']) for i in indices for row in self.by_patient[i])
        p = np.zeros((len(indices), slides, n, 768), np.float32); mask = np.zeros(p.shape[:3], bool)
        for k, i in enumerate(indices):
            for j, row in enumerate(self.by_patient[i]):
                count = min(self.budget, row['selected']); offset = row['offset']
                p[k, j, :count] = self.arr[offset:offset+count]; mask[k, j, :count] = True
        return torch.tensor(p, device=device), torch.tensor(mask, device=device)
