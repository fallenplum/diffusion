"""All transforms refitted within the current training partition."""
import numpy as np
from sklearn.decomposition import PCA
from src.phase10_trimodal.features import clinical, gene_fit, scale, ahash, CLINICAL


def prepare(d, fit, groups, seed=117301):
    fit = np.asarray(fit, dtype=np.int64)
    audit = {'fit_indices': fit.tolist(), 'fit_hash': ahash(fit), 'source_fields': CLINICAL,
             'selection_source': 'RNA expression only; no outcome'}
    out = {}; masks = {}
    for key, minimal in [('C', False), ('C_min', True)]:
        raw, coding = clinical(d, fit, minimal)
        out[key], state = scale(raw, fit)
        sizes = [1 if name == 'AGE' else len(v['levels']) for name, v in coding.items()]
        field_mask = np.zeros((len(sizes), raw.shape[1]), np.float32)
        start = 0
        for j, size in enumerate(sizes):
            field_mask[j, start:start + size] = 1; start += size
        masks[key] = field_mask
        audit[key] = {'coding': coding, 'mean': state['mean'].tolist(), 'sd': state['sd'].tolist()}
    out['S'], ss = scale(np.asarray(d['static'], float), fit)
    audit['S'] = {k: v.tolist() for k, v in ss.items()}
    x, st = gene_fit(d['rna_genes'], fit)
    eligible = np.flatnonzero(st['usable'])
    top = eligible[np.argsort(-st['sd'][eligible], kind='stable')[:512]]
    if len(top) != 512: raise ValueError('Top512 insufficient')
    out['R'] = x[:, top]
    audit['top_indices'] = top.tolist(); audit['top_symbols'] = d['rna_gene_symbols'][top].astype(str).tolist()
    audit['top_hash'] = ahash(top); audit['rna_fit_hash'] = ahash(np.r_[st['median'], st['mean'], st['sd']])
    audit['missing_mask_hash'] = ahash(np.isnan(d['rna_genes'][fit]))
    names = d['rna_gene_symbols'][top].astype(str)
    gm = np.array([[name in values for name in names] for values in groups.values()], np.float32)
    biologic_covered = int((gm.sum(0) > 0).sum())
    # All arms use the SAME Top512. Uncovered genes receive explicit nonbiologic
    # fallback slots; we do not silently drop them or claim Hallmark membership.
    for j in np.flatnonzero(gm.sum(0) == 0): gm[j % 8, j] = 1
    masks['true'] = gm
    ordered = np.zeros((8, 512), np.float32)
    for k in range(8): ordered[k, k*64:(k+1)*64] = 1
    masks['ordered'] = ordered
    rng = np.random.default_rng(seed)
    mean = st['mean'][top]; sd = st['sd'][top]
    bins = (np.argsort(np.argsort(mean, kind='stable'), kind='stable')*5//512)*5
    bins += np.argsort(np.argsort(sd, kind='stable'), kind='stable')*5//512
    perm = np.arange(512)
    for b in np.unique(bins):
        ix = rng.permutation(np.flatnonzero(bins == b))
        if len(ix) > 1: perm[ix] = np.roll(ix, 1)
    shuffled = np.zeros_like(gm); shuffled[:, perm] = gm
    masks['random'] = shuffled
    assert np.array_equal(gm @ gm.T, shuffled @ shuffled.T)
    assert (gm.sum(0) > 0).all() and (shuffled.sum(0) > 0).all()
    audit['groups'] = {'names': list(groups), 'biologic_covered': biologic_covered,
                       'fallback_genes': 512-biologic_covered,
                       'true_members': [np.flatnonzero(v).tolist() for v in gm],
                       'random_members': [np.flatnonzero(v).tolist() for v in shuffled],
                       'permutation': perm.tolist(), 'bins': bins.tolist(),
                       'singleton_bins': int(sum(np.sum(bins == b) == 1 for b in np.unique(bins)))}
    z, zs = scale(np.asarray(d['titan'], float), fit)
    pca = PCA(32, svd_solver='full').fit(z[fit]); out['T'] = pca.transform(z)
    audit['pca'] = {'fit_hash': ahash(fit), 'components_hash': ahash(pca.components_),
                    'mean_hash': ahash(zs['mean']), 'sd_hash': ahash(zs['sd'])}
    return {k: v.astype(np.float32) for k, v in out.items()}, masks, audit
