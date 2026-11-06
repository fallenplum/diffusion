"""R4冻结特征配方：Clinical + TITAN S32 + GRID-CONCH PCA32 + RNA512。

源逻辑对应 diffusion / src/bgb_en/protocol.py（R4/2026-10）。
本软件以特征为起点，不下载SVS，不装载第三方基础模型权重。
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA

CLINICAL = ('AGE', 'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 'PATH_T_STAGE', 'PATH_N_STAGE', 'PATH_M_STAGE')


def scale_fit(x, train):
    x, train = np.asarray(x, float), np.asarray(train, int)
    if not np.isfinite(x[train]).all():
        raise ValueError('训练输入存在非有限数值')
    mu, sigma = x[train].mean(0), x[train].std(0)
    return {'mean': mu, 'sd': np.where(sigma > 1e-8, sigma, 1.0)}


def scale_apply(x, state):
    return (np.asarray(x, float) - state['mean']) / state['sd']


def clinical_fit(raw, train):
    cols, coding = [], {}
    for name in CLINICAL:
        v = np.asarray(raw['clinical_raw'][name]).astype(str)
        if name == 'AGE':
            values = np.array([float(s) if s.replace('.', '', 1).isdigit() else np.nan for s in v])
            median = float(np.nanmedian(values[train])); coding[name] = {'median': median}
            cols.append(np.where(np.isfinite(values), values, median)[:, None])
        else:
            levels = np.unique(v[train]); coding[name] = {'levels': levels.tolist()}
            cols.append(np.stack([v == s for s in levels], 1))
    x = np.concatenate(cols, 1).astype(float)
    return scale_apply(x, (st := scale_fit(x, train))), {'coding': coding, 'scale': st}


def clinical_apply(raw, state):
    cols = []
    for name in CLINICAL:
        v = np.asarray(raw['clinical_raw'][name]).astype(str)
        if name == 'AGE':
            def age_parse(s):
                try: return float(s)
                except (ValueError, TypeError): return np.nan
            a = np.array([age_parse(s) for s in v]); m = state['coding'][name]['median']
            cols.append(np.where(np.isfinite(a), a, m)[:, None])
        else:
            cols.append(np.stack([v == s for s in state['coding'][name]['levels']], 1))
    return scale_apply(np.concatenate(cols, 1), state['scale'])


def fit_rna(raw, train, selected_count=512):
    x = np.asarray(raw['rna'], float); train = np.asarray(train, int)
    mask = np.isfinite(x)
    if np.isinf(x).any(): raise ValueError('RNA不可出现正负无穷')
    with np.errstate(all='ignore'):
        median = np.nanmedian(x[train], axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    measured = mask[train].sum(0)
    filled = np.where(mask, x, median)
    st = scale_fit(filled, train)
    # 冻结筛选：先Hallmark/direct universe，再补足非注释基因，训练折方差排序。
    annotated = np.asarray(raw['gene_universe'], bool)
    var = filled[train].var(0); eligible = (measured >= .9 * len(train)) & (var > 1e-16)
    order = np.argsort(-var, kind='stable')
    inside = order[eligible[order] & annotated[order]]
    outside = order[eligible[order] & ~annotated[order]]
    selected = np.r_[inside[:selected_count], outside[:max(0, selected_count-len(inside))]]
    if len(selected) != selected_count: raise ValueError('训练折不存在足够的可观测有效RNA基因')
    z = scale_apply(filled, st); z = np.where(mask & (measured[None] > 0), z, 0).astype(np.float32)
    partial = z[:, selected].astype(float); after = scale_fit(partial, train)
    return scale_apply(partial, after), {'median': median, 'measured': measured, 'before': st,
                                         'selected': selected, 'after': after, 'input_gene_count': x.shape[1]}


def apply_rna(raw, st):
    x = np.asarray(raw['rna'], float)
    if x.shape[1] != st['input_gene_count']: raise ValueError('RNA维度不匹配：必须保持基因列顺序')
    mask = np.isfinite(x)
    if np.isinf(x).any(): raise ValueError('RNA含无穷')
    filled = np.where(mask, x, st['median'])
    z = np.where(mask & (st['measured'][None] > 0), scale_apply(filled, st['before']), 0).astype(np.float32)
    return scale_apply(z[:, st['selected']].astype(float), st['after'])


def fit_features(raw, train):
    train = np.asarray(train, int)
    if len(train) <= 32: raise ValueError('PCA32需要训练患者数大于32')
    clinical, cs = clinical_fit(raw, train)
    sst = scale_fit(raw['S32'], train)
    s32 = scale_apply(raw['S32'], sst)
    conch = np.asarray(raw['conch_local'], float)
    before = scale_fit(conch, train)
    z = scale_apply(conch, before)
    pca = PCA(n_components=32, svd_solver='full').fit(z[train])
    pc = pca.transform(z); after = scale_fit(pc, train)
    cp = np.c_[clinical, s32, scale_apply(pc, after)]
    rna, rs = fit_rna(raw, train)
    return cp, rna, {'clinical': cs, 's32': sst, 'conch_before': before,
                      'pca_components': pca.components_, 'pca_mean': pca.mean_, 'conch_after': after, 'rna': rs}


def apply_features(raw, st):
    clinical = clinical_apply(raw, st['clinical']); s32 = scale_apply(raw['S32'], st['s32'])
    conch = scale_apply(raw['conch_local'], st['conch_before'])
    pc = (conch - st['pca_mean']) @ st['pca_components'].T
    cp = np.c_[clinical, s32, scale_apply(pc, st['conch_after'])]
    return cp, apply_rna(raw, st['rna'])
