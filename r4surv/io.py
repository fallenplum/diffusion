"""仅支持不含对象的NPZ特征输入；禁止自动加载真实患者数据。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .preprocess import CLINICAL

def read_npz(path, *, require_outcome=False):
    p=Path(path)
    with np.load(p,allow_pickle=False) as f:
        keys=set(f.files)
        required={'patient_id','S32','conch_local','rna','gene_universe'} | {'clinical_'+k for k in CLINICAL}
        if require_outcome:required|={'t','e'}
        missing=required-keys
        if missing:raise ValueError('缺少字段：'+','.join(sorted(missing)))
        raw={key:np.array(f[key]) for key in required}
    raw['clinical_raw']={key:raw.pop('clinical_'+key) for key in CLINICAL}
    n=len(raw['patient_id'])
    if any(len(raw[k])!=n for k in ['S32','conch_local','rna']):raise ValueError('患者行数不一致')
    if raw['S32'].shape!=(n,32) or raw['conch_local'].shape!=(n,768) or raw['rna'].ndim!=2:
        raise ValueError('S32[patient,32]/CONCH[patient,768]/RNA[patient,genes]维度异常')
    if raw['gene_universe'].shape!=(raw['rna'].shape[1],):raise ValueError('gene_universe形状异常')
    if len(np.unique(raw['patient_id']))!=n:raise ValueError('patient_id不唯一')
    if not np.isfinite(raw['S32']).all() or not np.isfinite(raw['conch_local']).all():
        raise ValueError('病理特征存在非有限值')
    if require_outcome:
        if (np.asarray(raw['t'])<=0).any() or not np.isin(raw['e'],[0,1]).all():raise ValueError('DSS标签无效')
    return raw
