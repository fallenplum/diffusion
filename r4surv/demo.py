"""可重复生成完全合成的示例患者表征，绝非TCGA或临床真实数据。"""
from __future__ import annotations
from pathlib import Path
import numpy as np

def create(path, n=144, genes=544, seed=20261107):
    if n<120 or genes<512:raise ValueError('演示样本或基因列不足')
    rng=np.random.default_rng(seed)
    latent=rng.standard_normal((n,5))
    s32=latent@rng.normal(size=(5,32))*.15+rng.normal(size=(n,32))*.5
    conch=latent@rng.normal(size=(5,768))*.1+rng.normal(size=(n,768))*.6
    rna=latent@rng.normal(size=(5,genes))*.12+rng.normal(size=(n,genes))*.7
    rna[rng.random((n,genes))<.01]=np.nan
    age=np.clip(rng.normal(65,12,n),30,90)
    stage=rng.choice(['I','II','III','IV'],size=n,p=[.15,.4,.3,.15])
    stage_num=np.array([{'I':0,'II':1,'III':2,'IV':3}[s] for s in stage])
    log_risk=.36*latent[:,0]-.2*latent[:,2]+.35*stage_num
    event_time=rng.exponential(scale=65,size=n)*np.exp(-log_risk)
    censor_time=rng.uniform(24,135,size=n)
    t=np.minimum(event_time,censor_time)+.05;e=(event_time<=censor_time).astype(np.int8)
    payload={'patient_id':np.asarray([f'SYNTH{i:04d}' for i in range(n)]),'S32':s32.astype('float32'),
             'conch_local':conch.astype('float32'),'rna':rna.astype('float32'),
             'gene_universe':np.array([True]*512+[False]*(genes-512)),
             'clinical_AGE':age.astype('float32').astype(str),
             'clinical_SEX':rng.choice(['Female','Male'],n),
             'clinical_AJCC_PATHOLOGIC_TUMOR_STAGE':stage,
             'clinical_PATH_T_STAGE':rng.choice(['T1','T2','T3','T4'],n),
             'clinical_PATH_N_STAGE':rng.choice(['N0','N1','N2'],n),
             'clinical_PATH_M_STAGE':rng.choice(['M0','M1'],n),
             't':t,'e':e}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**payload)
    return {'patients':n,'genes':genes,'events':int(e.sum()),'synthetic_only':True,'seed':seed}
