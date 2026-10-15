import hashlib,json
from pathlib import Path
import numpy as np,torch
from sklearn.decomposition import PCA
from src.phase16_semantic.core import base,parent,ROOT,sha,write
from src.phase16_semantic.prompts import molecular
from src.phase10_trimodal.features import scale,ahash

P=Path('/root/phase16_prompt_rna_diagnostic_private')
R=ROOT/'reports/phase16_prompt_rna_diagnostic'
CONFIG=ROOT/'configs/phase16_prompt_rna_diagnostic.json'

def cfg():return json.loads(CONFIG.read_text())
def source():
    files=[CONFIG,Path(__file__),ROOT/'scripts/run_phase16_rna_diagnostic.py']
    return hashlib.sha256((base.source()+''.join(sha(p) for p in files)).encode()).hexdigest()

def lock():
    P.mkdir(exist_ok=True);R.mkdir(exist_ok=True)
    z={'status':'LOCKED','parent_commit':cfg()['parent_commit'],'protocol_sha256':sha(CONFIG),'training_source':source(),'data_sha256':sha(parent.DATA),'patient_hash':ahash(parent.load()['patient_ids']),'fold_hash':ahash(parent.load()['folds'])}
    if (P/'LOCK.json').exists() and json.loads((P/'LOCK.json').read_text())!=z:raise ValueError('locked diagnostic input/source changed')
    write(P/'LOCK.json',z);write(R/'PROTOCOL_LOCK.json',z)
    return z

def features(d,fold,role):
    meta,z=parent.parent_features(fold,role);fit=parent.splits(d,'DSS',fold)['inner_train' if role=='inner' else 'outer_train']
    if not np.array_equal(fit,meta['fit_indices']):raise ValueError('historical preprocessing partition mismatch')
    top=z['f__top__R'].astype(float);hall=z['f__enriched__R'].astype(float);mask=z['gm_enriched_true'].astype(float)
    states,cont,state=molecular(d,fit);ordinal=np.zeros(cont.shape)
    ordinal[states=='low']=-1;ordinal[states=='high']=1
    onehot=np.stack([states==v for v in ['low','typical','high','unmeasured']],2).reshape(512,-1).astype(float)
    missing=np.stack([1-g['patient_coverage'] for g in state['groups']],1)
    # Real membership only: never turn a nonmember into a biological slot.
    symbols=d['rna_gene_symbols'].astype(str)[np.asarray(meta['phase12']['enriched_indices'])]
    groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text())
    true_mask=np.array([[s in values for s in symbols] for values in groups.values()],float)
    if np.any(true_mask.sum(1)==0):raise ValueError('empty same512 pathway')
    pca=PCA(8,svd_solver='full').fit(hall[fit])
    arrays={'TOP512':top,'HALL512':hall,'HALL_PCA8':pca.transform(hall),'HALL_POOL8':hall@true_mask.T/true_mask.sum(1),
            'PATH8_CONT':cont,'PATH8_QUANT_ORD':ordinal,'PATH8_QUANT_ONEHOT':onehot,
            'PATH8_CONT_COV':np.c_[cont,missing],'PATH8_QUANT_COV':np.c_[ordinal,missing]}
    standardized={};normalization={}
    for k,x in arrays.items():standardized[k],normalization[k]=scale(x,fit)
    return standardized,{'fit':fit,'molecular':state,'RNA_states':states,'continuous':cont,'normalization':normalization,
                        'same512_membership':true_mask,'same512_uncovered':int((true_mask.sum(0)==0).sum()),
                        'historical_npz_sha256':sha(parent.PARENT/'preprocessing'/f'fold{fold}_{role}.npz'),
                        'pca_components':pca.components_,'pca_mean':pca.mean_,'state_counts':{k:int((states==k).sum()) for k in ['low','typical','high','unmeasured']}}
