"""Private model artifacts with fit provenance; no patient arrays in reports."""
import json
from pathlib import Path
import numpy as np
from .data import PRIVATE,CONFIG,source_hash,digest,write_json,ahash


def stem(endpoint,model,seed,fold):return f'{endpoint}_{model}_seed{seed}_fold{fold}'


def save(endpoint,model,seed,fold,indices,time,event,risk,manifest,w=None,design=None,extra=None,reuse_checkpoint=None):
    name=stem(endpoint,model,seed,fold);root=PRIVATE/'formal'
    for folder in ['cells','oof','checkpoints','manifests']:(root/folder).mkdir(parents=True,exist_ok=True)
    cp=root/'cells'/f'{name}.json'
    if cp.exists():raise RuntimeError('refuse overwrite completed '+name)
    prediction=root/'oof'/f'{name}.npz';np.savez_compressed(prediction,test_indices=indices,risk=risk,time=time,event=event,endpoint=endpoint,model=model,seed=seed,fold=fold)
    metadata=root/'manifests'/f'{name}.json';write_json(metadata,manifest)
    ck=Path(reuse_checkpoint) if reuse_checkpoint else root/'checkpoints'/f'{name}.npz'
    if not reuse_checkpoint:
        if w is None:raise ValueError('new coefficient checkpoint absent')
        np.savez_compressed(ck,w=w)
    cell={'status':'COMPLETE','endpoint':endpoint,'model':model,'seed':seed,'fold':fold,'source_hash':source_hash(),'protocol_sha256':digest(CONFIG),
          'prediction_path':str(prediction),'prediction_sha256':digest(prediction),'manifest_path':str(metadata),'manifest_sha256':digest(metadata),
          'checkpoint_path':str(ck),'checkpoint_sha256':digest(ck),'design':design,'outer_test_selection':False,**(extra or {})}
    write_json(cp,cell)
    return cell


def absent(endpoint,model,seed,fold):
    path=PRIVATE/'formal/cells'/f'{stem(endpoint,model,seed,fold)}.json'
    if not path.exists():return True
    c=json.loads(path.read_text())
    if c['status']!='COMPLETE' or c['source_hash']!=source_hash() or digest(c['prediction_path'])!=c['prediction_sha256']:raise ValueError('stale or corrupted completed cell')
    return False


def provenance(partitions,inner_audit,outer_audit,d,producer=None):
    return {**{k:v.tolist() for k,v in partitions.items()},'patient_hash':ahash(d['patient_ids']),'fold_hash':ahash(d['folds']),
            'preprocessing_inner':inner_audit,'preprocessing_outer':outer_audit,'producer':producer}
