"""Private bounded repair infrastructure; never changes historical sources."""
from pathlib import Path
import hashlib,json,pickle,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123_worker import check_lock as parent_check,read,write,atomic
from scripts.semantic_b0b7_p2_matrix import raw_inputs,shared_preprocessing
from src.semantic_m123.data import prepare
from src.semantic_m123r.data import repair_inputs
from src.semantic_m123r.io import sha,stratified_cycle
from src.semantic_b0b7.p2_anchor import array_sha
from types import SimpleNamespace

PARENT='41f041bcb883d65285167a201183f40d41e9de1640ead2b40a9311f61f318af3'


def verify_lock(path,expected):
    if sha(path)!=expected:raise ValueError('REPAIR_MASTER_CHANGED')
    lock=read(path)
    if lock['status'] not in ('LOCKED_BEFORE_REPAIR_DSS_OPTIMIZATION','LOCKED_ARTIFACT_REPLAY_REPAIR_BEFORE_RETRY'):raise ValueError('REPAIR_NOT_LOCKED')
    parent=parent_check(lock['parent_master'],PARENT)
    for n,h in lock['sources'].items():
        if sha(ROOT/n)!=h:raise ValueError('REPAIR_SOURCE_CHANGED:'+n)
    return lock,parent


def identifier(job):
    c=job['config'];part='final' if job.get('role')=='final' else 'inner'+str(job['inner'])
    return f"outer{job['outer']}_{part}/{job['arm']}_seed{job['seed']}_a{c['a']}_b{c['b']}_n{c['n']}"


def folder(private,job):return Path(private)/'fits'/identifier(job)


def build(parent,split):
    raw,member,assets=raw_inputs(SimpleNamespace(assets=parent['assets_path'],legacy_root=parent['legacy_root']),endpoints=False)
    part=read(parent['parent_split_plan'])['partitions'][split]
    ap=Path(parent['parent_private'])/'anchors'/(split+'.private.pkl')
    if sha(ap)!=parent['partitions'][split]['anchor_sha256']:raise ValueError('CURRENT_TRAIN_TEACHER_CHANGED')
    with ap.open('rb') as f:anchor=pickle.load(f)
    frozen=torch.load(parent['frozen_images'],map_location='cpu',weights_only=False,mmap=True)
    if not np.array_equal(frozen['global_rows'].numpy(),np.arange(512)):raise ValueError('CANONICAL_FM_ORDER_REQUIRED')
    data=prepare(raw,frozen,anchor,part['fit'],part['evaluation'],member)
    raw['m123r_titan_native']=frozen['image']['titan_native'].numpy()
    raw['m123r_titan_aligned']=frozen['image']['titan_aligned'].numpy()
    batch,scale=repair_inputs(data,raw)
    digest=hashlib.sha256(json.dumps({k:array_sha(v.numpy()) for k,v in batch.items()},sort_keys=True).encode()).hexdigest()
    return batch,data,scale,raw,assets,part,digest


def verify_cell(private,job):
    f=folder(private,job);p=f/'COMPLETE.private.json'
    if not p.exists():return None
    r=read(p)
    if r['job']!=job or r['status']!='COMPLETE':raise ValueError('COMPLETION_CHANGED')
    for name,key in (('PREDICTIONS.private.npz','predictions_sha256'),('HISTORY.private.json.gz','history_sha256')):
        if sha(f/name)!=r[key]:raise ValueError('PRIVATE_FIT_OUTPUT_CHANGED')
    retained=read(f/'WEIGHT_RETENTION.private.json') if (f/'WEIGHT_RETENTION.private.json').exists() else None
    for c in r['checkpoints'].values():
        path=f/c['file']
        if path.exists():
            if sha(path)!=c['sha256']:raise ValueError('CHECKPOINT_BYTES_CHANGED')
        elif retained is None or retained['verified_removed'].get(c['file'])!=c['sha256']:
            raise ValueError('UNREGISTERED_MISSING_CHECKPOINT')
    if r['maximum_saved_replay_error']>1e-9:raise ValueError('SNAPSHOT_REPLAY_ERROR')
    return r
