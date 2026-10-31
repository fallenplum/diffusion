"""Bounded current-partition label decoding and protected snapshot storage."""
from pathlib import Path
import gzip,hashlib,pickle,zipfile
import numpy as np
import torch
from numpy.lib import format as fmt
from src.semantic_b0b7.p2_data import ENDPOINT_SHA256


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda:f.read(8*1024**2),b''):h.update(piece)
    return h.hexdigest()


def selected_npy_rows(stream,rows,n):
    version=fmt.read_magic(stream)
    if version==(1,0):shape,fortran,dtype=fmt.read_array_header_1_0(stream)
    elif version==(2,0):shape,fortran,dtype=fmt.read_array_header_2_0(stream)
    else:raise ValueError('UNSUPPORTED_NPY_VERSION')
    if fortran or shape!=(n,) or dtype.hasobject:raise ValueError('FIXED_ENDPOINT_VECTOR_REQUIRED')
    rows=np.asarray(rows,int)
    if len(np.unique(rows))!=len(rows) or (rows<0).any() or (rows>=n).any():raise ValueError('UNIQUE_BOUNDED_ROWS')
    start=stream.tell();out=np.empty(len(rows),dtype=dtype)
    for position in np.argsort(rows):
        stream.seek(start+int(rows[position])*dtype.itemsize)
        value=stream.read(dtype.itemsize)
        if len(value)!=dtype.itemsize:raise ValueError('TRUNCATED_ENDPOINT')
        out[position]=np.frombuffer(value,dtype=dtype,count=1)[0]
    return out


def partition_endpoints(path,canonical_ids,rows,allowed):
    if sha(path)!=ENDPOINT_SHA256:raise ValueError('ENDPOINT_CONTAINER_CHANGED')
    if not set(map(int,rows))<=set(map(int,allowed)):raise ValueError('LABEL_ROWS_OUTSIDE_CURRENT_PROTOCOL')
    with zipfile.ZipFile(path) as z:
        with z.open('patient_hash.npy') as f:patient_hash=str(fmt.read_array(f,allow_pickle=False))
        if patient_hash!=hashlib.sha256(np.ascontiguousarray(canonical_ids).tobytes()).hexdigest():raise ValueError('CANONICAL_LABEL_ORDER_CHANGED')
        result={}
        for key in ('DSS_time','DSS_event','DSS_valid'):
            with z.open(key+'.npy') as f:result[key]=selected_npy_rows(f,rows,512)
    t=np.asarray(result['DSS_time'],np.float64);e=np.asarray(result['DSS_event'],np.float64)
    if not result['DSS_valid'].all() or not np.isfinite(t).all() or (t<0).any() or not np.isin(e,[0,1]).all():raise ValueError('CURRENT_DSS_INVALID')
    return t,e


def stratified_cycle(rna_mask,cna_mask):
    groups={};count=len(rna_mask);order=np.arange(count);eligible=np.zeros(count,bool)
    for i in range(count):
        key=hashlib.sha256(np.packbits(np.r_[rna_mask[i],cna_mask[i]]).tobytes()).hexdigest()
        groups.setdefault(key,[]).append(i)
    for group in groups.values():
        if len(group)>1:order[group]=np.roll(group,1);eligible[group]=True
    return order,eligible


def save_checkpoint(path,model,epoch,job,dims,base_reference,input_sha):
    compact=epoch<=90 and model.arm not in ('R2','R3') or epoch==0
    state={n:v.detach().cpu() for n,v in model.state_dict().items() if not compact or not n.startswith('base.')}
    cp={'schema':'m123r_checkpoint_v1','state':state,'compact_frozen_base':compact,'base_reference':base_reference,
        'epoch':epoch,'job':job,'dims':dims,'input_sha256':input_sha}
    with gzip.open(path,'xb') as f:torch.save(cp,f)
    return {'file':Path(path).name,'sha256':sha(path),'bytes':Path(path).stat().st_size}


def load_checkpoint(path,private,model):
    with gzip.open(path,'rb') as f:cp=torch.load(f,map_location='cpu',weights_only=False)
    state=dict(cp['state'])
    if cp['compact_frozen_base']:
        p=Path(private)/'preprocessing'/cp['base_reference']['file']
        if sha(p)!=cp['base_reference']['sha256']:raise ValueError('FROZEN_INITIAL_BASE_CHANGED')
        with p.open('rb') as f:base=pickle.load(f)
        state.update({n:torch.as_tensor(v) for n,v in base.items()})
    model.load_state_dict(state,strict=True)
    # requires_grad flags are not stored in state_dict. Restore the original
    # stage before inference: CUDA may choose different FP32 kernels when
    # linear parameters have different trainability, even inside no_grad.
    model.stage(cp['epoch']);model.eval()
    return cp
