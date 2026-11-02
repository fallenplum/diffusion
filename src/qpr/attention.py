"""Independent FP64 mass-aware retrieval and compression qualification."""
import torch


def unit_ball(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(1.)


def retrieve(q,k,v,mass,*,tau=1.,chunk=1024):
    if q.ndim!=2 or k.ndim!=2 or v.ndim!=2 or len(k)!=len(v) or mass.shape!=(len(k),):raise ValueError('KV_SHAPE')
    if q.shape[1]!=k.shape[1] or q.dtype!=torch.float64 or k.dtype!=torch.float64 or v.dtype!=torch.float64:raise ValueError('FP64_REFERENCE_REQUIRED')
    if not all(torch.isfinite(a).all() for a in (q,k,v,mass)) or (mass<0).any() or mass.sum()<=0:raise ValueError('INVALID_MASS_OR_DATA')
    logs=[];answers=[]
    for start in range(0,len(k),chunk):
        w=mass[start:start+chunk];valid=w>0
        if not valid.any():continue
        logits=tau*q@k[start:start+chunk][valid].T+torch.log(w[valid])[None,:]
        lz=torch.logsumexp(logits,1);ans=torch.softmax(logits,1)@v[start:start+chunk][valid]
        logs.append(lz);answers.append(ans)
    logz=torch.logsumexp(torch.stack(logs,1),1)
    weights=torch.exp(torch.stack(logs,1)-logz[:,None])
    return (torch.stack(answers,1)*weights[...,None]).sum(1),logz


def compress(k,v,mass,assignment):
    if assignment.ndim!=2 or len(assignment)!=len(k) or (assignment<0).any():raise ValueError('ASSIGNMENT_SHAPE_OR_SIGN')
    if not torch.allclose(assignment.sum(1),torch.ones(len(k),dtype=assignment.dtype,device=k.device),atol=1e-12,rtol=0):raise ValueError('ROW_STOCHASTIC_REQUIRED')
    w=mass[:,None]*assignment;pi=w.sum(0)
    # Tiny positive masses remain positive; only exactly zero slots are null.
    safe=pi.clamp_min(1e-12)
    kb=w.T@k/safe[:,None];vb=w.T@v/safe[:,None]
    return pi,kb,vb
