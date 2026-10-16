"""Exact ZOH diagonal S6: u/dt [B,N,D], B/C [B,N,S], A [D,S].

CUDA path is a reverse-exclusive-cumsum closed form for final state.
All exp arguments are nonpositive. Unlike the official Mamba last_state
output, normal torch autograd differentiates every transition here.
"""
import torch


def zoh(dt,A):
    v=dt[...,None]*A
    # exprel(v), with safe evaluation even on the inactive where branch.
    safe=torch.where(v.abs()<1e-4,torch.ones_like(v),v)
    ratio=torch.expm1(v)/safe
    series=1+v/2+v*v/6+v*v*v/24
    return v,dt[...,None]*torch.where(v.abs()<1e-4,series,ratio)


def reference_scan(u,dt,B,A,h0=None,states=False):
    h=torch.zeros(u.shape[0],u.shape[2],A.shape[1],device=u.device,dtype=u.dtype) if h0 is None else h0
    out=[]
    for k in range(u.shape[1]):
        loga,b=zoh(dt[:,k],A)
        h=torch.exp(loga)*h+b*B[:,k,None,:]*u[:,k,:,None]
        if states:out.append(h)
    return torch.stack(out,1) if states else h


def cuda_final_scan(u,dt,B,A,h0=None):
    loga,b=zoh(dt,A)
    # Sum only transitions strictly after each input, avoiding total-prefix
    # cancellation and avoiding log(a) when a numerically underflows.
    # Deterministic parallel doubling scan on dt [B,N,D], before broadcasting
    # over states. torch CUDA cumsum is rejected in deterministic mode.
    suffix=dt
    step=1
    while step<dt.shape[1]:
        suffix=suffix+torch.cat([suffix[:,step:],torch.zeros_like(suffix[:,:step])],1)
        step*=2
    after_dt=torch.cat([suffix[:,1:],torch.zeros_like(suffix[:,:1])],1)
    after=after_dt[...,None]*A
    h=(torch.exp(after)*b*B[:,:,None,:]*u[:,:,:,None]).sum(1)
    if h0 is not None:h=h+torch.exp(loga.sum(1))*h0
    return h
