"""Two sparse endpoint-aggregation layers learn assignments, not teacher KV."""
import torch
from torch import nn
from torch.nn import functional as F


class SparseCompressor(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(96,64)
        self.own=nn.ModuleList([nn.Linear(64,64) for _ in range(2)])
        self.neighbor=nn.ModuleList([nn.Linear(64,64,bias=False) for _ in range(2)])
        self.output=nn.Linear(64,32)

    def forward(self,k,v,endpoints,nodes):
        h=F.gelu(self.input(torch.cat([k,v],-1)))
        for own,neighbor in zip(self.own,self.neighbor):
            summ=h.new_zeros(nodes,64);count=h.new_zeros(nodes,1)
            for axis in (0,1):
                summ.index_add_(0,endpoints[:,axis],h)
                count.index_add_(0,endpoints[:,axis],torch.ones_like(h[:,:1]))
            mean=summ/count.clamp_min(1.)
            context=(mean[endpoints[:,0]]+mean[endpoints[:,1]])/2
            h=F.gelu(own(h)+neighbor(context))
        return torch.softmax(self.output(h),-1)


def summary(k,v,s):
    pi=s.mean(0);den=s.sum(0).clamp_min(1e-12)
    return pi,s.T@k/den[:,None],s.T@v/den[:,None]


def answer(q,k,v,mass=None):
    logits=q@k.T
    if mass is not None:
        # Never evaluate log(0), even on torch.where's inactive branch: its
        # backward otherwise forms 0/0 and contaminates empty-slot gradients.
        positive=mass>0
        safe=torch.where(positive,mass,torch.ones_like(mass))
        logits=logits+torch.log(safe).masked_fill(~positive,-torch.inf)[None]
    else:logits=logits-torch.log(logits.new_tensor(len(k)))
    return torch.softmax(logits,1)@v,torch.logsumexp(logits,1)


def losses(k,v,s,q,teacher,qmask=None):
    pi,kb,vb=summary(k,v,s);pred,lz=answer(q,kb,vb,pi)
    target,tlz=teacher
    # Equal weight for the true-query and unseen-ball-probe groups.
    if qmask is None:qmask=torch.ones(len(q),dtype=torch.bool,device=q.device)
    def masked_mean(x,sl):
        w=qmask[sl].to(x.dtype);return (x[sl]*w).sum()/w.sum().clamp_min(1)
    mse=(pred-target).square().mean(-1);ze=(lz-tlz).square()
    a=.5*(masked_mean(mse,slice(0,50))+masked_mean(mse,slice(50,None)))
    z=.5*(masked_mean(ze,slice(0,50))+masked_mean(ze,slice(50,None)))
    dk=(s*((k.square().sum(1)[:,None]+kb.square().sum(1)[None]-2*k@kb.T).clamp_min(0))).sum()/len(k)
    dv=(s*((v.square().sum(1)[:,None]+vb.square().sum(1)[None]-2*v@vb.T).clamp_min(0))).sum()/len(v)
    return a+.1*z,dk+dv
