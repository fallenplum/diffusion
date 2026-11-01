"""Preserve complete R2 additive inputs/function, then add matched controls."""
from __future__ import annotations
import torch
from torch import nn
from src.semantic_m123r.models import RepairModel
from src.semantic_m123.models import slide_mean

ARMS=('S0','S1','S2','S3','S1S','S3_WRONG_TRAIN')


def encode(branch,x):
    h=branch.linear(x.double())
    return h+branch.residual(h)


class RSCIL(nn.Module):
    def __init__(self,arm,p_dim,g_dim,c_dim,rank=8):
        super().__init__()
        if arm not in ARMS or rank not in (4,8):raise ValueError('UNLOCKED_MODEL')
        self.arm=arm;self.rank=rank
        self.additive=RepairModel('R2',p_dim,g_dim,c_dim).double()
        if arm=='S1S':
            def one():return nn.Sequential(nn.Linear(64,rank),nn.LayerNorm(rank),nn.GELU(),nn.Linear(rank,1,bias=False))
            self.extra_p=one();self.extra_g=one()
            nn.init.zeros_(self.extra_p[-1].weight);nn.init.zeros_(self.extra_g[-1].weight)
        elif arm!='S0':
            self.project_p=nn.Sequential(nn.Linear(64,rank),nn.LayerNorm(rank))
            self.project_g=nn.Sequential(nn.Linear(64,rank),nn.LayerNorm(rank))
            self.interaction_head=nn.Linear(rank,1,bias=False);nn.init.zeros_(self.interaction_head.weight)
        self.double()
        self.register_buffer('A',torch.zeros(3,rank,dtype=torch.float64))
        self.register_buffer('A_ready',torch.tensor(False))
        self.initial_base=None

    def warm(self,warm):
        self.additive.warm(warm)
        self.initial_base={n:p.detach().clone() for n,p in self.named_parameters() if n.startswith('additive.base.')}

    def main(self,b):
        hidden={k:encode(self.additive.base[k],b[key]) for k,key in zip(('p','g','c'),('P','G','C'))}
        mean=b.get('mean_patch')
        if mean is None:mean=slide_mean(b['patch'].double(),b['mask'],b['slide'])
        local=self.additive.local_out(self.additive.mean(mean.double()))
        hidden['p']=hidden['p']+local
        bs=torch.stack([self.additive.base[k].head(hidden[k]).squeeze(-1) for k in ('p','g','c')],1)
        return hidden,bs

    def forward(self,b,*,a=None,interaction_g_hidden=None):
        hidden,bs=self.main(b);main=bs.sum(1);z=main[:,None]*0.;delta=main*0.
        if self.arm=='S1S':delta=self.extra_p(hidden['p']).squeeze(-1)+self.extra_g(hidden['g']).squeeze(-1)
        elif self.arm!='S0':
            hg=hidden['g'] if interaction_g_hidden is None else interaction_g_hidden
            z=self.project_p(hidden['p'])*self.project_g(hg)
            if self.arm in ('S2','S3','S3_WRONG_TRAIN'):
                if a is None:
                    if not bool(self.A_ready):raise ValueError('TRAIN_FITTED_A_REQUIRED')
                    a=self.A
                zp=z-bs.detach()@a.detach()
            else:zp=z
            delta=self.interaction_head(zp).squeeze(-1)
        return {'risk':main+delta,'main':main,'B':bs,'Z':z,'interaction':delta,'hidden_g':hidden['g']}

    def raw(self,b,*,wrong=None):
        hidden,bs=self.main(b);hg=hidden['g'] if wrong is None else hidden['g'][wrong]
        return bs,self.project_p(hidden['p'])*self.project_g(hg),hg

    def stage(self,epoch):
        for n,p in self.named_parameters():
            if self.arm=='S0':active=True
            elif n.startswith('additive.base.'):active=epoch>60
            elif epoch<=10:active=self.group(n)=='output'
            else:active=True
            p.requires_grad_(active)

    @staticmethod
    def group(name):
        if name.startswith('additive.base.'):return 'base'
        if name.startswith(('interaction_head.','extra_p.3.','extra_g.3.','additive.local_out.')):return 'output'
        return 'generator'

    def penalties(self):
        if self.initial_base is None:raise ValueError('WARM_REQUIRED')
        pb=sum((p-self.initial_base[n]).square().sum() for n,p in self.named_parameters() if n in self.initial_base)
        pn=sum(p.square().sum() for n,p in self.named_parameters() if n not in self.initial_base)
        return pb,pn

    def added_parameters(self):return sum(p.numel() for n,p in self.named_parameters() if not n.startswith('additive.'))
