"""Full relation and explicitly separated content/topology/query controls."""
import torch
from torch import nn
from src.qpr.query import MolecularQuery

ARMS=('B1','B2','B3','B4','B5','B3_QUERY_OFF')


class FullRelation(nn.Module):
    def __init__(self,c_dim,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('UNREGISTERED_G2_ARM')
        self.arm=arm;self.query=MolecularQuery(c_dim).double()
        self.unconditional=nn.Linear(64,1,bias=False).double()
        self.factors=nn.Parameter(torch.randn(4,64,dtype=torch.float64)*.02)
        self.beta=nn.Parameter(torch.zeros(50,4,dtype=torch.float64))
        nn.init.zeros_(self.unconditional.weight)
        if arm=='B1':
            for p in self.query.parameters():p.requires_grad_(False)
        if arm=='B3_QUERY_OFF':self.query.g.weight.requires_grad_(False)

    def stage(self,epoch):
        for name,p in self.named_parameters():
            active=not name.startswith('query.') or epoch>20
            if self.arm=='B1' and name.startswith('query.'):active=False
            if self.arm=='B3_QUERY_OFF' and name.startswith('query.g.'):active=False
            p.requires_grad_(active)

    def forward(self,program,clinical,s32,k,v,mask,program_mask,*,constant_query=None):
        if self.arm=='B1':q=constant_query[None].expand(len(program),-1,-1)
        else:q=self.query(program,clinical,s32,off=self.arm=='B3_QUERY_OFF')
        if self.arm in ('B1','B3_QUERY_OFF'):program_mask=torch.ones_like(program_mask)
        # Every row is a full sampled-edge distribution; padding has zero mass.
        logits=q@k.double().transpose(1,2)
        logits=logits.masked_fill(~mask[:,None,:],-torch.inf)
        f=torch.softmax(logits,dim=-1)@v.double()
        mean=(v.double()*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        b=self.factors/self.factors.norm(dim=-1,keepdim=True).clamp_min(1.)
        conditional=(torch.einsum('bkd,ld->bkl',f,b)*self.beta[None]*program_mask[:,:,None]).sum((1,2))
        return self.unconditional(mean).squeeze(-1)+conditional

    def effective_parameters(self):
        return sum(p.numel() for n,p in self.named_parameters() if not (self.arm=='B1' and n.startswith('query.')) and not(self.arm=='B3_QUERY_OFF' and n.startswith('query.g.')))
