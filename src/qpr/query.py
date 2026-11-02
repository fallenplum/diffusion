import torch
from torch import nn


class MolecularQuery(nn.Module):
    def __init__(self,c_dim):
        super().__init__();self.g=nn.Linear(3,32,bias=False);self.c=nn.Linear(c_dim,32,bias=False);self.s=nn.Linear(32,32,bias=False)
        self.identity=nn.Parameter(torch.randn(50,32)*.02)
        for m in (self.g,self.c,self.s):nn.init.normal_(m.weight,std=.02)

    def forward(self,program,clinical,s32,*,off=False):
        q=self.c(clinical)[:,None]+self.s(s32)[:,None]+self.identity[None]
        if not off:q=q+self.g(program)
        return q/q.norm(dim=-1,keepdim=True).clamp_min(1.)
