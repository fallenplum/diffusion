import torch
from torch import nn


class MolecularTokens(nn.Module):
    """8 verified memberships x 2 subtokens; RNA/CNA never equated.

    Separate trainable masked gene pools for RNA, CNA gain and CNA loss;
    coverage is measured, not inferred from an imputed zero.
    """
    def __init__(self,membership,d=64,subtokens=2):
        super().__init__();m=torch.as_tensor(membership,dtype=torch.bool)
        self.register_buffer('membership',m.repeat_interleave(subtokens,0))
        n=m.shape[1];k=m.shape[0]*subtokens
        self.logits=nn.Parameter(torch.randn(3,k,n)*0.02)
        self.slot=nn.Parameter(torch.randn(k,d)*0.02)
        self.evidence=nn.Sequential(nn.Linear(6,8),nn.SiLU(),nn.Linear(8,d))

    def forward(self,r,g,l,rm,cm):
        values=[];coverage=[];valid=[]
        for k,(x,mask) in enumerate([(r,rm),(g,cm),(l,cm)]):
            allow=self.membership[None,:,:]&mask[:,None,:].bool()
            logits=self.logits[k][None].expand(len(x),-1,-1)
            weights=logits.masked_fill(~allow,-1e4).softmax(-1)*allow
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            values.append((weights*x[:,None,:]).sum(-1))
            coverage.append(allow.float().sum(-1)/self.membership.sum(-1).clamp_min(1))
            valid.append(allow.any(-1))
        v=torch.stack(values+coverage,-1)
        return self.evidence(v)+self.slot[None],torch.stack(valid).any(0)
