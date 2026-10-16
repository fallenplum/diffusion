import math
import torch
from torch import nn
import torch.nn.functional as F
from .omics import MolecularTokens
from .ssm import cuda_final_scan,reference_scan


class ConditionalS6(nn.Module):
    def __init__(self,clinical_dim,membership,arm='A8',dropout=.1):
        super().__init__();self.arm=arm;d=64;s=16;q=16;r=8
        self.visual=nn.Sequential(nn.Linear(1536,d),nn.LayerNorm(d),nn.Dropout(dropout))
        self.clinical_dim=clinical_dim
        self.clinical=nn.Linear(clinical_dim,d*6)
        self.molecular=MolecularTokens(membership,d)
        self.q=nn.Linear(d,q,bias=False);self.k=nn.Linear(d,q,bias=False);self.v=nn.Linear(d,d,bias=False)
        self.attn_out=nn.Linear(d,d,bias=False)
        self.attn_input=nn.Linear(d,d,bias=False)
        self.film=nn.Linear(d,2*d)
        self.u=nn.Linear(d,d);self.dt=nn.Linear(d,d)
        self.B=nn.Linear(d,s);self.C=nn.Linear(d,s)
        self.log_A=nn.Parameter(torch.arange(1,s+1).float().log().repeat(d,1))
        self.delta_down=nn.Linear(d,r,bias=False);self.delta_up=nn.Linear(r,d,bias=False)
        self.B_down=nn.Linear(d,r,bias=False);self.B_up=nn.Linear(r,s,bias=False)
        self.write=nn.Linear(2*d,1)
        self.readout=nn.Linear(d,1,bias=False)
        self.late=nn.Linear(2*d,1,bias=False)
        self.direct=nn.Linear(512*3,1,bias=False)
        self.gamma=nn.Parameter(torch.tensor(.1))
        nn.init.zeros_(self.delta_up.weight);nn.init.zeros_(self.B_up.weight)
        nn.init.zeros_(self.write.weight);nn.init.constant_(self.write.bias,math.log(9))
        nn.init.zeros_(self.direct.weight);nn.init.normal_(self.readout.weight,std=.02)
        nn.init.normal_(self.late.weight,std=.002)
        nn.init.constant_(self.dt.bias,-3.)
        # Preserve prior RNG consumption for every common module, then replace
        # the FiLM comparator with rank8:512+1024+128 =1664 active parameters,
        # exactly the two low-rank delta/B adapters combined.
        del self.film
        self.film_down=nn.Linear(d,r,bias=False);self.film_up=nn.Linear(r,2*d)

    def forward(self,visual,mask,clinical,r,g,l,rm,cm,anchor,context_index=None,context_omics=None,gate_override=None,order=None,diagnostic=False,module_off=False):
        # One batch row is one WSI. Patient aggregation happens outside.
        own=torch.cat([r,g,l],-1)
        own_mt,own_mm=self.molecular(r,g,l,rm,cm)
        if context_omics is not None:r,g,l,rm,cm=context_omics
        if context_index is not None:r,g,l,rm,cm=[a[context_index] for a in [r,g,l,rm,cm]]
        if self.arm=='A9':
            # Same parameters, but no patient-specific molecular evidence.
            r=torch.ones_like(r);g=torch.ones_like(g);l=torch.ones_like(l)
            rm=torch.ones_like(rm);cm=torch.ones_like(cm)
        if self.arm=='A11_NO_CLIN':clinical=torch.zeros_like(clinical)
        mt,mm=self.molecular(r,g,l,rm,cm)
        ct=self.clinical(clinical).reshape(-1,6,64)
        context=torch.cat([ct,mt],1);context_mask=torch.cat([torch.ones(len(mt),6,device=mt.device,dtype=torch.bool),mm],1)
        x=self.visual(visual)
        logits=torch.einsum('bnd,bkd->bnk',self.q(x),self.k(context))/4
        attention=logits.masked_fill(~context_mask[:,None,:],-1e4).softmax(-1)
        gx=self.attn_out(attention@self.v(context))
        sx=x
        if self.arm=='A3':sx=x+self.attn_input(gx)
        if self.arm=='A4':
            scale,bias=self.film_up(F.silu(self.film_down(gx))).chunk(2,-1);sx=x*(1+.1*torch.tanh(scale))+.1*bias
        dd=self.delta_up(F.silu(self.delta_down(gx))) if self.arm in ['A5','A7','A8','A9','A10','A11_NO_S32','A11_NO_CLIN'] else torch.zeros_like(x)
        bb=self.B_up(F.silu(self.B_down(gx))) if self.arm in ['A6','A7','A8','A9','A10','A11_NO_S32','A11_NO_CLIN'] else torch.zeros(*x.shape[:2],16,device=x.device)
        m=torch.sigmoid(self.write(torch.cat([x,gx],-1))) if self.arm in ['A8','A9','A10','A11_NO_S32','A11_NO_CLIN'] else torch.ones(*x.shape[:2],1,device=x.device)
        if gate_override is not None:m=torch.full_like(m,gate_override)
        m=m*mask[:,:,None]
        dt=m*F.softplus(self.dt(sx)+dd)
        B=self.B(sx)+bb;C=self.C(sx);u=self.u(sx);A=-self.log_A.exp()
        h=cuda_final_scan(u,dt,B,A)
        Cpool=(C*m).sum(1)/m.sum(1).clamp_min(1e-8)
        y=(h*Cpool[:,None,:]).sum(-1)
        late=torch.cat([ct.mean(1),(own_mt*own_mm[:,:,None]).sum(1)/own_mm.sum(1,keepdim=True).clamp_min(1)],-1)
        # The common late main effects do not carry an ungated WSI skip.
        ssm=self.readout(y).squeeze(-1)
        risk=anchor+self.direct(own).squeeze(-1)+self.late(late).squeeze(-1)+(0 if module_off else self.gamma*ssm)
        if not diagnostic:return risk
        return risk,{'valid_mask':mask,'m':m,'dt':dt,'B':B,'conditional_delta':dd,'conditional_B':bb,'g':gx,'attention':attention,'h':h,'ssm_risk':self.gamma*ssm,'direct_risk':self.direct(own).squeeze(-1),'u':u,'A':A,'Cpool':Cpool}
