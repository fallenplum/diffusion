import torch
import torch.nn.functional as F
from torch import nn
from src.phase17_ssd.model import ConditionalS6
from src.phase17_ssd.ssm import cuda_final_scan

class RepairS6(ConditionalS6):
    def __init__(self,clinical_dim,membership,arm,scale=1.,dropout=.25):
        kind='A8' if 'A8' in arm else 'A3' if 'A3' in arm else 'A2'
        super().__init__(clinical_dim,membership,kind,dropout)
        self.repair_arm=arm;self.scale=scale;self.normalized=arm.startswith('R2')
        # Instantiate common parent first, preserving paired RNG initialization.
        if arm!='R1_A8_full':self.direct=None
        if self.normalized:
            self.input_down=nn.Linear(64,13,bias=False)
            self.input_up=nn.Linear(13,64,bias=False)
            nn.init.zeros_(self.input_up.weight)
            nn.init.zeros_(self.film_up.weight);nn.init.zeros_(self.film_up.bias)

    def forward(self,visual,mask,clinical,r,g,l,rm,cm,anchor,context_index=None,context_omics=None,gate_override=None,order=None,diagnostic=False,module_off=False):
        own=torch.cat([r,g,l],-1)
        own_mt,own_mm=self.molecular(r,g,l,rm,cm)
        if context_omics is not None:r,g,l,rm,cm=context_omics
        if context_index is not None:r,g,l,rm,cm=[a[context_index] for a in [r,g,l,rm,cm]]
        mt,mm=self.molecular(r,g,l,rm,cm);ct=self.clinical(clinical).reshape(-1,6,64)
        context=torch.cat([ct,mt],1)
        context_mask=torch.cat([torch.ones(len(mt),6,device=mt.device,dtype=torch.bool),mm],1)
        x=self.visual(visual)
        attention=(torch.einsum('bnd,bkd->bnk',self.q(x),self.k(context))/4).masked_fill(~context_mask[:,None,:],-1e4).softmax(-1)
        gx=self.attn_out(attention@self.v(context))
        z=F.layer_norm(gx,(64,)) if self.normalized else gx
        sx=x;arm=self.repair_arm
        if arm=='R1_A3_noDirect':sx=x+self.attn_input(gx)
        if arm=='R2_Input':sx=x+self.scale*torch.tanh(self.input_up(F.silu(self.input_down(z))))
        if arm=='R2_FiLM':
            a,b=self.film_up(F.silu(self.film_down(z))).chunk(2,-1)
            sx=x*(1+self.scale*torch.tanh(a))+self.scale*torch.tanh(b)
        condition='A8' in arm
        dd=self.delta_up(F.silu(self.delta_down(z))) if condition else torch.zeros_like(x)
        bb=self.B_up(F.silu(self.B_down(z))) if condition else torch.zeros(*x.shape[:2],16,device=x.device)
        if self.normalized:dd=self.scale*torch.tanh(dd);bb=self.scale*torch.tanh(bb)
        gated=condition or self.normalized
        m=torch.sigmoid(self.write(torch.cat([x,gx],-1))) if gated else torch.ones(*x.shape[:2],1,device=x.device)
        if gate_override is not None:m=torch.full_like(m,gate_override)
        m=m*mask[:,:,None];base_dt=self.dt(sx);base_B=self.B(sx)
        dt=m*F.softplus(base_dt+dd);B=base_B+bb;C=self.C(sx);u=self.u(sx);A=-self.log_A.exp()
        h=cuda_final_scan(u,dt,B,A)
        Cpool=(C*m).sum(1)/m.sum(1).clamp_min(1e-8);y=(h*Cpool[:,None,:]).sum(-1)
        late=torch.cat([ct.mean(1),(own_mt*own_mm[:,:,None]).sum(1)/own_mm.sum(1,keepdim=True).clamp_min(1)],-1)
        ssm_raw=self.readout(y).squeeze(-1);direct=self.direct(own).squeeze(-1) if self.direct is not None else torch.zeros_like(anchor)
        late_risk=self.late(late).squeeze(-1);ssm=self.gamma*ssm_raw
        risk=anchor+direct+late_risk+(0 if module_off else ssm)
        if not diagnostic:return risk
        return risk,{'valid_mask':mask,'m':m,'dt':dt,'B':B,'base_dt':base_dt,'base_B':base_B,'conditional_delta':dd,'conditional_B':bb,'g':gx,'normalized_g':z,'attention':attention,'h':h,'ssm_risk':ssm,'ssm_raw':ssm_raw,'direct_risk':direct,'late_risk':late_risk,'u':u,'A':A,'Cpool':Cpool}
