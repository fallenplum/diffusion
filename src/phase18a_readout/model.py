import torch
import torch.nn.functional as F
from src.phase17r_ssd.model import RepairS6
from src.phase17_ssd.ssm import cuda_final_scan
from .scan import local_readout_scan


class ReadoutS6(RepairS6):
    def __init__(self,clinical_dim,membership,arm,dropout=.25,chunk_size=32):
        base={'F0':'R1_S6_noDirect','F1':'R1_A8_noDirect','L0':'R1_S6_noDirect','L1':'R1_A8_noDirect'}[arm]
        super().__init__(clinical_dim,membership,base,1.0,dropout)
        self.readout_arm=arm;self.chunk_size=chunk_size

    def forward(self,visual,mask,clinical,r,g,l,rm,cm,anchor,context_index=None,context_omics=None,gate_override=None,order=None,diagnostic=False,module_off=False):
        # Same module construction, RNG consumption, numerical preprocessing and
        # Phase17R equations; the only production change is the y readout below.
        own=torch.cat([r,g,l],-1)
        own_mt,own_mm=self.molecular(r,g,l,rm,cm)
        if context_omics is not None:r,g,l,rm,cm=context_omics
        if context_index is not None:r,g,l,rm,cm=[a[context_index] for a in [r,g,l,rm,cm]]
        mt,mm=self.molecular(r,g,l,rm,cm);ct=self.clinical(clinical).reshape(-1,6,64)
        context=torch.cat([ct,mt],1)
        context_mask=torch.cat([torch.ones(len(mt),6,device=mt.device,dtype=torch.bool),mm],1)
        x=self.visual(visual)
        attention=(torch.einsum('bnd,bkd->bnk',self.q(x),self.k(context))/4).masked_fill(~context_mask[:,None,:],-1e4).softmax(-1)
        gx=self.attn_out(attention@self.v(context));z=gx;sx=x
        condition='A8' in self.repair_arm
        dd=self.delta_up(F.silu(self.delta_down(z))) if condition else torch.zeros_like(x)
        bb=self.B_up(F.silu(self.B_down(z))) if condition else torch.zeros(*x.shape[:2],16,device=x.device)
        m=torch.sigmoid(self.write(torch.cat([x,gx],-1))) if condition else torch.ones(*x.shape[:2],1,device=x.device)
        if gate_override is not None:m=torch.full_like(m,gate_override)
        m=m*mask[:,:,None];base_dt=self.dt(sx);base_B=self.B(sx)
        dt=m*F.softplus(base_dt+dd);B=base_B+bb;C=self.C(sx);u=self.u(sx);A=-self.log_A.exp()
        Cpool=(C*m).sum(1)/m.sum(1).clamp_min(1e-8)
        if self.readout_arm.startswith('L'):
            y,h=local_readout_scan(u,dt,B,C,A,m,chunk_size=self.chunk_size,checkpoint_chunks=True,return_final=True)
        else:
            h=cuda_final_scan(u,dt,B,A);y=(h*Cpool[:,None,:]).sum(-1)
        late=torch.cat([ct.mean(1),(own_mt*own_mm[:,:,None]).sum(1)/own_mm.sum(1,keepdim=True).clamp_min(1)],-1)
        ssm_raw=self.readout(y).squeeze(-1);late_risk=self.late(late).squeeze(-1);ssm=self.gamma*ssm_raw
        risk=anchor+late_risk+(0 if module_off else ssm)
        if not diagnostic:return risk
        return risk,{'valid_mask':mask,'m':m,'dt':dt,'B':B,'base_dt':base_dt,'base_B':base_B,'conditional_delta':dd,'conditional_B':bb,'g':gx,'normalized_g':z,'attention':attention,'h':h,'y':y,'ssm_risk':ssm,'ssm_raw':ssm_raw,'direct_risk':torch.zeros_like(anchor),'late_risk':late_risk,'u':u,'A':A,'C':C,'Cpool':Cpool}
