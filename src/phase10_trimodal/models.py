"""Auditable ADAPTED/INSPIRED mechanisms; never claimed paper-exact.

Common patch projection and transformer, equal slide weighting, clinical
linear skip. All ablations operate on identical input tensor layouts.
"""
import torch
from torch import nn


class PairAttention(nn.Module):
    """SurvPGC asymmetric self/cross attention, source token ordering explicit."""
    def __init__(self,d):
        super().__init__();self.norm=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d,bias=False)
        self.ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Dropout(.1),nn.Linear(d,d),nn.LayerNorm(d))
    def forward(self,x,y,ymask,swap=False,shared_qkv=None,shared_norm=None,xmask=None):
        nx=x.shape[1];z=torch.cat([x,y],1);norm=self.norm if shared_norm is None else shared_norm
        valid_x=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device) if xmask is None else xmask
        q,k,v=(self.qkv if shared_qkv is None else shared_qkv)(norm(z)).chunk(3,-1)
        q=q*(q.shape[-1]**-.5);qx,qy=q[:,:nx],q[:,nx:];kx,ky=k[:,:nx],k[:,nx:];vx,vy=v[:,:nx],v[:,nx:]
        if not swap:
            score=qx@k.transpose(-1,-2);mask=torch.cat([valid_x,ymask],1)
            ox=score.masked_fill(~mask[:,None,:],-torch.inf).softmax(-1)@v
            oy=(qy@kx.transpose(-1,-2)).masked_fill(~valid_x[:,None,:],-torch.inf).softmax(-1)@vx
        else:
            # Reverse which modality receives own self attention in addition to cross.
            ox=(qx@ky.transpose(-1,-2)).masked_fill(~ymask[:,None,:],-torch.inf).softmax(-1)@vy
            mask=torch.cat([valid_x,ymask],1)
            oy=(qy@k.transpose(-1,-2)).masked_fill(~mask[:,None,:],-torch.inf).softmax(-1)@v
        ox=self.ff(ox);oy=self.ff(oy)
        mx=ox.mean(1) if xmask is None or bool(valid_x.all()) else (ox*valid_x[:,:,None]).sum(1)/valid_x.sum(1,keepdim=True)
        return mx,(oy*ymask[:,:,None]).sum(1)/ymask.sum(1,keepdim=True)


class IFA(nn.Module):
    """MRePath AlignFusion token order repaired: original returns P then R."""
    def __init__(self,d):
        super().__init__();self.self_r=nn.MultiheadAttention(d,2,batch_first=True)
        self.r_to_p=nn.MultiheadAttention(d,2,batch_first=True);self.p_to_r=nn.MultiheadAttention(d,2,batch_first=True)
        self.final=nn.MultiheadAttention(d,2,batch_first=True)
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(4)]);self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.conf_r=nn.Linear(d,1);self.conf_p=nn.Linear(d,1)
    def forward(self,r,p,mask,gate=False,source_behavior=False):
        cr=self.conf_r(r.mean(1)).sigmoid().clamp(1e-6,1-1e-6)
        cp=self.conf_p((p*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)).sigmoid().clamp(1e-6,1-1e-6)
        # Paper confidence-weight probe: modality labels explicit. No detach so
        # a functioning trainable gate can be independently verified.
        # Paper Eq.8 uses OWN-modality mono confidence in the numerator.
        # Public code instead swaps these numerators, in addition to dropping
        # the weights from the prediction path.
        hp=torch.log(cp if not source_behavior else cr)/torch.log(cp*cr)
        hr=torch.log(cr if not source_behavior else cp)/torch.log(cp*cr)
        w=torch.cat([cp+hp,cr+hr],1).softmax(1)
        if gate and not source_behavior:p=p*w[:,0,None,None];r=r*w[:,1,None,None]
        rr=self.norms[0](r+self.self_r(r,r,r,need_weights=False)[0])
        rr=self.norms[1](rr+self.r_to_p(rr,p,p,key_padding_mask=~mask,need_weights=False)[0]);rr=self.norms[2](rr+self.ff(rr))
        pp=self.norms[3](p+self.p_to_r(p,rr,rr,need_weights=False)[0]);pp=pp+p;rr=rr+r
        rr=rr+self.final(rr,pp,pp,key_padding_mask=~mask,need_weights=False)[0]
        return rr.mean(1),(pp*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True),w


class MechanismModel(nn.Module):
    def __init__(self,arm,clinical_dim,patch_dim=768,d=64):
        super().__init__();self.arm=arm;self.d=d
        self.patch_projection=nn.Sequential(nn.LayerNorm(patch_dim),nn.Linear(patch_dim,d),nn.GELU())
        block=nn.TransformerEncoderLayer(d,2,dim_feedforward=2*d,dropout=.1,batch_first=True,norm_first=True)
        self.patch_backbone=nn.TransformerEncoder(block,1,enable_nested_tensor=False)
        self.clinical_projection=nn.Linear(clinical_dim,6*d)
        self.rna_projection=nn.ModuleList([nn.Sequential(nn.Linear(64,d),nn.ELU(),nn.Dropout(.1)) for _ in range(8)])
        rblock=nn.TransformerEncoderLayer(d,2,dim_feedforward=2*d,dropout=.1,batch_first=True,norm_first=True)
        # C4 retains full capacity with per-token FF rather than inter-token MSA.
        self.rna_attention=nn.TransformerEncoder(rblock,1,enable_nested_tensor=False)
        self.bottleneck=nn.Parameter(torch.randn(1,2,d)*.02)
        self.rna_cls=nn.Parameter(torch.randn(1,1,d)*.02)
        self.pair_c=PairAttention(d);self.pair_r=PairAttention(d)
        self.ifa=IFA(d) if arm.startswith('E') else None
        # Four slots are always produced; same head output dimension.
        self.head=nn.Sequential(nn.Linear(4*d,d),nn.GELU(),nn.Dropout(.1),nn.Linear(d,1))
        self.clinical_skip=nn.Linear(clinical_dim,1,bias=False)
        self.aux=nn.ModuleDict({k:nn.Linear(d,1) for k in ['C','R','P']})

    def forward(self,c,r,patch,mask,return_details=False):
        arm=self.arm;bs,slides,n,dim=patch.shape
        sv=mask.any(-1);valid=sv.reshape(-1).nonzero().squeeze(1)
        p=patch.reshape(bs*slides,n,dim)[valid];pm=mask.reshape(bs*slides,n)[valid]
        p=self.patch_projection(p);p=self.patch_backbone(p,src_key_padding_mask=~pm)
        owner=torch.div(valid,slides,rounding_mode='floor')
        ct=self.clinical_projection(c).reshape(bs,6,self.d)
        cp=c.repeat(1,(512+c.shape[1]-1)//c.shape[1])[:,:512].reshape(bs,8,64)
        # A2/A5 and B1 budget control only see pathology/clinical, never RNA.
        rr=cp if arm in ['A2','A5','B1'] else r
        if arm=='A2':ct=ct*0
        rt=torch.stack([self.rna_projection[k](rr[:,k]) for k in range(8)],1)
        rr0=torch.cat([self.bottleneck.expand(bs,-1,-1),self.rna_cls.expand(bs,-1,-1),rt],1)
        if arm=='C4':
            # Same parameter tensors as GroupMSA, but no content-dependent
            # inter-token attention: q/k/v become a tokenwise linear map.
            layer=self.rna_attention.layers[0];v=layer.norm1(rr0)
            q,k,v=nn.functional.linear(v,layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias).chunk(3,-1)
            rr1=rr0+layer.self_attn.out_proj((q+k+v)/3)
            rr1=rr1+layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(rr1)))))
            rr1=torch.cat([rr1[:,:3]+rr1[:,3:].mean(1,keepdim=True),rr1[:,3:]],1)
        else:rr1=self.rna_attention(rr0)
        bt=rr1[:,:2];rt=rr1[:,3:];rp=rr1[:,2]
        pc=(p*pm[:,:,None]).sum(1)/pm.sum(1,keepdim=True)
        gate=torch.ones((len(valid),2),device=p.device)*.5
        if arm in ['A2','A5','A6']:
            # Matched late fusion: same two PairAttention blocks, each intra-
            # modal. No C/P or R/P token exchange.
            c1,c2=self.pair_c(ct[owner],ct[owner],torch.ones(len(valid),6,dtype=torch.bool,device=p.device))
            p1,p2=self.pair_r(p,p,pm,xmask=pm)
            c2=c2+rp[owner] # Clinical-only surrogate for A2/A5, true RNA for A6.
            slots=[c1,c2,p1,p2]
        elif arm.startswith('C'):
            use=bt[owner]
            if arm=='C2':use=self.bottleneck.expand(len(valid),-1,-1)
            # Same backbone module is applied a second time for EVERY C arm.
            # C1 tokens are RNA independent, RNA still available to late head.
            if arm=='C1':use=self.bottleneck.expand(len(valid),-1,-1)
            mixed=torch.cat([use,p],1);mixedmask=torch.cat([torch.ones(len(valid),2,dtype=torch.bool,device=p.device),pm],1)
            guided=self.patch_backbone(mixed,src_key_padding_mask=~mixedmask)
            gp=(guided[:,2:]*pm[:,:,None]).sum(1)/pm.sum(1,keepdim=True)
            # Pair blocks act independently to match processing budget.
            ca,cb=self.pair_c(ct[owner],ct[owner],torch.ones(len(valid),6,dtype=torch.bool,device=p.device))
            ra,rb=self.pair_r(rt[owner],rt[owner],torch.ones(len(valid),8,dtype=torch.bool,device=p.device))
            if arm=='C2':ra=ra*0;rb=rb*0 # Clears all patient-specific RNA, not just the bottleneck.
            slots=[ca,gp,ra,rb]
        elif arm.startswith('E'):
            ra,pa,gate=self.ifa(rt[owner],p,pm,gate=(arm=='E1'))
            ca,cb=self.pair_c(ct[owner],ct[owner],torch.ones(len(valid),6,dtype=torch.bool,device=p.device));slots=[ca,cb,ra,pa]
        else:
            ca,pa=self.pair_c(ct[owner],p,pm)
            if arm=='B2':ca,pa=self.pair_c(ct[owner],ct[owner],torch.ones(len(valid),6,dtype=torch.bool,device=p.device))
            ra,pr=self.pair_r(rt[owner],p,pm,swap=(arm=='B3'),shared_qkv=self.pair_c.qkv if arm=='B4' else None,shared_norm=self.pair_c.norm if arm=='B4' else None)
            if arm=='B4':
                # Unshared second projection now supplies a branch-local map;
                # all parameter tensors remain connected to risk.
                q,k,v=self.pair_r.qkv(self.pair_r.norm(rt[owner])).chunk(3,-1);ra=ra+(q+k+v).mean(1)/3
            slots=[ca,pa,ra,pr]
        features=torch.cat(slots,1)
        aggregated=torch.zeros(bs,4*self.d,device=p.device).index_add(0,owner,features)/sv.sum(1,keepdim=True)
        preg=torch.zeros(bs,self.d,device=p.device).index_add(0,owner,pc)/sv.sum(1,keepdim=True)
        creg=ct.mean(1);rreg=rp
        risk=self.head(aggregated).squeeze(1)*.1
        if arm!='A2':risk=risk+self.clinical_skip(c).squeeze(1)
        aux={k:self.aux[k](v).squeeze(1) for k,v in {'C':creg,'R':rreg,'P':preg}.items()}
        if return_details:return risk,{'C':creg,'R':rreg,'P':preg},aux,gate
        return risk
