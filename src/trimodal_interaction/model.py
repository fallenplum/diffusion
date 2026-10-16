import math
import torch
from torch import nn


ARMS={'A0':'additive','A1':'all_pairwise','A2':'conditional','A3':'official_diff_adapted','A4':'mixed_ungated','A5':'mixed_gated','A6':'wrong_context','A7':'matched_norm_centered','F0':'pre_attention_FiLM','PR':'RNA_pathology_late_clinical'}


def mixed_difference(a11,a10,a01,a00):
    # Paired subtraction is essential to exact null cancellation in floating point.
    return (a11-a10)-(a01-a00)


def masked_softmax(scores,mask):
    if not bool(mask.any(-1).all()):raise ValueError('patient with no valid patch')
    return scores.masked_fill(~mask[:,None,None,:],-torch.inf).softmax(-1)


class Interaction(nn.Module):
    def __init__(self,nc,c0,r0,rm0,arm,hidden=64,heads=4,bound=.25):
        super().__init__();self.arm=arm;self.h=heads;self.d=hidden//heads;self.bound=bound
        self.register_buffer('c0',torch.as_tensor(c0).float());self.register_buffer('r0',torch.as_tensor(r0).float());self.register_buffer('rm0',torch.as_tensor(rm0).float())
        self.gene=nn.Parameter(torch.randn(512,hidden)*.02)
        self.rna=nn.Linear(2,hidden,bias=False)
        self.c=nn.Sequential(nn.Linear(nc,8,bias=False),nn.Linear(8,hidden,bias=False))
        self.qnorm=nn.LayerNorm(hidden);self.pnorm=nn.LayerNorm(64);self.k=nn.Linear(64,hidden,bias=False);self.v=nn.Linear(64,hidden,bias=False)
        self.q=nn.Linear(hidden,hidden,bias=False)
        self.out=nn.Linear(hidden,1,bias=False)
        self.gate=nn.Linear(hidden*2,hidden)
        self.cr=nn.Linear(hidden,1,bias=False)
        self.pair_heads=nn.ModuleList([nn.Linear(hidden,1,bias=False) for _ in range(2)])
        # Differential Transformer: 2 differential heads vs 4 ordinary heads.
        self.lambda_q1=nn.Parameter(torch.randn(self.d)*.1);self.lambda_k1=nn.Parameter(torch.randn(self.d)*.1)
        self.lambda_q2=nn.Parameter(torch.randn(self.d)*.1);self.lambda_k2=nn.Parameter(torch.randn(self.d)*.1)
        self.diff_norm=nn.RMSNorm(hidden//2,eps=1e-5,elementwise_affine=False)

    def queries(self,c,r,rm,mode='conditional'):
        qr=self.gene[None]+self.rna(torch.stack([r,rm],-1));cc=self.c(c)
        if mode=='rna':z=qr
        elif mode=='clinical':z=self.gene[None]+cc[:,None]
        elif mode=='film':z=qr*(1+.1*torch.tanh(cc[:,None]))
        else:z=qr*(1+.1*torch.tanh(cc[:,None]))+.1*cc[:,None]
        return self.q(self.qnorm(z)),qr.mean(1),cc

    def attention(self,q,k,mask):
        b,g,_=q.shape;n=k.shape[1]
        qq=q.view(b,g,self.h,self.d).transpose(1,2);kk=k.view(b,n,self.h,self.d).transpose(1,2)
        return masked_softmax((qq@kk.transpose(-1,-2))/math.sqrt(self.d),mask)

    def message(self,a,v):
        b,n,_=v.shape;vv=v.view(b,n,self.h,self.d).transpose(1,2)
        return (a@vv).transpose(1,2).reshape(b,a.shape[-2],-1).mean(1)

    def forward(self,c,r,rm,p,mask,diagnostic=False):
        b=len(c);c0=self.c0.expand_as(c);r0=self.r0.expand_as(r);rm0=self.rm0.expand_as(rm)
        pp=self.pnorm(p);k,v=self.k(pp),self.v(pp);q11,qr,cc=self.queries(c,r,rm)
        a11=self.attention(q11,k,mask);delta=None;ratio=torch.ones(b,device=p.device)
        if self.arm=='A1':
            cp=self.message(self.attention(self.queries(c,r,rm,'clinical')[0],k,mask),v)
            rp=self.message(self.attention(self.queries(c,r,rm,'rna')[0],k,mask),v)
            # Each nonlinear branch has only its named two modalities. Linear sum only.
            branches=[self.pair_heads[0](cp).squeeze(-1),self.pair_heads[1](rp).squeeze(-1),self.cr(torch.tanh(qr)*torch.tanh(cc)).squeeze(-1)]
            raw=sum(branches)
            pair_risk=self.bound/3*sum(torch.tanh(z) for z in branches)
            m=cp+rp
        elif self.arm=='A3':
            # Preserve the official signed-map lambda and headwise RMSNorm semantics.
            qd=q11.view(b,512,2,2*self.d).transpose(1,2);kd=k.view(b,-1,2,2*self.d).transpose(1,2)
            aa=masked_softmax(qd[...,:self.d]@kd[...,:self.d].transpose(-1,-2)/math.sqrt(self.d),mask)
            bb=masked_softmax(qd[...,self.d:]@kd[...,self.d:].transpose(-1,-2)/math.sqrt(self.d),mask)
            lam=torch.exp((self.lambda_q1*self.lambda_k1).sum())-torch.exp((self.lambda_q2*self.lambda_k2).sum())+.2
            vv=v.view(b,-1,2,2*self.d).transpose(1,2)
            m=(self.diff_norm((aa-lam*bb)@vv)*.8).transpose(1,2).reshape(b,512,64).mean(1)
            raw=self.out(m).squeeze(-1)
        elif self.arm in ['A4','A5','A6','A7']:
            a10=self.attention(self.queries(c,r0,rm0)[0],k,mask)
            a01=self.attention(self.queries(c0,r,rm)[0],k,mask)
            a00=self.attention(self.queries(c0,r0,rm0)[0],k,mask)
            delta=mixed_difference(a11,a10,a01,a00);m=self.message(delta,v)
            if self.arm=='A7':
                uniform=mask[:,None,None,:]/mask.sum(-1)[:,None,None,None]
                other=self.message(a11-uniform,v)
                target=m.detach().norm(dim=-1);den=other.detach().norm(dim=-1)
                if bool(((den<1e-15)&(target>1e-12)).any()):raise ValueError('norm control undefined')
                ratio=torch.where(den>1e-15,target/den,torch.zeros_like(den));m=other*ratio[:,None]
            if self.arm in ['A5','A6','A7']:m=m*torch.sigmoid(self.gate(torch.cat([qr,cc],-1)))
            raw=self.out(m).squeeze(-1)
        else:
            mode='film' if self.arm=='F0' else 'rna' if self.arm=='PR' else 'conditional'
            aa=self.attention(self.queries(c,r,rm,mode)[0],k,mask)
            m=self.message(aa,v);raw=self.out(m).squeeze(-1)
        risk=pair_risk if self.arm=='A1' else self.bound*torch.tanh(raw)
        if not diagnostic:return risk
        return risk,{'message_norm':m.norm(dim=-1),'risk_abs':risk.abs(),'raw_risk':raw,'score_range':torch.stack([a11.amin((-1,-2,-3)),a11.amax((-1,-2,-3))],-1),'delta_norm':delta.flatten(1).norm(dim=-1) if delta is not None else torch.zeros_like(raw),'row_sum_error':delta.sum(-1).abs().flatten(1).max(-1).values if delta is not None else torch.zeros_like(raw),'norm_control_scale':ratio}


class Model(nn.Module):
    def __init__(self,data,arm,w,config):
        super().__init__();tc=config['training'];self.arm=arm;self.main=nn.Parameter(torch.as_tensor(w).float())
        self.interaction=None if arm=='A0' else Interaction(data.f['C'].shape[1],data.C0,data.R0,data.RM0,arm,tc['hidden'],tc['heads'],tc['interaction_risk_bound'])

    def forward(self,x,c,r,rm,p,mask,diagnostic=False):
        main=x@self.main
        if self.interaction is None:return (main,{'risk_abs':torch.zeros_like(main)}) if diagnostic else main
        result=self.interaction(c,r,rm,p,mask,diagnostic)
        if diagnostic:return main+result[0],result[1]
        return main+result
