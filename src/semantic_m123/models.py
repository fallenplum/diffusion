"""M1--M3: trainable additive effects and controlled molecular patch readers.

Neither an old risk vector nor an offset is a model input. Warm initialization
copies *training-partition* coefficients into trainable weights. All original
values remain separate from query-only counterfactuals.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('M1_SCRATCH','M1_WARM','M2_MEAN_MATCHED','M2_SINGLE',
        'M2_EIGHT','M2_EIGHT_MEAN','M3_FULL','M3_LEARNED_QUERY',
        'M3_NO_PRODUCT','M3_BIDIRECTIONAL')


def masked_softmax(logits, mask, dim=-1):
    mask=torch.broadcast_to(mask,logits.shape)
    z=logits.masked_fill(~mask,-torch.inf)
    has=mask.any(dim=dim,keepdim=True)
    z=torch.where(has,z,torch.zeros_like(z))
    return torch.softmax(z,dim=dim)*mask.to(logits.dtype)


def slide_mean(x,mask,slide):
    """Within-slide mean, then equal valid-slide mean; never patch-count pooling."""
    if x.ndim!=3 or x.shape[:2]!=mask.shape or slide.shape!=mask.shape:
        raise ValueError('PATCH_MASK_SLIDE_SHAPE')
    result=torch.zeros_like(x[:,0]);count=x.new_zeros(len(x),1)
    for s in range(int(slide[mask].max().item())+1 if mask.any() else 0):
        m=mask&(slide==s)
        result=result+torch.where(m[...,None],x,0.).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
        count=count+m.any(1,keepdim=True)
    if (count==0).any():raise ValueError('PATIENT_WITHOUT_VALID_SLIDE')
    return result/count


class Branch(nn.Module):
    def __init__(self,input_dim,width=64):
        super().__init__()
        self.linear=nn.Linear(input_dim,width)
        self.residual=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),
            nn.GELU(),nn.Linear(width,width))
        self.head=nn.Linear(width,1,bias=False)
        nn.init.normal_(self.linear.weight,std=.01);nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        nn.init.normal_(self.head.weight,std=.02)

    def forward(self,x,extra=None):
        h=self.linear(x)
        h=h+self.residual(h)
        if extra is not None:h=h+extra
        return self.head(h).squeeze(-1)

    def warm(self,coef,bias=0.):
        if tuple(coef.shape)!=(self.linear.in_features,):raise ValueError('WARM_COEFFICIENT_DIMENSION')
        with torch.no_grad():
            self.linear.weight[0].copy_(coef)
            self.linear.bias[0]=bias
            self.head.weight.zero_();self.head.weight[0,0]=1.


class Attention(nn.Module):
    def __init__(self,width=64,heads=2):
        super().__init__()
        if width%heads:raise ValueError('HEAD_WIDTH')
        self.heads=heads;self.width=width
        self.q=nn.Linear(width,width,bias=False);self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False);self.out=nn.Linear(width,width,bias=False)

    def forward(self,q,x,mask):
        b,k,d=q.shape; n=x.shape[1];h=self.heads;dh=d//h
        qq=self.q(q).reshape(b,k,h,dh).transpose(1,2)
        kk=self.k(x).reshape(b,n,h,dh).transpose(1,2)
        vv=self.v(x).reshape(b,n,h,dh).transpose(1,2)
        a=masked_softmax(qq@kk.transpose(-1,-2)/math.sqrt(dh),mask[:,None,None,:])
        z=(a@vv).transpose(1,2).reshape(b,k,d)
        return self.out(z),a

    def slides(self,q,x,mask,slide):
        z=x.new_zeros(len(x),q.shape[1],x.shape[-1]);a=x.new_zeros(len(x),self.heads,q.shape[1],x.shape[1])
        count=x.new_zeros(len(x),1,1)
        for s in range(int(slide[mask].max().item())+1 if mask.any() else 0):
            m=mask&(slide==s)
            zz,aa=self(q,x,m)
            z=z+zz;a=a+aa;count=count+m.any(1)[:,None,None]
        if (count==0).any():raise ValueError('EMPTY_PATIENT_ATTENTION')
        return z/count,a/count[:,None]


class JointModel(nn.Module):
    def __init__(self,arm,p_dim,g_dim,c_dim,*,width=64,heads=2):
        super().__init__()
        if arm not in ARMS:raise ValueError('UNREGISTERED_M123_ARM')
        if width!=64 or heads!=2:raise ValueError('M123_FIXED_ENGINEERING_WIDTH_HEADS')
        self.arm=arm;self.width=width
        self.p=Branch(p_dim,width);self.g=Branch(g_dim,width);self.c=Branch(c_dim,width)
        self.local_out=nn.Linear(width,width,bias=False)
        nn.init.zeros_(self.local_out.weight)
        self.kind='mean' if arm.startswith('M1') else 'm2' if arm.startswith('M2') else 'm3'
        if self.kind in ('mean','m3'):
            self.mean=nn.Linear(768,width,bias=False)
        if self.kind!='mean':
            self.patch=nn.Sequential(nn.Linear(768,width),nn.LayerNorm(width))
        if self.kind=='m2':
            if arm=='M2_MEAN_MATCHED':
                self.mean_extra=nn.Sequential(nn.Linear(width,128),nn.GELU(),nn.Linear(128,width))
            else:
                n=1 if arm=='M2_SINGLE' else 8
                self.query=nn.Parameter(torch.randn(n,width)*.02)
                self.attention=Attention(width,heads)
                self.pool=nn.Linear(width,1,bias=False)
        if self.kind=='m3':
            self.program=nn.Sequential(nn.Linear(9,width),nn.LayerNorm(width),nn.GELU())
            self.identity=nn.Parameter(torch.randn(50,width)*.02)
            self.query_projection=nn.Linear(width,width,bias=False)
            self.attention=Attention(width,heads)
            if arm!='M3_BIDIRECTIONAL':self.u=nn.Linear(width,16,bias=False)
            self.v=nn.Linear(width,16,bias=False)
            self.program_pool=nn.Linear(width,1,bias=False)
            self.interaction_head=nn.Linear(16,1,bias=False)
            nn.init.zeros_(self.interaction_head.weight)
            if arm=='M3_BIDIRECTIONAL':
                # A conventional weight-shared bidirectional attention control;
                # same effective parameter budget as the Hadamard alternative.
                self.reverse_attention=self.attention
                self.reverse_out=nn.Linear(width,16,bias=False)

    def warm(self,p_coef,g_coef,c_coef,p_bias,g_bias):
        if self.arm=='M1_SCRATCH':raise ValueError('SCRATCH_MUST_NOT_READ_TEACHER_COEFFICIENTS')
        self.p.warm(p_coef,p_bias);self.g.warm(g_coef,g_bias);self.c.warm(c_coef)

    def head_parameters(self):
        return [p for name,p in self.named_parameters() if '.head.' in name or name.startswith('interaction_head.')]

    def set_stage(self,head_only):
        selected={id(p) for p in self.head_parameters()}
        for p in self.parameters():p.requires_grad_(not head_only or id(p) in selected)

    def forward(self,b,*,query_context=None,query_mode='matched',diagnostics=False):
        """Wrong context changes queries only, not owner values/main effects.

        M3_NO_PRODUCT removes the explicit Hadamard operator. Its conditioned
        reader remains nonlinear across modalities: it is not a globally
        additive model and is not reported as one.
        """
        mask=b['mask'];slide=b['slide']
        x=torch.where(mask[...,None],b['patch'],0.)
        mean=b['mean'];local=self.mean(mean) if self.kind=='mean' else None
        interaction=x.new_zeros(len(x));att=None;tokens=None
        if self.kind=='m2':
            xp=self.patch(x)
            if self.arm=='M2_MEAN_MATCHED':
                local=self.mean_extra(slide_mean(xp,mask,slide))
            else:
                q=self.query[None].expand(len(x),-1,-1)
                tokens,att=self.attention.slides(q,xp,mask,slide)
                pool=torch.softmax(self.pool(tokens).squeeze(-1),dim=1)
                local=(tokens*pool[...,None]).sum(1)
                if self.arm=='M2_EIGHT_MEAN':local=local+self.patch[0](mean)
        elif self.kind=='m3':
            xp=self.patch(x)
            valid=b['program_mask'].any(-1)
            info=torch.cat([torch.where(b['program_mask'],b['program'],0.),b['program_mask'].to(x.dtype),b['coverage']],dim=-1)
            owner_g=self.program(info)+self.identity[None]
            owner_g=owner_g*valid[...,None]
            context=owner_g
            if query_context is not None:
                qinfo=torch.cat([torch.where(query_context['program_mask'],query_context['program'],0.),query_context['program_mask'].to(x.dtype),query_context['coverage']],dim=-1)
                context=self.program(qinfo)+self.identity[None]
            if self.arm=='M3_LEARNED_QUERY':context=self.identity[None].expand_as(owner_g)
            elif query_mode=='zero_context':context=torch.zeros_like(context)
            elif query_mode=='mean_context' and query_context is None:raise ValueError('MEAN_QUERY_REQUIRES_TRAIN_ONLY_CONTEXT')
            elif query_mode not in ('matched','mean_context'):raise ValueError('UNKNOWN_QUERY_INTERVENTION')
            q=self.query_projection(context)+self.identity[None]
            tokens,att=self.attention.slides(q,xp,mask,slide)
            scores=self.program_pool(owner_g).squeeze(-1)
            weight=masked_softmax(scores,valid)
            v=self.v(tokens)
            if self.arm=='M3_NO_PRODUCT':message=self.u(owner_g)+v
            elif self.arm=='M3_BIDIRECTIONAL':
                reverse,_=self.reverse_attention(xp,owner_g,valid)
                rv=slide_mean(reverse,mask,slide)
                # Standard two-way attention readout, without Hadamard product.
                message=v+self.reverse_out(rv)[:,None]
            else:message=self.u(owner_g)*v
            pooled=(message*weight[...,None]).sum(1)
            interaction=self.interaction_head(pooled).squeeze(-1)
            # Independent raw mean morphology is retained for every M3 arm.
            local=self.mean(mean)
        extra=self.local_out(local)
        rp=self.p(b['P'],extra);rg=self.g(b['G']);rc=self.c(b['C'])
        result={'risk':rp+rg+rc+interaction,'pathology_risk':rp,'omics_risk':rg,
                'clinical_risk':rc,'interaction_risk':interaction}
        if diagnostics:result.update(patch_attention=att,local_tokens=tokens,local_message=extra)
        return result

    def regularizer(self):
        return sum(p.square().sum() for p in self.parameters())/2

    def parameter_audit(self):
        return {'parameters':sum(p.numel() for p in self.parameters()),
                'head_parameters':sum(p.numel() for p in self.head_parameters()),
                'frozen_foundation_parameters_trained':0,
                'pretrained_rank_or_medical_slot_names_claimed':False}
