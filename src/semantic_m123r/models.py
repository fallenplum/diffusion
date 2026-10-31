"""Repair the trained pathway, retaining the exact R4 main-effect function."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from src.semantic_m123.models import Branch,masked_softmax,slide_mean

ARMS=('R2','R3','R4','R5','R6','R7','R8','R9')


class NormalizedAttention(nn.Module):
    def __init__(self,width=64,heads=2):
        super().__init__();self.heads=heads;self.width=width
        self.q=nn.Linear(width,width,bias=False);self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False);self.out=nn.Linear(width,width,bias=False)

    def forward(self,q,x,mask):
        b,n,d=x.shape;h=self.heads;dh=d//h
        qq=self.q(q).reshape(b,-1,h,dh).transpose(1,2)
        kk=self.k(x).reshape(b,n,h,dh).transpose(1,2)
        qq=F.normalize(F.layer_norm(qq,(dh,)),dim=-1,eps=1e-6)
        kk=F.normalize(F.layer_norm(kk,(dh,)),dim=-1,eps=1e-6)
        logits=5.*(qq@kk.transpose(-1,-2))
        aa=masked_softmax(logits,mask[:,None,None,:])
        vv=self.v(x).reshape(b,n,h,dh).transpose(1,2)
        z=(aa@vv).transpose(1,2).reshape(b,-1,d)
        return self.out(z),aa

    def slides(self,q,x,mask,slide):
        z=x.new_zeros(len(x),q.shape[1],x.shape[-1]);a=x.new_zeros(len(x),self.heads,q.shape[1],x.shape[1])
        count=x.new_zeros(len(x),1,1)
        for s in range(int(slide[mask].max())+1):
            m=mask&(slide==s);zz,aa=self(q,x,m);z+=zz;a+=aa;count+=m.any(1)[:,None,None]
        if (count==0).any():raise ValueError('EMPTY_SLIDE_PATIENT')
        return z/count,a/count[:,None]


class RepairModel(nn.Module):
    def __init__(self,arm,p_dim,g_dim,c_dim):
        super().__init__()
        if arm not in ARMS:raise ValueError('UNREGISTERED_REPAIR_ARM')
        self.arm=arm
        # Keep the previous 64D additive branch parameterization. FP64 permits
        # faithful S0 output instead of relying on rounded float32 covariates.
        self.base=nn.ModuleDict({k:Branch(d).double() for k,d in zip(('p','g','c'),(p_dim,g_dim,c_dim))})
        self.local_out=nn.Linear(64,64,bias=False);nn.init.zeros_(self.local_out.weight)
        if arm in ('R2','R3'):
            self.mean=nn.Linear(768,64,bias=False)
        else:
            self.patch=nn.Sequential(nn.Linear(768,64),nn.LayerNorm(64))
            if arm=='R4':self.mean_extra=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))
            else:
                q=torch.empty(8,64);nn.init.orthogonal_(q)
                self.mil_query=nn.Parameter(q)
                self.mil_attention=NormalizedAttention();self.mil_pool=nn.Linear(64,1,bias=False)
        if arm in ('R6','R7','R8','R9'):
            self.program=nn.Sequential(nn.Linear(9,64),nn.LayerNorm(64),nn.GELU())
            self.identity=nn.Parameter(torch.randn(50,64)*.02)
            self.query_projection=nn.Linear(64,64,bias=False)
            self.cross_attention=NormalizedAttention()
            self.u=nn.Linear(64,16,bias=False) if arm!='R9' else None
            self.v=nn.Linear(64,16,bias=False);self.program_pool=nn.Linear(64,1,bias=False)
            self.interaction_head=nn.Linear(16,1,bias=False);nn.init.zeros_(self.interaction_head.weight)
            if arm=='R9':self.reverse_out=nn.Linear(64,16,bias=False)
        self.initial_base=None

    def warm(self,warm):
        device=next(self.parameters()).device
        for k in ('p','g','c'):
            self.base[k].warm(torch.as_tensor(warm[k+'_coef'],dtype=torch.float64,device=device),warm.get(k+'_bias',0.))
        self.initial_base={n:p.detach().clone() for n,p in self.named_parameters() if n.startswith('base.')}

    def group(self,name):
        return [(n,p) for n,p in self.named_parameters() if n.startswith('base.')==(name=='base')]

    def output_names(self):
        return ('local_out.','interaction_head.')

    def stage(self,epoch):
        for n,p in self.named_parameters():
            if self.arm in ('R2','R3'):active=True
            elif n.startswith('base.'):active=epoch>90
            else:active=epoch>10 or n.startswith(self.output_names())
            p.requires_grad_(active)
        return 'MAIN_MATCHED' if self.arm in ('R2','R3') else 'S1' if epoch<=10 else 'S2' if epoch<=90 else 'S3'

    def penalties(self):
        if self.initial_base is None:raise ValueError('WARM_REQUIRED')
        b=sum((p-self.initial_base[n]).square().sum() for n,p in self.group('base'))
        new=sum(p.double().square().sum() for n,p in self.group('new'))
        return b,new

    def forward(self,b,*,query_context=None,disable_local=False,disable_interaction=False,disable_mil=False,diagnostics=False):
        main={k:self.base[k](b[key].double()) for k,key in zip(('p','g','c'),('P','G','C'))}
        interaction=b['patch'].new_zeros(len(b['P']));mil_att=mil_tokens=cross_att=cross_tokens=None
        if self.arm in ('R2','R3'):local=self.mean(slide_mean(b['patch'],b['mask'],b['slide']))
        else:
            xp=self.patch(torch.where(b['mask'][...,None],b['patch'],0.))
            mean=slide_mean(xp,b['mask'],b['slide'])
            if self.arm=='R4':local=mean+self.mean_extra(mean)
            else:
                q=self.mil_query[None].expand(len(xp),-1,-1)
                mil_tokens,mil_att=self.mil_attention.slides(q,xp,b['mask'],b['slide'])
                weight=torch.softmax(self.mil_pool(mil_tokens).squeeze(-1),-1)
                local=mean if disable_mil else mean+(weight[...,None]*mil_tokens).sum(1)
            if self.arm in ('R6','R7','R8','R9'):
                def encode(context):
                    info=torch.cat([torch.where(context['program_mask'],context['program'],0.),context['program_mask'].float(),context['coverage']],-1)
                    return self.program(info)
                valid=b['program_mask'].any(-1)
                measured=encode(b);owner=(measured+self.identity[None])*valid[...,None]
                identity=F.normalize(self.identity,dim=-1,eps=1e-6)[None]
                if self.arm=='R7':q=(F.normalize(self.query_projection(identity),dim=-1,eps=1e-6)+identity).expand_as(owner)
                else:
                    context=measured if query_context is None else encode(query_context)
                    q=F.normalize(self.query_projection(context),dim=-1,eps=1e-6)+identity
                cross_tokens,cross_att=self.cross_attention.slides(q,xp,b['mask'],b['slide'])
                weight=masked_softmax(self.program_pool(owner).squeeze(-1),valid)
                v=self.v(cross_tokens)
                if self.arm=='R9':
                    reverse,_=self.cross_attention(xp,owner,valid)
                    message=v+self.reverse_out(slide_mean(reverse,b['mask'],b['slide']))[:,None]
                elif self.arm=='R8':message=self.u(owner)+v
                else:message=self.u(owner)*v
                interaction=self.interaction_head((weight[...,None]*message).sum(1)).squeeze(-1)
        extra=self.local_out(local).double()
        local_risk=self.base['p'].head(extra).squeeze(-1)
        base=main['p']+main['g']+main['c']
        risk=base+(0. if disable_local else local_risk)+(0. if disable_interaction else interaction.double())
        out={'risk':risk,'base_risk':base,'local_risk':local_risk,'interaction_risk':interaction.double(),
            'pathology_main_risk':main['p'],'omics_main_risk':main['g'],'clinical_main_risk':main['c']}
        if diagnostics:out.update(mil_attention=mil_att,mil_tokens=mil_tokens,cross_attention=cross_att,cross_tokens=cross_tokens)
        return out
