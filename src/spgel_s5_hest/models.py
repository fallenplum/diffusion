"""Image-only prediction; optional paired supervision uses training 50-gene RNA.

The four queries are anonymous learned slots, not medical programs. The extra
paired branch never contributes its RNA-conditioned values to prediction.
These are feature-based adaptations, not BLEEP/TRIPLEX or the old full A5.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..spgel_hest_neural.models import build_model as old_model, _linear
from .protocol import ARMS

class SharedFeatureModel(nn.Module):
    def __init__(self, arm, seed=1):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("UNREGISTERED_S5_ARM")
        self.arm=arm
        if arm in ("MLP","GRAPHSAGE","IDENTITY"):
            self.baseline=old_model({"MLP":"MLP","GRAPHSAGE":"SPATIAL_GNN","IDENTITY":"IDENTITY_GNN"}[arm],seed)
            return
        self.input_projection=_linear(256,128,seed,"input")
        self.output_projection=_linear(128,50,seed,"output")
        self.image_key=_linear(128,32,seed,"s5_image_key")
        self.image_value=_linear(128,32,seed,"s5_image_value")
        self.hidden=_linear(128,128,seed,"s5_query_hidden")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+501)
            self.queries=nn.Parameter(torch.randn(4,32)*0.02)
        if arm=="SPGEL50_ADAPT":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+502)
                self.gene_id=nn.Embedding(50,32)
                nn.init.normal_(self.gene_id.weight,std=0.02)
            self.gene_value=_linear(2,32,seed,"s5_gene_measurement")
            self.gene_key=_linear(32,32,seed,"s5_gene_key")
            self.gene_to_image=_linear(32,32,seed,"s5_gene_to_image",bias=False)
            self.image_to_gene=_linear(32,32,seed,"s5_image_to_gene",bias=False)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+503)
                nn.init.normal_(self.gene_to_image.weight,std=0.002)
                nn.init.normal_(self.image_to_gene.weight,std=0.002)
            self.pair_seed=int(seed)

    def forward(self, image, context, expression=None,epoch=1):
        """Expression is optional and never modifies returned image prediction."""
        if self.arm in ("MLP","GRAPHSAGE","IDENTITY"):
            adjacency=context["identity_adjacency"] if self.arm=="IDENTITY" else context["neighbor_adjacency"]
            pred=self.baseline(image,adjacency)
            return dict(prediction=pred,auxiliary_loss=pred.sum()*0)
        h=torch.relu(self.input_projection(image))
        local=h[context["indices"]]
        keys=self.image_key(local);values=self.image_value(local)
        scores=torch.einsum("qd,nkd->nqk",self.queries,keys)/(32**0.5)
        scores=scores.masked_fill(~context["valid"][:,None,:],-torch.inf)
        weights=torch.softmax(scores,dim=-1)
        slots=torch.einsum("nqk,nkd->nqd",weights,values)
        pred=self.output_projection(torch.relu(self.hidden(slots.flatten(1))))
        auxiliary=pred.sum()*0
        audit=None
        if self.arm=="SPGEL50_ADAPT" and expression is not None:
            if expression.shape!=(len(image),50) or not torch.isfinite(expression).all():
                raise ValueError("S5_TRAIN_FIFTY_GENE_VALUES_REQUIRED")
            bank=dict(keys=keys,values=values,slots=slots,valid=context["valid"])
            auxiliary,audit=self.paired_itc(bank,expression,epoch=epoch)
        return dict(prediction=pred,auxiliary_loss=auxiliary,pairing_audit=audit)

    def gene_read(self,expression,observed,query):
        """Exact linear key/value read, factorized without N*50*32 evidence.

        The explicit per-gene value is geneID + Linear([value,observed]).
        Linearity permits weighting its identity and scalar terms separately;
        every conditional candidate still rereads its own original measurements.
        """
        if expression.shape[-1]!=50 or observed.shape!=expression.shape or observed.dtype!=torch.bool:
            raise ValueError("FIFTY_GENE_VALUES_AND_REAL_BOOLEAN_MEASUREMENT_MASK_REQUIRED")
        safe=torch.where(observed,expression,torch.zeros_like(expression))
        identity=self.gene_id.weight+self.gene_value.weight[:,1]+self.gene_value.bias
        value_vector=self.gene_value.weight[:,0]
        id_key=self.gene_key(identity)
        value_key=F.linear(value_vector,self.gene_key.weight,None)
        score=torch.einsum("...qd,gd->...qg",query,id_key)
        score=score+torch.einsum("...qd,d->...q",query,value_key)[...,None]*safe[...,None,:]
        score=score/(32**0.5)
        score=score.masked_fill(~observed[...,None,:],-torch.inf)
        active=observed.any(-1)
        score=torch.where(active[...,None,None],score,torch.zeros_like(score))
        attention=torch.softmax(score,-1)*observed[...,None,:]
        token=torch.einsum("...qg,gd->...qd",attention,identity)
        token=token+(attention*safe[...,None,:]).sum(-1)[...,None]*value_vector
        return dict(tokens=token,attention=attention,valid=active)

    def paired_scores(self,image_bank,expression,image_rows,gene_rows,*,observed=None,gene_base=None,return_details=False):
        if self.arm!="SPGEL50_ADAPT" or image_rows.ndim!=2 or image_rows.shape!=gene_rows.shape:
            raise ValueError("EXPLICIT_PAIRED_CANDIDATE_ROW_MATRICES_REQUIRED")
        n=len(expression)
        if image_rows.dtype!=torch.long or gene_rows.dtype!=torch.long or not ((image_rows>=0)&(image_rows<n)&(gene_rows>=0)&(gene_rows<n)).all():
            raise ValueError("TRAIN_PAIRED_ROW_RANGE")
        observed=torch.ones_like(expression,dtype=torch.bool) if observed is None else observed
        if gene_base is None:
            gene_base=self.gene_read(expression,observed,self.queries.expand(n,-1,-1))["tokens"]
        # Only unconditional single-modality banks may be cached/reused.
        g_context=gene_base[gene_rows];v_context=image_bank["slots"][image_rows]
        query_image=self.queries+self.gene_to_image(g_context)
        keys=image_bank["keys"][image_rows];values=image_bank["values"][image_rows];valid=image_bank["valid"][image_rows]
        scores=torch.einsum("acqd,ackd->acqk",query_image,keys)/(32**0.5)
        image_attention=torch.softmax(scores.masked_fill(~valid[:,:,None,:],-torch.inf),-1)
        cond_image=torch.einsum("acqk,ackd->acqd",image_attention,values)
        query_gene=self.queries+self.image_to_gene(v_context)
        cond_gene=self.gene_read(expression[gene_rows],observed[gene_rows],query_gene)
        compatible=F.cosine_similarity(cond_image,cond_gene["tokens"],dim=-1,eps=1e-12).mean(-1)
        active=valid.any(-1)&cond_gene["valid"]
        result=dict(scores=torch.where(active,compatible,torch.zeros_like(compatible)),valid=active)
        if return_details:result.update(conditioned_image_tokens=cond_image,conditioned_gene_tokens=cond_gene["tokens"],image_attention=image_attention,gene_attention=cond_gene["attention"])
        return result

    def paired_itc(self,image_bank,expression,*,epoch):
        anchors,candidates=paired_candidates(len(expression),epoch,seed=self.pair_seed,device=expression.device)
        observed=torch.ones_like(expression,dtype=torch.bool)
        gene_base=self.gene_read(expression,observed,self.queries.expand(len(expression),-1,-1))["tokens"]
        repeated=anchors[:,None].expand_as(candidates)
        forward=self.paired_scores(image_bank,expression,repeated,candidates,observed=observed,gene_base=gene_base)
        reverse=self.paired_scores(image_bank,expression,candidates,repeated,observed=observed,gene_base=gene_base)
        if not forward["valid"].all() or not reverse["valid"].all():raise ValueError("REGISTERED_HVG_PAIRED_CANDIDATE_UNMEASURED")
        truth=torch.zeros(len(anchors),dtype=torch.long,device=expression.device)
        image_loss=F.cross_entropy(forward["scores"]/0.1,truth)
        gene_loss=F.cross_entropy(reverse["scores"]/0.1,truth)
        return (image_loss+gene_loss)/2,dict(anchor_count=len(anchors),negative_count=candidates.shape[1]-1,
            train_rows=len(expression),pairings_per_direction=int(candidates.numel()),temperature=0.1,
            image_anchor_CE=float(image_loss.detach()),gene_anchor_CE=float(gene_loss.detach()),
            candidate_condition_recomputed_both_directions=True,positive_conditioned_embeddings_reused=False)

def paired_candidates(n,epoch,*,seed=1,device="cpu"):
    if n<2 or int(epoch)!=epoch or epoch<1:raise ValueError("TWO_TRAIN_ROWS_AND_POSITIVE_EPOCH_REQUIRED")
    cap=min(512,n);negative_count=min(8,n-1)
    anchors=(np.arange(cap,dtype=np.int64)*n//cap+int(epoch)-1)%n
    negative=[]
    for row in anchors:
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(row)]))
        offsets=rng.choice(n-1,size=negative_count,replace=False)+1
        negative.append((int(row)+offsets)%n)
    candidates=np.concatenate((anchors[:,None],np.asarray(negative,np.int64)),axis=1)
    return torch.as_tensor(anchors,dtype=torch.long,device=device),torch.as_tensor(candidates,dtype=torch.long,device=device)

def build_model(arm,seed=1):
    return SharedFeatureModel(arm,seed)

def parameter_counts(model):
    total=sum(p.numel() for p in model.parameters() if p.requires_grad)
    extra=sum(p.numel() for name,p in model.named_parameters() if (name.startswith("gene_") or name.startswith("image_to_gene")) and p.requires_grad)
    return dict(total=total,image_prediction=total-extra,training_only_gene_auxiliary=extra)
