"""Exact inherited encoder, selector, values and heads; only update differs."""
import math
import torch
from src.phase20_selective.model import SelectiveModel, normalize_gate_scores
from src.phase20_selective.operators import dense_budget_gate, capped_simplex
from src.phase19_attention.attention import masked_softmax
from src.phase19_attention.model import masked_mean
from .operators import direct_update


class DirectModel(SelectiveModel):
    """S16 parameter/state layout; direct globally normalized relative update."""
    def forward(self,visual,mask,clinical,r,gain,loss,rm,cm,main_x,
                context_clinical=None,context_omics=None,diagnostic=False,
                interaction_off=False,main_only=False,anchor_override=None,pair_uniform=False,**unused):
        main = ((main_x.double()@self.main).to(visual.dtype) if anchor_override is None
                else anchor_override.detach().to(visual.dtype))
        if main.shape != (len(visual),):
            raise ValueError('anchor must have one scalar per WSI')
        if main_only:
            return (main,{'main_risk':main}) if diagnostic else main
        p = self.visual(torch.where(mask[...,None],visual,torch.zeros_like(visual)))
        o, om, token_diag = self._tokens(r,gain,loss,rm,cm,diagnostic)
        c_aug = torch.cat([torch.ones_like(clinical[:,:1]),clinical],-1)
        q,k = self.pair_attention.q(p), self.pair_attention.k(o)
        ap,bo,cc = self.pair_attention.a(p),self.pair_attention.b(o),self.pair_attention.c(c_aug)
        pc = torch.einsum('bnr,br->bn',ap,cc); oc = torch.einsum('bkr,br->bk',bo,cc)
        base_scores = q@k.transpose(-1,-2)/math.sqrt(32) + pc[:,:,None] + oc[:,None,:]
        valid = mask[:,:,None] & om[:,None,:]
        pair_scores = torch.zeros_like(base_scores) if pair_uniform else base_scores
        a0 = masked_softmax(pair_scores,valid,-1)
        b0 = masked_softmax(pair_scores.transpose(1,2),valid.transpose(1,2),-1)
        pv,ov = self.p_value(p),self.o_value(o)
        message0 = torch.cat([masked_mean(a0@ov,mask),masked_mean(b0@pv,om)],-1)
        pair = self.risk_scale*torch.tanh(self.head(message0).squeeze(-1))
        correction = torch.zeros_like(main)
        a1,b1 = a0,b0
        gate_a,gate_b = torch.zeros_like(a0),torch.zeros_like(b0)
        gate_score,delta = torch.zeros_like(a0),torch.zeros_like(a0)
        normalized_a,normalized_b = torch.zeros_like(a0),torch.zeros_like(b0)
        eligible = valid
        update_a,update_b = {},{}
        if self.has_correction and not interaction_off:
            c = clinical if context_clinical is None else context_clinical
            if context_omics is None:
                context_o,context_mask = o,om
            else:
                context_o,context_mask,_ = self._tokens(*context_omics)
            eligible = valid & context_mask[:,None,:]
            po = self.Up(p)[:,:,None,:] + self.Uo(context_o)[:,None,:,:]
            # Center before the bias-free linear map.  This is algebraically
            # Uc(c), but guarantees an exactly zero change at c == c_ref even
            # when batched and vector GEMMs have different rounding paths.
            reference_pre = po+self.Uc(self.clinical_mean)[None,None,None,:]
            h0 = torch.tanh(reference_pre)
            h = torch.tanh(reference_pre+self.Uc(c-self.clinical_mean)[:,None,None,:])
            gate_score = self.wg(h0 if self.arm == 'F16' else h).squeeze(-1)
            delta = .5*torch.tanh(self.wd(h-h0).squeeze(-1))
            project = dense_budget_gate if self.arm == 'D16' else capped_simplex
            normalized_a = normalize_gate_scores(gate_score,eligible)
            normalized_b = normalize_gate_scores(gate_score.transpose(1,2),eligible.transpose(1,2))
            gate_a = project(normalized_a,eligible,.25)
            gate_b = project(normalized_b,eligible.transpose(1,2),.25)
            a1,update_a = direct_update(a0,gate_a,delta,eligible,.5,True)
            b1,update_b = direct_update(b0,gate_b,delta.transpose(1,2),eligible.transpose(1,2),.5,True)
            message_delta = torch.cat([masked_mean((a1-a0)@ov,mask),masked_mean((b1-b0)@pv,om)],-1)
            correction = self.correction_cap*torch.tanh(self.correction_head(message_delta).squeeze(-1))
        risk = main+pair+correction
        if not diagnostic:
            return risk
        return risk, {'main_risk':main,'pair_risk':pair,'correction_risk':correction,
            'interaction_risk':pair+correction,'tokens':o,'patch_mask':mask,'omics_mask':om,
            'base_p_to_o':a0,'base_o_to_p':b0,'p_to_o':a1,'o_to_p':b1,
            'base_scores':base_scores,'gate_score':gate_score,'delta':delta,
            'normalized_gate_score_p':normalized_a,'normalized_gate_score_o':normalized_b,
            'gate_p_to_o':gate_a,'gate_o_to_p':gate_b,
            'correction_eligible_mask':eligible,
            'eligible_omics_per_patch':eligible.sum(-1),'eligible_patches_per_omics':eligible.sum(-2),
            'update_p_to_o':update_a,'update_o_to_p':update_b, **token_diag}
