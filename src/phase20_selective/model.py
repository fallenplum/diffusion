"""Frozen raw anchor, pairwise-logit backbone and selective Clinical correction."""
import math
import torch
from torch import nn

from src.phase19_attention.model import SeparateMolecularTokens, masked_mean
from src.phase19_attention.attention import RankConditionedAttention, masked_softmax
from .operators import capped_simplex, dense_budget_gate, conservative_update, rank_one_energy_projection

ARMS = ['L0','P1','P16','S1','S16','D16','F16','WC16','WO16']


def normalize_gate_scores(scores, mask, floor=1e-6):
    """Valid-row population z-score, retaining mean/variance gradients.

    Clamping the variance before sqrt implements max(population_SD,floor)
    without a singular sqrt derivative at a constant or empty row.
    """
    if mask.shape != scores.shape or mask.dtype != torch.bool or floor <= 0.:
        raise ValueError('invalid gate normalization mask or floor')
    safe = torch.where(mask,scores,torch.zeros_like(scores)).double()
    count = mask.sum(-1,keepdim=True).clamp_min(1)
    mean = safe.sum(-1,keepdim=True)/count
    centered = torch.where(mask,safe-mean,torch.zeros_like(safe))
    variance = centered.square().sum(-1,keepdim=True)/count
    scale = variance.clamp_min(float(floor)**2).sqrt()
    return (centered/scale).to(scores.dtype)


class SignedPathwayTokens(nn.Module):
    """Sixteen signed measured-gene filters per modality/program, not PCA."""
    def __init__(self, membership, d=32):
        super().__init__()
        member = torch.as_tensor(membership, dtype=torch.bool)
        if member.ndim != 2 or member.shape[0] != 8 or not member.any(-1).all():
            raise ValueError('exactly eight nonempty programs required')
        self.register_buffer('membership', member)
        self.filters = nn.ParameterList()
        for channel in range(3):
            for k in range(8):
                ix = torch.nonzero(member[k],as_tuple=True)[0]
                self.register_buffer(f'gene_indices_{channel*8+k}', ix)
                weight = nn.Parameter(torch.empty(16, len(ix)))
                nn.init.orthogonal_(weight)
                self.filters.append(weight)
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(17,16), nn.SiLU(), nn.Linear(16,d)) for _ in range(3)])
        self.slot = nn.Parameter(torch.randn(3,8,d)*.02)

    def forward(self, r, gain, loss, rm, cm, rank_one=False, diagnostic=False):
        tokens, masks, summaries, transformed, norms, coverages = [], [], [], [], [], []
        for channel, (values, measured) in enumerate([(r,rm),(gain,cm),(loss,cm)]):
            group_values, group_masks, group_norms, group_coverage = [], [], [], []
            for k in range(8):
                index = channel*8+k
                ix = getattr(self, f'gene_indices_{index}')
                available = measured[:,ix].bool()
                x = torch.where(available, values[:,ix], torch.zeros_like(values[:,ix]))
                weight = self.filters[index]
                square_norm = available.to(values.dtype) @ weight.square().T
                positive = square_norm > 0.
                safe_norm = torch.where(positive, square_norm, torch.ones_like(square_norm)).sqrt()
                z = torch.where(positive, (x@weight.T)/safe_norm, torch.zeros_like(square_norm))
                group_values.append(z)
                group_masks.append(available.any(-1))
                group_norms.append(torch.where(positive, safe_norm, torch.zeros_like(safe_norm)))
                group_coverage.append(available.to(values.dtype).mean(-1))
            z0 = torch.stack(group_values,1)
            z = rank_one_energy_projection(z0) if rank_one else z0
            coverage = torch.stack(group_coverage,1)
            valid = torch.stack(group_masks,1)
            token = self.encoders[channel](torch.cat([z,coverage[...,None]],-1)) + self.slot[channel]
            tokens.append(torch.where(valid[...,None],token,torch.zeros_like(token)))
            masks.append(valid); summaries.append(z0); transformed.append(z)
            norms.append(torch.stack(group_norms,1)); coverages.append(coverage)
        result = torch.cat(tokens,1), torch.cat(masks,1)
        if not diagnostic:
            return result
        return (*result, {'summary16_before':torch.cat(summaries,1),
                         'summary16_after':torch.cat(transformed,1),
                         'filter_measured_norms':torch.cat(norms,1),
                         'token_coverage':torch.cat(coverages,1)})


class SelectiveModel(nn.Module):
    def __init__(self, clinical_dim, membership, anchor_w, clinical_mean, anchor_sd, arm, config):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        d = int(config.get('hidden',32)); spec = config.get('selective',{})
        locked = {'width':8,'budget_fraction':.25,'rho':.5,'delta_bound':.5,
                  'risk_cap_fraction':.25,'gate_solver_iterations':64,'pair_rank':4,
                  'gate_score_normalization':'valid_row_population_zscore',
                  'gate_score_std_floor':1e-6}
        if d != 32 or any(spec.get(k,v) != v for k,v in locked.items()):
            raise ValueError('Phase20 model constants differ from locked design')
        if config.get('tokenizer',{}).get('filters',16) != 16:
            raise ValueError('exactly sixteen signed filters required')
        self.arm, self.rank_one = arm, arm in ['P1','S1']
        self.has_correction = arm in ['S1','S16','D16','F16','WC16','WO16']
        self.visual = nn.Sequential(nn.Linear(1536,d),nn.LayerNorm(d))
        # One draw for an independent molecular stream, no arm-dependent effect
        # on the shared fusion/correction initialization.
        token_seed = int(torch.randint(0,2**31-1,()).item())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(token_seed)
            self.molecular = (SeparateMolecularTokens(membership,d) if arm == 'L0'
                              else SignedPathwayTokens(membership,d))
        self.p_value = nn.Linear(d,d,bias=False); self.o_value = nn.Linear(d,d,bias=False)
        self.head = nn.Sequential(nn.Linear(2*d,16),nn.SiLU(),nn.Dropout(config['training']['dropout']),nn.Linear(16,1,bias=False))
        self.pair_attention = RankConditionedAttention(d,d,clinical_dim+1,d=d,rank=4)
        with torch.no_grad():
            self.pair_attention.c.weight.zero_(); self.pair_attention.c.weight[:,0] = 1.
        self.Up = nn.Linear(d,8,bias=False); self.Uo = nn.Linear(d,8,bias=False)
        self.Uc = nn.Linear(clinical_dim,8,bias=False)
        self.wg = nn.Linear(8,1,bias=False); self.wd = nn.Linear(8,1,bias=False)
        self.correction_head = nn.Linear(2*d,1,bias=False)
        nn.init.normal_(self.head[-1].weight,std=.02)
        nn.init.normal_(self.correction_head.weight,std=.02)
        self.register_buffer('main',torch.as_tensor(anchor_w,dtype=torch.float64).clone())
        self.register_buffer('clinical_mean',torch.as_tensor(clinical_mean,dtype=torch.float32).clone())
        scale = float(anchor_sd)
        if not math.isfinite(scale) or scale < 0.:
            raise ValueError('anchor population SD must be finite and nonnegative')
        self.register_buffer('risk_scale',torch.tensor(scale,dtype=torch.float32))
        self.register_buffer('correction_cap',torch.tensor(.25*scale,dtype=torch.float32))

    def _tokens(self,r,g,l,rm,cm,diagnostic=False):
        if self.arm == 'L0':
            tokens, valid = self.molecular(r,g,l,rm,cm)
            return tokens, valid, {}
        return self.molecular(r,g,l,rm,cm,rank_one=self.rank_one,diagnostic=True)

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
            a1,update_a = conservative_update(a0,gate_a,delta,eligible,.5,True)
            b1,update_b = conservative_update(b0,gate_b,delta.transpose(1,2),eligible.transpose(1,2),.5,True)
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
