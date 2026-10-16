"""One input-preserving model family and explicitly limited mechanism controls."""
import torch
from torch import nn
from .attention import RankConditionedAttention, masked_softmax


class SeparateMolecularTokens(nn.Module):
    """Keep RNA, CNA gain and CNA loss as distinct measured-evidence tokens."""
    def __init__(self, membership, d=32):
        super().__init__()
        member = torch.as_tensor(membership, dtype=torch.bool)
        if member.ndim != 2 or (member.sum(-1) == 0).any():
            raise ValueError('empty or malformed gene membership')
        self.register_buffer('membership', member)
        k, n = member.shape
        self.logits = nn.Parameter(torch.randn(3, k, n) * .02)
        self.slot = nn.Parameter(torch.randn(3, k, d) * .02)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, d))
            for _ in range(3)])

    def forward(self, r, gain, loss, rm, cm):
        tokens, masks = [], []
        for j, (x, measured) in enumerate([(r, rm), (gain, cm), (loss, cm)]):
            allow = self.membership[None] & measured[:, None].bool()
            weights = masked_softmax(self.logits[j][None].expand(len(x), -1, -1), allow, -1)
            safe_x = torch.where(measured.bool(), x, torch.zeros_like(x))
            value = (weights * safe_x[:, None]).sum(-1)
            coverage = allow.float().sum(-1) / self.membership.sum(-1).clamp_min(1)
            token = self.encoders[j](torch.stack([value, coverage], -1)) + self.slot[j]
            valid = allow.any(-1)
            tokens.append(torch.where(valid[..., None], token, torch.zeros_like(token)))
            masks.append(valid)
        return torch.cat(tokens, 1), torch.cat(masks, 1)


def masked_mean(x, mask):
    x = torch.where(mask[..., None], x, torch.zeros_like(x))
    return x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class AttentionModel(nn.Module):
    def __init__(self, clinical_dim, membership, main_w, clinical_mean, arm, config):
        super().__init__()
        if arm not in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']:
            raise ValueError(arm)
        self.arm = arm
        d, rank = config['hidden'], config['rank']
        dropout = config['training']['dropout']
        # Instantiate the same modules, in the same order, for every mechanism arm.
        self.visual = nn.Sequential(nn.Linear(1536, d), nn.LayerNorm(d))
        self.molecular = SeparateMolecularTokens(membership, d)
        self.p_value = nn.Linear(d, d, bias=False)
        self.o_value = nn.Linear(d, d, bias=False)
        self.head = nn.Sequential(nn.Linear(2*d, 16), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(16, 1, bias=False))
        self.pc_head = nn.Linear(d, 1, bias=False)
        self.oc_head = nn.Linear(d, 1, bias=False)
        self.attention = RankConditionedAttention(d, d, clinical_dim + 1, d=d, rank=rank)
        with torch.no_grad():
            self.attention.c.weight.zero_()
            self.attention.c.weight[:, 0] = 1.
        nn.init.normal_(self.head[-1].weight, std=.02)
        nn.init.normal_(self.pc_head.weight, std=.02)
        nn.init.normal_(self.oc_head.weight, std=.02)
        self.main = nn.Parameter(torch.as_tensor(main_w, dtype=torch.float32).clone())
        self.register_buffer('clinical_mean', torch.as_tensor(clinical_mean, dtype=torch.float32))

    def forward(self, visual, mask, clinical, r, gain, loss, rm, cm, main_x,
                context_clinical=None, context_omics=None, diagnostic=False,
                interaction_off=False, **unused):
        main = main_x @ self.main
        if interaction_off and not diagnostic:
            return main
        c = clinical if context_clinical is None else context_clinical
        if self.arm == 'A2':
            c = self.clinical_mean.expand_as(c)
        if context_omics is not None:
            r, gain, loss, rm, cm = context_omics
        p = self.visual(torch.where(mask[..., None], visual, torch.zeros_like(visual)))
        o, om = self.molecular(r, gain, loss, rm, cm)
        c_aug = torch.cat([torch.ones_like(c[:, :1]), c], -1)
        scores = self.attention(p, o, c_aug, mask, om)
        ap, bo, cc = self.attention.a(p), self.attention.b(o), self.attention.c(c_aug)
        pc = torch.einsum('bnr,br->bn', ap, cc)
        oc = torch.einsum('bkr,br->bk', bo, cc)
        base = scores['base_scores']
        if self.arm in ['A0', 'A6']:
            combined = base
        elif self.arm == 'A1':
            combined = base + pc[:, :, None] + oc[:, None, :]
        else:
            combined = scores['scores']
        valid = scores['valid_pairs']
        po = masked_softmax(combined, valid, -1)
        op = masked_softmax(combined.transpose(1, 2), valid.transpose(1, 2), -1)
        pv, ov = self.p_value(p), self.o_value(o)
        p_message, o_message = po @ ov, op @ pv
        interaction = self.head(torch.cat([masked_mean(p_message, mask), masked_mean(o_message, om)], -1)).squeeze(-1)
        if self.arm == 'A6':
            # Each separate branch reads exactly two modalities. No joint MLP.
            pc_message = (masked_softmax(pc, mask, -1)[..., None] * pv).sum(1)
            oc_message = (masked_softmax(oc, om, -1)[..., None] * ov).sum(1)
            interaction = interaction + self.pc_head(pc_message).squeeze(-1) + self.oc_head(oc_message).squeeze(-1)
        risk = main + (torch.zeros_like(interaction) if interaction_off else interaction)
        if not diagnostic:
            return risk
        return risk, {'p_to_o': po, 'o_to_p': op, 'base_scores': base,
                      'trilinear_scores': scores['trilinear'], 'scores': combined,
                      'patch_mask': mask, 'omics_mask': om,
                      'interaction_risk': interaction, 'main_risk': main,
                      'tokens': o}


class NativeModel(nn.Module):
    """An explicitly adapted official method, with the same raw main effects."""
    def __init__(self, clinical_dim, membership, main_w, arm, config):
        super().__init__()
        from .baselines import build_baseline
        self.arm = arm
        self.core = build_baseline(arm, clinical_dim, membership, config)
        self.main = nn.Parameter(torch.as_tensor(main_w, dtype=torch.float32).clone())

    def forward(self, visual, mask, clinical, r, gain, loss, rm, cm, main_x,
                context_clinical=None, context_omics=None, diagnostic=False,
                interaction_off=False, native_features=None, wsi_indices=None,
                return_extra=False, **unused):
        main = main_x @ self.main
        if interaction_off:
            return (main, {}) if diagnostic else main
        f = dict(native_features or {})
        cr, cg, cl, crm, ccm = (r, gain, loss, rm, cm) if context_omics is None else context_omics
        f.update(C=clinical if context_clinical is None else context_clinical,
                 R=cr, G=cg, L=cl, RM=crm, CM=ccm)
        result = self.core(visual, mask, f, wsi_indices=wsi_indices, diagnostic=diagnostic)
        risk = main + result['risk']
        if return_extra:
            return risk, torch.as_tensor(result.get('extra_loss', 0.), device=risk.device, dtype=risk.dtype)
        if diagnostic:
            return risk, {'interaction_risk': result['risk'], 'main_risk': main,
                          **result.get('diagnostics', {})}
        return risk
