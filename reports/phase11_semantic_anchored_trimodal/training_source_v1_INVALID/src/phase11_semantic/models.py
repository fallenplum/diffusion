"""Small conditional-query network; no quadratic patch self-attention.

Registered budgets are common; inactive/zero-gradient elements are reported,
not disguised as equal effective capacity. ADAPTED/INSPIRED, never paper EXACT.
"""
import torch
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, d):
        super().__init__(); self.d = d
        self.norm = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3*d, bias=False)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2*d), nn.GELU(),
                                nn.Dropout(.1), nn.Linear(2*d, d))

    def forward(self, query, key_value, mask=None, return_attention=False):
        q = self.qkv(self.norm(query))[..., :self.d]
        _, k, v = self.qkv(self.norm(key_value)).chunk(3, -1)
        score = q @ k.transpose(-1, -2) / self.d**.5
        if mask is not None:
            if not mask.any(-1).all(): raise ValueError('all keys masked')
            score = score.masked_fill(~mask[:, None, :], -torch.inf)
        weights = score.softmax(-1)
        h = query + weights @ v; h = h + self.ff(h)
        return (h, weights) if return_attention else h

    def tokenwise(self, x):
        q, k, v = self.qkv(self.norm(x)).chunk(3, -1)
        h = x + (q+k+v)/3
        return h + self.ff(h)


class ConceptTokenizer(nn.Module):
    def __init__(self, d):
        super().__init__(); self.scalar = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.identity = nn.Parameter(torch.randn(32, d)*.02); self.norm = nn.LayerNorm(d)
    def forward(self, scores): return self.norm(self.scalar(scores[..., None]) + self.identity)


class RNATokenizer(nn.Module):
    def __init__(self, d, membership):
        super().__init__(); self.register_buffer('membership', torch.tensor(membership))
        self.weight = nn.Parameter(torch.randn(8, 512, d)/8)
        self.bias = nn.Parameter(torch.zeros(8, d)); self.gate = nn.Parameter(torch.zeros(8, 512))
        self.patient_gate = nn.Parameter(torch.ones(8, 1)*.1)
        self.norm = nn.LayerNorm(d); self.inter = CrossAttention(d)
    def forward(self, r, mode='msa'):
        if mode == 'soft':
            gates = (self.gate[None, :, :] + self.patient_gate[None, :, :]*r[:, None, :]).softmax(-1)*512
            raw = torch.einsum('bg,bkg,kgd->bkd', r, gates, self.weight)/8 + self.bias
        else:
            raw = torch.einsum('bg,kg,kgd->bkd', r, self.membership, self.weight) / 8 + self.bias
        raw = self.norm(torch.nn.functional.elu(raw))
        tokens = self.inter.tokenwise(raw) if mode == 'tokenwise' else self.inter(raw, raw)
        return raw, tokens


def masked_mean(x, mask): return (x*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)


class SemanticModel(nn.Module):
    def __init__(self, arm, clinical_mask, rna_membership, d=64):
        super().__init__(); self.arm = arm; self.d = d
        clinical_mask = torch.tensor(clinical_mask); dc = clinical_mask.shape[1]
        self.register_buffer('clinical_mask', clinical_mask)
        self.clinical_weight = nn.Parameter(torch.randn(len(clinical_mask), dc, d)/dc**.5)
        self.clinical_bias = nn.Parameter(torch.zeros(len(clinical_mask), d))
        self.concept = ConceptTokenizer(d)
        self.patch_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d), nn.GELU(),
                                              nn.Linear(d, d), nn.LayerNorm(d))
        self.rna = RNATokenizer(d, rna_membership)
        self.cp = CrossAttention(d); self.rp = CrossAttention(d); self.cr = CrossAttention(d)
        self.unguided = nn.Parameter(torch.randn(32, d)*.02)
        self.head = nn.Sequential(nn.Linear(5*d, d), nn.GELU(), nn.Dropout(.1), nn.Linear(d, 1))
        self.anchor = nn.Linear(dc+32, 1, bias=False)
        self.aux = nn.Linear(d, 1)

    def forward(self, c, s, r, patch, mask, details=False):
        a = self.arm; bs, slides, n, _ = patch.shape
        sv = mask.any(-1); valid = sv.reshape(-1).nonzero().squeeze(1)
        if not sv.any(-1).all(): raise ValueError('patient without real slide')
        owner = torch.div(valid, slides, rounding_mode='floor')
        p = patch.reshape(bs*slides, n, 768)[valid]; pm = mask.reshape(bs*slides, n)[valid]
        # Exclude padding before projection too, even if padded input is extreme.
        p = self.patch_projection(p.masked_fill(~pm[..., None], 0))
        ct = torch.einsum('bc,kc,kcd->bkd', c, self.clinical_mask, self.clinical_weight) + self.clinical_bias
        st = self.concept(s)
        no_rna = a in ['A5', 'A6', 'B6', 'D3']
        mode = 'tokenwise' if a in ['C1', 'C5'] else ('soft' if a == 'C2' else 'msa')
        if no_rna and a != 'D3':
            raw = rt = torch.zeros(bs, 8, self.d, device=c.device)
        else: raw, rt = self.rna(r, mode)
        late_semantic = a in ['B5', 'D1']
        queries = torch.cat([ct, self.unguided.expand(bs, -1, -1) if late_semantic else st], 1)
        if a == 'A5':
            queries = self.unguided.expand(bs, -1, -1)
        hp, att = self.cp(queries[owner], p, pm, True)
        if a == 'B2':
            # Clinical queries select RNA; RNA token outputs receive this context.
            selected = self.cr(ct, rt).mean(1, keepdim=True)
            rt = rt + selected
        if no_rna:
            ri = torch.zeros(len(valid), self.d, device=c.device)
        elif a == 'A7': ri = self.rp.tokenwise(rt[owner]).mean(1)
        elif a == 'B1': ri = self.rp(rt, st).mean(1)[owner]
        elif a == 'B3': ri = masked_mean(self.rp(p, rt[owner]), pm)
        elif a == 'B4':
            ri = (self.rp(rt[owner], p, pm).mean(1) + masked_mean(self.rp(p, rt[owner]), pm))/2
        else: ri = self.rp(rt[owner], p, pm).mean(1)
        pp = masked_mean(p, pm)
        slots = torch.cat([ct.mean(1)[owner], st.mean(1)[owner], hp.mean(1), ri, pp], 1)
        joined = torch.zeros(bs, 5*self.d, device=c.device).index_add(0, owner, slots)/sv.sum(1, keepdim=True)
        anchor = self.anchor(torch.cat([c, s], 1)).squeeze(1)
        correction = self.head(joined).squeeze(1)*.1
        risk = anchor + correction
        if not details: return risk
        representations = torch.cat([raw.mean(1), rt.mean(1)], 1)
        aux = self.aux(rt.mean(1)).squeeze(1)
        entropy = -(att.clamp_min(1e-30).log()*att).sum(-1).mean()
        return risk, representations, aux, correction, entropy
