"""Frozen-kernel dual-evidence feasibility extractor; not end-to-end CDEI.

New rich-token/CP weights are required. The historical P16/S16 logits are not
the original Phase19 CP law. No historical implementation is modified here.
"""
from __future__ import annotations
import hashlib
import math
import torch
from torch import nn
from src.phase19_attention.attention import RankConditionedAttention
from src.phase19_attention.model import masked_mean
from src.phase20_selective.model import SignedPathwayTokens


def module_fingerprint(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        x = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
        h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class CompatibilityBackbone(nn.Module):
    """Phase20 rich tokens and the unchanged Phase19 CP arithmetic."""
    def __init__(self, clinical_dim, membership, hidden=32, rank=4):
        super().__init__()
        if hidden != 32 or rank != 4 or clinical_dim < 1:
            raise ValueError("qualification fixes hidden32/rank4 and positive clinical dimension")
        self.clinical_dim = clinical_dim
        self.visual = nn.Sequential(nn.Linear(1536, hidden), nn.LayerNorm(hidden))
        self.molecular = SignedPathwayTokens(membership, hidden)
        self.compatibility = RankConditionedAttention(hidden, hidden, clinical_dim + 1, d=hidden, rank=rank)
        self.p_value, self.o_value = nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, hidden, bias=False)

    def encode_pathology(self, visual, mask):
        if visual.ndim != 3 or visual.shape[-1] != 1536 or mask.shape != visual.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("invalid pathology feature/mask contract")
        if not torch.isfinite(visual[mask]).all():
            raise ValueError("nonfinite observed pathology feature")
        p = self.visual(torch.where(mask[..., None], visual, torch.zeros_like(visual)))
        return torch.where(mask[..., None], p, torch.zeros_like(p))

    def encode_omics(self, r, gain, loss, rm, cm):
        if r.ndim != 2 or any(x.shape != r.shape for x in (gain, loss, rm, cm)):
            raise ValueError("RNA/CNA/mask shapes differ")
        for x, m in ((r, rm), (gain, cm), (loss, cm)):
            if m.dtype != torch.bool or not torch.isfinite(x[m]).all():
                raise ValueError("omics needs Boolean availability and finite observed values")
        return self.molecular(r, gain, loss, rm, cm)

    def score_encoded(self, p, o, clinical, patch_mask, omics_mask):
        if clinical.shape != (len(p), self.clinical_dim) or not torch.isfinite(clinical).all():
            raise ValueError("invalid clinical context")
        augmented = torch.cat((torch.ones_like(clinical[:, :1]), clinical), -1)
        return self.compatibility(p, o, augmented, patch_mask, omics_mask)

    def relative_encoded(self, p, o, clinical, patch_mask, omics_mask):
        scored = self.score_encoded(p, o, clinical, patch_mask, omics_mask)
        relative = torch.cat((masked_mean(scored["p_to_o"] @ self.o_value(o), patch_mask),
                              masked_mean(scored["o_to_p"] @ self.p_value(p), omics_mask)), -1)
        return relative, scored

    def forward(self, visual, mask, clinical, r, gain, loss, rm, cm):
        p = self.encode_pathology(visual, mask)
        o, om = self.encode_omics(r, gain, loss, rm, cm)
        return self.relative_encoded(p, o, clinical, mask, om)


class FrozenDualEvidence(nn.Module):
    """Context interventions change Absolute only; Relative stays matched.

    Reference query indices identify the omics source. Paired interventions
    must use a common exclusion union, so a training comparison does not also
    change the reference-bank composition.
    Equal-WSI patient aggregation belongs to the calling data pipeline.
    """
    def __init__(self, backbone, reference, *, tau=1.0):
        super().__init__()
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("temperature must be finite and positive")
        if any(p.requires_grad for p in backbone.parameters()) or backbone.training:
            raise ValueError("freeze every encoder/kernel and use eval before reference calibration")
        self.backbone, self.reference, self.tau = backbone, reference, float(tau)
        self._kernel_fingerprint = module_fingerprint(backbone)
        if getattr(reference, "provenance", {}).get("kernel_sha256") != self._kernel_fingerprint:
            raise ValueError("reference has a different or unspecified frozen kernel")
        self.eval()

    def train(self, mode=True):
        if mode:
            raise ValueError("frozen evidence extractor cannot enter training mode")
        return super().train(False)

    def verify_frozen(self):
        if (self.backbone.training or any(p.requires_grad for p in self.backbone.parameters())
                or module_fingerprint(self.backbone) != self._kernel_fingerprint):
            raise ValueError("frozen compatibility kernel drift")

    @torch.no_grad()
    def extract(self, visual, mask, clinical, r, gain, loss, rm, cm, patient_indices,
                *, absolute_clinical=None, absolute_omics=None, omics_source_indices=None,
                reference_exclusions=None, absolute_token_mask=None):
        from .operators import dual_evidence_core, masked_mean_std_pool
        self.verify_frozen()
        p = self.backbone.encode_pathology(visual, mask)
        o, om = self.backbone.encode_omics(r, gain, loss, rm, cm)
        relative, matched = self.backbone.relative_encoded(p, o, clinical, mask, om)
        context_o, context_mask = ((o, om) if absolute_omics is None else
                                   self.backbone.encode_omics(*absolute_omics))
        if absolute_omics is not None and omics_source_indices is None:
            raise ValueError("explicit omics source indices required for an intervention")
        sources = patient_indices if omics_source_indices is None else omics_source_indices
        if absolute_omics is not None:
            if reference_exclusions is None:
                raise ValueError("wrong-omics comparisons need a shared reference exclusion plan")
            extra = torch.as_tensor(reference_exclusions)
            own = torch.as_tensor(patient_indices).to(extra.device)
            donor = torch.as_tensor(sources).to(extra.device)
            if (extra.ndim != 2 or len(extra) != len(p) or own.shape != (len(p),)
                    or donor.shape != own.shape
                    or not (extra == own[:, None]).any(-1).all()
                    or not (extra == donor[:, None]).any(-1).all()):
                raise ValueError("paired reference plan must include both recipient and omics source")
            expected_mask = om & context_mask
            if (absolute_token_mask is None or absolute_token_mask.dtype != torch.bool
                    or absolute_token_mask.shape != expected_mask.shape
                    or not torch.equal(absolute_token_mask.to(expected_mask.device), expected_mask)):
                raise ValueError("wrong-omics comparison needs its shared observed-slot intersection")
        ref = self.reference.query_reference_mean(sources, additional_exclusions=reference_exclusions)
        cc = clinical if absolute_clinical is None else absolute_clinical
        context = self.backbone.score_encoded(p, context_o, cc, mask, context_mask)
        null = self.backbone.score_encoded(p, ref.tokens.to(p), cc, mask, ref.mask.to(mask.device))
        available = context["valid_pairs"] & null["valid_pairs"]
        if absolute_token_mask is not None:
            if absolute_token_mask.dtype != torch.bool or absolute_token_mask.shape != context_mask.shape:
                raise ValueError("invalid shared absolute token mask")
            available = available & absolute_token_mask.to(available.device)[:, None, :]
        absolute = dual_evidence_core(context["scores"], null["scores"], available, self.tau)
        return {"relative": relative.detach(),
                "absolute": masked_mean_std_pool(absolute["evidence"], absolute["has_valid"]),
                "mean_only": masked_mean_std_pool(absolute["residual_mean"], absolute["has_valid"]),
                "centered_only": masked_mean_std_pool(absolute["centered_lme"], absolute["has_valid"]),
                "absolute_available": absolute["has_valid"].any(-1),
                "relative_p_to_o": matched["p_to_o"].detach(),
                "relative_o_to_p": matched["o_to_p"].detach(),
                "reference_counts": ref.counts.detach()}
