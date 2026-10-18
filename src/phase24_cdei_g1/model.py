"""Joint current-kernel/local-evidence models for the locked CDEI-G1 audit.

The reference contains raw FIT-patient omics, never encoded stale tokens or
scores. Each call re-encodes it with the same trainable encoder used for the
recipient and propagates gradients through the patient-equal reference mean.
Reference means are computational controls, not medical no-association labels.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from src.phase19_attention.model import masked_mean
from src.phase23_cdei.model import CompatibilityBackbone


ARMS = ("REL", "POOL", "LOCAL", "CTRL")
_RAW_KEYS = frozenset(("R", "G", "L", "RM", "CM", "indices"))


def _indices(values, *, device, length, name, matrix=False):
    out = torch.as_tensor(values, device=device)
    if out.dtype == torch.bool or out.dtype.is_floating_point or out.dtype.is_complex:
        raise ValueError(f"{name} must contain integer patient indices")
    expected_ndim = 2 if matrix else 1
    if out.ndim != expected_ndim or out.shape[0] != length:
        raise ValueError(f"invalid {name} dimensions")
    return out.to(dtype=torch.long)


def _masked_scalar_mean_std(values, valid):
    """Patch mean/population SD, including a finite derivative at zero SD."""
    x = torch.where(valid, values, torch.zeros_like(values))
    n = valid.sum(-1).clamp_min(1).to(x.dtype)
    mean = x.sum(-1) / n
    centered = torch.where(valid, x - mean[:, None], torch.zeros_like(x))
    variance = centered.square().sum(-1) / n
    positive = variance > 0
    # Taking sqrt(0) before torch.where would still leave an infinite backward.
    sd = torch.where(positive, torch.where(positive, variance, torch.ones_like(variance)).sqrt(),
                     torch.zeros_like(variance))
    return torch.stack((mean, sd), -1)


class G1Model(nn.Module):
    """Relative, pooled-absolute, local-absolute and capacity-matched control.

    Interventions alter only the extra branch. Matched attention ``A`` and the
    relative 64-vector always use the recipient's original P/O/C. LOCAL uses
    signed ``A*tanh(D/tau)`` weights without a second softmax or normalization.
    Its CTRL uses ``A*sigmoid(s_context/tau)`` and no null subtraction. Neither
    branch rescales the resulting message to unit length.

    This model returns a bounded *unit* risk correction. The caller adds the
    unchanged frozen main-effect risk and its fit-only risk scale once per
    patient after equal-slide aggregation.
    """
    def __init__(self, clinical_dim, membership, arm="LOCAL", tau=1.0):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unknown CDEI-G1 arm")
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("temperature must be finite and positive")
        self.arm, self.tau = arm, float(tau)
        self.backbone = CompatibilityBackbone(clinical_dim, membership, hidden=32, rank=4)
        # The same RNG state creates every common parameter for all four arms.
        self.relative_linear = nn.Linear(64, 16)
        self.output = nn.Linear(16, 1, bias=False)
        self.extra_dim = {"REL": 0, "POOL": 2, "LOCAL": 32, "CTRL": 32}[arm]
        # Preserve the post-common initialization RNG for common dropout draws.
        # LOCAL/CTRL have exactly identical extra initialization and budget.
        with torch.random.fork_rng(devices=[]):
            self.extra_linear = nn.Linear(self.extra_dim, 16, bias=False) if self.extra_dim else None
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def _validate_raw_reference(self, reference_raw, device):
        if not isinstance(reference_raw, dict) or set(reference_raw) != _RAW_KEYS:
            raise ValueError("reference must contain only raw fit R/G/L/RM/CM and indices")
        data = {k: torch.as_tensor(reference_raw[k], device=device) for k in ("R", "G", "L", "RM", "CM")}
        genes = self.backbone.molecular.membership.shape[1]
        shape = data["R"].shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] != genes or any(x.shape != shape for x in data.values()):
            raise ValueError("malformed fit-only raw reference shapes")
        ids = _indices(reference_raw["indices"], device=device, length=shape[0], name="reference indices")
        if ids.unique().numel() != ids.numel():
            raise ValueError("reference must contain one row per unique fit patient")
        return data, ids

    def current_reference(self, reference_raw, patient_indices, *, additional_exclusions=None):
        """Recompute query-specific, patient-equal, slot-masked current means.

        Global patient indices, rather than bank offsets, identify exclusions.
        Every FIT query excludes itself; heldout queries are absent from the
        bank and therefore use all FIT patients unless extra exclusions apply.
        Duplicate exclusions count once. No detach/cached scores are used.
        """
        device = next(self.parameters()).device
        raw, fit_ids = self._validate_raw_reference(reference_raw, device)
        query = torch.as_tensor(patient_indices, device=device)
        query = _indices(query, device=device, length=len(query), name="query indices")
        tokens, token_mask = self.backbone.encode_omics(*(raw[k] for k in ("R", "G", "L", "RM", "CM")))
        allow = fit_ids[None, :] != query[:, None]
        if additional_exclusions is not None:
            extra = _indices(additional_exclusions, device=device, length=len(query),
                             name="reference exclusions", matrix=True)
            allow = allow & ~(fit_ids[None, None, :] == extra[:, :, None]).any(1)
        # A row is one patient, irrespective of WSI count. Slot-specific counts
        # prevent a missing gene program being treated as observed zero.
        weights = allow.to(tokens.dtype)
        counts = weights @ token_mask.to(tokens.dtype)
        sums = torch.einsum("bf,fkd->bkd", weights, tokens)
        available = counts > 0
        means = sums / counts.clamp_min(1)[..., None]
        means = torch.where(available[..., None], means, torch.zeros_like(means))
        return {"tokens": means, "mask": available, "counts": counts.to(torch.long),
                "encoded_fit_tokens": tokens, "encoded_fit_mask": token_mask,
                "fit_indices": fit_ids}

    def forward(self, visual, mask, clinical, r, g, l, rm, cm, patient_indices, reference_raw,
                *, absolute_clinical=None, absolute_omics=None, omics_source_indices=None,
                reference_exclusions=None, absolute_token_mask=None, absolute_off=False,
                return_details=False):
        genes = self.backbone.molecular.membership.shape[1]
        if r.ndim != 2 or r.shape[1] != genes:
            raise ValueError("omics gene width differs from the locked membership")
        own = _indices(patient_indices, device=visual.device, length=len(visual), name="patient indices")
        p = self.backbone.encode_pathology(visual, mask)
        o, om = self.backbone.encode_omics(r, g, l, rm, cm)
        relative, matched = self.backbone.relative_encoded(p, o, clinical, mask, om)
        extra = relative.new_zeros((len(relative), self.extra_dim))
        details = {"scores": matched["scores"], "A": matched["p_to_o"], "scored": matched,
                   "relative_p_to_o": matched["p_to_o"],
                   "relative_o_to_p": matched["o_to_p"],
                   "relative_valid_pairs": matched["valid_pairs"]}
        if self.extra_dim:
            cc = clinical if absolute_clinical is None else absolute_clinical
            context_o, context_mask = ((o, om) if absolute_omics is None else
                                      self.backbone.encode_omics(*absolute_omics))
            if omics_source_indices is not None:
                sources = _indices(omics_source_indices, device=visual.device, length=len(own),
                                   name="omics source indices")
                if absolute_omics is None and not torch.equal(sources, own):
                    raise ValueError("different source indices require explicit omics intervention")
            else:
                sources = own
            if absolute_omics is not None:
                if omics_source_indices is None or reference_exclusions is None:
                    raise ValueError("wrong-omics intervention needs source indices and shared exclusion plan")
                exclusions = _indices(reference_exclusions, device=visual.device, length=len(own),
                                      name="reference exclusions", matrix=True)
                if not ((exclusions == own[:, None]).any(-1).all()
                        and (exclusions == sources[:, None]).any(-1).all()):
                    raise ValueError("reference exclusion union must include recipient and omics donor")
                intersection = om & context_mask
                if (absolute_token_mask is None or absolute_token_mask.dtype != torch.bool
                        or absolute_token_mask.shape != intersection.shape
                        or not torch.equal(absolute_token_mask.to(intersection.device), intersection)):
                    raise ValueError("wrong-omics intervention needs the shared observed-slot intersection")
            if absolute_token_mask is not None:
                if absolute_token_mask.dtype != torch.bool or absolute_token_mask.shape != om.shape:
                    raise ValueError("invalid absolute token mask")
                if (absolute_token_mask.to(om.device) & ~(om & context_mask)).any():
                    raise ValueError("absolute mask cannot activate unobserved recipient/context slots")
            ref = self.current_reference(reference_raw, own, additional_exclusions=reference_exclusions)
            context = self.backbone.score_encoded(p, context_o, cc, mask, context_mask)
            null = self.backbone.score_encoded(p, ref["tokens"], cc, mask, ref["mask"])
            eligible = matched["valid_pairs"] & context["valid_pairs"]
            if absolute_token_mask is not None:
                eligible = eligible & absolute_token_mask.to(om.device)[:, None, :]
            if self.arm != "CTRL":
                eligible = eligible & null["valid_pairs"]
            raw_difference = context["scores"] - null["scores"]
            difference = torch.where(eligible, raw_difference, torch.zeros_like(raw_difference))
            if not torch.isfinite(difference[eligible]).all():
                raise ValueError("nonfinite local compatibility residual")
            signed = torch.where(eligible, torch.tanh(difference / self.tau), torch.zeros_like(difference))
            if self.arm == "POOL":
                scaled = difference.double() / self.tau
                if not torch.isfinite(scaled[eligible]).all():
                    raise ValueError("nonfinite temperature-scaled compatibility residual")
                row_valid = eligible.any(-1)
                logits = scaled.masked_fill(~eligible, -torch.inf)
                logits = torch.where(row_valid[..., None], logits, torch.zeros_like(logits))
                count = eligible.sum(-1).clamp_min(1).to(scaled.dtype)
                evidence = self.tau * (torch.logsumexp(logits, -1) - count.log())
                evidence = torch.where(row_valid, evidence, torch.zeros_like(evidence))
                extra = _masked_scalar_mean_std(evidence, row_valid).to(relative.dtype)
                weighted = matched["p_to_o"] * signed
                messages = weighted @ self.backbone.o_value(context_o)
                details.update(evidence=evidence, evidence_valid=row_valid)
            else:
                modulation = signed if self.arm == "LOCAL" else torch.where(
                    eligible, torch.sigmoid(context["scores"] / self.tau), torch.zeros_like(difference))
                weighted = matched["p_to_o"] * modulation
                messages = weighted @ self.backbone.o_value(context_o)
                extra = masked_mean(messages, mask)
            details.update(context_scores=context["scores"], null=null["scores"], D=difference,
                           signed_modulation=signed, absolute_valid_pairs=eligible,
                           weighted_weights=weighted, weightedmessages=messages, local_patch=messages,
                           reference_counts=ref["counts"], reference_tokens=ref["tokens"],
                           reference_mask=ref["mask"], context_omics_tokens=context_o,
                           context_omics_values=self.backbone.o_value(context_o))
            if absolute_off:
                extra = torch.zeros_like(extra)
        elif any(x is not None for x in (absolute_clinical, absolute_omics, omics_source_indices,
                                        reference_exclusions, absolute_token_mask)) or absolute_off:
            raise ValueError("REL has no absolute branch to intervene on")
        hidden = self.relative_linear(relative)
        if self.extra_linear is not None:
            hidden = hidden + self.extra_linear(extra)
        correction = torch.tanh(self.output(self.dropout(self.activation(hidden))).squeeze(-1))
        result = {"risk_delta_unit": correction, "relative": relative, "extra": extra}
        if return_details:
            details["extra"] = extra
            result["details"] = details
        return result
