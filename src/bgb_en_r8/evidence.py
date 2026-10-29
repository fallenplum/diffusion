"""Outcome-free source evidence reads with equal weighting of nonempty WSIs.

The frozen source visual forward has no molecular input. Molecular context
changes only read weights; values, patch masks and source predictions remain
the recipient's original visual evidence. No patient cohort is loaded here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import torch
from torch.nn import functional as F


def require(value, code):
    if not bool(value):
        raise ValueError(code)


@dataclass(frozen=True)
class FrozenWSIEvidence:
    states: torch.Tensor
    program_keys: torch.Tensor
    program_scores: torch.Tensor
    patch_mask: torch.Tensor

    def validate(self):
        h, q, s, m = self.states, self.program_keys, self.program_scores, self.patch_mask
        require(h.ndim == 4 and h.shape[-1] == 64 and min(h.shape[:3]) > 0
                and h.dtype == torch.float32, "R8_SOURCE_STATES_REQUIRE_B_S_N_64_FLOAT32")
        require(q.shape == (*h.shape[:3], 4, 64) and s.shape == (*h.shape[:3], 4)
                and m.shape == h.shape[:3] and m.dtype == torch.bool
                and q.dtype == s.dtype == h.dtype
                and q.device == s.device == m.device == h.device,
                "R8_SOURCE_EVIDENCE_SHAPE_DTYPE_OR_DEVICE")
        require(torch.isfinite(h[m]).all() and torch.isfinite(q[m]).all()
                and torch.isfinite(s[m]).all(), "R8_NONFINITE_VALID_VISUAL_EVIDENCE")
        require(m.any((1, 2)).all(), "R8_PATIENT_HAS_NO_VALID_VISUAL_EVIDENCE")
        require(not any(x.requires_grad for x in (h, q, s)), "R8_EVIDENCE_MUST_BE_FROZEN")
        return self

    @classmethod
    def from_encoded_bank(cls, encoded):
        require("program_key_prior" in encoded, "R8_CACHE_KEYS_MUST_BE_RECONSTRUCTED_FROM_FROZEN_SOURCE")
        return cls(encoded["states"], encoded["program_key_prior"],
                   encoded["program_score_prior"], encoded["patch_mask"]).validate()


def decode_frozen_source_cache(source_model, encoded, *, expected_source_state_sha256,
                               replay_tolerance=2e-6, batch_size=1024):
    """Reconstruct omitted source keys; independently replay the saved scores.

    R6 READY caches contain only h, source scores and masks. A learned target
    projector must not be invented to fill the missing key field. Recompute
    the source tissue/query heads on h, and require its score to equal the
    original source prediction before treating the key as source-aligned.
    """
    from src.bgb_en.r6_source_model import SourceProgramModel
    from src.bgb_en.r6_pathology import source_model_state_sha256

    require(isinstance(source_model, SourceProgramModel) and source_model.arm == "S7"
            and source_model.width == 64 and source_model.input_dim == 1536
            and not source_model.training and not any(p.requires_grad for p in source_model.parameters()),
            "R8_COMPLETE_SOURCE_MUST_BE_S7_FROZEN_EVAL")
    require(isinstance(expected_source_state_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected_source_state_sha256)
            and source_model_state_sha256(source_model) == expected_source_state_sha256,
            "R8_SOURCE_STATE_SHA256_CHANGED")
    require(type(batch_size) is int and batch_size > 0 and 0 < replay_tolerance <= 2e-6,
            "R8_SOURCE_DECODE_BUDGET_OR_TOLERANCE_INVALID")
    h, scores, m = encoded["states"], encoded["program_score_prior"], encoded["patch_mask"]
    require(h.ndim == 4 and h.shape[-1] == 64 and scores.shape == (*h.shape[:3], 4)
            and m.shape == h.shape[:3] and m.dtype == torch.bool
            and h.dtype == scores.dtype == torch.float32 and h.device == scores.device == m.device
            and torch.isfinite(h[m]).all() and torch.isfinite(scores[m]).all(),
            "R8_FROZEN_CACHE_SHAPE_OR_VALID_VALUES")
    device = next(source_model.parameters()).device
    require(h.device == device, "R8_SOURCE_CACHE_MODEL_DEVICE_MISMATCH")
    flat = h.reshape(-1, 64); positions = m.flatten().nonzero(as_tuple=False).flatten()
    keys = h.new_zeros((*h.shape[:3], 4, 64)); flat_keys = keys.reshape(-1, 4, 64)
    replayed = scores.new_zeros(scores.shape); flat_replayed = replayed.reshape(-1, 4)
    with torch.no_grad():
        for ids in positions.split(batch_size):
            z = flat[ids]
            q = source_model.queries(source_model.tissue_head(z).softmax(-1))
            flat_keys[ids] = q
            flat_replayed[ids] = torch.einsum("nd,nkd->nk", z, q)/math.sqrt(64)+source_model.program_bias
    error = float((replayed[m]-scores[m]).abs().max()) if m.any() else 0.
    require(error <= replay_tolerance, "R8_CACHED_SOURCE_PREDICTIONS_DO_NOT_REPLAY")
    if "program_key_prior" in encoded:
        require(encoded["program_key_prior"].shape == keys.shape
                and float((encoded["program_key_prior"][m]-keys[m]).abs().max()) <= replay_tolerance,
                "R8_SUPPLIED_SOURCE_KEYS_DO_NOT_MATCH_FROZEN_SOURCE")
    result = FrozenWSIEvidence(h, keys, scores, m).validate()
    return result, dict(status="PASS_FROZEN_SOURCE_HEAD_SCORE_REPLAY", valid_patches=int(m.sum()),
        score_max_abs_error=error, tolerance=replay_tolerance,
        source_state_sha256=expected_source_state_sha256,
        missing_keys_reconstructed_by="same_frozen_source_tissue_head_and_queries",
        raw_1536_features_reencoded=False, target_parameters_learned=0,
        target_or_source_outcome_labels_used=False, new_optimizer_fits=0,
        source_visual_input_uses_RNA=False)


def _masked_softmax(scores, valid):
    present = valid.any(-1, keepdim=True)
    safe = torch.where(present, scores.masked_fill(~valid, -torch.inf), torch.zeros_like(scores))
    return torch.softmax(safe, -1) * valid


def pool_frozen_evidence(evidence: FrozenWSIEvidence, *, query=None, query_mask=None,
                         source_prior_scale: float, cosine_temperature: float):
    """Return four scores, four 64D summaries and slide-local attention.

    Without molecular context the only logits are fixed source predictions.
    With context, cosine compatibility uses the *source-aligned* visual
    program embedding normalize(h + q_source), and normalize(g_source).
    A missing molecular query exactly falls back to the unconditional read.
    All temperatures/scales must be supplied by a pre-registered caller.
    """
    evidence.validate()
    require(math.isfinite(source_prior_scale) and source_prior_scale >= 0
            and math.isfinite(cosine_temperature) and cosine_temperature > 0,
            "R8_EXPLICIT_FINITE_POOLING_SCALE_REQUIRED")
    require((query is None) == (query_mask is None), "R8_QUERY_REQUIRES_ITS_EXPLICIT_MASK")
    h, source_q, scores, m = (evidence.states, evidence.program_keys,
                             evidence.program_scores, evidence.patch_mask)
    b, slides, n, width = h.shape
    safe_h = torch.where(m[..., None], h, torch.zeros_like(h))
    safe_q = torch.where(m[..., None, None], source_q, torch.zeros_like(source_q))
    safe_scores = torch.where(m[..., None], scores, torch.zeros_like(scores))
    logits = source_prior_scale * safe_scores.permute(0, 1, 3, 2)
    if query is not None:
        require(query.shape == (b, 4, width) and query_mask.shape == (b, 4)
                and query.dtype == h.dtype and query_mask.dtype == torch.bool
                and query.device == query_mask.device == h.device
                and torch.isfinite(query[query_mask]).all()
                and not query.requires_grad, "R8_FROZEN_PROGRAM_QUERY_SHAPE_OR_VALUES")
        v = F.normalize(safe_h[..., None, :] + safe_q, dim=-1, eps=1e-12)
        g = F.normalize(torch.where(query_mask[..., None], query, torch.zeros_like(query)), dim=-1, eps=1e-12)
        compatibility = (v * g[:, None, None]).sum(-1).permute(0, 1, 3, 2)
        logits = logits + compatibility * query_mask[:, None, :, None] / cosine_temperature
    valid = m[:, :, None, :].expand(-1, -1, 4, -1)
    attention = _masked_softmax(logits, valid)
    slide_values = torch.einsum("bskn,bsnd->bskd", attention, safe_h)
    slide_scores = torch.einsum("bskn,bsnk->bsk", attention, safe_scores)
    slide_valid = m.any(-1)
    count = slide_valid.sum(1).clamp_min(1)
    values = (slide_values * slide_valid[..., None, None]).sum(1) / count[:, None, None]
    pooled_scores = (slide_scores * slide_valid[..., None]).sum(1) / count[:, None]
    log_prob = torch.where(attention > 0, attention.clamp_min(1e-30).log(), torch.zeros_like(attention))
    entropy = -(attention * log_prob).sum(-1)
    denominator = m.sum(-1).clamp_min(2).to(h.dtype).log()
    normalized_entropy = entropy / denominator[..., None]
    normalized_entropy = torch.where(slide_valid[..., None], normalized_entropy, torch.zeros_like(normalized_entropy))
    proxy = (normalized_entropy * slide_valid[..., None]).sum(1) / count[:, None]
    require(torch.isfinite(values).all() and torch.isfinite(pooled_scores).all(), "R8_NONFINITE_FROZEN_POOL")
    return dict(image_program_scores=pooled_scores, image_program_embeddings=values,
                attention=attention, valid_slide_count=count, uncertainty_proxy=proxy,
                diagnostics=dict(aggregation="softmax_inside_each_slide_then_equal_nonempty_slides",
                    source_values_unchanged=True, source_patch_mask_unchanged=True,
                    source_visual_forward_uses_RNA=False, query_changes_weights_only=query is not None,
                    proxy_is_calibrated_medical_confidence=False,
                    raw_per_patient_outputs_must_remain_private=True))
