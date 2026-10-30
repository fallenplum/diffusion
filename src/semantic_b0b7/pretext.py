"""Outcome-free auxiliary losses and train-scope patient pairing.

Inputs contain no event indicators or follow-up times. Supervised survival
scorers and loaders are intentionally not imported by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .models import masked_softmax


@dataclass(frozen=True)
class PairMetadata:
    patient: np.ndarray
    stage: np.ndarray
    cohort: np.ndarray
    missing_pattern: np.ndarray

    def validate(self):
        n = len(self.patient)
        if n < 2 or any(np.asarray(x).shape != (n,) for x in
                        (self.patient, self.stage, self.cohort, self.missing_pattern)):
            raise ValueError("PAIR_METADATA_SHAPE_OR_TOO_SMALL")
        # Caller must construct metadata exclusively from its current train
        # partition; scope proof lives in the caller's immutable manifest.
        if any(np.asarray(x).dtype.kind == "O" for x in
               (self.patient, self.stage, self.cohort, self.missing_pattern)):
            raise ValueError("USE_TYPED_METADATA_NOT_OBJECT_ARRAYS")


def matched_derangement(metadata: PairMetadata, *, seed: int):
    """Train-only Hungarian wrong positives with explicitly counted fallbacks.

    Same Stage, missing pattern, and COAD/READ are preferred in that order;
    patient identity is an absolute exclusion. No patient group is silently
    dropped if an exact stratum cannot be deranged.
    """
    from scipy.optimize import linear_sum_assignment

    metadata.validate()
    n = len(metadata.patient)
    same_patient = metadata.patient[:, None] == metadata.patient[None, :]
    stage = metadata.stage[:, None] != metadata.stage[None, :]
    missing = metadata.missing_pattern[:, None] != metadata.missing_pattern[None, :]
    cohort = metadata.cohort[:, None] != metadata.cohort[None, :]
    rng = np.random.default_rng(seed)
    # Lexicographic at the assignment level: one higher-priority mismatch
    # costs more than all lower-priority mismatches across the partition.
    base = n + 1
    cost = stage * (base ** 2) + missing * base + cohort
    cost = cost.astype(float) + rng.random((n, n)) / (10 * n)
    cost[same_patient] = 1e15
    rows, cols = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(n)) or same_patient[rows, cols].any():
        raise ValueError("NO_VALID_WRONG_PATIENT_DERANGEMENT")
    return cols.astype(np.int64), {
        "anchors": n, "self_or_same_patient_pairs": 0,
        "same_stage": int((~stage[rows, cols]).sum()),
        "same_missing_pattern": int((~missing[rows, cols]).sum()),
        "same_cohort": int((~cohort[rows, cols]).sum()),
        "exact_all_three": int((~(stage | missing | cohort)[rows, cols]).sum()),
        "fallback_anchors": int((stage | missing | cohort)[rows, cols].sum()),
        "preference_order": ["stage", "missing_pattern", "cohort"],
        "scope_is_callers_current_train_partition_required": True,
    }


def negative_support(metadata: PairMetadata, *, mode: str = "matched"):
    """Return an allowed negative matrix and explicit per-anchor coverage.

    Matched candidates must satisfy all three grouping variables. Anchors
    without such candidates are excluded from that loss, not assigned a false
    negative. A separate random loss uses all different-patient candidates.
    """
    metadata.validate()
    allowed = metadata.patient[:, None] != metadata.patient[None, :]
    if mode == "matched":
        allowed &= metadata.stage[:, None] == metadata.stage[None, :]
        allowed &= metadata.cohort[:, None] == metadata.cohort[None, :]
        allowed &= metadata.missing_pattern[:, None] == metadata.missing_pattern[None, :]
    elif mode != "random":
        raise ValueError("UNKNOWN_NEGATIVE_SUPPORT_MODE")
    return allowed, {"mode": mode, "anchors": len(allowed),
                     "anchors_with_negative": int(allowed.any(1).sum()),
                     "candidate_negative_pairs": int(allowed.sum())}


def masked_program_loss(prediction: Tensor, target: Tensor, measured: Tensor,
                        *, delta: float = 1.):
    if prediction.shape != target.shape or measured.shape != target.shape or measured.dtype != torch.bool:
        raise ValueError("PROGRAM_TARGET_OR_MASK_SHAPE")
    if delta <= 0 or not torch.isfinite(prediction).all() or not torch.isfinite(target[measured]).all():
        raise ValueError("NONFINITE_PROGRAM_PREDICTION_OR_MEASUREMENT")
    clean = torch.where(measured, target, 0.)
    loss = F.huber_loss(prediction, clean, reduction="none", delta=delta)
    # Average measured programs within each patient, then equal patients.
    counts = measured.sum(-1)
    valid = counts > 0
    per_patient = (loss * measured).sum(-1) / counts.clamp_min(1)
    mean = per_patient[valid].mean() if valid.any() else prediction.sum() * 0.
    return {"loss": mean, "valid_patients": int(valid.sum().item()),
            "measured_cells": int(measured.sum().item())}


def paired_info_nce(scores: Tensor, score_valid: Tensor, negatives: Tensor,
                    *, positive: Tensor | None = None):
    """Bidirectional patient InfoNCE with arbitrary bijective positive pairing.

    B6a changes ONLY ``positive``. It must not alter measured program targets,
    image arrays, gene arrays, masks, or clinical covariates. ``negative``
    support explicitly excludes the assigned positive before each direction.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("PAIR_LOSS_EXPECTS_SQUARE_PATIENT_BANK")
    n = len(scores)
    if (score_valid.shape != scores.shape or negatives.shape != scores.shape
            or score_valid.dtype != torch.bool or negatives.dtype != torch.bool):
        raise ValueError("PAIR_MASK_SHAPE_OR_DTYPE")
    if not torch.isfinite(scores[score_valid]).all():
        raise ValueError("NONFINITE_VALID_PAIR_SCORE")
    if positive is None:
        positive = torch.arange(n, device=scores.device)
    if (positive.shape != (n,) or positive.dtype != torch.long
            or not torch.equal(torch.sort(positive).values,
                               torch.arange(n, device=scores.device))):
        raise ValueError("POSITIVE_ASSIGNMENT_MUST_BE_A_BIJECTION")
    rows = torch.arange(n, device=scores.device)
    positive_mask = torch.zeros_like(score_valid)
    positive_mask[rows, positive] = True
    negative_mask = negatives & ~positive_mask & score_valid
    support = negative_mask | (positive_mask & score_valid)
    valid_i = score_valid[rows, positive] & negative_mask.any(-1)
    inverse = torch.argsort(positive)
    valid_g = score_valid[inverse, rows] & negative_mask.any(0)
    clean = torch.where(score_valid, scores, 0.)
    # masked logsumexp without all-masked NaNs, including excluded anchors.
    masked = clean.masked_fill(~support, torch.finfo(scores.dtype).min)
    row = torch.logsumexp(masked, dim=1) - clean[rows, positive]
    col = torch.logsumexp(masked, dim=0) - clean[inverse, rows]
    pieces = []
    if valid_i.any():
        pieces.append(row[valid_i].mean())
    if valid_g.any():
        pieces.append(col[valid_g].mean())
    loss = torch.stack(pieces).mean() if pieces else clean.sum() * 0.
    probability = masked_softmax(clean, support, dim=-1)
    return {"loss": loss,
            "image_anchors_used": int(valid_i.sum().item()),
            "gene_anchors_used": int(valid_g.sum().item()),
            "positive_probability": probability[rows, positive],
            "positive_is_identity": bool(torch.equal(positive, rows)),
            "candidate_support": support}


def auxiliary_loss(outputs: dict, program_target: Tensor, program_measured: Tensor,
                   negatives: Tensor, *, positive: Tensor | None = None,
                   lambda_pair: float = .1):
    if lambda_pair < 0:
        raise ValueError("NEGATIVE_AUXILIARY_PAIR_WEIGHT")
    program = masked_program_loss(outputs["program_prediction"], program_target, program_measured)
    pair = paired_info_nce(outputs["pair"]["scores"], outputs["pair"]["valid"],
                           negatives, positive=positive)
    return {"loss": program["loss"] + lambda_pair * pair["loss"],
            "program": program, "pair": pair, "lambda_pair": lambda_pair}
