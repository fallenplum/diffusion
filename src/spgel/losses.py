"""Fixed pilot objectives and optional masked-gene utilities (disabled in pilot)."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .contracts import GeneBatch, require


def source_objective(model, image_outputs, target_z, target_mask, tissue, class_weights,
                     *, genes=None, negative_index=None, align_weight=.1, temperature=.1,
                     tissue_weight=.1):
    prediction = image_outputs["program_predictions"]
    require(prediction.shape == target_z.shape == target_mask.shape and prediction.shape[1] == 4
            and target_mask.dtype == torch.bool and torch.isfinite(target_z[target_mask]).all(),
            "SOURCE_OBJECTIVE_PROGRAM_SHAPES")
    support = target_mask.sum(0)
    require((support > 0).all(), "SOURCE_MINIBATCH_NO_PROGRAM_TARGET_SUPPORT")
    error = F.huber_loss(prediction, torch.where(target_mask, target_z, 0.), delta=1., reduction="none")
    prediction_loss = (torch.where(target_mask, error, 0.).sum(0) / support).mean()
    tissue_loss = prediction.sum() * 0.
    if model.has_slots:
        require(tissue.shape == (len(prediction),) and tissue.dtype == torch.long
                and class_weights.shape == (3,), "SOURCE_TISSUE_SUPERVISION_SHAPES")
        tissue_loss = F.cross_entropy(image_outputs["tissue_logits"], tissue, weight=class_weights, reduction="none").mean()
    contrast = prediction.sum() * 0.
    contrast_audit = dict(active_image_anchors=0, active_gene_anchors=0)
    if model.has_genes:
        require(genes is not None and negative_index is not None, "PAIRED_CONTRAST_REQUIRES_CURRENT_TRAIN_GENES")
        contrast, contrast_audit = bidirectional_contrastive_loss(model, image_outputs, genes, negative_index,
                                                                temperature=temperature)
    total = prediction_loss + tissue_weight * tissue_loss + align_weight * contrast
    return total, dict(prediction_Huber=prediction_loss, tissue_CE=tissue_loss, ITC=contrast,
        contrastive=contrast_audit, masked_gene_loss_enabled=False, matching_loss_enabled=False,
        spatial_smoothing_loss_enabled=False, RNA_enters_image_prediction=False,
        library_depth_is_model_input=False)


def bidirectional_contrastive_loss(model, image_outputs, genes, negative_index, *, temperature=.1):
    n, device = len(genes.values), genes.values.device
    require(negative_index.ndim == 2 and negative_index.shape[0] == n and negative_index.dtype == torch.long
            and negative_index.device == device and ((negative_index >= -1) & (negative_index < n)).all()
            and temperature > 0, "CURRENT_MINIBATCH_CONTRASTIVE_NEGATIVE_INDEX")
    valid = negative_index >= 0
    own = torch.arange(n, device=device)[:, None]
    require(not (valid & (negative_index == own)).any(), "CONTRASTIVE_SELF_IS_NEGATIVE")
    candidate = torch.cat((own, negative_index.clamp_min(0)), 1)
    allowed = torch.cat((torch.ones((n, 1), device=device, dtype=torch.bool), valid), 1)
    # Reverse direction is recomputed as score(image_candidate, gene_anchor).
    # The matrix is not a transpose of positively conditioned embeddings.
    image_anchor = own.expand_as(candidate)
    gene_base = model.gene_encoder(genes)
    a = model.paired_scores(image_outputs, genes, image_anchor, candidate, paired_mode=True, _gene_base=gene_base)
    b = model.paired_scores(image_outputs, genes, candidate, image_anchor, paired_mode=True, _gene_base=gene_base)
    losses, counts = [], []
    for result in (a, b):
        mask = allowed & result["valid"]
        active = mask[:, 0] & mask[:, 1:].any(1)
        counts.append(int(active.sum()))
        if active.any():
            score = (result["scores"] / temperature).masked_fill(~mask, -torch.inf)
            losses.append(F.cross_entropy(score[active], torch.zeros(int(active.sum()), device=device, dtype=torch.long)))
    loss = sum(losses) / len(losses) if losses else image_outputs["program_predictions"].sum() * 0.
    return loss, dict(active_image_anchors=counts[0], active_gene_anchors=counts[1],
        paired_mode=True, each_negative_pair_reconditions_both_rereads=model.bidirectional,
        no_dense_full_source_pair_matrix=True)


def fixed_gene_mask(observed, *, fraction=.15, seed=31):
    """One bank mask used by every overlapping program; not enabled in pilot."""
    require(observed.ndim == 2 and observed.dtype == torch.bool and .15 <= fraction <= .3,
            "MASKED_GENE_FRACTION_OR_MEASUREMENT_MASK")
    generator = torch.Generator(device=observed.device)
    generator.manual_seed(int(seed))
    return (torch.rand(observed.shape, generator=generator, device=observed.device) < fraction) & observed


def measured_masked_huber(prediction, target, original_observed, hidden):
    require(prediction.shape == target.shape == original_observed.shape == hidden.shape
            and original_observed.dtype == hidden.dtype == torch.bool
            and not (hidden & ~original_observed).any()
            and torch.isfinite(target[hidden]).all(), "MASKED_GENE_TARGET_NOT_REAL_MEASUREMENT")
    return F.huber_loss(prediction[hidden], target[hidden], delta=1.) if hidden.any() else prediction.sum() * 0.


def optional_pair_matching_bce(pair_logits, matched, valid):
    """Standalone pair-match objective; NOT_RUN and weight zero in pilot.

    The caller must construct source-training pairs. This function does not
    create pseudo tissue/program truth or treat near spatial spots as negatives.
    """
    require(pair_logits.shape == matched.shape == valid.shape and valid.dtype == torch.bool
            and torch.isfinite(pair_logits[valid]).all()
            and ((matched[valid] == 0) | (matched[valid] == 1)).all(),
            "OPTIONAL_PAIR_MATCH_LABEL_OR_SUPPORT")
    return F.binary_cross_entropy_with_logits(pair_logits[valid], matched[valid].to(pair_logits.dtype)) \
        if valid.any() else pair_logits.sum() * 0.


def optional_spatial_huber(program_values, edge_index, edge_weight, allowed_rows):
    """Standalone same-partition edge penalty; NOT_RUN in fixed pilot.

    Geometry validation (same section, true um radius) precedes this utility.
    It only accepts already-qualified edges and rejects a partition crossing.
    """
    require(program_values.ndim == 2 and edge_index.ndim == 2 and edge_index.shape[0] == 2
            and edge_index.dtype == torch.long and edge_weight.shape == (edge_index.shape[1],)
            and allowed_rows.shape == (len(program_values),) and allowed_rows.dtype == torch.bool
            and ((edge_index >= 0) & (edge_index < len(program_values))).all()
            and torch.isfinite(edge_weight).all() and (edge_weight >= 0).all(),
            "OPTIONAL_SPATIAL_EDGE_SHAPES")
    require(allowed_rows[edge_index].all(), "OPTIONAL_SPATIAL_EDGE_CROSSES_PARTITION")
    if not len(edge_weight) or not (edge_weight > 0).any():
        return program_values.sum() * 0.
    require(torch.isfinite(program_values[edge_index]).all(), "OPTIONAL_SPATIAL_NONFINITE_VALUES")
    errors = F.huber_loss(program_values[edge_index[0]], program_values[edge_index[1]],
                          delta=1., reduction="none").mean(-1)
    return (errors * edge_weight).sum() / edge_weight.sum()
