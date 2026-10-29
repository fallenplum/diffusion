"""S5 shared main targets and candidate-specific, support-matched paired losses."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from src.spgel.contracts import require


def paired_loss(model, outputs, genes, negative_image, negative_gene, active, *, wrong=False):
    """Both anchor directions stay present in every structural ablation.

    The bank contains B original then B matched-wrong source measurements.
    Only the positive gene index changes for M5. Main image targets never move.
    Common negative support excludes either possible positive source identity.
    """
    b, device = len(outputs["program_predictions"]), genes.values.device
    require(len(genes.values) == 2*b and active.shape == (b,) and active.dtype == torch.bool,
            "S5_ORIGINAL_PLUS_WRONG_GENE_BANK_REQUIRED")
    require(negative_image.shape == negative_gene.shape == (b, 8)
            and negative_image.dtype == negative_gene.dtype == torch.long
            and ((negative_image >= -1) & (negative_image < b)).all()
            and ((negative_gene >= -1) & (negative_gene < b)).all(), "S5_NEGATIVE_CANDIDATE_SHAPE")
    own = torch.arange(b, device=device)[:, None]
    positive_gene = own + (b if wrong else 0)
    image_rows = own.expand(b, 9)
    gene_rows = torch.cat((positive_gene, negative_gene.clamp_min(0)), 1)
    reverse_image = torch.cat((own, negative_image.clamp_min(0)), 1)
    reverse_gene = positive_gene.expand(b, 9)
    base = model.gene_encoder(genes)
    forward = model.paired_scores(outputs, genes, image_rows, gene_rows, paired_mode=True, _gene_base=base)
    reverse = model.paired_scores(outputs, genes, reverse_image, reverse_gene, paired_mode=True, _gene_base=base)
    losses, counts = [], []
    for result, negative in ((forward, negative_gene), (reverse, negative_image)):
        allowed = torch.cat((active[:, None], negative >= 0), 1) & result["valid"]
        selected = active & allowed[:, 0] & allowed[:, 1:].any(1)
        counts.append(int(selected.sum()))
        if selected.any():
            logits = (result["scores"]/0.1).masked_fill(~allowed, -torch.inf)
            losses.append(F.cross_entropy(logits[selected], torch.zeros(int(selected.sum()), dtype=torch.long, device=device)))
    loss = sum(losses)/len(losses) if losses else outputs["program_predictions"].sum()*0
    return loss, dict(active_image_anchors=counts[0], active_gene_anchors=counts[1],
        two_anchor_losses_retained=True, wrong_positive_gene_bank=wrong,
        metadata_enters_model_forward=False, positive_and_negative_each_reconditioned=True)


def objective(model, outputs, targets, mask, tissue, weights, genes, negative_image, negative_gene, active):
    prediction = outputs["program_predictions"]
    require(prediction.shape == targets.shape == mask.shape and mask.dtype == torch.bool
        and prediction.shape[1] == 4 and (mask.sum(0) > 0).all(), "S5_PROGRAM_TARGET_SUPPORT")
    error = F.huber_loss(prediction, torch.where(mask, targets, 0), delta=1., reduction="none")
    regression = (torch.where(mask, error, 0).sum(0)/mask.sum(0)).mean()
    ce = F.cross_entropy(outputs["tissue_logits"], tissue, weight=weights, reduction="none").mean()
    itc, pair_audit = paired_loss(model, outputs, genes, negative_image, negative_gene, active,
                                 wrong=(model.arm == "M5"))
    return regression + .1*ce + .1*itc, dict(prediction_Huber=regression, tissue_CE=ce, ITC=itc,
        contrastive=pair_audit, regression_targets_mismatched=False, true_tissue_is_prediction_input=False)
