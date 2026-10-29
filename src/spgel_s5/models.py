"""Attribution of the existing paired-only A5 reread, not a new survival model.

TASK-A/C prediction stays image-only. Conditional rereading participates in the
training paired objective and TASK-B scoring. M1 spends the same two 128x128
matrices on SELF context; M2 spends them on the opposite modality. The legacy
matrix names are preserved for exact initialization and checkpoint comparison.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from ..spgel.contracts import GeneBatch, require
from ..spgel.models import SpatialProgramEncoder


ARMS = tuple(f"M{i}" for i in range(7))


class S5ProgramEncoder(SpatialProgramEncoder):
    def __init__(self, arm, up, down, symbols, *, seed=31, tissue_program_mean=None):
        require(arm in ARMS, "S5_UNKNOWN_REGISTERED_ARM")
        inherited = "S_A4" if arm == "M0" else "S_A5"
        super().__init__(inherited, up, down, symbols, width=128, hidden=128,
                         seed=seed, tissue_program_mean=tissue_program_mean)
        self.inherited_arm, self.arm = inherited, arm
        if arm == "M3":
            self.image_to_gene = None
        elif arm == "M4":
            self.gene_to_image = None
        self.bidirectional = arm in ("M1", "M2", "M5", "M6")

    def paired_scores(self, image_outputs, genes: GeneBatch, image_rows, gene_rows,
                      *, paired_mode=False, return_details=False, _gene_base=None):
        require(paired_mode is True, "PAIRED_SCORING_MODE_REQUIRED")
        require(image_rows.ndim == 2 and image_rows.shape == gene_rows.shape
                and image_rows.dtype == gene_rows.dtype == torch.long,
                "PAIRED_SCORE_ROW_SHAPES")
        n = len(genes.values)
        visual = image_outputs["visual_program_tokens"]
        windows = image_outputs["window_tokens"]
        window_mask = image_outputs["window_mask"]
        b = len(visual)
        require(visual.shape == (b, 4, self.width)
                and windows.ndim == 3 and windows.shape[0] == b
                and windows.shape[2] == self.width
                and window_mask.shape == windows.shape[:-1]
                and window_mask.dtype == torch.bool,
                "S5_UNCONDITIONAL_IMAGE_BANK_SHAPES")
        require(image_rows.device == gene_rows.device == genes.values.device == visual.device
                and ((image_rows >= 0) & (image_rows < b)).all()
                and ((gene_rows >= 0) & (gene_rows < n)).all(),
                "S5_SEPARATE_IMAGE_GENE_BANK_ROW_RANGE")
        molecular = self.gene_encoder(genes) if _gene_base is None else _gene_base
        require(molecular["tokens"].shape == (n, 4, self.width)
                and molecular["program_mask"].shape == (n, 4),
                "UNIMODAL_GENE_CACHE_SHAPE")
        g_context = molecular["tokens"][gene_rows]
        v_context = visual[image_rows]
        selected_windows, selected_valid = windows[image_rows], window_mask[image_rows]
        g_mask = molecular["program_mask"][gene_rows]

        if self.arm == "M0":
            # Preserve the old A4 normalization/dot operation order exactly.
            v = F.normalize(visual, dim=-1, eps=1e-12)[image_rows]
            g = F.normalize(molecular["tokens"], dim=-1, eps=1e-12)[gene_rows]
            scores = torch.where(g_mask, (v * g).sum(-1), 0.).sum(-1) / g_mask.sum(-1).clamp_min(1)
            result = dict(scores=scores, valid=g_mask.any(-1))
            if return_details:
                result.update(conditioned_image_tokens=v_context,
                              conditioned_gene_tokens=g_context,
                              condition_is_recomputed_for_each_candidate=True,
                              image_context_source="none", gene_context_source="none")
            return result

        # M1: both queries depend only on their own modality. M2/M5/M6:
        # opposite-modality query context. M3/M4 keep only one cross direction.
        if self.gene_to_image is not None:
            image_context = v_context if self.arm == "M1" else g_context
            query = self.window.queries + self.gene_to_image(image_context)
            cond_v = self.window(selected_windows, selected_valid, query=query,
                                 return_attention=return_details)
            image_source = "self_image" if self.arm == "M1" else "measured_genes"
        else:
            cond_v = dict(tokens=v_context,
                program_mask=selected_valid.any(-1)[..., None].expand(*image_rows.shape, 4))
            image_source = "none"
        if self.image_to_gene is not None:
            gene_context = g_context if self.arm == "M1" else v_context
            query = self.gene_encoder.queries + self.image_to_gene(gene_context)
            cond_g = self.gene_encoder.read(genes, query=query, candidates=gene_rows,
                                           return_attention=return_details)
            gene_source = "self_genes" if self.arm == "M1" else "image_window"
        else:
            cond_g = dict(tokens=g_context, program_mask=g_mask)
            gene_source = "none"
        mask = cond_v["program_mask"] & cond_g["program_mask"]
        score = F.cosine_similarity(cond_v["tokens"], cond_g["tokens"], dim=-1, eps=1e-12)
        result = dict(scores=torch.where(mask, score, 0.).sum(-1) / mask.sum(-1).clamp_min(1),
                      valid=mask.any(-1))
        if return_details:
            result.update(conditioned_image_tokens=cond_v["tokens"],
                          conditioned_gene_tokens=cond_g["tokens"],
                          image_attention=cond_v.get("attention"),
                          gene_up_attention=cond_g.get("up_attention"),
                          gene_down_attention=cond_g.get("down_attention"),
                          condition_is_recomputed_for_each_candidate=True,
                          image_context_source=image_source, gene_context_source=gene_source)
        return result

    def descriptor(self):
        output = super().descriptor()
        output.update(schema="spgel_s5_source_program_encoder_v1", inherited_arm=self.inherited_arm,
            query_context_mode=("none" if self.arm == "M0" else "self_modality" if self.arm == "M1"
                                else "gene_to_image_only" if self.arm == "M3"
                                else "image_to_gene_only" if self.arm == "M4" else "cross_modal_both"),
            image_only_prediction_uses_conditional_reread=False,
            conditional_reread_scope="train_paired_objective_and_TASK_B_only",
            two_anchor_contrastive_losses_preserved=True,
            wrong_pair_mapping_is_training_data_contract=self.arm == "M5",
            shuffled_membership_is_external_model_input=self.arm == "M6",
            simultaneous_program_target_and_slot_relabel_is_valid_negative_control=False,
            extra_capacity_is_dummy_or_detached=False,
            official_fifty_HVG_head=False)
        return output
