"""P2 ordinary-attention controls on shared, nonsemantic visual readers.

Both use raw512 pathology via eight image-only random visual readers, followed
by 8x50 ordinary cross-attention. NATIVE names the independent-QKVO operator;
neither class is an official paper reproduction. The fixed B2 matched operator
uses a single-head eight-dimensional attention subspace inside the same width64 encoder.
Its FLOP envelope must be measured and may fail independently of parameter
matching. No biological relation prior or medical text Query enters B2.
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .models import ModelConfig, masked_softmax
from .p1r_models import RepairRepresentation, RepairSpec


class NarrowSharedAttention(nn.Module):
    """Identity Q/K/V on a prelocked subspace, with zero-parameter padding."""
    def __init__(self, encoder_width, heads, attention_width):
        super().__init__()
        if attention_width > encoder_width or attention_width % heads:
            raise ValueError("INVALID_MATCHED_ATTENTION_SUBSPACE")
        self.encoder_width, self.heads, self.attention_width = encoder_width, heads, attention_width

    def forward(self, query, values, valid):
        b, q = query.shape[:2]
        n, h, d = values.shape[1], self.heads, self.attention_width
        qq = query[..., :d].reshape(b, q, h, d // h).transpose(1, 2)
        kk = values[..., :d].reshape(b, n, h, d // h).transpose(1, 2)
        weights = masked_softmax(qq @ kk.transpose(-1, -2) / math.sqrt(d // h), valid[:, None, None, :])
        message = (weights @ kk).transpose(1, 2).reshape(b, q, d)
        return F.pad(message, (0, self.encoder_width - d)), weights


class CrossAttentionRepresentation(RepairRepresentation):
    def __init__(self, conch_prototypes, titan_prototypes, relation_prior, *,
                 spec: RepairSpec, config: ModelConfig | None = None,
                 seed=31, score_surrogate=None):
        c = config or ModelConfig()
        if spec.query_mode != "visual_random" or spec.score_route != "actual" or spec.relation_mode != "legacy":
            raise ValueError("B2_REQUIRES_RANDOM_VISUAL_READERS_SHARED_ACTUAL_SCORE_AND_NO_BIOLOGICAL_RELATION")
        if spec.shuffled_map or score_surrogate is not None:
            raise ValueError("B2_HAS_NO_BIOLOGICAL_MAP_OR_SCORE_ROUTE_ABLATION")
        native = spec.source_arm in ("B2", "B2_NATIVE", "B2_NATIVE_QKVO")
        if not native and spec.source_arm not in ("B2_MATCHED", "B2_MATCHED_4H", "B2_MATCHED_WIDTH8"):
            raise ValueError("UNKNOWN_B2_OPERATOR_VERSION")
        construction = replace(spec, source_arm="B2" if native else "B2_MATCHED_4H")
        # Dense support does not use any values of the biological input prior.
        dense = torch.ones(c.slots, c.programs, dtype=torch.float32)
        super().__init__(None, None, dense, spec=construction, config=c, seed=seed)
        self.spec = spec
        self.native_qkvo = native
        self.pair_row_block = 8
        self.attention_heads = c.heads if native else 1
        self.attention_width = c.width if native else min(8, c.width)
        if not native:
            self.g_to_p = NarrowSharedAttention(c.width, self.attention_heads, self.attention_width)
            self.p_to_g = NarrowSharedAttention(c.width, self.attention_heads, self.attention_width)

    def _conditional(self, visual, gene, gene_mask):
        image_mask = torch.ones(visual.shape[:2], device=visual.device, dtype=torch.bool)
        image_message, _ = self.g_to_p(visual, gene, gene_mask)
        gene_message, _ = self.p_to_g(gene, visual, image_mask)
        return visual + image_message, (gene + gene_message) * gene_mask[..., None]

    def _pair_score_block(self, image_rows, gene_bank, gene_mask):
        """Unfactorized ordinary-QKVO oracle retained for numerical review."""
        r, n, s, k, d = len(image_rows), len(gene_bank), image_rows.shape[1], gene_bank.shape[1], image_rows.shape[2]
        visual = image_rows[:, None].expand(r, n, s, d).reshape(r * n, s, d)
        gene = gene_bank[None].expand(r, n, k, d).reshape(r * n, k, d)
        valid = gene_mask[None].expand(r, n, k).reshape(r * n, k)
        conditional_visual, conditional_gene = self._conditional(visual, gene, valid)
        u = F.normalize(self.u(conditional_visual), dim=-1).mean(1)
        v = F.normalize(self.v(conditional_gene), dim=-1) * valid[..., None]
        v = v.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        return (u * v).sum(-1).reshape(r, n)

    def _factorized_banks(self, image, gene):
        """Compute linear maps once per unique patient, not once per pair."""
        h, width = self.attention_heads, self.attention_width
        part = width // h
        def split(value):
            return value.reshape(len(value), value.shape[1], h, part).transpose(1, 2)
        if self.native_qkvo:
            qp = split(self.g_to_p.q(image))
            kg = split(self.g_to_p.k(gene))
            value_gene = split(self.g_to_p.v(gene))
            qg = split(self.p_to_g.q(gene))
            kp = split(self.p_to_g.k(image))
            value_image = split(self.p_to_g.v(image))
            out_to_u = self.u.weight @ self.g_to_p.out.weight
            out_to_v = self.v.weight @ self.p_to_g.out.weight
        else:
            qp = kp = split(image[..., :width])
            qg = kg = split(gene[..., :width])
            value_gene, value_image = kg, kp
            out_to_u, out_to_v = self.u.weight[:, :width], self.v.weight[:, :width]
        # For each head: U WO_h (A_h V_h) = A_h (V_h (U WO_h)^T).
        # Only rank2 messages, rather than width64 Value tensors, are expanded
        # across the image×gene candidate grid. Normalize stays AFTER adding
        # u(P)/v(G), exactly as in the unfactorized operator.
        gene_rank_value = torch.einsum("nhkd,rhd->nhkr", value_gene,
                                      out_to_u.reshape(self.config.rank, h, part))
        image_rank_value = torch.einsum("nhsd,rhd->nhsr", value_image,
                                       out_to_v.reshape(self.config.rank, h, part))
        return qp, kg, qg, kp, gene_rank_value, image_rank_value, self.u(image), self.v(gene)

    def _factorized_score_block(self, qp, kg, qg, kp, gene_rank_value,
                                image_rank_value, u_base, v_base, gene_mask):
        scale = math.sqrt(self.attention_width // self.attention_heads)
        pg_logits = torch.einsum("ihsd,jhkd->ijhsk", qp, kg) / scale
        pg = masked_softmax(pg_logits, gene_mask[None, :, None, None, :])
        image_rank_message = torch.einsum("ijhsk,jhkr->ijsr", pg, gene_rank_value)
        gp_logits = torch.einsum("jhkd,ihsd->ijhks", qg, kp) / scale
        gp = torch.softmax(gp_logits, dim=-1)
        gene_rank_message = torch.einsum("ijhks,ihsr->ijkr", gp, image_rank_value)
        u = F.normalize(u_base[:, None] + image_rank_message, dim=-1).mean(2)
        v = F.normalize(v_base[None] + gene_rank_message, dim=-1) * gene_mask[None, :, :, None]
        v = v.sum(2) / gene_mask.sum(1).clamp_min(1)[None, :, None]
        return (u * v).sum(-1)

    def pair_scores(self, image_outputs, gene_outputs):
        image, gene, mask = image_outputs["visual_tokens"], gene_outputs["gene_tokens"], gene_outputs["gene_mask"]
        qp, kg, qg, kp, gr, pr, ub, vb = self._factorized_banks(image, gene)
        pieces = []
        for start in range(0, len(image), self.pair_row_block):
            stop = start + self.pair_row_block
            banks = (qp[start:stop], kg, qg, kp[start:stop], gr, pr[start:stop], ub[start:stop], vb, mask)
            if self.native_qkvo and self.training and torch.is_grad_enabled():
                # The checkpointed function has no dropout or random operation.
                # Recomputed backward does not introduce optimizer updates or
                # change positives/negative support; every candidate is scored.
                score = checkpoint(self._factorized_score_block, *banks,
                                   use_reentrant=False, preserve_rng_state=False)
            else:
                score = self._factorized_score_block(*banks)
            pieces.append(score)
        scores = torch.cat(pieces) / self.config.temperature
        return {"scores": scores, "valid": mask.any(-1)[None, :].expand_as(scores)}

    def mechanism_from_tokens(self, visual, gene, gene_mask, *, additive=False):
        if additive:
            raise ValueError("ORDINARY_CROSS_ATTENTION_IS_NOT_AN_ADDITIVE_CONTROL")
        visual, gene = self._conditional(visual, gene, gene_mask)
        p = self.u(visual).mean(1)
        g = self.v(gene) * gene_mask[..., None]
        g = g.sum(1) / gene_mask.sum(1).clamp_min(1)[:, None]
        return p * g

    def architecture_descriptor(self):
        return {"pathology_raw_input_budget": 512,
                "cross_attention_domain": "8_image_only_nonsemantic_readers_by_50_measured_program_tokens",
                "encoder_width": self.config.width, "attention_width": self.attention_width,
                "attention_heads": self.attention_heads,
                "qkvo": "independent_trainable" if self.native_qkvo else "identity_shared_projection_subspace",
                "medical_text_query_used": False, "biological_relation_prior_used": False,
                "shared_actual_TITAN32_scores_used": True,
                "official_paper_reproduction": False,
                "pair_score_row_block": self.pair_row_block,
                "all_pair_candidates_scored": True,
                "activation_checkpoint_enabled_during_training": self.native_qkvo,
                "activation_checkpoint_backward_recomputes_without_extra_optimizer_steps": self.native_qkvo,
                "matched_uses_direct_autograd_without_checkpoint_recomputation": not self.native_qkvo,
                "pair_linear_factorization": "unique_bank_QKV_and_headwise_U_WO_V_projection_before_pair_expansion",
                "factorization_exact_in_real_arithmetic_FP32_reordering_error_requires_audit": True,
                "parameter_match_requires_actual_gradient_audit": True,
                "FLOP_match_within10percent_requires_real_profile_not_assumed": True}
