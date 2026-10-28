"""R6 target core: frozen source evidence, native reads and linear risk deltas.

This module performs no fitting, endpoint access, cohort loading or model
selection. The caller supplies fold-local gene measurements and the unchanged
CP+RNA scalar offset. A cached pathology descriptor must identify the complete
same frozen S7 source model. Four source programs and optional Hallmark50
unimodal auxiliary tokens have deliberately different interfaces.

All arms construct the same modules in the same order. Unused target weights
are frozen and reported, rather than counted as effective training capacity.
The only synchrony is RNA-query -> original patch read -> reverse gene read;
there is no repeated state iteration. Wrong/mean RNA affects the query and its
source mask, never this patient's patch/gene values, main effects or offset.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .r6_omics import GeneMeasurements, OriginalGeneRereader, SparseProgramEncoder
from .r6_pathology import NativeSpatialAdapter, WSIProgramReader, source_model_state_sha256
from .r6_source_model import SourceProgramModel


TARGET_ARMS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B11", "B13")
SOURCE_KINDS = ("MATCHED_ST", "WRONG_DONOR_ST", "RANDOM_NO_ST")


def require(value, code):
    if not bool(value):
        raise ValueError(code)


def masked_mean(value, mask, dim):
    require(mask.shape == value.shape[:-1] and mask.dtype == torch.bool,
            "R6_MASKED_MEAN_SHAPE_OR_DTYPE_CHANGED")
    safe = torch.where(mask[..., None], value, torch.zeros_like(value))
    return safe.sum(dim) / mask.sum(dim).clamp_min(1)[..., None]


def patient_patch_mean(states, mask):
    """Mean patches inside each WSI, then equal valid-WSI patient mean."""
    per_wsi = masked_mean(states, mask, 2)
    return masked_mean(per_wsi, mask.any(-1), 1)


@dataclass(frozen=True)
class MeanGeneQueryContext:
    """Outcome-free mean; caller must provide only current training patients."""
    tokens: torch.Tensor
    program_mask: torch.Tensor
    training_patients: int

    def expand(self, batch_size):
        require(self.tokens.shape == (4, 64) and self.program_mask.shape == (4,)
                and self.program_mask.dtype == torch.bool and self.training_patients > 0,
                "R6_TRAIN_MEAN_QUERY_SCHEMA_CHANGED")
        return (self.tokens[None].expand(batch_size, -1, -1),
                self.program_mask[None].expand(batch_size, -1))


class R6SurvivalModel(nn.Module):
    """Tensor-level target network; no raw pathology or survival labels loaded.

    Query adapter: bias-free 64->4->64 GELU; down uses PyTorch's fixed
    Linear initialization, up N(0,.02). Base queries use N(0,.02). Every linear
    risk head uses N(0,.002), no bias. These are fixed engineering defaults,
    not outcome-selected settings. OriginalGeneRereader supplies the common
    64x64 reverse query matrix for both token and original-gene controls.
    """
    def __init__(self, arm, source_model, *, source_kind, init_seed=31):
        super().__init__()
        require(arm in TARGET_ARMS and source_kind in SOURCE_KINDS,
                "R6_UNKNOWN_TARGET_ARM_OR_SOURCE_KIND")
        expected_kind = ("RANDOM_NO_ST" if arm in ("B2", "B13") else
                         "WRONG_DONOR_ST" if arm == "B8" else "MATCHED_ST")
        require(source_kind == expected_kind, "R6_SOURCE_CONTROL_DOES_NOT_MATCH_TARGET_ARM")
        require(isinstance(source_model, SourceProgramModel) and source_model.arm == "S7"
                and source_model.width == 64 and source_model.input_dim == 1536
                and isinstance(source_model.gene_reader, SparseProgramEncoder)
                and len(source_model.gene_reader.mapping.program_names) == 4,
                "R6_COMPLETE_S7_WITH_ORIGINAL_GENE_READER_REQUIRED")
        require(isinstance(init_seed, int) and not isinstance(init_seed, bool),
                "R6_EXPLICIT_INTEGER_INITIALIZATION_SEED_REQUIRED")
        self.arm, self.source_kind, self.init_seed = arm, source_kind, init_seed
        self.foundation = copy.deepcopy(source_model).requires_grad_(False).eval()
        self.source_state_sha256 = source_model_state_sha256(self.foundation)
        self.forward_enabled = arm not in ("B2", "B3")
        self.reverse_kind = "token" if arm == "B5" else (
            "gene" if arm in ("B6", "B7", "B8", "B9", "B13") else "none")
        self.use_source_prior = arm in ("B7", "B8", "B9", "B11", "B13")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.spatial_adapter = NativeSpatialAdapter(width=64, rank=4, scale=.1)
            self.base_queries = nn.Parameter(torch.randn(4, 64) * .02)
            self.query_condition = nn.Sequential(nn.Linear(64, 4, bias=False), nn.GELU(),
                                                 nn.Linear(4, 64, bias=False))
            nn.init.normal_(self.query_condition[2].weight, std=.02)
            self.patch_reader = WSIProgramReader(width=64, source_prior_scale=.1)
            self.gene_rereader = OriginalGeneRereader(self.foundation.gene_reader.mapping,
                gene_encoder=self.foundation.gene_reader.gene_encoder, width=64)
            self.path_main = nn.Linear(64, 1, bias=False)
            self.gene_main = nn.Linear(64, 1, bias=False)
            self.hallmark_main = nn.Linear(32, 1, bias=False)
            self.path_cross = nn.Linear(64, 1, bias=False)
            self.gene_cross = nn.Linear(64, 1, bias=False)
            for head in (self.path_main, self.gene_main, self.hallmark_main,
                         self.path_cross, self.gene_cross):
                nn.init.normal_(head.weight, std=.002)
        self.base_queries.requires_grad_(self.forward_enabled)
        self.query_condition.requires_grad_(self.forward_enabled)
        self.path_cross.requires_grad_(self.forward_enabled)
        self.gene_rereader.query_projection.requires_grad_(self.reverse_kind != "none")
        self.gene_cross.requires_grad_(self.reverse_kind != "none")
        # Target heads/adapters follow the already placed foundation.
        parameter = next(self.foundation.parameters())
        self.to(device=parameter.device, dtype=parameter.dtype)

    def train(self, mode=True):
        super().train(mode)
        self.foundation.eval()
        self.gene_rereader.gene_encoder.eval()
        return self

    def mean_query_context(self, training_genes):
        """No labels; mean is over valid programs of training_genes only.

        Scope cannot be inferred from tensors: the fold-owning caller must
        establish their patient lineage before calling this helper.
        """
        require(isinstance(training_genes, GeneMeasurements), "R6_MEAN_REQUIRES_RAW_TRAIN_GENES")
        with torch.no_grad():
            result = self.foundation.encode_gene_programs(training_genes)
            tokens = masked_mean(result["tokens"], result["program_mask"], 0).detach()
            mask = result["program_mask"].any(0).detach()
        return MeanGeneQueryContext(tokens, mask, len(training_genes.values))

    def _check_patch(self, patch, genes):
        require(isinstance(patch, dict) and isinstance(genes, GeneMeasurements),
                "R6_ORIGINAL_PATCH_DESCRIPTOR_AND_GENE_MEASUREMENTS_REQUIRED")
        require(patch.get("diagnostics", {}).get("source_state_sha256") == self.source_state_sha256,
                "R6_PATCH_CACHE_AND_COMPLETE_SOURCE_STATE_DIFFER")
        states, mask = patch["states"], patch["patch_mask"]
        require(states.ndim == 4 and states.shape[-1] == 64 and len(states) == len(genes.values)
                and states.dtype == torch.float32 and mask.shape == states.shape[:3]
                and mask.dtype == torch.bool and states.device == mask.device == genes.values.device,
                "R6_NATIVE_PATCH_BATCH_OR_MASK_CHANGED")
        require(torch.isfinite(states[mask]).all(), "R6_NONFINITE_VALID_PATCH_STATE")
        require(not self.foundation.training and not any(p.requires_grad for p in self.foundation.parameters()),
                "R6_FOUNDATION_MUST_STAY_COMPLETELY_FROZEN")
        genes.validate()

    def _token_reverse(self, evidence, evidence_mask, tokens, token_mask):
        """Four original G0 token values, common 64x64 reverse Q, zero null."""
        safe_e = torch.where(evidence_mask[..., None], evidence, torch.zeros_like(evidence))
        safe_g = torch.where(token_mask[..., None], tokens, torch.zeros_like(tokens))
        query = self.gene_rereader.query_projection(safe_e)
        logits = torch.einsum("bkd,bjd->bkj", query, safe_g) / math.sqrt(64)
        valid = evidence_mask[:, :, None] & token_mask[:, None, :]
        distribution = torch.cat((logits.masked_fill(~valid, -torch.inf),
                                  logits.new_zeros((*logits.shape[:-1], 1))), -1).softmax(-1)
        weights = distribution[..., :-1]
        result = torch.einsum("bkj,bjd->bkd", weights, safe_g)
        mask = evidence_mask & token_mask.any(-1)[:, None]
        return dict(tokens=result, program_mask=mask, weights=weights, null=distribution[..., -1],
                    query=query, values=safe_g, original_gene_reread=False)

    def forward(self, patch_encoded, own_genes, offset, *, neighborhood,
                hallmark_tokens=None, hallmark_mask=None, query_context=None, query_mask=None,
                mean_context=None, intervention=None, assignment_shuffle_seed=None):
        self._check_patch(patch_encoded, own_genes)
        b = len(own_genes.values)
        require(offset.shape == (b,) and offset.dtype == torch.float32
                and offset.device == own_genes.values.device and torch.isfinite(offset).all(),
                "R6_UNCHANGED_FINITE_CP_SCALAR_OFFSET_REQUIRED")
        if intervention is None:
            intervention = "wrong_query" if self.arm == "B9" else "matched"
        require(intervention in ("matched", "wrong_query", "mean_query", "off", "value_off"),
                "R6_UNKNOWN_INTERVENTION_WHOLE_OMICS_REQUIRES_OUTER_WRAPPER")
        require((query_context is None) == (query_mask is None),
                "R6_QUERY_CONTEXT_REQUIRES_ITS_OWN_SOURCE_MASK")
        if mean_context is not None:
            require(intervention == "mean_query" and query_context is None
                    and isinstance(mean_context, MeanGeneQueryContext), "R6_MEAN_CONTEXT_CONTROL_CONFLICT")
            query_context, query_mask = mean_context.expand(b)
        require(intervention not in ("wrong_query", "mean_query") or query_context is not None,
                "R6_WRONG_OR_MEAN_QUERY_NEEDS_EXPLICIT_SOURCE_CONTEXT_AND_MASK")
        require(intervention != "matched" or query_context is None,
                "R6_MATCHED_QUERY_CANNOT_SILENTLY_USE_OTHER_CONTEXT")
        adapted = self.spatial_adapter(patch_encoded["states"], patch_encoded["patch_mask"], neighborhood)
        patch_values, patch_mask = adapted["states"], patch_encoded["patch_mask"]
        # This is not a cached pooled-token replacement: it reencodes own raw
        # gene values and identities on every target forward, with frozen weights.
        own = self.foundation.encode_gene_programs(own_genes)
        g0, own_mask = own["tokens"], own["program_mask"]
        context, source_mask = g0, own_mask
        if query_context is not None:
            require(query_context.shape == (b, 4, 64) and query_mask.shape == (b, 4)
                    and query_mask.dtype == torch.bool and query_context.dtype == g0.dtype
                    and query_context.device == query_mask.device == g0.device
                    and torch.isfinite(query_context[query_mask]).all(),
                    "R6_QUERY_SOURCE_SHAPE_DTYPE_DEVICE_OR_FINITE_CHANGED")
            context, source_mask = query_context, query_mask
        safe_context = torch.where(source_mask[..., None], context, torch.zeros_like(context))
        q_base = self.base_queries[None].expand(b, -1, -1)
        q_cond = q_base + self.query_condition(safe_context)
        kwargs = dict(use_source_prior=self.use_source_prior)
        if assignment_shuffle_seed is not None:
            kwargs["assignment_shuffle_seed"] = assignment_shuffle_seed
        if self.use_source_prior:
            require("program_score_prior" in patch_encoded, "R6_EXACT_FROZEN_SOURCE_PROGRAM_SCORE_REQUIRED")
            kwargs["program_score_prior"] = patch_encoded["program_score_prior"].detach()
        base = self.patch_reader(q_base, patch_values, patch_mask, source_mask, **kwargs)
        if self.forward_enabled and intervention != "off":
            conditioned = self.patch_reader(q_cond, patch_values, patch_mask, source_mask, **kwargs)
        else:
            conditioned = base
        effective_base, effective_cond = base["states"], conditioned["states"]
        effective_mask = base["program_mask"] & conditioned["program_mask"]
        if intervention == "value_off":
            effective_base = torch.zeros_like(effective_base)
            effective_cond = torch.zeros_like(effective_cond)
        d_e = torch.where(effective_mask[..., None], effective_cond - effective_base,
                          torch.zeros_like(effective_base))
        reverse_base = reverse_cond = None
        d_g = torch.zeros_like(g0)
        reverse_mask = torch.zeros_like(own_mask)
        if self.reverse_kind != "none":
            if self.reverse_kind == "token":
                reverse_base = self._token_reverse(effective_base, effective_mask, g0, own_mask)
                reverse_cond = self._token_reverse(effective_cond, effective_mask, g0, own_mask)
            else:
                reverse_base = self.gene_rereader(effective_base, own_genes, evidence_mask=effective_mask)
                reverse_cond = self.gene_rereader(effective_cond, own_genes, evidence_mask=effective_mask)
            reverse_mask = reverse_base["program_mask"] & reverse_cond["program_mask"]
            d_g = torch.where(reverse_mask[..., None], reverse_cond["tokens"] - reverse_base["tokens"],
                              torch.zeros_like(g0))
        patch_mean = patient_patch_mean(patch_values, patch_mask)
        gene_mean = masked_mean(g0, own_mask, 1)
        if hallmark_tokens is None:
            require(hallmark_mask is None, "R6_HALLMARK_MASK_WITHOUT_SEPARATE_TOKENS")
            hallmark_mean = g0.new_zeros((b, 32))
        else:
            require(hallmark_tokens.shape == (b, 50, 32) and hallmark_tokens.dtype == g0.dtype
                    and hallmark_tokens.device == g0.device and hallmark_mask is not None
                    and hallmark_mask.shape == (b, 50) and hallmark_mask.dtype == torch.bool
                    and hallmark_mask.device == g0.device
                    and torch.isfinite(hallmark_tokens[hallmark_mask]).all(),
                    "R6_HALLMARK_AUXILIARY_REQUIRES_EXPLICIT_B_50_32_VALUES_AND_MASK")
            hallmark_mean = masked_mean(hallmark_tokens, hallmark_mask, 1)
        p_main = .025 * self.path_main(patch_mean).squeeze(-1)
        g_main = .025 * self.gene_main(gene_mean).squeeze(-1)
        h_main = .025 * self.hallmark_main(hallmark_mean).squeeze(-1)
        main_delta = p_main + g_main + h_main
        cross_p = .05 * self.path_cross(masked_mean(d_e, effective_mask, 1)).squeeze(-1)
        cross_g = .05 * self.gene_cross(masked_mean(d_g, reverse_mask, 1)).squeeze(-1)
        cross_delta = cross_p + cross_g
        delta = main_delta + cross_delta
        gene_weight_change = (reverse_cond["weights"] - reverse_base["weights"]
                              if self.reverse_kind == "gene" else None)
        return dict(risk=offset + delta, offset=offset, delta=delta, total_delta=delta,
            main_delta=main_delta, cross_delta=cross_delta, path_main_delta=p_main,
            gene_main_delta=g_main, hallmark_main_delta=h_main, path_cross_delta=cross_p,
            gene_cross_delta=cross_g, delta_E=d_e, delta_G=d_g, Ebase=base["states"],
            Econd=conditioned["states"], G0=g0, own_program_mask=own_mask,
            query_context_used=safe_context, query_source_mask=source_mask,
            base_queries=q_base, conditioned_queries=q_cond, patch_values=patch_values,
            gene_values=own_genes.values, gene_measured_mask=own_genes.measured_mask,
            patch_mean=patch_mean, gene_mean=gene_mean, hallmark_mean=hallmark_mean,
            patch_base_read=base, patch_cond_read=conditioned,
            reverse_base_read=reverse_base, reverse_cond_read=reverse_cond,
            gene_weight_change=gene_weight_change, source_gene_read=own, native_adapter=adapted,
            attention_change=conditioned["weights"] - base["weights"],
            coverage=own["coverage"], intervention=intervention,
            diagnostics=dict(arm=self.arm, source_kind=self.source_kind,
                complete_frozen_source_state_sha256=self.source_state_sha256,
                one_synchronous_update=True, original_genes_reencoded=True,
                reverse_kind=self.reverse_kind, source_prior_in_both_patch_reads=self.use_source_prior,
                query_only_intervention=intervention in ("wrong_query", "mean_query"),
                own_gene_values_replaced=False, own_patch_values_replaced=False,
                RNA_main_effect_replaced=False, offset_untouched=True,
                local_value_message_disabled=intervention == "value_off",
                Hallmark_tokens_claimed_ST_aligned=False,
                source_pretraining_success_requires_separate_receipt=True))

    @staticmethod
    def residual_penalty(output):
        """Mean over complete risk set, or correctly weighted microbatches."""
        return .1 * output["delta"].square().mean() + .1 * output["cross_delta"].square().mean()

    def parameter_regularization(self):
        """Add once per full-risk optimization step, not once per microbatch."""
        return 1e-4 * sum(p.square().sum() for p in self.parameters() if p.requires_grad)

    regularization = parameter_regularization

    def parameter_budget(self):
        return dict(registered=sum(p.numel() for p in self.parameters()),
                    requires_grad=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    frozen_foundation=sum(p.numel() for p in self.foundation.parameters()),
                    shared_source_gene_stem_frozen=True,
                    reverse_query_parameters=4096 if self.reverse_kind != "none" else 0,
                    unsupported_FLOP_equivalence_claimed=False)
