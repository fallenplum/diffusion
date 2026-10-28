"""Finite R7 tensor operators. No data loading, endpoint access or fitting.

RAW delegates to the unchanged R6 B7 operator. Other modes use only fixed
positive training-RMS scalars inside interaction; raw main effects stay intact.
Independent reverse reads are candidate increments, not statistically pure
interactions. Foreign contexts change routing only, never own evidence values.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .r6_omics import masked_softmax
from .r6_survival import R6SurvivalModel, MeanGeneQueryContext, masked_mean, patient_patch_mean, require


R7_ARMS = ("ADD", "RAW", "NORM", "INDEP", "SLOT", "GATE1", "GATE2",
           "SLOT_WRONG_RNA", "GATE1_WRONG_RNA", "GATE1_WRONG_PATH", "GATE2_WRONG_RNA")
TARGET_ARMS = R7_ARMS
SCALE_NAMES = ("context_rms", "patch_rms", "query_rms", "evidence_rms", "gene_key_rms", "reverse_query_rms")


class R7SurvivalModel(R6SurvivalModel):
    def __init__(self, arm, source_model, *, source_kind="MATCHED_ST", init_seed=31):
        require(arm in R7_ARMS and source_kind == "MATCHED_ST", "R7_UNKNOWN_ARM_OR_SOURCE")
        # Construct every inherited parameter in precisely the R6 order/seed.
        super().__init__("B7", source_model, source_kind=source_kind, init_seed=init_seed)
        self.arm = arm
        self.family = arm.replace("_WRONG_RNA", "").replace("_WRONG_PATH", "")
        self.normalized_interaction = self.family not in ("RAW", "ADD")
        self.independent_reverse = self.family in ("INDEP", "SLOT", "GATE1", "GATE2")
        self.slot_readout = self.family in ("SLOT", "GATE1", "GATE2")
        self.gated_states = self.family in ("GATE1", "GATE2")
        self.rounds = 2 if self.family == "GATE2" else 1
        for name in SCALE_NAMES:
            self.register_buffer(name, torch.tensor(1., dtype=torch.float32))
        if self.slot_readout:
            self.slot_path_weight = nn.Parameter(self.path_cross.weight.detach().repeat(4, 1))
            self.slot_gene_weight = nn.Parameter(self.gene_cross.weight.detach().repeat(4, 1))
            self.path_cross.requires_grad_(False); self.gene_cross.requires_grad_(False)
        if self.gated_states:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed + 700003)
                self.path_state_gate = nn.Linear(128, 1)
                self.gene_state_gate = nn.Linear(128, 1)
                for gate in (self.path_state_gate, self.gene_state_gate):
                    nn.init.zeros_(gate.weight); nn.init.zeros_(gate.bias)
        if self.family == "ADD":
            self.forward_enabled = False; self.reverse_kind = "none"
            for module in (self.query_condition, self.path_cross, self.gene_cross, self.gene_rereader.query_projection):
                module.requires_grad_(False)
            self.base_queries.requires_grad_(False)
        self.to(device=next(self.foundation.parameters()).device, dtype=next(self.foundation.parameters()).dtype)

    def set_calibration_scales(self, **scales):
        require(set(scales) <= set(SCALE_NAMES), "R7_UNKNOWN_RMS_BUFFER")
        with torch.no_grad():
            for name, value in scales.items():
                number = float(value)
                require(math.isfinite(number) and number >= 1e-6, "R7_RMS_MUST_BE_FINITE_POSITIVE_MIN_1E6")
                getattr(self, name).fill_(number)

    def _scale(self, name, normalized):
        value = getattr(self, name)
        require(value.ndim == 0 and torch.isfinite(value) and value >= 1e-6, "R7_INVALID_RMS_BUFFER")
        return value if normalized else value.new_ones(())

    def _parts(self, patch, genes, neighborhood, *, normalized):
        self._check_patch(patch, genes)
        adapted = self.spatial_adapter(patch["states"], patch["patch_mask"], neighborhood)
        own = self.foundation.encode_gene_programs(genes)
        raw_patch, raw_g = adapted["states"], own["tokens"]
        return dict(adapted=adapted, own=own, raw_patch=raw_patch, raw_G0=raw_g,
                    patch=raw_patch / self._scale("patch_rms", normalized),
                    G0=raw_g / self._scale("context_rms", normalized))

    def _patch_read(self, query, patch_values, patch, program_mask, *, assignment_shuffle_seed=None):
        options = dict(use_source_prior=True, program_score_prior=patch["program_score_prior"].detach())
        if assignment_shuffle_seed is not None:
            options["assignment_shuffle_seed"] = assignment_shuffle_seed
        return self.patch_reader(query, patch_values, patch["patch_mask"], program_mask, **options)

    def unconditional_patch_base(self, patch_encoded, own_genes, *, neighborhood):
        """Current-kernel P0, own mask, interaction units; not an updated state."""
        parts = self._parts(patch_encoded, own_genes, neighborhood, normalized=self.normalized_interaction)
        query = self.base_queries[None].expand(len(own_genes.values), -1, -1) / self._scale("query_rms", self.normalized_interaction)
        return self._patch_read(query, parts["patch"], patch_encoded, parts["own"]["program_mask"])

    @staticmethod
    def _statistics(value, valid):
        require(value.dtype == torch.float32 and valid.dtype == torch.bool and value.shape[:-1] == valid.shape,
                "R7_RMS_STAT_FLOAT32_VALID_MASK_REQUIRED")
        selected = value[valid]
        require(torch.isfinite(selected).all(), "R7_RMS_NONFINITE_VALID_ACTIVATION")
        return dict(sum_squares=selected.double().square().sum(),
                    count=torch.tensor(selected.numel(), device=value.device, dtype=torch.long))

    @torch.no_grad()
    def calibration_batch(self, patch_encoded, own_genes, *, neighborhood, stage):
        """Caller must supply only untrained-model training rows, no labels.

        RAW normal forwards ignore buffers, but calibration always uses the
        common normalized recipe, so one staged contract is shared by all arms.
        """
        require(stage in ("raw", "base", "reverse"), "R7_UNKNOWN_CALIBRATION_STAGE")
        parts = self._parts(patch_encoded, own_genes, neighborhood, normalized=True)
        own_mask = parts["own"]["program_mask"]
        raw_base = self.base_queries[None].expand(len(own_genes.values), -1, -1)
        if stage == "raw":
            encoded = self.gene_rereader.gene_encoder(own_genes)
            stats = dict(raw_G0=self._statistics(parts["raw_G0"], own_mask),
                raw_adapted_patch=self._statistics(parts["raw_patch"], patch_encoded["patch_mask"]),
                raw_base_query=self._statistics(raw_base, own_mask),
                raw_gene_key=self._statistics(encoded["keys"], encoded["gene_mask"]),
                raw_gene_value=self._statistics(encoded["values"], encoded["gene_mask"]))
        else:
            base = self._patch_read(raw_base / self.query_rms, parts["patch"], patch_encoded, own_mask)
            if stage == "base":
                stats = dict(scaled_P0=self._statistics(base["states"], base["program_mask"]))
            else:
                query = self.gene_rereader.query_projection(base["states"] / self.evidence_rms)
                stats = dict(projected_reverse_query=self._statistics(query, base["program_mask"]))
        return dict(stage=stage, statistics=stats, labels_used=False, buffer_values_changed=False,
                    normalized_calibration_path=True, gene_key_values_count_union_genes_not_signed_edges=True)

    def _gene_read(self, evidence, genes, mask, *, normalized):
        require(evidence.shape == (len(genes.values), 4, 64) and mask.shape == evidence.shape[:-1], "R7_BAD_GENE_QUERY")
        safe = torch.where(mask[..., None], evidence, torch.zeros_like(evidence))
        query = self.gene_rereader.query_projection(safe / self._scale("evidence_rms", normalized))
        query = query / self._scale("reverse_query_rms", normalized)
        encoded = self.gene_rereader.gene_encoder(genes)  # Own original values on every read.
        keys = encoded["keys"] / self._scale("gene_key_rms", normalized)
        values = encoded["values"] / self._scale("context_rms", normalized)
        ep, eg, sign = self.gene_rereader.edge_program, self.gene_rereader.edge_gene, self.gene_rereader.edge_sign
        measured = encoded["gene_mask"][:, eg]
        scores = (query[:, ep] * keys[:, eg]).sum(-1) / math.sqrt(64)
        valid = measured & mask[:, ep]
        weights = torch.zeros_like(scores)
        states, component_masks, coverages, counts, kept_counts, defined = [], [], [], [], [], []
        for k in range(4):
            component, active, coverage, count, kept, definition = [], [], [], [], [], []
            for direction in (1, -1):
                columns = torch.where((ep == k) & (sign == direction))[0]
                definition.append(bool(len(columns)))
                if len(columns):
                    probability = masked_softmax(scores[:, columns], valid[:, columns])
                    weights[:, columns] = probability
                    component.append((probability[..., None] * values[:, eg[columns]]).sum(1))
                    active.append(valid[:, columns].any(-1)); count.append(measured[:, columns].sum(-1))
                    kept.append(valid[:, columns].sum(-1))
                    coverage.append(count[-1].to(scores.dtype) / len(columns))
                else:
                    component.append(values.new_zeros(len(values), 64))
                    active.append(torch.zeros(len(values), device=values.device, dtype=torch.bool))
                    count.append(torch.zeros(len(values), device=values.device, dtype=torch.long))
                    kept.append(torch.zeros(len(values), device=values.device, dtype=torch.long))
                    coverage.append(scores.new_zeros(len(values)))
            states.append(torch.stack(component, 1)); component_masks.append(torch.stack(active, 1))
            coverages.append(torch.stack(coverage, 1)); counts.append(torch.stack(count, 1))
            kept_counts.append(torch.stack(kept, 1)); defined.append(definition)
        states, active = torch.stack(states, 1), torch.stack(component_masks, 1)
        return dict(tokens=states[:, :, 0] - states[:, :, 1], program_mask=active.any(-1), component_mask=active,
            up_tokens=states[:, :, 0], down_tokens=states[:, :, 1], coverage=torch.stack(coverages, 1),
            measured_counts=torch.stack(counts, 1), retained_counts=torch.stack(kept_counts, 1),
            component_defined=torch.tensor(defined, dtype=torch.bool, device=values.device), weights=weights,
            signed_weights=weights * sign[None], query=query, all_members_used=True, topk_per_program_before_signed_normalization=None,
            original_gene_values_reencoded_this_call=True, sparse_membership_edges=len(ep),
            own_gene_values_replaced=False, dense_gene_gene_attention_used=False)

    def _main(self, raw_patch, patch_mask, raw_g0, own_mask, hallmark_tokens, hallmark_mask):
        b = len(raw_g0)
        pmean, gmean = patient_patch_mean(raw_patch, patch_mask), masked_mean(raw_g0, own_mask, 1)
        if hallmark_tokens is None:
            require(hallmark_mask is None, "R7_HALLMARK_MASK_WITHOUT_TOKENS")
            hmean = raw_g0.new_zeros((b, 32))
        else:
            require(hallmark_tokens.shape == (b, 50, 32) and hallmark_mask is not None and hallmark_mask.shape == (b, 50)
                    and hallmark_mask.dtype == torch.bool and torch.isfinite(hallmark_tokens[hallmark_mask]).all(), "R7_BAD_HALLMARK_INPUT")
            hmean = masked_mean(hallmark_tokens, hallmark_mask, 1)
        pm, gm, hm = (.025 * head(value).squeeze(-1) for head, value in
                      ((self.path_main, pmean), (self.gene_main, gmean), (self.hallmark_main, hmean)))
        return pm, gm, hm, pmean, gmean, hmean

    def _risk_contributions(self, delta, mask, direction):
        weight = (getattr(self, "slot_" + direction + "_weight") if self.slot_readout
                  else getattr(self, direction + "_cross").weight.expand(4, -1))
        values = .05 * (delta * weight[None]).sum(-1)
        return torch.where(mask, values, torch.zeros_like(values)) / mask.sum(-1).clamp_min(1)[:, None]

    def forward(self, patch_encoded, own_genes, offset, *, neighborhood, hallmark_tokens=None, hallmark_mask=None,
                query_context=None, query_mask=None, mean_context=None, intervention=None, assignment_shuffle_seed=None,
                path_query_context=None, path_query_mask=None):
        if intervention is None:
            intervention = "wrong_query" if self.arm.endswith("_WRONG_RNA") else "wrong_path" if self.arm.endswith("_WRONG_PATH") else "matched"
        if self.family in ("RAW", "ADD"):
            require(path_query_context is None and path_query_mask is None, "R7_RAW_HAS_NO_FOREIGN_PATH_OPERATOR")
            out = super().forward(patch_encoded, own_genes, offset, neighborhood=neighborhood,
                hallmark_tokens=hallmark_tokens, hallmark_mask=hallmark_mask, query_context=query_context,
                query_mask=query_mask, mean_context=mean_context, intervention=intervention,
                assignment_shuffle_seed=assignment_shuffle_seed)
            out.update(state_trace=[], round_updates=[], program_risk_contributions=None,
                slot_path_contributions=None, slot_gene_contributions=None, gates=None, G0_interaction=out["G0"],
                diagnostics=out["diagnostics"] | dict(R7_arm=self.arm, normalized_interaction=False, independent_reverse=False,
                    source_scaler_applied_to_target=False, R7_RAW_equals_R6_B7=self.family == "RAW"))
            return out
        require(intervention in ("matched", "wrong_query", "wrong_path", "mean_query", "off", "value_off"), "R7_UNKNOWN_INTERVENTION")
        require((query_context is None) == (query_mask is None) and (path_query_context is None) == (path_query_mask is None),
                "R7_ROUTE_CONTEXT_NEEDS_SOURCE_MASK")
        require(intervention != "matched" or (query_context is None and path_query_context is None and mean_context is None),
                "R7_MATCHED_CANNOT_USE_FOREIGN_OR_MEAN_CONTEXT")
        b = len(own_genes.values)
        require(offset.shape == (b,) and offset.dtype == torch.float32 and torch.isfinite(offset).all(), "R7_BAD_UNCHANGED_OFFSET")
        if mean_context is not None:
            require(intervention == "mean_query" and query_context is None and isinstance(mean_context, MeanGeneQueryContext), "R7_MEAN_CONTEXT_CONFLICT")
            query_context, query_mask = mean_context.expand(b)
        require(intervention not in ("wrong_query", "mean_query") or query_context is not None, "R7_FOREIGN_OR_MEAN_RNA_CONTEXT_REQUIRED")
        require(intervention != "wrong_path" or path_query_context is not None, "R7_FOREIGN_PATH_CONTEXT_REQUIRED")
        parts = self._parts(patch_encoded, own_genes, neighborhood, normalized=True)
        own, g0 = parts["own"], parts["G0"]
        own_mask = own["program_mask"]
        qbase = self.base_queries[None].expand(b, -1, -1)
        base = self._patch_read(qbase / self.query_rms, parts["patch"], patch_encoded, own_mask,
                                assignment_shuffle_seed=assignment_shuffle_seed)
        p0, pvalid = base["states"], base["program_mask"]
        for context, mask in ((query_context, query_mask), (path_query_context, path_query_mask)):
            if context is not None:
                require(context.shape == (b, 4, 64) and context.dtype == torch.float32 and mask.shape == (b, 4)
                        and mask.dtype == torch.bool and context.device == g0.device and mask.device == g0.device
                        and torch.isfinite(context[mask]).all(), "R7_BAD_FOREIGN_ROUTE_TENSOR")
        initial_rna = g0 if query_context is None else query_context / self.context_rms
        rna_mask = own_mask if query_mask is None else query_mask
        initial_rna = torch.where(rna_mask[..., None], initial_rna, torch.zeros_like(initial_rna))
        initial_path = p0 if path_query_context is None else path_query_context
        path_mask = pvalid if path_query_mask is None else path_query_mask
        initial_path = torch.where(path_mask[..., None], initial_path, torch.zeros_like(initial_path))
        pstate, gstate, trace = p0, g0, []
        conditioned = base
        reverse_base = reverse_cond = None
        pmask, gmask = pvalid, own_mask
        for step in range(self.rounds):
            rna_route = gstate if query_context is None else initial_rna + (gstate - g0)
            path_route = pstate if path_query_context is None else initial_path + (pstate - p0)
            rna_route = torch.where(rna_mask[..., None], rna_route, torch.zeros_like(rna_route))
            path_route = torch.where(path_mask[..., None], path_route, torch.zeros_like(path_route))
            qcond = qbase + self.query_condition(rna_route)
            conditioned = self._patch_read(qcond / self.query_rms, parts["patch"], patch_encoded, rna_mask,
                                           assignment_shuffle_seed=assignment_shuffle_seed)
            pmask = pvalid & conditioned["program_mask"]
            dp = torch.where(pmask[..., None], conditioned["states"] - p0, torch.zeros_like(p0))
            if self.independent_reverse:
                gmask = path_mask & own_mask
                reverse_base = self._gene_read(torch.zeros_like(p0), own_genes, gmask, normalized=True)
                reverse_cond = self._gene_read(path_route, own_genes, gmask, normalized=True)
            else:
                gmask = pmask & own_mask
                reference = initial_path
                actual = initial_path + dp
                reverse_base = self._gene_read(reference, own_genes, gmask, normalized=True)
                reverse_cond = self._gene_read(actual, own_genes, gmask, normalized=True)
            gmask = gmask & reverse_base["program_mask"] & reverse_cond["program_mask"]
            dg = torch.where(gmask[..., None], reverse_cond["tokens"] - reverse_base["tokens"], torch.zeros_like(g0))
            if intervention in ("off", "value_off"):
                dp, dg = torch.zeros_like(dp), torch.zeros_like(dg)
            gp = 2 * self.path_state_gate(torch.cat((pstate, rna_route), -1)).sigmoid() if self.gated_states else torch.ones_like(dp[..., :1])
            gg = 2 * self.gene_state_gate(torch.cat((gstate, path_route), -1)).sigmoid() if self.gated_states else torch.ones_like(dg[..., :1])
            pnext = p0 + torch.where(pmask[..., None], gp * dp, torch.zeros_like(dp))
            gnext = g0 + torch.where(gmask[..., None], gg * dg, torch.zeros_like(dg))
            trace.append(dict(round=step + 1, RNA_route=rna_route, RNA_route_mask=rna_mask, PATH_route=path_route,
                PATH_route_mask=path_mask, P_before=pstate, G_before=gstate, raw_delta_P=dp, raw_delta_G=dg,
                path_gate=gp, gene_gate=gg, P_after=pnext, G_after=gnext))
            pstate, gstate = pnext, gnext
        de, dg = pstate - p0, gstate - g0
        pc, gc = self._risk_contributions(de, pmask, "path"), self._risk_contributions(dg, gmask, "gene")
        cross_p, cross_g = pc.sum(-1), gc.sum(-1)
        cross = cross_p + cross_g
        pmain, gmain, hmain, pmean, gmean, hmean = self._main(parts["raw_patch"], patch_encoded["patch_mask"],
            parts["raw_G0"], own_mask, hallmark_tokens, hallmark_mask)
        main = pmain + gmain + hmain; delta = main + cross
        return dict(risk=offset + delta, offset=offset, delta=delta, total_delta=delta, main_delta=main, cross_delta=cross,
            path_main_delta=pmain, gene_main_delta=gmain, hallmark_main_delta=hmain, path_cross_delta=cross_p, gene_cross_delta=cross_g,
            delta_E=de, delta_G=dg, Ebase=p0, Econd=pstate, G0=parts["raw_G0"], G0_interaction=g0, Gfinal=gstate,
            own_program_mask=own_mask, query_context_used=trace[-1]["RNA_route"], query_source_mask=rna_mask,
            base_queries=qbase / self.query_rms, conditioned_queries=qcond / self.query_rms,
            patch_values=parts["raw_patch"], interaction_patch_values=parts["patch"], gene_values=own_genes.values,
            gene_measured_mask=own_genes.measured_mask, patch_mean=pmean, gene_mean=gmean, hallmark_mean=hmean,
            patch_base_read=base, patch_cond_read=conditioned, reverse_base_read=reverse_base, reverse_cond_read=reverse_cond,
            gene_weight_change=reverse_cond["weights"] - reverse_base["weights"], source_gene_read=own,
            native_adapter=parts["adapted"], attention_change=conditioned["weights"] - base["weights"],
            coverage=own["coverage"], intervention=intervention, state_trace=trace, round_updates=trace,
            program_risk_contributions=dict(path=pc, gene=gc), gates=dict(path=gp, gene=gg),
            slot_path_contributions=pc, slot_gene_contributions=gc,
            diagnostics=dict(arm=self.arm, R7_arm=self.arm, source_kind=self.source_kind,
                normalized_interaction=True, independent_reverse=self.independent_reverse, rounds=self.rounds,
                reverse_kind="gene", forward_structure_enabled=True, source_prior_in_both_patch_reads=True,
                own_gene_values_replaced=False, own_patch_values_replaced=False, RNA_main_effect_replaced=False,
                offset_untouched=True, raw_main_inputs_unchanged=True, RNA_route_is_foreign=query_context is not None,
                PATH_route_is_foreign=path_query_context is not None, foreign_initial_context_fixed_across_rounds=True,
                no_foreign_updated_states=True, no_per_patient_normalization=True,
                reverse_reference="zero_evidence_uniform_signed_members" if self.independent_reverse else "unconditional_P0",
                independent_reverse_may_contain_remaining_main_effects=True, source_scaler_applied_to_target=False,
                Hallmark_tokens_claimed_ST_aligned=False, whole_modality_replacement=False))

    def parameter_l2_sum(self):
        total = self.base_queries.new_zeros(())
        for name, value in self.named_parameters():
            if value.requires_grad:
                total = total + value.square().sum() * (.25 if name in ("slot_path_weight", "slot_gene_weight") else 1.)
        return total

    def parameter_regularization(self):
        return 1e-4 * self.parameter_l2_sum()

    def regularization(self):
        return self.parameter_regularization()

    def parameter_budget(self):
        result = super().parameter_budget()
        result.update(R7_arm=self.arm, slot_heads=self.slot_readout, rounds=self.rounds,
            slot_L2_factor=.25 if self.slot_readout else None, gate_parameters=258 if self.gated_states else 0,
            source_stem_frozen=True, calibration_buffers_are_not_parameters=True)
        return result
