"""Frozen R8 feature lineage and finite, pre-registered arm construction.

This module never opens cohort, endpoint or source files. Caller-owned arrays
are already restricted to the allowed development patients. All PCA fitting
uses only the supplied train row indices; the model constructor has no DSS.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import torch

from .features import BlockPCA, MaskedPCA, build_low_capacity_features

ARMS = ("R8-2", "R8-3", "R8-4", "R8-5", "R8-6", "R8-8a_RANDOM_SOURCE",
        "R8-8b_WRONG_PROGRAM", "R8-9_WRONG_PATIENT", "R8-10_NO_INTERACTION")
PARAMETERS = {"R8-2": 4, "R8-3": 8, "R8-4": 16, "R8-5": 8, "R8-6": 32,
              "R8-8a_RANDOM_SOURCE": 32, "R8-8b_WRONG_PROGRAM": 32,
              "R8-9_WRONG_PATIENT": 32, "R8-10_NO_INTERACTION": 32}
WRONG_PROGRAM = (1, 2, 3, 0)


def require(value, code):
    if not bool(value):
        raise ValueError(code)


@dataclass(frozen=True)
class FrozenPatientEvidence:
    image_scores: np.ndarray
    image_tokens: np.ndarray
    conditioned_image_tokens: np.ndarray
    gene_tokens: np.ndarray
    gene_program_mask: np.ndarray
    source_state_sha256: str
    source_version: str
    wrong_conditioned_image_tokens: np.ndarray | None = None

    def validate(self):
        h, hg, g = map(lambda x: np.asarray(x, np.float64),
                       (self.image_tokens, self.conditioned_image_tokens, self.gene_tokens))
        scores, mask = np.asarray(self.image_scores), np.asarray(self.gene_program_mask)
        require(h.ndim == 3 and h.shape[1:] == (4, 64) and len(h) > 1
                and hg.shape == g.shape == h.shape and scores.shape == (len(h), 4)
                and mask.shape == h.shape[:2] and mask.dtype == bool,
                "R8_FROZEN_PATIENT_EVIDENCE_SHAPE")
        require(np.isfinite(h).all() and np.isfinite(hg).all() and np.isfinite(scores).all()
                and np.isfinite(g[mask]).all(), "R8_NONFINITE_FROZEN_PATIENT_EVIDENCE")
        if self.wrong_conditioned_image_tokens is not None:
            wrong=np.asarray(self.wrong_conditioned_image_tokens)
            require(wrong.shape==h.shape and np.isfinite(wrong).all(),
                    "R8_FROZEN_WRONG_CONDITIONED_IMAGE_SHAPE")
        require(isinstance(self.source_state_sha256, str)
                and re.fullmatch(r"[0-9a-f]{64}", self.source_state_sha256)
                and self.source_version in ("R6_S7_locked", "R6_S7_RANDOM_NO_ST_locked"),
                "R8_EXPLICIT_FROZEN_SOURCE_VERSION_REQUIRED")
        return self

    @property
    def rows(self):
        return len(self.image_scores)


@dataclass
class FrozenPartitionTransform:
    image_pca8: MaskedPCA
    conditional_pca8: MaskedPCA
    image_blocks2: BlockPCA
    gene_blocks2: BlockPCA
    train_indices: np.ndarray
    fit_identity_sha256: str
    source_state_sha256: str

    @classmethod
    def fit(cls, data: FrozenPatientEvidence, train_indices, *, fit_identity_sha256):
        data.validate()
        idx = np.asarray(train_indices)
        require(idx.ndim == 1 and idx.dtype.kind in "iu" and len(idx) >= 8
                and len(np.unique(idx)) == len(idx) and ((idx >= 0) & (idx < data.rows)).all(),
                "R8_PCA_TRAIN_PARTITION_INVALID")
        h, hg, g, gm = (np.asarray(data.image_tokens), np.asarray(data.conditioned_image_tokens),
                        np.asarray(data.gene_tokens), np.asarray(data.gene_program_mask))
        return cls(MaskedPCA.fit(h[idx].reshape(len(idx), -1), components=8, fit_identity_sha256=fit_identity_sha256),
            MaskedPCA.fit(hg[idx].reshape(len(idx), -1), components=8, fit_identity_sha256=fit_identity_sha256),
            BlockPCA.fit(h[idx], components_per_program=2, fit_identity_sha256=fit_identity_sha256),
            BlockPCA.fit(g[idx], components_per_program=2, program_mask=gm[idx], fit_identity_sha256=fit_identity_sha256),
            idx.copy(), fit_identity_sha256, data.source_state_sha256)

    def features(self, data: FrozenPatientEvidence, arm, *, query_mapping=None):
        data.validate()
        require(data.source_state_sha256 == self.source_state_sha256,
                "R8_FEATURE_TRANSFORM_SOURCE_STATE_CHANGED")
        require(arm in ARMS, "R8_ARM_OUTSIDE_PRELOCKED_FINITE_MATRIX")
        require((arm == "R8-8a_RANDOM_SOURCE") == (data.source_version == "R6_S7_RANDOM_NO_ST_locked"),
                "R8_RANDOM_SOURCE_ARM_OR_VERSION_MISMATCH")
        if arm == "R8-2":
            require(query_mapping is None, "R8_IMAGE_ONLY_HAS_NO_QUERY_MAPPING")
            out = np.asarray(data.image_scores, np.float64).copy()
        elif arm == "R8-3":
            require(query_mapping is None, "R8_IMAGE_ONLY_HAS_NO_QUERY_MAPPING")
            out = self.image_pca8.transform(np.asarray(data.image_tokens).reshape(data.rows, -1))
        elif arm == "R8-5":
            require(query_mapping is None, "R8_CONDITIONAL_IMAGE_CACHE_MAPPING_IS_FIXED")
            out = self.conditional_pca8.transform(np.asarray(data.conditioned_image_tokens).reshape(data.rows, -1))
        else:
            h = self.image_blocks2.transform(data.image_tokens)
            g = self.gene_blocks2.transform(data.gene_tokens, data.gene_program_mask)
            if arm == "R8-4":
                require(query_mapping is None, "R8_ADDITIVE_HAS_NO_QUERY_MAPPING")
                out = build_low_capacity_features(h, g, mode="additive")
            elif arm == "R8-10_NO_INTERACTION":
                require(query_mapping is None, "R8_NO_INTERACTION_HAS_NO_QUERY_MAPPING")
                out = build_low_capacity_features(h, g, mode="additive_quadratic")
            elif arm == "R8-8b_WRONG_PROGRAM":
                require(query_mapping is None, "R8_WRONG_PROGRAM_MUST_NOT_ALSO_CHANGE_PATIENT")
                out = build_low_capacity_features(h, g, mode="bilinear_rank2", program_permutation=WRONG_PROGRAM)
            elif arm == "R8-9_WRONG_PATIENT":
                p = np.asarray(query_mapping) if query_mapping is not None else np.empty(0)
                require(p.shape == (data.rows,) and p.dtype.kind in "iu"
                        and np.array_equal(np.sort(p), np.arange(data.rows)), "R8_QUERY_MAPPING_NOT_A_BIJECTION")
                require(np.all(p != np.arange(data.rows)), "R8_QUERY_MAPPING_HAS_SELF_PAIRS")
                out = build_low_capacity_features(h, g, mode="bilinear_rank2", query_gene2=g[p])
            else:
                require(query_mapping is None, "R8_MATCHED_BILINEAR_HAS_NO_QUERY_MAPPING")
                out = build_low_capacity_features(h, g, mode="bilinear_rank2")
        require(out.shape == (data.rows, PARAMETERS[arm]) and np.isfinite(out).all(),
                "R8_COEFFICIENT_BUDGET_OR_FEATURE_VALUES_CHANGED")
        return out

    def audit(self):
        return dict(train_rows=len(self.train_indices), train_identity_sha256=self.fit_identity_sha256,
                    source_state_sha256=self.source_state_sha256,
                    unsupervised_PCA_fits=10, flat_image_components=8, conditional_image_components=8,
                    program_components=2, fixed_program_positions=4,
                    image_program_PCA=[p.audit() for p in self.image_blocks2.blocks],
                    gene_program_PCA=[p.audit() for p in self.gene_blocks2.blocks],
                    target_outcomes_used=False, train_only=True, validation_moments_used=False)


def validate_query_partition_mapping(mapping, train_indices, validation_indices, rows):
    p, tr, va = np.asarray(mapping), np.asarray(train_indices), np.asarray(validation_indices)
    require(p.shape == (rows,) and p.dtype.kind in "iu" and tr.ndim == va.ndim == 1
            and tr.dtype.kind in "iu" and va.dtype.kind in "iu"
            and len(np.unique(tr)) == len(tr) and len(np.unique(va)) == len(va)
            and len(np.intersect1d(tr, va)) == 0,
            "R8_QUERY_PARTITIONS_OR_MAPPING_INVALID")
    require(set(tr) | set(va) == set(range(rows)) and np.array_equal(np.sort(p), np.arange(rows)),
            "R8_QUERY_MAPPING_OUTSIDE_CURRENT_DEVELOPMENT_PATIENTS")
    for idx in (tr, va):
        require(len(idx) > 1 and set(p[idx]) == set(idx) and np.all(p[idx] != idx),
                "R8_WRONG_QUERY_CROSSES_TRAIN_VALIDATION_OR_HAS_SELF_PAIR")
    return dict(train_rows=len(tr), validation_rows=len(va), self_pairs=0,
                train_validation_crossings=0, query_only_control=True,
                recipient_additive_genes_masks_offset_and_images_unchanged=True,
                target_survival_labels_used_for_mapping=False)


def encode_frozen_bulk_programs(source_model, measurements, *, expected_source_state_sha256):
    """Consume caller's train-scaled bulk values; never apply spot scalers."""
    from src.bgb_en.r6_omics import GeneMeasurements
    from src.bgb_en.r6_pathology import source_model_state_sha256
    require(isinstance(measurements, GeneMeasurements), "R8_ORIGINAL_MEASURED_BULK_GENES_REQUIRED")
    measurements.validate()
    require(not measurements.cna_enabled and not source_model.training
            and not any(p.requires_grad for p in source_model.parameters())
            and source_model_state_sha256(source_model) == expected_source_state_sha256,
            "R8_BULK_SOURCE_NOT_FROZEN_OR_CNA_ENABLED")
    with torch.no_grad():
        result = source_model.encode_gene_programs(measurements)
    require(result["tokens"].shape == (len(measurements.values), 4, 64)
            and torch.isfinite(result["tokens"]).all(), "R8_FROZEN_GENE_TOKEN_INTERFACE_CHANGED")
    measured = measurements.measured_mask
    observed = measurements.values[measured]
    return result, dict(source_state_sha256=expected_source_state_sha256,
        input_domain="TCGA_bulk_inner_train_scaled_RNA_not_Visium_counts",
        source_training_encoder_input="source_train_observed_only_per_gene_Z_of_log1p_CP10k_full_library",
        source_actual_training_path="r6_pretrain.SourceFitData.gene_z_then_GeneValueBank_not_model_gene_values_helper",
        target_encoder_input="TCGA_inner_train_median_imputed_moments_per_gene_Z_then_real_missing_mask",
        common_Z_form_proves_cross_domain_biological_alignment=False,
        source_spot_scaler_applied=False, raw_measurement_mask_unchanged=True,
        measured_values=int(measured.sum()), observed_zero_values=int((observed == 0).sum()),
        observed_coordinate_RMS=float(observed.double().square().mean().sqrt()) if observed.numel() else None,
        source_weights_updated=False, target_parameters_learned=0, optimizer_fits=0,
        uncertainty_proxy_is_medical_confidence=False)


def signed_mean_dispersion_audit(measurements, mapping):
    """Unlabelled domain proxy only; not part of the locked DSS design matrix.

    Up/down means and variances use only actually measured members. An empty
    *defined* signed component makes the proxy missing; an undefined down set
    contributes the definition's exact zero, never an imputed observation.
    """
    measurements.validate()
    require(not measurements.cna_enabled and measurements.gene_symbols == mapping.gene_symbols,
            "R8_SIGNED_PROXY_REQUIRES_CANONICAL_RNA_ONLY_GENE_BANK")
    x, mask = measurements.values[..., 0], measurements.measured_mask[..., 0]
    values=[];valid=[];coverage=[]
    for k in range(4):
        means=[];variances=[];present=[];fractions=[]
        for sign in (1,-1):
            genes=mapping.edge_gene[(mapping.edge_program==k)&(mapping.edge_sign==sign)].to(x.device)
            if not len(genes):
                means.append(x.new_zeros(len(x)));variances.append(x.new_zeros(len(x)))
                present.append(torch.ones(len(x),device=x.device,dtype=torch.bool))
                fractions.append(x.new_zeros(len(x)));continue
            m=mask[:,genes];n=m.sum(-1);v=torch.where(m,x[:,genes],torch.zeros_like(x[:,genes]))
            mu=v.sum(-1)/n.clamp_min(1)
            variance=torch.where(m,(v-mu[:,None]).square(),torch.zeros_like(v)).sum(-1)/n.clamp_min(1)
            means.append(mu);variances.append(variance);present.append(n>0);fractions.append(n.to(x.dtype)/len(genes))
        usable=present[0]&present[1]
        pair=torch.stack((means[0]-means[1],(variances[0]+variances[1]).clamp_min(0).sqrt()),-1)
        values.append(torch.where(usable[:,None],pair,torch.zeros_like(pair)));valid.append(usable)
        coverage.append(torch.stack(fractions,-1))
    return dict(values=torch.stack(values,1),program_mask=torch.stack(valid,1),coverage=torch.stack(coverage,1),
        diagnostics=dict(representation="signed_up_minus_down_mean_and_sqrt_sum_member_variances",
            measured_members_only=True, source_or_target_outcomes_used=False,
            fed_to_any_locked_DSS_readout=False, source_gene_encoder_replaced=False,
            invalid_program_values_require_explicit_mask=True))
