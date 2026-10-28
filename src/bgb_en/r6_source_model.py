"""Outcome-free R6 source model with complete transferable program semantics.

Forward inference accepts visual evidence and optional validated neighbors only.
Tissue truth and measured genes are loss/evaluation inputs, never substitutes
for the visual program prediction. No R5 module or survival endpoint is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .external import PROGRAMS

SOURCE_ARMS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
FEATURE_UNIT = "float32_per_row_L2_norm_division_reject_norm_le_1e-12"
GATE_PERMUTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
RANDOM_GATE_ALGORITHM = "unit_UNI2_float32_little_endian_row_SHA256_first64little_mod6_class_permutation_v1"
SOURCE_DEFAULTS = dict(epochs=80, tissue_CE_weight=.1, alignment_weight=.01,
                       neighborhood_weight=.01, Huber_delta=1.,
                       training_spot_weights="uniform_current_source_fit_spots",
                       heldout_report_unit="equal_donor_not_independent_spots",
                       target_scale="current_training_donors_only",
                       source_vs_TCGA_gene_scalers_are_not_interchangeable=True)


def require(value: Any, code: str) -> None:
    if not bool(value):
        raise ValueError(code)


def array_sha(value: np.ndarray) -> str:
    x = np.ascontiguousarray(value)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def fixed_feature_gate_permutation(features):
    """Stateless per-row null: no labels, learned state, row IDs or resampling.

    Every bit of the original unit float32 feature row contributes to its SHA.
    Duplicate feature rows deliberately receive the same class permutation.
    Callers can precompute N-by-3 indices; the fallback is intended for QA.
    """
    tensor_input = torch.is_tensor(features)
    raw = features.detach().cpu().numpy() if tensor_input else np.asarray(features)
    require(raw.ndim == 2 and len(raw) > 0 and raw.dtype == np.dtype(np.float32)
            and np.isfinite(raw).all(), "HASH_GATE_REQUIRES_UNIT_FLOAT32_ROWS")
    require(np.max(np.abs(np.linalg.norm(raw, axis=1) - 1.)) <= 3e-6, "HASH_GATE_FEATURES_NOT_UNIT_L2")
    raw = np.ascontiguousarray(raw, dtype="<f4")
    indices = [int.from_bytes(hashlib.sha256(row.tobytes()).digest()[:8], "little") % 6 for row in raw]
    result = torch.as_tensor(np.asarray(GATE_PERMUTATIONS, dtype=np.int64)[indices], dtype=torch.long)
    return result.to(features.device) if tensor_input else result


def gate_probability_audit(original, used, permutation, *, algorithm):
    """Aggregates only; exact row multisets do not imply matched model marginals."""
    def numpy(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    p, q, order = numpy(original).astype(np.float64), numpy(used).astype(np.float64), numpy(permutation)
    require(p.ndim == 2 and p.shape[1] == 3 and len(p) > 0 and p.shape == q.shape == order.shape
            and np.isfinite(p).all() and np.isfinite(q).all() and (p >= 0).all() and (q >= 0).all()
            and np.allclose(p.sum(1), 1., atol=3e-6, rtol=0) and np.allclose(q.sum(1), 1., atol=3e-6, rtol=0)
            and order.dtype.kind in "iu" and np.array_equal(np.sort(order, axis=1), np.tile(np.arange(3), (len(p), 1))),
            "BAD_GATE_PROBABILITY_AUDIT_INPUT")
    entropy = lambda value: -np.sum(np.where(value > 0, value * np.log(np.where(value > 0, value, 1.)), 0.), axis=1)
    hp, hq = entropy(p), entropy(q)
    counts = {"".join(map(str, pattern)): int(np.all(order == np.asarray(pattern), axis=1).sum()) for pattern in GATE_PERMUTATIONS}
    exact = bool(np.array_equal(np.sort(p, axis=1), np.sort(q, axis=1)))
    return dict(schema="bgb_en_r6_row_gate_audit_v1", status="RECORDED", rows=len(p), algorithm=algorithm,
        row_probability_multiset_max_abs_difference=float(np.max(np.abs(np.sort(p, axis=1) - np.sort(q, axis=1)))),
        row_probability_multiset_exact=exact,
        used_probabilities_follow_reported_permutation=bool(np.array_equal(np.take_along_axis(p, order, 1), q)),
        original_entropy_mean=float(hp.mean()), used_entropy_mean=float(hq.mean()),
        row_entropy_max_abs_difference=float(np.max(np.abs(hp - hq))),
        original_perclass_probability_mean=p.mean(0).tolist(), used_perclass_probability_mean=q.mean(0).tolist(),
        permutation_counts=counts, identity_rate=counts["012"] / len(p), unique_permutations=sum(v > 0 for v in counts.values()),
        permutation_uses_labels=False, permutation_uses_patient_or_spot_ids=False,
        permutation_uses_learned_features=False, between_model_distributions_exactly_matched_claimed=False,
        confidence_information_removed_claimed=False)


def donor_derangement_feasibility(donor_codes, indices=None):
    donor = np.asarray(donor_codes)
    require(donor.ndim == 1 and donor.dtype.kind in "iu", "SOURCE_DONORS_REQUIRE_INTEGER_CODES")
    idx = np.arange(len(donor)) if indices is None else np.asarray(indices)
    require(idx.ndim == 1 and idx.dtype.kind in "iu" and len(idx) >= 2
            and len(np.unique(idx)) == len(idx) and ((idx >= 0) & (idx < len(donor))).all(),
            "INVALID_SOURCE_TRAIN_INDICES")
    names, counts = np.unique(donor[idx], return_counts=True)
    feasible = int(counts.max()) * 2 <= len(idx) and len(names) >= 2
    return dict(status="PASS_STRICT_CROSS_DONOR_BIJECTION_FEASIBLE" if feasible
                else "BLOCKED_STRICT_CROSS_DONOR_BIJECTION", train_spots=len(idx),
                train_donors=len(names), anonymous_donor_counts=counts.tolist(),
                maximum_donor_fraction=float(counts.max() / len(idx)),
                no_same_donor_bijection_possible=feasible, outcomes_or_program_targets_used=False)


def lodo_derangement_feasibility(donor_codes):
    donor = np.asarray(donor_codes)
    return [dict(heldout_donor_code=int(code), **donor_derangement_feasibility(
        donor, np.flatnonzero(donor != code))) for code in np.unique(donor)]


def _depth_pairs(recipient, target, depth, rng):
    """Match the smaller side to uniform depth quantiles of the larger side."""
    recipient = np.asarray(recipient, np.int64)
    target = np.asarray(target, np.int64)
    if not len(recipient) or not len(target):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    recipient = recipient[np.lexsort((rng.random(len(recipient)), depth[recipient]))]
    target = target[np.lexsort((rng.random(len(target)), depth[target]))]
    count = min(len(recipient), len(target))
    a = np.floor((np.arange(count) + .5) * len(recipient) / count).astype(int)
    b = np.floor((np.arange(count) + .5) * len(target) / count).astype(int)
    return recipient[a], target[b]


def wrong_donor_permutation(donor_codes, tissue, library_size, train_indices, *, seed):
    """O(N log N) joint-target permutation; no dense N-by-N assignment matrix.

    A maximum donor-block cyclic shift gives a strict derangement. Within each
    target donor's allocated recipients, exact tissue matches are used first,
    then depth-ranked pairing. Unavoidable mismatches remain explicit.
    """
    donor = np.asarray(donor_codes)
    tissue = np.asarray(tissue)
    library = np.asarray(library_size, np.float64)
    idx = np.asarray(train_indices)
    audit = donor_derangement_feasibility(donor, idx)
    require(audit["no_same_donor_bijection_possible"], "BLOCKED_STRICT_CROSS_DONOR_BIJECTION")
    require(tissue.shape == library.shape == donor.shape and np.isfinite(library[idx]).all()
            and (library[idx] > 0).all(), "INVALID_SOURCE_MATCHING_METADATA")
    rng = np.random.default_rng(int(seed))
    blocks = [rng.permutation(idx[donor[idx] == code]) for code in rng.permutation(np.unique(donor[idx]))]
    order = np.concatenate(blocks)
    shifted = np.roll(order, -max(map(len, blocks)))
    mapping = np.arange(len(donor), dtype=np.int64)
    mapping[order] = shifted
    require(np.all(donor[mapping[idx]] != donor[idx]), "CYCLIC_DONOR_DERANGEMENT_FAILED")
    depth = np.log1p(library)
    initial_matches = int(np.sum(tissue[idx] == tissue[mapping[idx]]))
    for target_donor in np.unique(donor[idx]):
        recipients = idx[donor[mapping[idx]] == target_donor]
        targets = mapping[recipients].copy()
        remaining_r, remaining_t = set(recipients.tolist()), set(targets.tolist())
        for label in np.unique(tissue[idx]):
            left = np.array(sorted(i for i in remaining_r if tissue[i] == label), dtype=np.int64)
            right = np.array(sorted(i for i in remaining_t if tissue[i] == label), dtype=np.int64)
            a, b = _depth_pairs(left, right, depth, rng)
            mapping[a] = b
            remaining_r.difference_update(a.tolist())
            remaining_t.difference_update(b.tolist())
        left, right = _depth_pairs(sorted(remaining_r), sorted(remaining_t), depth, rng)
        require(len(left) == len(remaining_r) == len(remaining_t), "SOURCE_PERMUTATION_MATCHING_LOST_ROWS")
        mapping[left] = right
    require(set(mapping[idx]) == set(idx) and np.all(mapping[idx] != idx)
            and np.all(donor[mapping[idx]] != donor[idx]), "SOURCE_PERMUTATION_NOT_A_STRICT_BIJECTION")
    boundaries = np.quantile(depth[idx], [.25, .5, .75])
    bins = np.searchsorted(boundaries, depth, side="right")
    audit.update(seed=int(seed), mapping_sha256=array_sha(mapping),
                 initial_tissue_matches=initial_matches,
                 final_tissue_matches=int(np.sum(tissue[idx] == tissue[mapping[idx]])),
                 depth_quartile_matches=int(np.sum(bins[idx] == bins[mapping[idx]])),
                 mean_absolute_log_library_gap=float(np.mean(np.abs(depth[idx] - depth[mapping[idx]]))),
                 target_four_vector_and_mask_must_move_together=True,
                 algorithm="maximum_donor_block_shift_then_tissue_first_depth_quantile_pairing",
                 weighted_joint_marginals_not_automatically_preserved=True,
                 dense_N_squared_allocation=False)
    return mapping, audit


def joint_permutation_audit(targets, valid_mask, mapping, indices, *, weights=None):
    """Report both exact unweighted joint preservation and optional weight drift."""
    y, mask = np.asarray(targets, np.float64), np.asarray(valid_mask, bool)
    idx, mapping = np.asarray(indices, int), np.asarray(mapping, int)
    require(y.ndim == 2 and y.shape[1] == 4 and mask.shape == y.shape
            and np.isfinite(y[mask]).all() and set(mapping[idx]) == set(idx), "BAD_JOINT_PROGRAM_PERMUTATION")
    a, b = np.where(mask[idx], y[idx], 0.), np.where(mask[mapping[idx]], y[mapping[idx]], 0.)
    ma, mb = mask[idx], mask[mapping[idx]]
    if weights is None:
        w = np.full(len(idx), 1 / len(idx))
    else:
        w = np.asarray(weights, np.float64)
        require(w.shape == (len(idx),) and np.isfinite(w).all() and (w > 0).all(), "BAD_SOURCE_AUDIT_WEIGHTS")
        w = w / w.sum()

    def joint_digest(values, observed):
        columns = np.c_[values, observed.astype(float)]
        order = np.lexsort(tuple(columns[:, k] for k in range(columns.shape[1] - 1, -1, -1)))
        return array_sha(columns[order])

    def moments(values, observed):
        mass = np.sum(w[:, None] * observed, 0)
        mean = np.sum(w[:, None] * values, 0) / np.maximum(mass, 1e-15)
        complete = observed.all(1)
        cw = w[complete] / w[complete].sum() if complete.any() else np.empty(0)
        centered = values[complete] - (cw @ values[complete] if len(cw) else 0.)
        covariance = (centered * cw[:, None]).T @ centered if len(cw) else np.zeros((4, 4))
        return mean, covariance

    am, ac = moments(a, ma)
    bm, bc = moments(b, mb)
    require(joint_digest(a, ma) == joint_digest(b, mb), "JOINT_PROGRAM_ROWS_OR_MASKS_NOT_PRESERVED")
    return dict(unweighted_joint_vector_and_mask_multiset_preserved=True,
                joint_multiset_sha256=joint_digest(a, ma),
                sample_weights_uniform=bool(np.allclose(w, w[0], atol=1e-15, rtol=0)),
                weighted_mean_max_abs_drift=float(np.max(np.abs(am - bm))),
                weighted_covariance_max_abs_drift=float(np.max(np.abs(ac - bc))),
                weighted_marginal_preservation_claimed=bool(np.allclose(w, w[0], atol=1e-15, rtol=0)))


def local_neighbor_indices(global_neighbors, selected_indices, *, section_codes=None, donor_codes=None):
    graph, selected = np.asarray(global_neighbors), np.asarray(selected_indices)
    require(graph.ndim == 2 and graph.dtype.kind in "iu" and 1 <= graph.shape[1] <= 16
            and selected.ndim == 1 and selected.dtype.kind in "iu" and len(np.unique(selected)) == len(selected),
            "BAD_SOURCE_GRAPH_OR_SELECTED_ROWS")
    require(((selected >= 0) & (selected < len(graph))).all()
            and ((graph >= -1) & (graph < len(graph))).all(), "SOURCE_GRAPH_INDEX_OUT_OF_RANGE")
    lookup = np.full(len(graph), -1, np.int64)
    lookup[selected] = np.arange(len(selected))
    chosen = graph[selected]
    mask = chosen >= 0
    safe = np.maximum(chosen, 0)
    require((lookup[safe][mask] >= 0).all(), "SOURCE_GRAPH_CROSSES_TRAIN_HELDOUT_PARTITION")
    for metadata in (section_codes, donor_codes):
        if metadata is not None:
            codes = np.asarray(metadata)
            require(codes.shape == (len(graph),)
                    and np.all(codes[safe][mask] == np.broadcast_to(codes[selected, None], chosen.shape)[mask]),
                    "SOURCE_GRAPH_CROSSES_SECTION_OR_DONOR")
    return np.where(mask, lookup[safe], -1)


@dataclass
class SourceFitStatistics:
    target_mean: np.ndarray
    target_scale: np.ndarray
    tissue_prior: np.ndarray
    class_weights: np.ndarray
    fit_indices_sha256: str
    fit_donor_codes: tuple[int, ...]
    gene_mean: np.ndarray | None = None
    gene_scale: np.ndarray | None = None

    @classmethod
    def fit(cls, targets, tissue, donor_codes, train_indices, *, heldout_donors=(),
            valid_mask=None, gene_values=None, gene_mask=None):
        y, labels, donor = np.asarray(targets, np.float64), np.asarray(tissue), np.asarray(donor_codes)
        idx = np.asarray(train_indices)
        require(y.ndim == 2 and y.shape[1] == 4 and labels.shape == donor.shape == (len(y),)
                and idx.ndim == 1 and idx.dtype.kind in "iu" and len(idx) > 1
                and len(np.unique(idx)) == len(idx) and ((idx >= 0) & (idx < len(y))).all(),
                "BAD_SOURCE_FIT_SCOPE")
        require(not np.intersect1d(donor[idx], np.asarray(heldout_donors)).size,
                "HELDOUT_DONOR_ENTERED_SOURCE_STATISTICS")
        mask = np.isfinite(y) if valid_mask is None else np.asarray(valid_mask, bool)
        require(mask.shape == y.shape and np.isfinite(y[idx][mask[idx]]).all(), "NONFINITE_TRAIN_SOURCE_TARGET")
        require(np.isin(labels[idx], [0, 1, 2]).all(), "BAD_TRAIN_TISSUE_LABELS")
        count = mask[idx].sum(0)
        require((count > 1).all(), "INSUFFICIENT_OBSERVED_TRAIN_PROGRAMS")
        safe = np.where(mask[idx], y[idx], 0.)
        mean = safe.sum(0) / count
        scale = np.sqrt(np.where(mask[idx], (safe - mean)**2, 0.).sum(0) / count)
        require((scale > 1e-12).all() and np.isfinite(scale).all(), "CONSTANT_TRAIN_PROGRAM_TARGET")
        mass = np.bincount(labels[idx].astype(int), minlength=3) / len(idx)
        require((mass > 0).all(), "ALL_THREE_TISSUES_REQUIRED_IN_SOURCE_TRAIN")
        gm = gs = None
        if gene_values is not None:
            x, xm = np.asarray(gene_values, np.float64), np.asarray(gene_mask, bool)
            require(x.ndim == 2 and len(x) == len(y) and xm.shape == x.shape
                    and np.isfinite(x[idx][xm[idx]]).all(), "BAD_TRAIN_SOURCE_GENE_VALUES")
            observed = xm[idx].sum(0)
            values = np.where(xm[idx], x[idx], 0.)
            gm = values.sum(0) / np.maximum(observed, 1)
            gs = np.sqrt(np.where(xm[idx], (values - gm)**2, 0.).sum(0) / np.maximum(observed, 1))
            gs = np.where(gs > 1e-12, gs, 1.)
        return cls(mean, scale, mass, 1 / (3 * mass), array_sha(idx), tuple(map(int, np.unique(donor[idx]))), gm, gs)

    def transform_targets(self, targets, valid_mask=None):
        y = np.asarray(targets, np.float64)
        mask = np.isfinite(y) if valid_mask is None else np.asarray(valid_mask, bool)
        require(y.ndim == 2 and y.shape[1] == 4 and mask.shape == y.shape
                and np.isfinite(y[mask]).all(), "BAD_SOURCE_TARGET_TRANSFORM")
        return np.where(mask, (np.where(mask, y, self.target_mean) - self.target_mean) / self.target_scale, 0.)

    def transform_genes(self, gene_values, measured_mask):
        require(self.gene_mean is not None and self.gene_scale is not None, "SOURCE_GENE_SCALER_NOT_FITTED")
        x, mask = np.asarray(gene_values, np.float64), np.asarray(measured_mask, bool)
        require(x.ndim == 2 and mask.shape == x.shape and x.shape[1] == len(self.gene_mean)
                and np.isfinite(x[mask]).all(), "BAD_SOURCE_GENE_TRANSFORM")
        return np.where(mask, (np.where(mask, x, self.gene_mean) - self.gene_mean) / self.gene_scale, 0.).astype(np.float32)

    def audit(self):
        return dict(training_spot_weights="uniform", fit_indices_sha256=self.fit_indices_sha256,
                    fit_donors=len(self.fit_donor_codes), target_mean=self.target_mean.tolist(),
                    target_scale=self.target_scale.tolist(), tissue_prior=self.tissue_prior.tolist(),
                    class_weights=self.class_weights.tolist(), gene_scaler_fitted=self.gene_mean is not None,
                    source_gene_scaler_must_not_be_applied_to_TCGA_bulk=True)


class SourceProgramModel(nn.Module):
    """Visual predictor and transferable four-program query/tissue module."""
    def __init__(self, arm="S4", *, input_dim=1536, width=64, condition_rank=2,
                 condition_scale=.1, spatial_rank=4, spatial_scale=.1,
                 tissue_prior=None, gate_mode="predicted", gene_reader=None,
                 init_seed=None, program_order=PROGRAMS):
        super().__init__()
        require(arm in SOURCE_ARMS and tuple(program_order) == PROGRAMS, "UNKNOWN_SOURCE_ARM_OR_PROGRAM_ORDER")
        require(gate_mode in ("predicted", "random") and condition_rank == 2,
                "UNKNOWN_SOURCE_GATE_OR_CONDITION_RANK")
        self.arm, self.width, self.input_dim = arm, int(width), int(input_dim)
        self.gate_mode, self.condition_scale, self.spatial_scale = gate_mode, float(condition_scale), float(spatial_scale)
        self.use_condition = arm in ("S4", "S5", "S6", "S7")
        self.use_spatial = arm == "S6"
        self.use_alignment = arm == "S7"
        with torch.random.fork_rng(devices=[]):
            if init_seed is not None:
                torch.manual_seed(int(init_seed))
            self.project = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
            self.tissue_head = nn.Linear(width, 3)
            self.program_queries = nn.Parameter(torch.randn(4, width) * .02)
            self.program_bias = nn.Parameter(torch.zeros(4))
            self.condition_gate = nn.Linear(3, condition_rank, bias=False)
            nn.init.normal_(self.condition_gate.weight, std=.02)
            self.condition_basis = nn.Parameter(torch.randn(4, condition_rank, width) * .02)
            self.spatial_down = nn.Linear(width, spatial_rank, bias=False)
            self.spatial_up = nn.Linear(spatial_rank, width, bias=False)
            nn.init.normal_(self.spatial_up.weight, std=.02)
        prior = torch.full((3,), 1 / 3) if tissue_prior is None else torch.as_tensor(tissue_prior, dtype=torch.float32)
        require(prior.shape == (3,) and torch.isfinite(prior).all() and (prior > 0).all()
                and torch.allclose(prior.sum(), prior.new_tensor(1.), atol=1e-6, rtol=0), "BAD_SOURCE_TRAIN_TISSUE_PRIOR")
        require(not self.use_condition or tissue_prior is not None, "CONDITIONAL_SOURCE_REQUIRES_TRAIN_ONLY_PRIOR")
        self.register_buffer("train_tissue_prior", prior.clone())
        self.gene_reader = gene_reader
        require(not self.use_alignment or gene_reader is not None, "S7_REQUIRES_SHARED_R6_GENE_READER")
        if gene_reader is not None:
            require(hasattr(gene_reader, "mapping") and tuple(gene_reader.mapping.program_names) == PROGRAMS
                    and gene_reader.width == self.width, "SOURCE_REQUIRES_SAME_FOUR_PROGRAM_SHARED_GENE_READER")
        self._configure_trainable()

    def _configure_trainable(self):
        if self.arm == "S0":
            self.requires_grad_(False)
            return
        self.tissue_head.requires_grad_(self.arm != "S2")
        self.program_queries.requires_grad_(self.arm != "S1")
        self.program_bias.requires_grad_(self.arm != "S1")
        self.condition_gate.requires_grad_(self.use_condition)
        self.condition_basis.requires_grad_(self.use_condition)
        self.spatial_down.requires_grad_(self.use_spatial)
        self.spatial_up.requires_grad_(self.use_spatial)
        if self.gene_reader is not None:
            self.gene_reader.requires_grad_(self.use_alignment)

    def encode(self, features, neighbor_index=None, *, neighbor_features=None, neighbor_mask=None):
        require(features.ndim == 2 and len(features) > 0 and features.shape[1] == self.input_dim
                and features.dtype == torch.float32 and torch.isfinite(features).all(), "SOURCE_REQUIRES_FINITE_FLOAT32_FEATURES")
        require(torch.max(torch.abs(features.norm(dim=-1) - 1)).detach() <= 3e-6, "SOURCE_FEATURES_NOT_UNIT_L2")
        z = self.project(features)
        if self.use_spatial:
            require((neighbor_index is None) != (neighbor_features is None), "SOURCE_SPATIAL_USE_INDEX_OR_EXPLICIT_FEATURES")
            if neighbor_features is not None:
                require(neighbor_features.ndim == 3 and neighbor_features.shape[0] == len(features)
                        and neighbor_features.shape[-1] == self.input_dim
                        and 1 <= neighbor_features.shape[1] <= 16
                        and neighbor_features.dtype == torch.float32 and neighbor_features.device == z.device
                        and neighbor_mask is not None and neighbor_mask.shape == neighbor_features.shape[:2]
                        and neighbor_mask.dtype == torch.bool and neighbor_mask.device == z.device,
                        "BAD_EXPLICIT_SOURCE_NEIGHBOR_FEATURES")
                require(torch.isfinite(neighbor_features[neighbor_mask]).all(), "NONFINITE_OBSERVED_SOURCE_NEIGHBOR")
                if neighbor_mask.any():
                    require(torch.max(torch.abs(neighbor_features[neighbor_mask].norm(dim=-1) - 1)).detach() <= 3e-6,
                            "SOURCE_NEIGHBOR_FEATURES_NOT_UNIT_L2")
                valid = neighbor_mask
                safe = torch.where(valid[..., None], neighbor_features, torch.zeros_like(neighbor_features))
                neighbor = self.project(safe.reshape(-1, self.input_dim)).reshape(len(features), safe.shape[1], self.width)
            else:
                require(neighbor_index.ndim == 2 and neighbor_index.shape[0] == len(z)
                        and 1 <= neighbor_index.shape[1] <= 16 and neighbor_index.dtype == torch.long
                        and neighbor_index.device == z.device, "SOURCE_SPATIAL_GRAPH_REQUIRED")
                valid = neighbor_index >= 0
                require(((neighbor_index >= -1) & (neighbor_index < len(z))).all(), "SOURCE_LOCAL_GRAPH_INDEX_OUT_OF_RANGE")
                neighbor = z[neighbor_index.clamp_min(0)]
            mean = torch.where(valid[..., None], neighbor, torch.zeros_like(neighbor)).sum(1) / valid.sum(1).clamp_min(1)[:, None]
            difference = torch.where(valid.any(1)[:, None], mean - z, torch.zeros_like(z))
            z = z + self.spatial_scale * self.spatial_up(F.gelu(self.spatial_down(difference)))
        return z

    def queries(self, tissue_probability):
        require(tissue_probability.ndim == 2 and tissue_probability.shape[1] == 3
                and torch.isfinite(tissue_probability).all() and (tissue_probability >= 0).all()
                and torch.allclose(tissue_probability.sum(-1), tissue_probability.new_ones(len(tissue_probability)),
                                   atol=1e-6, rtol=0), "BAD_PREDICTED_SOURCE_TISSUE_PROBABILITIES")
        base = self.program_queries.unsqueeze(0).expand(len(tissue_probability), -1, -1)
        if not self.use_condition:
            return base
        coefficients = self.condition_gate(tissue_probability - self.train_tissue_prior)
        return base + self.condition_scale * torch.einsum("nr,krd->nkd", coefficients, self.condition_basis)

    def forward(self, features, neighbor_index=None, *, neighbor_features=None, neighbor_mask=None, gate_permutation=None):
        z = self.encode(features, neighbor_index, neighbor_features=neighbor_features, neighbor_mask=neighbor_mask)
        tissue_logits = self.tissue_head(z)
        tissue_probability = tissue_logits.softmax(-1)
        require(gate_permutation is None or self.gate_mode == "random", "GATE_PERMUTATION_ONLY_ALLOWED_FOR_RANDOM_MODE")
        if self.gate_mode == "random":
            order = fixed_feature_gate_permutation(features) if gate_permutation is None else gate_permutation
            require(order.shape == tissue_probability.shape and order.dtype == torch.long and order.device == features.device
                    and torch.equal(order.sort(-1).values, torch.arange(3, device=features.device).expand(len(features), -1)),
                    "BAD_FIXED_CLASS_GATE_PERMUTATION")
            # Only the index is nondifferentiable. Do not detach probabilities:
            # the program objective must reach the tissue classifier in both arms.
            gate_probability = tissue_probability.gather(-1, order)
        else:
            order = torch.arange(3, device=features.device).expand(len(features), -1)
            gate_probability = tissue_probability
        query = self.queries(gate_probability)
        prediction = torch.einsum("nd,nkd->nk", z, query) / math.sqrt(self.width) + self.program_bias
        visual_program = F.normalize(z[:, None, :] + query, dim=-1, eps=1e-12)
        return dict(embedding=z, tissue_logits=tissue_logits, tissue_probs=tissue_probability,
                    tissue_probabilities=tissue_probability, queries=query,
                    program_gate_probs=gate_probability, gate_permutation=order, program_queries=query,
                    program_predictions=prediction, visual_program_embeddings=visual_program)

    def encode_gene_programs(self, measurements):
        require(self.gene_reader is not None, "SOURCE_GENE_READER_NOT_ATTACHED")
        result = self.gene_reader(measurements)
        require("tokens" in result and "program_mask" in result, "BAD_SHARED_GENE_READER_INTERFACE")
        return result

    def descriptor(self):
        gene_signature = None
        if self.gene_reader is not None:
            mapping = self.gene_reader.mapping
            digest = hashlib.sha256("\n".join(mapping.gene_symbols).encode())
            for value in (mapping.edge_program, mapping.edge_gene, mapping.edge_sign):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            gene_signature = digest.hexdigest()
        return dict(schema="bgb_en_r6_source_model_v1", arm=self.arm, input_dim=self.input_dim,
                    width=self.width, program_order=list(PROGRAMS), tissue_classes=3,
                    condition_rank=2, condition_scale=self.condition_scale, gate_mode=self.gate_mode,
                    random_gate_algorithm=RANDOM_GATE_ALGORITHM, legacy_Gaussian_random_gate_used=False,
                    random_gate_preserves_own_probability_row_multiset=True,
                    random_gate_matches_between_model_probability_distributions=False,
                    random_gate_retains_confidence_information=True,
                    use_spatial=self.use_spatial, spatial_scale=self.spatial_scale,
                    gene_reader_attached=self.gene_reader is not None,
                    gene_mapping_signature=gene_signature,
                    input_feature_unit=FEATURE_UNIT, tissue_truth_is_forward_input=False,
                    gene_truth_is_visual_forward_input=False,
                    queries_and_tissue_model_included_in_transfer=True,
                    program_query_training_objective_enabled=self.arm not in ("S0", "S1"),
                    pretrained_status_requires_separate_fit_and_replay_receipt=True)

    def transfer_state(self):
        return dict(schema="bgb_en_r6_complete_source_state_v1", descriptor=self.descriptor(),
                    state_dict={name: value.detach().cpu().clone() for name, value in self.state_dict().items()},
                    projection_only_transfer=False)

    def load_transfer_state(self, artifact, *, freeze=True):
        require(artifact.get("schema") == "bgb_en_r6_complete_source_state_v1"
                and artifact.get("projection_only_transfer") is False
                and artifact.get("descriptor") == self.descriptor(), "INCOMPLETE_OR_DIFFERENT_R6_TRANSFER_CONTRACT")
        expected, incoming = self.state_dict(), artifact.get("state_dict", {})
        require(set(expected) == set(incoming) and all(expected[key].shape == incoming[key].shape
                and torch.isfinite(incoming[key]).all() for key in expected), "R6_FULL_STATE_MISSING_OR_NONFINITE")
        self.load_state_dict(incoming, strict=True)
        if freeze:
            self.requires_grad_(False)
            self.eval()
        return dict(status="PASS_FULL_SOURCE_STATE_TRANSFER", tensors=len(incoming),
                    source_frozen=freeze, projection_program_queries_and_tissue_transferred=True)

    def parameter_budget(self):
        return dict(registered=sum(p.numel() for p in self.parameters()),
                    requires_grad=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    conditional_parameters=sum(p.numel() for p in self.condition_gate.parameters()) + self.condition_basis.numel(),
                    same_effective_capacity_or_FLOPs_claimed=False)


def spot_alignment_loss(visual, molecular, program_mask, negative_index, *, temperature=.1):
    require(visual.shape == molecular.shape and visual.ndim == 3 and visual.shape[1] == 4
            and program_mask.shape == visual.shape[:2] and program_mask.dtype == torch.bool
            and negative_index.ndim == 2
            and negative_index.shape[0] == len(visual) and negative_index.dtype == torch.long,
            "BAD_SOURCE_SPOT_ALIGNMENT_SHAPES")
    require(visual.device == molecular.device == program_mask.device == negative_index.device
            and torch.isfinite(visual[program_mask]).all() and torch.isfinite(molecular[program_mask]).all(),
            "BAD_SOURCE_ALIGNMENT_DEVICE_OR_OBSERVED_VALUES")
    require(temperature > 0 and ((negative_index >= -1) & (negative_index < len(visual))).all(),
            "BAD_SOURCE_ALIGNMENT_NEGATIVES")
    valid_negative = negative_index >= 0
    require(not (valid_negative & (negative_index == torch.arange(len(visual), device=visual.device)[:, None])).any(),
            "SOURCE_ALIGNMENT_NEGATIVE_IS_SAME_SPOT")
    v = F.normalize(torch.where(program_mask[..., None], visual, torch.zeros_like(visual)), dim=-1, eps=1e-12)
    g = F.normalize(torch.where(program_mask[..., None], molecular, torch.zeros_like(molecular)), dim=-1, eps=1e-12)
    positive = (v * g).sum(-1)
    negative = torch.einsum("nkd,nmkd->nkm", v, g[negative_index.clamp_min(0)])
    valid = valid_negative[:, None, :] & program_mask[negative_index.clamp_min(0)].transpose(1, 2)
    score = torch.cat((positive[..., None], negative), -1) / temperature
    allowed = torch.cat((program_mask[..., None], valid & program_mask[..., None]), -1)
    active = program_mask & valid.any(-1)
    if not active.any():
        return (v.sum() + g.sum()) * 0., dict(active_spot_programs=0, no_valid_negative_anchors=True)
    score = score.masked_fill(~allowed, -torch.inf)
    loss = F.cross_entropy(score[active], torch.zeros(int(active.sum()), dtype=torch.long, device=visual.device))
    return loss, dict(active_spot_programs=int(active.sum()), no_valid_negative_anchors=False,
                      negatives_compare_same_program_only=True, dense_N_squared_matrix=False)


def neighborhood_difference_loss(prediction, targets, valid_mask, true_neighbor_index):
    require(prediction.shape == targets.shape == valid_mask.shape and prediction.ndim == 2
            and prediction.shape[1] == 4 and true_neighbor_index.ndim == 2
            and true_neighbor_index.shape[0] == len(prediction) and true_neighbor_index.dtype == torch.long
            and prediction.device == targets.device == valid_mask.device == true_neighbor_index.device
            and torch.isfinite(prediction).all()
            and valid_mask.dtype == torch.bool and torch.isfinite(targets[valid_mask]).all(),
            "BAD_SOURCE_NEIGHBOR_LOSS_SHAPES")
    edge = true_neighbor_index >= 0
    require(((true_neighbor_index >= -1) & (true_neighbor_index < len(prediction))).all(), "BAD_SOURCE_NEIGHBOR_LOSS_INDEX")
    other = true_neighbor_index.clamp_min(0)
    edge &= other != torch.arange(len(prediction), device=prediction.device)[:, None]
    observed = edge[..., None] & valid_mask[:, None, :] & valid_mask[other]
    pred_diff = prediction[:, None, :] - prediction[other]
    true_diff = targets[:, None, :] - targets[other]
    safe = torch.where(observed, true_diff, torch.zeros_like(true_diff))
    error = F.huber_loss(pred_diff, safe, reduction="none", delta=1.)
    per_spot = torch.where(observed, error, torch.zeros_like(error)).sum((1, 2)) / observed.sum((1, 2)).clamp_min(1)
    active = observed.any((1, 2))
    return (per_spot[active].mean() if active.any() else prediction.sum() * 0.), dict(
        active_spots=int(active.sum()), directed_nonself_edges=int(edge.sum()),
        targets_are_neighbor_differences_not_forced_smoothness=True)


def source_loss(model, output, target_z, tissue_y, *, valid_mask=None, class_weights=None,
                program_target_permutation=None, gene_programs=None, negative_index=None,
                true_neighbor_index=None, training_donor_codes=None,
                tissue_weight=.1, align_weight=.01, neighbor_weight=.01):
    """Full-source uniform-spot objective; callers supply training rows only."""
    prediction = output["program_predictions"]
    mask = torch.isfinite(target_z) if valid_mask is None else valid_mask
    require(prediction.shape == target_z.shape == mask.shape and prediction.shape[1] == 4,
            "SOURCE_TARGET_SHAPE_MISMATCH")
    require(mask.dtype == torch.bool and prediction.device == target_z.device == mask.device
            and torch.isfinite(target_z[mask]).all(), "NONFINITE_OBSERVED_SOURCE_TARGET")
    target = torch.where(mask, target_z, torch.zeros_like(target_z))
    if program_target_permutation is not None:
        require(model.arm == "S5" and program_target_permutation.dtype == torch.long
                and torch.equal(program_target_permutation.sort().values, torch.arange(len(target), device=target.device)),
                "WRONG_SOURCE_TARGET_MAPPING_NOT_A_BIJECTION")
        require(training_donor_codes is not None and training_donor_codes.shape == (len(target),)
                and torch.all(training_donor_codes[program_target_permutation] != training_donor_codes),
                "S5_TARGET_MAPPING_MUST_CHANGE_DONOR_EVERY_ROW")
        target, mask = target[program_target_permutation], mask[program_target_permutation]
    require(model.arm != "S5" or program_target_permutation is not None, "S5_REQUIRES_EXPLICIT_WRONG_DONOR_TARGET_MAPPING")
    huber = F.huber_loss(prediction, target, delta=1., reduction="none")
    observed = mask.sum(0)
    require((observed > 0).all(), "MISSING_WHOLE_SOURCE_PROGRAM_TARGET")
    program = (torch.where(mask, huber, torch.zeros_like(huber)).sum(0) / observed).mean()
    tissue = F.cross_entropy(output["tissue_logits"], tissue_y, weight=class_weights, reduction="none").mean()
    loss = prediction.sum() * 0.
    if model.arm not in ("S0", "S1"):
        loss = loss + program
    if model.arm not in ("S0", "S2"):
        loss = loss + tissue_weight * tissue
    alignment = prediction.sum() * 0.
    alignment_audit = dict(active_spot_programs=0)
    if model.use_alignment:
        require(gene_programs is not None and negative_index is not None, "S7_REQUIRES_TRUE_GENE_PROGRAMS_AND_MATCHED_NEGATIVES")
        alignment, alignment_audit = spot_alignment_loss(output["visual_program_embeddings"], gene_programs["tokens"],
                                                        gene_programs["program_mask"], negative_index)
        loss = loss + align_weight * alignment
    neighborhood = prediction.sum() * 0.
    neighborhood_audit = dict(active_spots=0)
    if model.use_spatial:
        require(true_neighbor_index is not None, "SPATIAL_SOURCE_REQUIRES_TRUE_EDGE_LOSS_SCOPE")
        neighborhood, neighborhood_audit = neighborhood_difference_loss(prediction, target_z, valid_mask if valid_mask is not None else torch.isfinite(target_z), true_neighbor_index)
        loss = loss + neighbor_weight * neighborhood
    return loss, dict(program_huber=program, tissue_CE=tissue, alignment=alignment,
                      neighborhood=neighborhood, alignment_audit=alignment_audit,
                      neighborhood_audit=neighborhood_audit,
                      program_target_is_wrong_donor=model.arm == "S5",
                      training_spot_weights="uniform", tissue_truth_was_forward_input=False)
