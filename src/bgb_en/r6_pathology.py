"""Native-coordinate UNI2 evidence and WSI-balanced R6 program reading.

This module does not load cohorts, outcomes or checkpoints. The caller must
verify the checkpoint and fit/replay receipt bytes before supplying their
digests. Complete frozen source state is independently checked here; a valid
digest alone does not establish that a source model was successfully trained.
Native level-0 distances are never presented as micrometers.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from .r6_source_model import FEATURE_UNIT, PROGRAMS, SourceProgramModel


def require(value, code):
    if not bool(value):
        raise ValueError(code)


def _patch_shape(x, mask, width):
    require(isinstance(x, torch.Tensor) and x.ndim == 4 and x.shape[-1] == width
            and x.dtype == torch.float32 and x.shape[0] > 0 and x.shape[1] > 0,
            "PATCH_REQUIRES_B_S_N_FLOAT32_FEATURES")
    require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
            and mask.shape == x.shape[:3] and mask.device == x.device,
            "PATCH_MASK_SHAPE_DTYPE_OR_DEVICE")
    require(torch.isfinite(x[mask]).all(), "NONFINITE_VALID_PATCH_FEATURE")


def normalize_uni2_patches(features, patch_mask):
    """Stateless float32 row-L2; invalid positions are exact zero."""
    _patch_shape(features, patch_mask, 1536)
    observed = features[patch_mask]
    norms = observed.norm(dim=-1, keepdim=True)
    require(torch.isfinite(norms).all() and (norms > 1e-12).all(),
            "ZERO_OR_NONFINITE_VALID_PATCH_NORM")
    result = torch.zeros_like(features)
    result[patch_mask] = observed / norms
    return result


@dataclass(frozen=True)
class NativeNeighborhood:
    """Indices are local N indices: no edge can change B or S."""
    neighbor_indices: torch.Tensor
    neighbor_counts: torch.Tensor
    patch_mask: torch.Tensor
    input_sha256: str
    native_hop: float = 512.
    radius: float = 768.

    def clone(self):
        return NativeNeighborhood(self.neighbor_indices.clone(), self.neighbor_counts.clone(),
                                  self.patch_mask.clone(), self.input_sha256, self.native_hop, self.radius)

    def public_audit(self):
        counts = self.neighbor_counts[self.patch_mask]
        return dict(schema="bgb_en_r6_native_patch_geometry_v1", input_sha256=self.input_sha256,
                    coordinate_unit="native_level0_coordinate_units_not_micrometers",
                    native_hop=self.native_hop, radius=self.radius,
                    maximum_neighbors=self.neighbor_indices.shape[-1],
                    valid_patches=int(self.patch_mask.sum()),
                    count_min=int(counts.min()) if len(counts) else 0,
                    count_max=int(counts.max()) if len(counts) else 0,
                    cross_patient_or_WSI_edges=0, self_included=True,
                    true_coordinates_or_identifiers_exported=False)


class NativePatchGeometry:
    """CPU-only, optionally cached, independent WSI radius neighborhoods."""
    def __init__(self, *, maximum_neighbors=16, native_hop=512., cache=True):
        require(isinstance(maximum_neighbors, int) and 1 <= maximum_neighbors <= 16,
                "NATIVE_NEIGHBOR_BUDGET_MUST_BE_AT_MOST_16")
        require(native_hop == 512., "AUDITED_UNI2_NATIVE_HOP_MUST_BE_512")
        self.maximum_neighbors, self.native_hop = maximum_neighbors, float(native_hop)
        self.radius, self.cache_enabled, self._cache = 1.5 * self.native_hop, bool(cache), {}

    def build(self, coordinates, patch_mask):
        coordinate = (coordinates.detach().cpu().numpy() if isinstance(coordinates, torch.Tensor)
                      else np.asarray(coordinates))
        mask = (patch_mask.detach().cpu().numpy() if isinstance(patch_mask, torch.Tensor)
                else np.asarray(patch_mask))
        require(coordinate.ndim == 4 and coordinate.shape[-1] == 2 and coordinate.shape[0] > 0
                and coordinate.shape[1] > 0 and coordinate.dtype.kind in "fiu"
                and mask.shape == coordinate.shape[:3] and mask.dtype == np.bool_,
                "NATIVE_COORDINATE_OR_MASK_SHAPE")
        require(np.isfinite(coordinate[mask]).all(), "NONFINITE_VALID_NATIVE_COORDINATE")
        safe = np.zeros(coordinate.shape, np.float64)
        safe[mask] = coordinate[mask]
        digest = hashlib.sha256(json.dumps(dict(shape=list(safe.shape), native_hop=self.native_hop,
                                                radius=self.radius, k=self.maximum_neighbors),
                                          sort_keys=True).encode())
        digest.update(mask.tobytes()); digest.update(safe.tobytes())
        key = digest.hexdigest()
        if self.cache_enabled and key in self._cache:
            return self._cache[key].clone()
        indices = np.full((*mask.shape, self.maximum_neighbors), -1, np.int64)
        counts = np.zeros(mask.shape, np.int64)
        for b in range(mask.shape[0]):
            for s in range(mask.shape[1]):
                positions = np.flatnonzero(mask[b, s])
                xy = safe[b, s, positions]
                require(len(np.unique(xy, axis=0)) == len(xy), "DUPLICATE_VALID_COORDINATE_WITHIN_WSI")
                if not len(positions):
                    continue
                tree = cKDTree(xy)
                for local, candidates in enumerate(tree.query_ball_point(xy, self.radius)):
                    # Coordinate tie breaks make geometric reading permutation equivariant.
                    ordered = sorted(candidates, key=lambda j: (float(np.linalg.norm(xy[j] - xy[local])),
                                                                float(xy[j, 0]), float(xy[j, 1])))
                    chosen = positions[np.asarray(ordered[:self.maximum_neighbors], np.int64)]
                    n = positions[local]
                    indices[b, s, n, :len(chosen)] = chosen
                    counts[b, s, n] = len(chosen)
        result = NativeNeighborhood(torch.from_numpy(indices), torch.from_numpy(counts),
                                    torch.from_numpy(mask.copy()), key, self.native_hop, self.radius)
        if self.cache_enabled:
            self._cache[key] = result.clone()
        return result

    __call__ = build


def source_model_state_sha256(source_model):
    """Fingerprint the entire descriptor/state, including queries and gene stem."""
    require(isinstance(source_model, SourceProgramModel), "COMPLETE_R6_SOURCE_MODEL_REQUIRED")
    artifact = source_model.transfer_state()
    require(artifact.get("schema") == "bgb_en_r6_complete_source_state_v1"
            and artifact.get("projection_only_transfer") is False, "COMPLETE_SOURCE_TRANSFER_REQUIRED")
    digest = hashlib.sha256(json.dumps(artifact["descriptor"], sort_keys=True,
                                      separators=(",", ":")).encode())
    for name, tensor in sorted(artifact["state_dict"].items()):
        require(torch.isfinite(tensor).all(), "NONFINITE_SOURCE_STATE")
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenSourceProvenance:
    checkpoint_sha256: str
    fit_replay_receipt_sha256: str
    complete_source_state_sha256: str
    source_arm: str = "S7"
    feature_unit: str = FEATURE_UNIT
    program_order: tuple[str, ...] = tuple(PROGRAMS)
    projection_only_transfer: bool = False

    def __post_init__(self):
        require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in (self.checkpoint_sha256, self.fit_replay_receipt_sha256,
                                  self.complete_source_state_sha256)), "SOURCE_PROVENANCE_REQUIRES_EXACT_SHA256")
        require(self.source_arm == "S7" and self.feature_unit == FEATURE_UNIT
                and tuple(self.program_order) == tuple(PROGRAMS) and self.projection_only_transfer is False,
                "WRONG_SOURCE_PATCH_TRANSFER_CONTRACT")

    def check(self, source_model):
        require(isinstance(source_model, SourceProgramModel) and source_model.arm == self.source_arm
                and source_model.width == 64 and source_model.input_dim == 1536
                and tuple(source_model.descriptor()["program_order"]) == self.program_order
                and source_model.descriptor()["input_feature_unit"] == self.feature_unit,
                "SOURCE_MODEL_DESCRIPTOR_MISMATCH")
        require(not source_model.training and not any(p.requires_grad for p in source_model.parameters()),
                "SOURCE_MODEL_MUST_BE_FROZEN_AND_EVAL")
        require(source_model_state_sha256(source_model) == self.complete_source_state_sha256,
                "COMPLETE_SOURCE_STATE_SHA_MISMATCH")


def frozen_source_patch_encode(source_model, features, patch_mask, *, source_provenance,
                               batch=1024, enable_grad=False):
    """Encode only valid patches; enable_grad audits features, never source weights."""
    require(isinstance(source_provenance, FrozenSourceProvenance), "EXPLICIT_SOURCE_PROVENANCE_REQUIRED")
    source_provenance.check(source_model)
    require(isinstance(batch, int) and batch > 0, "SOURCE_PATCH_BATCH_MUST_BE_POSITIVE")
    parameter = next(source_model.parameters())
    require(parameter.device == features.device and parameter.dtype == torch.float32,
            "SOURCE_PATCH_MODEL_DEVICE_OR_DTYPE_MISMATCH")
    with torch.enable_grad() if enable_grad else torch.no_grad():
        normalized = normalize_uni2_patches(features, patch_mask)
        shape = normalized.shape[:3]
        positions = patch_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        flat = normalized.reshape(-1, 1536)
        states = normalized.new_zeros((flat.shape[0], 64))
        tissues = normalized.new_zeros((flat.shape[0], 3))
        keys = normalized.new_zeros((flat.shape[0], 4, 64))
        scores = normalized.new_zeros((flat.shape[0], 4))
        for ids in positions.split(batch):
            if not len(ids):
                continue
            output = source_model(flat[ids])
            require(all(torch.isfinite(output[name]).all() for name in
                        ("embedding", "tissue_probs", "queries", "program_predictions")),
                    "NONFINITE_FROZEN_SOURCE_PATCH_OUTPUT")
            states[ids] = output["embedding"]
            tissues[ids] = output["tissue_probs"]
            keys[ids] = output["queries"]
            scores[ids] = output["program_predictions"]
        # Shape restoration must remain inside enable_grad even when the caller
        # is in no_grad; otherwise a view can disconnect the feature audit path.
        return dict(states=states.reshape(*shape, 64), tissue_probabilities=tissues.reshape(*shape, 3),
                    program_key_prior=keys.reshape(*shape, 4, 64), program_score_prior=scores.reshape(*shape, 4),
                    program_bias_prior=source_model.program_bias.detach().clone(), patch_mask=patch_mask.clone(),
                    diagnostics=dict(valid_patches=int(patch_mask.sum()), only_valid_source_inputs=True,
                        feature_unit=FEATURE_UNIT, source_parameter_gradients_enabled=False,
                        feature_gradient_enabled=bool(enable_grad),
                        source_state_sha256=source_provenance.complete_source_state_sha256,
                        source_checkpoint_sha256=source_provenance.checkpoint_sha256,
                        source_fit_replay_receipt_sha256=source_provenance.fit_replay_receipt_sha256,
                        source_fit_success_is_not_inferred_from_digest=True))


def _checked_neighbor_indices(z, patch_mask, neighbor_indices):
    if isinstance(neighbor_indices, NativeNeighborhood):
        require(torch.equal(neighbor_indices.patch_mask, patch_mask.detach().cpu()),
                "NATIVE_GEOMETRY_MASK_CHANGED")
        neighbor_indices = neighbor_indices.neighbor_indices
    require(isinstance(neighbor_indices, torch.Tensor) and neighbor_indices.dtype == torch.long
            and neighbor_indices.ndim == 4 and neighbor_indices.shape[:3] == patch_mask.shape
            and 1 <= neighbor_indices.shape[-1] <= 16, "BAD_NATIVE_NEIGHBOR_INDEX_SHAPE")
    indices = neighbor_indices.to(z.device)
    n = z.shape[2]
    require(((indices >= -1) & (indices < n)).all(), "NATIVE_NEIGHBOR_INDEX_OUT_OF_RANGE")
    require((indices[~patch_mask] == -1).all(), "PADDING_CENTER_CANNOT_HAVE_NEIGHBORS")
    if n:
        selected_mask = patch_mask.reshape(-1, n).gather(1, indices.reshape(-1, n * indices.shape[-1]).clamp_min(0))
        require((selected_mask.reshape_as(indices) | (indices < 0)).all(), "NATIVE_NEIGHBOR_POINTS_TO_PADDING")
    return indices


class NativeSpatialAdapter(nn.Module):
    def __init__(self, *, width=64, rank=4, scale=.1):
        super().__init__()
        require(width == 64 and rank == 4 and scale == .1, "FIXED_NATIVE_SPATIAL_ADAPTER_BUDGET")
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.up.weight, std=.02)
        self.scale = float(scale)

    def forward(self, z, patch_mask, neighbor_indices):
        _patch_shape(z, patch_mask, 64)
        indices = _checked_neighbor_indices(z, patch_mask, neighbor_indices)
        safe = torch.where(patch_mask[..., None], z, torch.zeros_like(z))
        b, s, n, d = z.shape
        valid = indices >= 0
        if n:
            neighbor = safe.reshape(b * s, n, d).gather(1, indices.reshape(b * s, -1, 1).clamp_min(0).expand(-1, -1, d))
            neighbor = neighbor.reshape(*indices.shape, d)
            mean = torch.where(valid[..., None], neighbor, torch.zeros_like(neighbor)).sum(-2)
            mean = mean / valid.sum(-1).clamp_min(1)[..., None]
        else:
            mean = torch.zeros_like(safe)
        difference = torch.where(valid.any(-1)[..., None], mean - safe, torch.zeros_like(safe))
        delta = self.scale * self.up(F.gelu(self.down(difference)))
        states = torch.where(patch_mask[..., None], safe + delta, torch.zeros_like(safe))
        return dict(states=states, delta=delta, neighbor_mean=mean, neighbor_counts=valid.sum(-1),
                    diagnostics=dict(rank=4, scale=self.scale, trainable_parameters=sum(p.numel() for p in self.parameters()),
                                     coordinate_unit="native_level0_coordinate_units_not_micrometers",
                                     no_neighbor_delta_exact_zero=True, cross_patient_or_WSI_edges=0))


def attention_assignment_permute(values, patch_mask, *, seed):
    """Shuffle value assignment within each WSI; queries/logits/weights stay put."""
    require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
            "ASSIGNMENT_PERMUTATION_REQUIRES_NONNEGATIVE_SEED")
    require(values.ndim == 4 and values.dtype == torch.float32 and patch_mask.shape == values.shape[:3]
            and patch_mask.dtype == torch.bool and patch_mask.device == values.device,
            "BAD_ASSIGNMENT_PERMUTATION_INPUT")
    require(torch.isfinite(values[patch_mask]).all(), "NONFINITE_VALID_ASSIGNMENT_VALUE")
    result = torch.zeros_like(values)
    assignment = torch.full(patch_mask.shape, -1, dtype=torch.long, device=values.device)
    for b in range(values.shape[0]):
        for s in range(values.shape[1]):
            ids = patch_mask[b, s].nonzero(as_tuple=False).flatten()
            target = ids
            if len(ids) > 1:
                generator = torch.Generator(device="cpu").manual_seed((seed + 104729 * b + 1009 * s) % (2**63 - 1))
                order = torch.randperm(len(ids), generator=generator).to(ids.device)
                target = torch.empty_like(ids)
                target[order] = ids[order.roll(1)]
            result[b, s, ids] = values[b, s, target]
            assignment[b, s, ids] = target
    changed = (assignment != torch.arange(values.shape[2], device=values.device)[None, None, :]) & patch_mask
    return dict(values=result, assignment=assignment, diagnostics=dict(seed=seed,
        changed_valid_patches=int(changed.sum()), valid_patches=int(patch_mask.sum()),
        preserves_each_WSI_value_multiset=True, changes_queries_logits_or_attention_weights=False,
        cross_patient_or_WSI_assignment=False))


class WSIProgramReader(nn.Module):
    """Read original patch values, append a zero-value null, then equal-weight WSIs."""
    def __init__(self, *, width=64, source_prior_scale=.1):
        super().__init__()
        require(width == 64 and source_prior_scale == .1, "FIXED_WSI_READER_WIDTH_OR_PRIOR_SCALE")
        self.width, self.source_prior_scale = width, float(source_prior_scale)

    def forward(self, query, z, patch_mask, program_mask, *, use_source_prior=False,
                program_score_prior=None, program_key_prior=None, program_bias_prior=None,
                source_states_prior=None, assignment_shuffle_seed=None):
        _patch_shape(z, patch_mask, self.width)
        b, s, n, d = z.shape
        require(query.ndim == 3 and query.shape[0] == b and query.shape[2] == d
                and query.shape[1] > 0 and query.dtype == torch.float32 and query.device == z.device,
                "BAD_PATIENT_PROGRAM_QUERY")
        k = query.shape[1]
        require(program_mask.shape == (b, k) and program_mask.dtype == torch.bool
                and program_mask.device == z.device, "BAD_PATIENT_PROGRAM_MASK")
        require(torch.isfinite(query[program_mask]).all(), "NONFINITE_VALID_PROGRAM_QUERY")
        safe_z = torch.where(patch_mask[..., None], z, torch.zeros_like(z))
        safe_q = torch.where(program_mask[..., None], query, torch.zeros_like(query))
        observed = patch_mask[:, :, :, None] & program_mask[:, None, None, :]
        logits = torch.einsum("bkd,bsnd->bskn", safe_q, safe_z) / math.sqrt(d)
        prior_kind = "none"
        if use_source_prior:
            require((program_score_prior is not None) != (program_key_prior is not None),
                    "CHOOSE_EXACTLY_ONE_SOURCE_PRIOR_INTERFACE")
            if program_score_prior is not None:
                require(program_bias_prior is None and source_states_prior is None,
                        "EXACT_SOURCE_SCORE_ALREADY_INCLUDES_QUERY_AND_BIAS")
                prior = program_score_prior
                prior_kind = "exact_frozen_source_program_prediction_including_bias"
            else:
                require(program_key_prior.shape == (b, s, n, k, d)
                        and program_key_prior.dtype == torch.float32 and program_key_prior.device == z.device
                        and program_bias_prior is not None and program_bias_prior.shape == (k,)
                        and program_bias_prior.dtype == torch.float32 and program_bias_prior.device == z.device
                        and torch.isfinite(program_bias_prior).all() and source_states_prior is not None,
                        "KEY_PRIOR_REQUIRES_EXACT_SOURCE_STATES_QUERIES_AND_BIAS")
                _patch_shape(source_states_prior, patch_mask, d)
                require(source_states_prior.device == z.device and torch.isfinite(program_key_prior[observed]).all(),
                        "INVALID_SOURCE_STATE_QUERY_PRIOR")
                keys = torch.where(observed[..., None], program_key_prior, torch.zeros_like(program_key_prior))
                source_states = torch.where(patch_mask[..., None], source_states_prior, torch.zeros_like(source_states_prior))
                prior = torch.einsum("bsnd,bsnkd->bsnk", source_states, keys) / math.sqrt(d) + program_bias_prior
                prior_kind = "explicit_frozen_source_states_queries_and_bias_not_target_adapter_states"
            require(prior.shape == (b, s, n, k) and prior.dtype == torch.float32
                    and prior.device == z.device and torch.isfinite(prior[observed]).all(),
                    "BAD_FROZEN_SOURCE_SCORE_PRIOR")
            prior = torch.where(observed, prior, torch.zeros_like(prior))
            logits = logits + self.source_prior_scale * prior.permute(0, 1, 3, 2)
        else:
            require(all(value is None for value in (program_score_prior, program_key_prior,
                    program_bias_prior, source_states_prior)), "SOURCE_PRIOR_SUPPLIED_BUT_NOT_ENABLED")
        valid = observed.permute(0, 1, 3, 2)
        logits = torch.where(valid, logits, torch.zeros_like(logits))
        real_logits = logits.masked_fill(~valid, -torch.inf)
        distribution = torch.cat((real_logits, logits.new_zeros((b, s, k, 1))), dim=-1).softmax(-1)
        weights, null = distribution[..., :n], distribution[..., n]
        value_assignment = None
        values = safe_z
        if assignment_shuffle_seed is not None:
            value_assignment = attention_assignment_permute(safe_z, patch_mask, seed=assignment_shuffle_seed)
            values = value_assignment["values"]
        per_WSI = torch.einsum("bskn,bsnd->bskd", weights, values)
        valid_WSI = patch_mask.any(-1)[..., None] & program_mask[:, None, :]
        valid_WSI_count = valid_WSI.sum(1)
        states = per_WSI.sum(1) / valid_WSI_count.clamp_min(1)[..., None]
        states = torch.where((valid_WSI_count > 0)[..., None], states, torch.zeros_like(states))
        return dict(states=states, weights=weights, null=null, logits=logits, per_WSI=per_WSI,
                    valid_WSI=valid_WSI, valid_WSI_count=valid_WSI_count,
                    program_mask=program_mask & (valid_WSI_count > 0),
                    value_assignment=None if value_assignment is None else value_assignment["assignment"],
                    diagnostics=dict(source_prior_kind=prior_kind, source_prior_scale=self.source_prior_scale,
                        null_logit_fixed=0., null_value_fixed=0., WSI_aggregation="equal_valid_WSI_mean",
                        attention_normalization_axis="valid_patches_plus_null_within_each_WSI",
                        query_condition_changes_read_weights_only=True,
                        query_condition_replaces_original_values=False,
                        assignment_shuffle_enabled=assignment_shuffle_seed is not None,
                        null_is_calibrated_medical_confidence=False))
