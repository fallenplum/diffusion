"""Locked outcome-free source training and independent complete-state replay.

This module deliberately knows no target-cohort context. A fit requires an
external launch receipt and a complete source/code lock. Importing it performs
no IO, training, model selection or device allocation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Callable

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bgb_en.r6_source import SourceContract, load_private_cache, sha256, canonical_hash
from src.bgb_en.r6_source_model import (
    SourceProgramModel, SourceFitStatistics, PROGRAMS, require, array_sha,
    wrong_donor_permutation, joint_permutation_audit, local_neighbor_indices,
    source_loss, fixed_feature_gate_permutation, gate_probability_audit, RANDOM_GATE_ALGORITHM,
)
from src.bgb_en.r6_omics import (
    load_source_program_mapping, GeneValueBank, GeneEvidenceTransform, SparseProgramEncoder,
)


CONTROLS = ("MATCHED", "WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH",
            "RANDOM_GATE", "WRONG_NEIGHBOR", "WRONG_SPOT_ALIGNMENT")
CODE_PATHS = (
    "scripts/bgb_en_r6_pretrain.py", "src/bgb_en/r6_source.py",
    "src/bgb_en/r6_source_model.py", "src/bgb_en/r6_omics.py",
    "src/bgb_en/external.py", "scripts/bgb_en_r6_schedule_source.py",
)
FIXED_DEFAULTS = dict(epochs=80, checkpoint_epochs=[20, 40, 80],
    optimizer="Adam", learning_rate=.001, weight_decay=0., parameter_L2=.0001,
    gradient_clip_norm=5., source_microbatch=128, tissue_CE_weight=.1,
    alignment_weight=.01, neighborhood_weight=.01, alignment_negatives=8,
    alignment_temperature=.1, target_permutation_depth_quantiles=4,
    full_output_replay_tolerance=2e-6, training_spot_weights="uniform",
    heldout_unit="equal_donor", fixed_training_updates=80,
    output_gradient_cache="two_pass_deterministic_all_negative_key_gradients")


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _private_directory(path, *, cache_root=None):
    path = Path(path).resolve()
    require(not any((parent / ".git").exists() for parent in (path, *path.parents)),
            "SOURCE_ARTIFACTS_MUST_BE_OUTSIDE_ANY_GIT_TREE")
    if cache_root is not None:
        require(not path.is_relative_to(Path(cache_root).resolve()), "SOURCE_OUTPUT_CANNOT_MODIFY_INPUT_CACHE")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def current_code_hashes():
    return {relative: sha256(ROOT / relative) for relative in CODE_PATHS}


def validate_lock(config):
    require(config.get("schema") == "bgb_en_r6_source_pretrain_lock_v1"
            and config.get("status") == "LOCKED_BEFORE_SOURCE_FITS", "UNLOCKED_SOURCE_PRETRAIN_CONFIG")
    require(config.get("source_code_sha256") == current_code_hashes(), "SOURCE_CODE_SHA_GUARD_FAILED")
    fixed = dict(FIXED_DEFAULTS)
    micro = config.get("fixed_defaults", {}).get("source_microbatch")
    require(micro in (64, 128), "SOURCE_MICROBATCH_MUST_BE_PRELOCKED_64_OR_128")
    fixed["source_microbatch"] = micro
    require(config.get("fixed_defaults") == fixed, "SOURCE_FIXED_TRAINING_CONTRACT_CHANGED")
    require(len(config.get("source_cache_manifest_sha256", "")) == 64
            and len(config.get("fresh_certificate_sha256", "")) == 64, "SOURCE_INPUT_SHA_LOCK_REQUIRED")
    cells = config.get("cells", [])
    require(bool(cells) and len({canonical_hash(c) for c in cells}) == len(cells), "SOURCE_CELL_MATRIX_EMPTY_OR_DUPLICATE")
    for cell in cells:
        validate_cell(cell)
    return fixed


def validate_cell(cell):
    require(set(cell) in ({"arm", "control", "heldout_donor", "seed"},
                         {"arm", "control", "heldout_donor", "seed", "schedule_phase"})
            and cell["arm"] in tuple("S" + str(i) for i in range(8))
            and cell["control"] in CONTROLS and cell["seed"] in (31, 47)
            and cell["heldout_donor"] in (-1, 0, 1, 2, 3, 4, 5), "INVALID_LOCKED_SOURCE_CELL")
    if "schedule_phase" in cell:
        require(cell["schedule_phase"] in ("PRIMARY", "SPATIAL", "SPOT_ALIGNMENT", "FULL_TRANSFER"),
                "INVALID_SOURCE_SCHEDULE_PHASE_METADATA")
    allowed = dict(S0=("MATCHED",), S1=("MATCHED",), S2=("MATCHED",), S3=("MATCHED",),
        S4=("MATCHED", "WITHIN_SECTION_TISSUE_DEPTH", "RANDOM_GATE"), S5=("WRONG_DONOR",),
        S6=("MATCHED", "WRONG_NEIGHBOR"),
        S7=("MATCHED", "WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH", "WRONG_SPOT_ALIGNMENT"))
    require(cell["control"] in allowed[cell["arm"]], "SOURCE_ARM_CONTROL_NOT_PREDEFINED")
    require(cell["heldout_donor"] != -1 or cell["arm"] == "S7", "FULL_SOURCE_TRANSFER_IS_FIXED_S7_ONLY")


def verify_launch_gate(path, expected_sha256):
    require(sha256(path) == expected_sha256, "SOURCE_LAUNCH_GATE_SHA_CHANGED")
    gate = json.loads(Path(path).read_text(encoding="utf-8"))
    require(gate.get("r6_formal_training_allowed") is True, "SOURCE_FORMAL_TRAINING_GATE_NOT_PASSED")
    return dict(sha256=expected_sha256, r6_formal_training_allowed=True)


def within_section_tissue_depth_permutation(section, tissue, library, indices, *, seed, quantiles=4):
    """Joint rows within fixed train-only section/tissue/depth strata, no outcome."""
    section, tissue, library = np.asarray(section), np.asarray(tissue), np.asarray(library, float)
    idx = np.asarray(indices, np.int64)
    require(len(idx) > 1 and len(np.unique(idx)) == len(idx) and (library[idx] > 0).all()
            and np.isfinite(library[idx]).all() and quantiles == 4, "BAD_WITHIN_STRATUM_SOURCE_SCOPE")
    depth = np.log1p(library)
    bins = np.full(len(section), -1, np.int64)
    for sec in np.unique(section[idx]):
        rows = idx[section[idx] == sec]
        boundary = np.quantile(depth[rows], [1 / quantiles, 2 / quantiles, 3 / quantiles])
        bins[rows] = np.searchsorted(boundary, depth[rows], side="right")
    rng = np.random.default_rng(seed)
    mapping = np.arange(len(section))
    counts, singletons = [], 0
    for sec, label, bin_ in sorted(set(zip(section[idx], tissue[idx], bins[idx]))):
        rows = idx[(section[idx] == sec) & (tissue[idx] == label) & (bins[idx] == bin_)]
        counts.append(len(rows))
        if len(rows) == 1:
            singletons += 1
            continue
        order = rng.permutation(rows)
        mapping[order] = np.roll(order, -1)
    require(np.array_equal(np.sort(mapping[idx]), np.sort(idx)), "WITHIN_STRATUM_MAPPING_NOT_BIJECTIVE")
    changed = mapping[idx] != idx
    require(changed.any(), "BLOCKED_NO_SHUFFLABLE_WITHIN_SOURCE_STRATUM")
    return mapping, dict(mapping_sha256=array_sha(mapping), training_rows=len(idx),
        changed_rows=int(changed.sum()), singleton_rows=singletons, strata=len(counts),
        section_tissue_depth_bins_preserved=True, target_four_vector_and_mask_must_move_together=True,
        bins_fitted_on_training_section_only=True, unshufflable_rows_retained_explicitly=True)


def geometry_sham_graph(true_graph, section_codes, *, seed):
    """Section-preserving conjugation; no tissue/program/depth label enters."""
    graph = np.asarray(true_graph, np.int64)
    section = np.asarray(section_codes)
    require(graph.ndim == 2 and len(graph) == len(section)
            and ((graph >= -1) & (graph < len(graph))).all(), "BAD_GEOMETRY_SHAM_INPUT")
    permutation = np.arange(len(graph))
    for code in np.unique(section):
        # Every section's random stream is independent. Its heldout geometry
        # cannot change a later training section's random permutation.
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(code)]))
        rows = np.flatnonzero(section == code)
        order = rng.permutation(rows)
        permutation[order] = np.roll(order, -1) if len(order) > 1 else order
    inverse = np.argsort(permutation)
    selected = graph[permutation]
    sham = np.where(selected >= 0, inverse[np.maximum(selected, 0)], -1)
    observed = sham >= 0
    require(np.all(section[sham[observed]] == np.broadcast_to(section[:, None], sham.shape)[observed]),
            "GEOMETRY_SHAM_CROSSES_SECTION")
    require(np.array_equal(np.sort((graph >= 0).sum(1)), np.sort(observed.sum(1))),
            "GEOMETRY_SHAM_CHANGED_DEGREE_DISTRIBUTION")
    return sham, dict(permutation_sha256=array_sha(permutation), graph_sha256=array_sha(sham),
        type="within_section_node_conjugation", degree_histogram_preserved=True,
        self_loops_preserved=True, cross_section_edges=0, tissue_or_depth_labels_used=False,
        composition_matched_claimed=False, physically_real_300um_neighbors_claimed=False)


def hard_negative_indices(donor, section, tissue, library, indices, *, count=8):
    """Same donor/section/tissue, nearest measured depth, no replacement/self."""
    donor, section, tissue = map(np.asarray, (donor, section, tissue))
    depth = np.log1p(np.asarray(library, float))
    idx = np.asarray(indices, np.int64)
    require(count == 8 and np.isfinite(depth[idx]).all(), "HARD_NEGATIVE_CONTRACT_CHANGED")
    result = np.full((len(idx), count), -1, np.int64)
    for key in sorted(set(zip(donor[idx], section[idx], tissue[idx]))):
        rows = np.flatnonzero((donor[idx] == key[0]) & (section[idx] == key[1]) & (tissue[idx] == key[2]))
        rows = rows[np.lexsort((idx[rows], depth[idx[rows]]))]
        for pos, local in enumerate(rows):
            candidates = rows[max(0, pos - count):min(len(rows), pos + count + 1)]
            candidates = candidates[candidates != local]
            order = np.lexsort((idx[candidates], np.abs(depth[idx[candidates]] - depth[idx[local]])))
            chosen = candidates[order[:count]]
            result[local, :len(chosen)] = chosen
    observed = result >= 0
    require(not np.any(result == np.arange(len(idx))[:, None]), "HARD_NEGATIVE_IS_ANCHOR")
    return result, dict(candidate_count=count, eligible_spots=int(observed.any(1).sum()),
        total_spots=len(idx), complete_candidate_spots=int(observed.all(1).sum()),
        singleton_strata_spots=int((~observed.any(1)).sum()),
        no_replacement=True, same_donor_section_tissue=True,
        negative_order="nearest_absolute_log1p_full_library_size_then_source_row",
        labels_used_only_for_source_loss_or_retrieval_evaluation_not_visual_forward=True)


def deterministic_output_gradient_cache_backward(model, forward_chunk: Callable, rows: int,
                                                objective: Callable, *, microbatch=128,
                                                parameter_L2=0., tolerance=2e-6):
    """One exact full-source backward via small-output leaves and replay.

    ``forward_chunk(start,end)`` returns only floating differentiable outputs.
    All cross-microbatch dependencies (contrastive negatives and true-edge
    losses) are computed on complete output leaves. Their complete gradients
    are then applied to each replayed chunk, including negative key gradients.
    No optimizer step, clipping or repeated L2 occurs inside a microbatch.
    """
    require(rows > 0 and microbatch > 0 and parameter_L2 >= 0, "BAD_SOURCE_GRADIENT_CACHE_SCOPE")
    require(not any(isinstance(module, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm))
                    for module in model.modules()), "SOURCE_REPLAY_REQUIRES_DETERMINISTIC_FORWARD")
    pieces = {}
    with torch.no_grad():
        for start in range(0, rows, microbatch):
            output = forward_chunk(start, min(start + microbatch, rows))
            require(output and (not pieces or set(output) == set(pieces)), "SOURCE_OUTPUT_CACHE_KEYS_CHANGED")
            for key, value in output.items():
                require(value.shape[0] == min(microbatch, rows - start)
                        and value.is_floating_point() and torch.isfinite(value).all(), "BAD_SOURCE_CACHED_OUTPUT")
                pieces.setdefault(key, []).append(value.detach())
    leaves = {key: torch.cat(parts, 0).detach().requires_grad_(True) for key, parts in pieces.items()}
    objective_value, audit = objective(leaves)
    require(objective_value.ndim == 0 and torch.isfinite(objective_value), "NONFINITE_FULL_SOURCE_OBJECTIVE")
    objective_value.backward()
    output_gradients = {key: value.grad.detach().clone() if value.grad is not None else torch.zeros_like(value)
                        for key, value in leaves.items()}
    model.zero_grad(set_to_none=True)
    max_error = 0.
    for start in range(0, rows, microbatch):
        end = min(start + microbatch, rows)
        actual = forward_chunk(start, end)
        scalar = None
        for key, value in actual.items():
            error = (value.detach() - leaves[key][start:end].detach()).abs().max().item()
            max_error = max(max_error, error)
            require(error <= tolerance, "SOURCE_DETERMINISTIC_OUTPUT_REPLAY_FAILED")
            if value.requires_grad:
                term = (value * output_gradients[key][start:end]).sum()
                scalar = term if scalar is None else scalar + term
        require(scalar is not None, "NO_SOURCE_TRAINABLE_OUTPUT_PATH")
        scalar.backward()
    penalty = sum(p.square().sum() for p in model.parameters() if p.requires_grad) * parameter_L2
    if parameter_L2:
        penalty.backward()
    require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "NONFINITE_SOURCE_REPLAY_GRADIENT")
    return dict(objective=float(objective_value.detach()), parameter_L2=float(penalty.detach()),
        replay_max_abs_error=max_error, microbatches=math.ceil(rows / microbatch),
        complete_source_updates_represented=1, negative_key_gradients_included=True,
        component_audit=detach_audit(audit))


def detach_audit(value):
    if torch.is_tensor(value):
        return float(value.detach()) if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: detach_audit(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [detach_audit(v) for v in value]
    return value


def _finite_number(value):
    return float(value) if np.isfinite(value) else None


def _correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    keep = np.isfinite(x) & np.isfinite(y)
    return _finite_number(spearmanr(x[keep], y[keep]).statistic) if keep.sum() > 2 and np.ptp(x[keep]) and np.ptp(y[keep]) else None


def evaluate_source_outputs(outputs, targets_z, tissue, library, *, class_weights, valid_mask=None,
                            donor_codes=None, gate_algorithm=None):
    """True heldout labels only score already-produced image-only outputs."""
    prediction = np.asarray(outputs["program_predictions"], float)
    y = np.asarray(targets_z, float)
    mask = np.isfinite(y) if valid_mask is None else np.asarray(valid_mask, bool)
    require(prediction.shape == y.shape == mask.shape and np.isfinite(prediction).all(), "BAD_SOURCE_METRIC_SHAPES")
    err = np.abs(prediction - np.where(mask, y, 0.))
    huber = np.where(err <= 1, .5 * err**2, err - .5)
    logits = np.asarray(outputs["tissue_logits"], float)
    logits = logits - logits.max(1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    tissue = np.asarray(tissue, int)
    pred_y = logits.argmax(1)
    class_recall = [float(np.mean(pred_y[tissue == c] == c)) if np.any(tissue == c) else None for c in range(3)]
    per_program = []
    for k, name in enumerate(PROGRAMS):
        observed = mask[:, k]
        groups = []
        for c in range(3):
            use = observed & (tissue == c)
            groups.append(dict(tissue_class=c, support=int(use.sum()),
                               Huber=float(huber[use, k].mean()) if use.any() else None,
                               Spearman=_correlation(prediction[use, k], y[use, k])))
        per_program.append(dict(program=name, support=int(observed.sum()),
            Huber=float(huber[observed, k].mean()) if observed.any() else None,
            Spearman=_correlation(prediction[observed, k], y[observed, k]), within_tissue=groups,
            log_library_Spearman=_correlation(prediction[:, k], np.log1p(library))))
    valid_huber = [r["Huber"] for r in per_program if r["Huber"] is not None]
    valid_rho = [r["Spearman"] for r in per_program if r["Spearman"] is not None]
    gate = dict(status="NOT_RECORDED")
    by_donor = []
    if all(key in outputs for key in ("tissue_probs", "program_gate_probs", "gate_permutation")):
        gate = gate_probability_audit(outputs["tissue_probs"], outputs["program_gate_probs"], outputs["gate_permutation"],
                                     algorithm=gate_algorithm)
        if donor_codes is not None:
            donor = np.asarray(donor_codes)
            require(donor.shape == (len(y),) and donor.dtype.kind in "iu", "BAD_GATE_AUDIT_DONOR_CODES")
            for code in np.unique(donor):
                rows = donor == code
                by_donor.append(dict(donor_code=int(code), diagnostics=gate_probability_audit(
                    outputs["tissue_probs"][rows], outputs["program_gate_probs"][rows], outputs["gate_permutation"][rows],
                    algorithm=gate_algorithm)))
    return dict(spots=len(y), program_Huber=float(np.mean(valid_huber)) if valid_huber else None,
        mean_program_Spearman=float(np.mean(valid_rho)) if valid_rho else None, per_program=per_program,
        tissue_BA=float(np.mean([v for v in class_recall if v is not None])),
        tissue_present_classes=sum(v is not None for v in class_recall), tissue_class_recall=class_recall,
        tissue_CE=float(np.mean(-logp[np.arange(len(y)), tissue] * np.asarray(class_weights)[tissue])),
        embedding_norm_log_library_Spearman=_correlation(np.linalg.norm(outputs["embedding"], axis=1), np.log1p(library)),
        true_labels_used_for_scoring_only=True, visual_program_forward_is_gene_and_tissue_truth_free=True,
        gate_diagnostics=gate, gate_diagnostics_by_donor=by_donor,
        row_gate_matching_is_within_model_not_across_trained_models=True)


def retrieval_metrics(visual, molecular, program_mask, negatives):
    v = F.normalize(torch.as_tensor(visual), dim=-1)
    g = F.normalize(torch.as_tensor(molecular), dim=-1)
    observed = torch.as_tensor(program_mask, dtype=torch.bool)
    neg = torch.as_tensor(negatives, dtype=torch.long)
    valid = (neg >= 0)[:, None, :] & observed[neg.clamp_min(0)].transpose(1, 2)
    active = observed & valid.any(-1)
    positive = (v * g).sum(-1)
    negative = torch.einsum("nkd,nmkd->nkm", v, g[neg.clamp_min(0)])
    ranks = 1 + ((negative >= positive[..., None]) & valid).sum(-1)
    return dict(active_spot_programs=int(active.sum()),
        top1=float((ranks[active] == 1).float().mean()) if active.any() else None,
        mean_rank=float(ranks[active].float().mean()) if active.any() else None,
        scope="positive_vs_fixed_same_section_tissue_depth_hard_negatives_not_all_spots",
        program_identity_is_shared_across_positive_and_negatives=True)


class SourceFitData:
    """Current fit's true donor-local scales, immutable mappings and raw inputs."""
    def __init__(self, dataset, cell, mapping, device):
        self.dataset, self.cell, self.device = dataset, dict(cell), torch.device(device)
        heldout = cell["heldout_donor"]
        self.train = np.flatnonzero(dataset.donor_codes != heldout) if heldout >= 0 else np.arange(len(dataset))
        self.heldout = np.flatnonzero(dataset.donor_codes == heldout) if heldout >= 0 else np.array([], np.int64)
        self.stats = SourceFitStatistics.fit(dataset.program_targets, dataset.tissue_y, dataset.donor_codes,
            self.train, heldout_donors=([heldout] if heldout >= 0 else ()),
            gene_values=dataset.log1p_cp10k if cell["arm"] == "S7" else None,
            gene_mask=dataset.measured_mask if cell["arm"] == "S7" else None)
        self.target_valid = np.isfinite(dataset.program_targets)
        self.target_z = self.stats.transform_targets(dataset.program_targets, self.target_valid)
        # The row-byte function is stateless: precomputing heldout rows does
        # not fit a scaler, use labels, or couple any patient's assignment.
        self.gate_permutation = fixed_feature_gate_permutation(dataset.features_unit) if cell["control"] == "RANDOM_GATE" else None
        self.control_mapping = np.arange(len(dataset))
        self.mapping_audit = dict(type="identity", training_rows=len(self.train))
        control = cell["control"]
        if control == "WRONG_DONOR":
            self.control_mapping, self.mapping_audit = wrong_donor_permutation(
                dataset.donor_codes, dataset.tissue_y, dataset.library_size, self.train, seed=cell["seed"] + 61001)
        if control in ("WITHIN_SECTION_TISSUE_DEPTH", "WRONG_SPOT_ALIGNMENT"):
            self.control_mapping, self.mapping_audit = within_section_tissue_depth_permutation(
                dataset.section_codes, dataset.tissue_y, dataset.library_size, self.train, seed=cell["seed"] + 62001)
        self.joint_target_audit = joint_permutation_audit(self.target_z, self.target_valid,
            self.control_mapping, self.train)
        self.true_graph = dataset.neighbors
        self.visual_graph, self.graph_audit = dataset.neighbors, dict(type="true_qualified_geometry")
        if control == "WRONG_NEIGHBOR":
            self.visual_graph, self.graph_audit = geometry_sham_graph(dataset.neighbors,
                dataset.section_codes, seed=cell["seed"] + 63001)
        # Scope checks precede any graph use, including explicit-feature batches.
        self.true_train_graph = local_neighbor_indices(self.true_graph, self.train,
            section_codes=dataset.section_codes, donor_codes=dataset.donor_codes)
        local_neighbor_indices(self.visual_graph, self.train,
            section_codes=dataset.section_codes, donor_codes=dataset.donor_codes)
        if len(self.heldout):
            local_neighbor_indices(self.visual_graph, self.heldout,
                section_codes=dataset.section_codes, donor_codes=dataset.donor_codes)
        self.negatives, self.negative_audit = hard_negative_indices(dataset.donor_codes,
            dataset.section_codes, dataset.tissue_y, dataset.library_size, self.train)
        self.gene_bank = GeneValueBank(dataset.gene_symbols.tolist(), mapping.gene_symbols).to(self.device)
        self.gene_z = self.stats.transform_genes(dataset.log1p_cp10k, dataset.measured_mask) if cell["arm"] == "S7" else None

    def chunk(self, model, indices, *, training_alignment=False):
        indices = np.asarray(indices, np.int64)
        dataset = self.dataset
        x = torch.as_tensor(dataset.features_unit[indices], device=self.device)
        spatial = {}
        if model.use_spatial:
            neighbor = self.visual_graph[indices]
            present = neighbor >= 0
            spatial = dict(neighbor_features=torch.as_tensor(dataset.features_unit[np.maximum(neighbor, 0)], device=self.device),
                           neighbor_mask=torch.as_tensor(present, device=self.device))
        gate = dict(gate_permutation=self.gate_permutation[indices].to(self.device)) if self.gate_permutation is not None else {}
        out = model(x, **spatial, **gate)
        if model.use_alignment and training_alignment:
            molecular_rows = self.control_mapping[indices] if self.cell["control"] in (
                "WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH", "WRONG_SPOT_ALIGNMENT") else indices
            measurements = self.gene_bank(torch.as_tensor(self.gene_z[molecular_rows], device=self.device),
                torch.as_tensor(dataset.measured_mask[molecular_rows], device=self.device))
            gene = model.encode_gene_programs(measurements)
            out["gene_tokens"] = gene["tokens"]
            out["gene_program_mask"] = gene["program_mask"]
        return out

    def objective(self, model):
        target_rows = self.train
        # S5 low-level objective validates its explicit cross-donor permutation.
        permutation = None
        if self.cell["control"] in ("WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH") and model.arm != "S5":
            target_rows = self.control_mapping[self.train]
        if model.arm == "S5":
            lookup = np.full(len(self.dataset), -1, np.int64); lookup[self.train] = np.arange(len(self.train))
            permutation = torch.as_tensor(lookup[self.control_mapping[self.train]], device=self.device)
        target = torch.as_tensor(self.target_z[target_rows], dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(self.target_valid[target_rows], device=self.device)
        tissue = torch.as_tensor(self.dataset.tissue_y[self.train], dtype=torch.long, device=self.device)
        weights = torch.as_tensor(self.stats.class_weights, dtype=torch.float32, device=self.device)
        negatives = torch.as_tensor(self.negatives, device=self.device)
        true_graph = torch.as_tensor(self.true_train_graph, device=self.device)
        donor = torch.as_tensor(self.dataset.donor_codes[self.train], dtype=torch.long, device=self.device)
        gene_rows = self.control_mapping[self.train] if self.cell["control"] in (
            "WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH", "WRONG_SPOT_ALIGNMENT") else self.train
        # Program masks are computed directly from actual measured source genes.
        gene_mask = None
        if model.use_alignment:
            with torch.no_grad():
                masks = []
                for start in range(0, len(self.train), 128):
                    out = self.chunk(model, self.train[start:start + 128], training_alignment=True)
                    masks.append(out["gene_program_mask"])
                gene_mask = torch.cat(masks)

        def objective(outputs):
            gene = dict(tokens=outputs["gene_tokens"], program_mask=gene_mask) if model.use_alignment else None
            loss, audit = source_loss(model, outputs, target, tissue, valid_mask=mask, class_weights=weights,
                program_target_permutation=permutation, training_donor_codes=donor,
                gene_programs=gene, negative_index=negatives if model.use_alignment else None,
                true_neighbor_index=true_graph if model.use_spatial else None)
            audit.update(program_target_is_wrong_donor=self.cell["control"] == "WRONG_DONOR",
                control=self.cell["control"], joint_target_and_gene_sham=(model.use_alignment and
                self.cell["control"] in ("WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH")),
                alignment_gene_rows_sha256=array_sha(gene_rows) if model.use_alignment else None)
            return loss, audit
        return objective


def _forward_small(model, data, start, end):
    out = data.chunk(model, data.train[start:end], training_alignment=True)
    result = {key: out[key] for key in ("program_predictions", "tissue_logits")}
    if model.use_alignment:
        result.update(visual_program_embeddings=out["visual_program_embeddings"], gene_tokens=out["gene_tokens"])
    return result


def collect_outputs(model, data, indices, *, microbatch, include_gene_for_retrieval=False):
    """Program inference is always called before any optional measured genes."""
    pieces = {}
    with torch.no_grad():
        for start in range(0, len(indices), microbatch):
            rows = indices[start:start + microbatch]
            out = data.chunk(model, rows, training_alignment=False)
            selected = {key: out[key] for key in ("embedding", "tissue_logits", "program_predictions", "queries",
                                                  "visual_program_embeddings", "tissue_probs", "program_gate_probs", "gate_permutation")}
            if model.use_alignment and include_gene_for_retrieval:
                measurements = data.gene_bank(torch.as_tensor(data.gene_z[rows], device=data.device),
                    torch.as_tensor(data.dataset.measured_mask[rows], device=data.device))
                gene = model.encode_gene_programs(measurements)
                selected.update(gene_tokens=gene["tokens"], gene_program_mask=gene["program_mask"])
            for key, value in selected.items():
                pieces.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(values) for key, values in pieces.items()}


def _model_for_cell(cell, data, mapping, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cell["seed"])
        reader = SparseProgramEncoder(mapping, gene_encoder=GeneEvidenceTransform()) if cell["arm"] == "S7" else None
    return SourceProgramModel(cell["arm"], tissue_prior=data.stats.tissue_prior,
        gate_mode="random" if cell["control"] == "RANDOM_GATE" else "predicted",
        gene_reader=reader, init_seed=cell["seed"]).to(device)


def _statistics_payload(stats):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(stats).items()}


def snapshot(model, optimizer, data, directory, epoch, *, config_sha, source_hashes, microbatch, launch_gate):
    model.eval()
    outputs, metrics = {}, {}
    for split, rows in (("train", data.train), ("heldout", data.heldout)):
        if not len(rows):
            metrics[split] = dict(status="NOT_APPLICABLE_FULL_SOURCE_NO_HELDOUT")
            continue
        out = collect_outputs(model, data, rows, microbatch=microbatch, include_gene_for_retrieval=model.use_alignment)
        outputs[split] = out
        metrics[split] = evaluate_source_outputs(out, data.target_z[rows], data.dataset.tissue_y[rows],
            data.dataset.library_size[rows], class_weights=data.stats.class_weights,
            valid_mask=data.target_valid[rows], donor_codes=data.dataset.donor_codes[rows],
            gate_algorithm=RANDOM_GATE_ALGORITHM if model.gate_mode == "random" else "predicted_identity")
        if model.use_alignment:
            neg, support = hard_negative_indices(data.dataset.donor_codes, data.dataset.section_codes,
                data.dataset.tissue_y, data.dataset.library_size, rows)
            metrics[split]["retrieval"] = retrieval_metrics(out["visual_program_embeddings"], out["gene_tokens"],
                                                           out["gene_program_mask"], neg)
            metrics[split]["retrieval_support"] = support
        filename = f"outputs_epoch{epoch:03d}_{split}.private.npz"
        np.savez_compressed(directory / filename, **out, row_indices=rows,
                            true_target_z=data.target_z[rows], true_target_mask=data.target_valid[rows])
    state_path = directory / f"state_epoch{epoch:03d}.private.pt"
    _atomic_torch(state_path, dict(schema="bgb_en_r6_source_checkpoint_v1", cell=data.cell,
        epochs=epoch, optimizer_updates=epoch if data.cell["arm"] != "S0" else 0,
        config_sha256=config_sha, source_code_sha256=source_hashes,
        feature_unit="float32_per_row_L2_norm_division_reject_norm_le_1e-12",
        transfer=model.transfer_state(), optimizer=optimizer.state_dict() if optimizer is not None else None,
        statistics=_statistics_payload(data.stats), launch_gate=launch_gate))
    receipt = dict(schema="bgb_en_r6_source_snapshot_v1", status="COMPLETE_SOURCE_SNAPSHOT",
        cell=data.cell, epoch=epoch, optimizer_updates=epoch if data.cell["arm"] != "S0" else 0,
        metrics=metrics, config_sha256=config_sha, source_code_sha256=source_hashes,
        artifact_sha256={p.name: sha256(p) for p in [state_path, *[directory / f"outputs_epoch{epoch:03d}_{s}.private.npz" for s in outputs]]},
        true_training_scaler_fitted_before_sham=True,
        heldout_program_forward_was_visual_only=True, full_source_no_heldout=data.cell["heldout_donor"] < 0)
    _atomic_json(directory / f"SNAPSHOT_{epoch:03d}.json", receipt)
    model.train()
    return receipt


def fit_cell(config, cell_index, dataset, mapping, private_root, *, device="cuda:0", launch_gate=None, config_sha=None,
             resume=False):
    """Called only after CLI lock+launch validation; never invoked on import."""
    fixed = validate_lock(config)
    require(launch_gate is not None and launch_gate.get("r6_formal_training_allowed") is True,
            "SOURCE_FIT_REQUIRES_VERIFIED_EXTERNAL_LAUNCH_GATE")
    cell = config["cells"][cell_index]
    name = f"{cell['arm']}_{cell['control']}_h{cell['heldout_donor']}_seed{cell['seed']}"
    directory = _private_directory(Path(private_root) / "source_cells" / name)
    config_sha = config_sha or canonical_hash(config)
    if (directory / "COMPLETE.json").exists():
        require(resume, "EXISTING_SOURCE_CELL_REQUIRES_VERIFIED_RESUME")
        result = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
        require(result["config_sha256"] == config_sha and result.get("source_code_sha256") == current_code_hashes()
                and result.get("cell") == cell and result.get("status") in (
                "COMPLETE_R6_SOURCE_FIT", "COMPLETE_R6_RANDOM_REFERENCE_NO_OPTIMIZER"),
                "EXISTING_SOURCE_CELL_FROM_DIFFERENT_LOCK")
        require(result.get("epochs") == 80 and result.get("optimizer_updates") == (0 if cell["arm"] == "S0" else 80),
                "EXISTING_SOURCE_CELL_UPDATE_BUDGET_CHANGED")
        for file, digest in result["artifact_sha256"].items():
            require(sha256(directory / file) == digest, "EXISTING_SOURCE_CELL_ARTIFACT_CHANGED")
        return result
    require(not any(directory.iterdir()), "INCOMPLETE_SOURCE_CELL_PRESERVED_NO_AUTOMATIC_RETRAIN")
    hashes = current_code_hashes()
    start_time = time.monotonic()
    optimizer_updates = 0
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if str(device).startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        data = SourceFitData(dataset, cell, mapping, device)
        model = _model_for_cell(cell, data, mapping, device)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0.) if cell["arm"] != "S0" else None
        _atomic_json(directory / "FIT_CONTRACT.json", dict(cell=cell, config_sha256=config_sha,
            source_code_sha256=hashes, statistics=data.stats.audit(), mapping=data.mapping_audit,
            joint_target_preservation=data.joint_target_audit, visual_graph=data.graph_audit,
            hard_negatives=data.negative_audit, fixed_defaults=fixed, launch_gate=launch_gate,
            source_cache_manifest_sha256=config["source_cache_manifest_sha256"],
            actual_gene_values_auxiliary_only=model.use_alignment,
            source_condition_is_predicted_tissue_not_tissue_truth=True,
            random_gate_algorithm=RANDOM_GATE_ALGORITHM if data.gate_permutation is not None else "not_used",
            random_gate_permutation_sha256=array_sha(data.gate_permutation.numpy()) if data.gate_permutation is not None else None,
            random_gate_indices_stateless_from_original_unit_feature_bytes=True,
            cross_model_probability_distribution_matching_claimed=False))
        curves = []
        if optimizer is None:
            final_snapshot = snapshot(model, None, data, directory, 80, config_sha=config_sha,
                source_hashes=hashes, microbatch=fixed["source_microbatch"], launch_gate=launch_gate)
        else:
            objective = data.objective(model)
            for epoch in range(1, 81):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                audit = deterministic_output_gradient_cache_backward(model,
                    lambda start, end: _forward_small(model, data, start, end), len(data.train), objective,
                    microbatch=fixed["source_microbatch"], parameter_L2=.0001,
                    tolerance=fixed["full_output_replay_tolerance"])
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
                require(torch.isfinite(norm), "SOURCE_GRADIENT_NORM_NONFINITE")
                optimizer.step()
                optimizer_updates = epoch
                curves.append(dict(epoch=epoch, optimizer_updates=epoch, preclip_gradient_norm=float(norm), **audit))
                _atomic_json(directory / "TRAINING_CURVE.json", curves)
                if epoch in (20, 40, 80):
                    require(current_code_hashes() == hashes, "SOURCE_CODE_CHANGED_DURING_FIT")
                    final_snapshot = snapshot(model, optimizer, data, directory, epoch, config_sha=config_sha,
                        source_hashes=hashes, microbatch=fixed["source_microbatch"], launch_gate=launch_gate)
        result = dict(status="COMPLETE_R6_SOURCE_FIT" if optimizer is not None else "COMPLETE_R6_RANDOM_REFERENCE_NO_OPTIMIZER",
            cell=cell, config_sha256=config_sha, source_code_sha256=hashes,
            actual_new_optimizer_fits=1 if optimizer is not None else 0,
            optimizer_updates=80 if optimizer is not None else 0, epochs=80,
            metrics=final_snapshot["metrics"], parameter_budget=model.parameter_budget(),
            artifact_sha256={p.name: sha256(p) for p in directory.iterdir() if p.is_file() and p.name not in ("COMPLETE.json", "ERROR.json", "REPLAY.json")},
            elapsed_seconds=time.monotonic() - start_time, independent_replay_complete=False,
            full_source_no_heldout=cell["heldout_donor"] < 0)
        _atomic_json(directory / "COMPLETE.json", result)
        return result
    except BaseException as error:
        _atomic_json(directory / "ERROR.json", dict(status="ERROR_SOURCE_FIT", cell=cell,
            config_sha256=config_sha, error_class=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), source_code_sha256=hashes,
            completed_optimizer_updates=optimizer_updates,
            actual_new_optimizer_fits=1 if optimizer_updates > 0 else 0,
            artifact_sha256={p.name: sha256(p) for p in directory.iterdir() if p.is_file()
                and p.name not in ("COMPLETE.json", "ERROR.json", "REPLAY.json")},
            elapsed_seconds=time.monotonic() - start_time, completion_claimed=False))
        raise


def replay_cell(config, cell_index, dataset, mapping, private_root, *, device="cpu", config_sha=None):
    """Independent state reload + forward + true metric recomputation, zero updates."""
    fixed = validate_lock(config)
    cell = config["cells"][cell_index]
    name = f"{cell['arm']}_{cell['control']}_h{cell['heldout_donor']}_seed{cell['seed']}"
    directory = Path(private_root) / "source_cells" / name
    receipt = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
    config_sha = config_sha or canonical_hash(config)
    require(receipt["config_sha256"] == config_sha and receipt["source_code_sha256"] == current_code_hashes(),
            "SOURCE_REPLAY_CODE_OR_CONFIG_CHANGED")
    updates = 0 if cell["arm"] == "S0" else 80
    require(receipt.get("cell") == cell and receipt.get("epochs") == 80
            and receipt.get("optimizer_updates") == updates and receipt.get("status") == (
            "COMPLETE_R6_RANDOM_REFERENCE_NO_OPTIMIZER" if cell["arm"] == "S0" else "COMPLETE_R6_SOURCE_FIT"),
            "SOURCE_REPLAY_RECEIPT_LINEAGE_MISMATCH")
    for filename, digest in receipt["artifact_sha256"].items():
        require(sha256(directory / filename) == digest, "SOURCE_REPLAY_ARTIFACT_SHA_CHANGED")
    state = torch.load(directory / "state_epoch080.private.pt", map_location=device, weights_only=False)
    require(state.get("schema") == "bgb_en_r6_source_checkpoint_v1" and state.get("cell") == cell
            and state.get("epochs") == 80 and state.get("optimizer_updates") == updates
            and state.get("config_sha256") == config_sha and state.get("source_code_sha256") == current_code_hashes()
            and state.get("feature_unit") == "float32_per_row_L2_norm_division_reject_norm_le_1e-12",
            "SOURCE_REPLAY_STATE_LINEAGE_OR_FEATURE_CONTRACT_MISMATCH")
    data = SourceFitData(dataset, cell, mapping, device)
    require(state["statistics"] == _statistics_payload(data.stats), "SOURCE_REPLAY_TRAIN_ONLY_SCALER_CHANGED")
    model = _model_for_cell(cell, data, mapping, device)
    model.load_transfer_state(state["transfer"], freeze=True)
    errors, metrics = {}, {}
    for split, rows in (("train", data.train), ("heldout", data.heldout)):
        if not len(rows):
            metrics[split] = dict(status="NOT_APPLICABLE_FULL_SOURCE_NO_HELDOUT")
            continue
        actual = collect_outputs(model, data, rows, microbatch=fixed["source_microbatch"], include_gene_for_retrieval=model.use_alignment)
        with np.load(directory / f"outputs_epoch080_{split}.private.npz", allow_pickle=False) as saved:
            require(np.array_equal(saved["row_indices"], rows), "SOURCE_REPLAY_ROWS_CHANGED")
            per_key = {}
            for key, value in actual.items():
                error = float(np.max(np.abs(value.astype(float) - saved[key].astype(float))))
                require(error <= fixed["full_output_replay_tolerance"], "SOURCE_REPLAY_FORWARD_VALUES_CHANGED")
                per_key[key] = error
            errors[split] = per_key
        metrics[split] = evaluate_source_outputs(actual, data.target_z[rows], data.dataset.tissue_y[rows],
            data.dataset.library_size[rows], class_weights=data.stats.class_weights, valid_mask=data.target_valid[rows],
            donor_codes=data.dataset.donor_codes[rows],
            gate_algorithm=RANDOM_GATE_ALGORITHM if model.gate_mode == "random" else "predicted_identity")
        expected = receipt["metrics"][split]
        # Retrieval is separate loss/evaluation input, not part of visual prediction.
        if model.use_alignment:
            negatives, support = hard_negative_indices(data.dataset.donor_codes, data.dataset.section_codes,
                data.dataset.tissue_y, data.dataset.library_size, rows)
            metrics[split]["retrieval"] = retrieval_metrics(actual["visual_program_embeddings"], actual["gene_tokens"], actual["gene_program_mask"], negatives)
            metrics[split]["retrieval_support"] = support
        require(_metric_difference(metrics[split], expected) <= 2e-6, "SOURCE_REPLAY_METRICS_CHANGED")
    result = dict(status="PASS_COMPLETE_R6_SOURCE_CHECKPOINT_REPLAY", cell=cell,
        source_code_sha256=current_code_hashes(), config_sha256=config_sha,
        completion_sha256=sha256(directory / "COMPLETE.json"), optimizer_updates=0,
        new_optimizer_fits=0, model_frozen=True, forward_max_abs_errors=errors, metrics=metrics,
        image_only_heldout_program_prediction=True, original_fit_artifacts_modified=False)
    _atomic_json(directory / "REPLAY.json", result)
    return result


def _metric_difference(actual, expected):
    if isinstance(actual, dict):
        require(set(actual) == set(expected), "SOURCE_REPLAY_METRIC_KEYS_CHANGED")
        return max((_metric_difference(actual[k], expected[k]) for k in actual), default=0.)
    if isinstance(actual, list):
        require(len(actual) == len(expected), "SOURCE_REPLAY_METRIC_LENGTH_CHANGED")
        return max((_metric_difference(a, b) for a, b in zip(actual, expected)), default=0.)
    if actual is None or isinstance(actual, (str, bool)):
        require(actual == expected, "SOURCE_REPLAY_METRIC_NONNUMERIC_CHANGED")
        return 0.
    return abs(float(actual) - float(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fit", "replay"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true", help="Reuse only a hash-verified COMPLETE cell; never retry an unfinished/error cell")
    parser.add_argument("--launch-gate", type=Path)
    parser.add_argument("--launch-gate-sha256")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_lock(config)
    require(0 <= args.cell_index < len(config["cells"]), "SOURCE_CELL_INDEX_NOT_IN_LOCK")
    private = _private_directory(args.private_root, cache_root=args.source_cache)
    gate = None
    if args.mode == "fit":
        require(args.launch_gate is not None and args.launch_gate_sha256 is not None, "SOURCE_FIT_REQUIRES_LAUNCH_RECEIPT")
        gate = verify_launch_gate(args.launch_gate, args.launch_gate_sha256)
    contract = SourceContract(fresh_certificate_sha256=config["fresh_certificate_sha256"])
    dataset = load_private_cache(args.source_cache, contract, expected_manifest_sha256=config["source_cache_manifest_sha256"])
    mapping = load_source_program_mapping(ROOT / "configs/sami_surv_stage_b_programs_v1.json")
    common = dict(device=args.device, config_sha=sha256(args.config))
    result = fit_cell(config, args.cell_index, dataset, mapping, private, launch_gate=gate, resume=args.resume, **common) if args.mode == "fit" else replay_cell(
        config, args.cell_index, dataset, mapping, private, **common)
    # Console exposes anonymous completion state only, never private arrays/paths.
    print(json.dumps(dict(status=result["status"], cell=result["cell"],
        optimizer_updates=result["optimizer_updates"]), sort_keys=True))


if __name__ == "__main__":
    main()
