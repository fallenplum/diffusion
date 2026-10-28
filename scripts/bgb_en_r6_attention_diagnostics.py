"""Post-replay, first-eight-held-out R6 diagnostics: no fit or backward.

This is an inference-only appendix outside targetLOCK. It records its own SHA,
checks the complete actual-replay chain before loading target inputs, and never
exports patient/slide identifiers, coordinates, attention arrays or weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B11", "B13")
PARTITIONS, SEEDS = (0, 1, 2), (31, 47, 83)
CONTROLS = ("matched_query", "wrong_query", "mean_query", "interaction_off", "attention_permutation")
MAX_REPORT_BYTES = 2 * 1024**2
INDEX_NAME = "R6_ATTENTION_DIAGNOSTICS.json"
STORAGE_FIELDS = frozenset(("storage_schema", "logical_object_sha256", "rows_shards", "rows_count"))


def require(value, code):
    if not value:
        raise ValueError(code)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _shard_name(arm):
    require(arm in ARMS, "R6_DIAGNOSTIC_UNKNOWN_SHARD_ARM")
    return f"R6_ATTENTION_DIAGNOSTICS_{arm}.json"


def _check_rows(rows):
    expected = {(arm, part, seed) for arm in ARMS for part in PARTITIONS for seed in SEEDS}
    require(len(rows) == 90 and {(row["arm"], row["partition"], row["seed"]) for row in rows} == expected,
            "R6_DIAGNOSTIC_SHARD_MATRIX_CHANGED")


def write_diagnostics(result, output):
    """Lossless 10x9-row packaging. Write index last; never replace a file."""
    require("rows" in result and not (set(result) & STORAGE_FIELDS), "R6_DIAGNOSTIC_RESERVED_STORAGE_FIELD")
    rows = result["rows"]; _check_rows(rows)
    output = Path(output)
    files = [output / INDEX_NAME, *(output / _shard_name(arm) for arm in ARMS)]
    require(not any(path.exists() for path in files), "R6_DIAGNOSTIC_OUTPUT_PRESERVED_NO_OVERWRITE")
    logical_sha = canonical(result)
    payloads, descriptors = [], []
    for arm in ARMS:
        positions = [i for i, row in enumerate(rows) if row["arm"] == arm]
        require(len(positions) == 9, "R6_DIAGNOSTIC_NINE_ROWS_PER_ARM_REQUIRED")
        part = dict(schema="bgb_en_r6_attention_diagnostics_arm_shard_v1", arm=arm,
                    contract_sha256=result["contract_sha256"], logical_object_sha256=logical_sha,
                    rows=[rows[i] for i in positions])
        data = _json_bytes(part)
        require(len(data) <= MAX_REPORT_BYTES, "R6_DIAGNOSTIC_SHARD_EXCEEDS_2MIB")
        name = _shard_name(arm)
        descriptors.append(dict(arm=arm, file=name, sha256=hashlib.sha256(data).hexdigest(),
                                row_count=9, row_positions=positions, bytes=len(data)))
        payloads.append((output / name, data))
    index = {key: value for key, value in result.items() if key != "rows"}
    index.update(storage_schema="bgb_en_r6_attention_diagnostics_sharded_index_v1",
                 logical_object_sha256=logical_sha, rows_count=90, rows_shards=descriptors)
    data = _json_bytes(index)
    require(len(data) <= MAX_REPORT_BYTES, "R6_DIAGNOSTIC_INDEX_EXCEEDS_2MIB")
    output.mkdir(parents=True, exist_ok=True)
    for path, content in [*payloads, (output / INDEX_NAME, data)]:
        with path.open("xb") as stream:
            stream.write(content)
    return index


def read_diagnostics(index_path):
    """Read only: verify paths, sizes, SHA, 10x9 identity and logical SHA."""
    index_path = Path(index_path).resolve()
    require(index_path.stat().st_size <= MAX_REPORT_BYTES, "R6_DIAGNOSTIC_INDEX_EXCEEDS_2MIB")
    index = read(index_path)
    require(index.get("storage_schema") == "bgb_en_r6_attention_diagnostics_sharded_index_v1"
            and "rows" not in index and index.get("rows_count") == 90, "R6_DIAGNOSTIC_BAD_SHARD_INDEX")
    descriptors = index["rows_shards"]
    require(len(descriptors) == 10 and {part["arm"] for part in descriptors} == set(ARMS),
            "R6_DIAGNOSTIC_TEN_ARM_SHARDS_REQUIRED")
    rows = [None] * 90
    for descriptor in descriptors:
        arm = descriptor["arm"]
        require(descriptor["file"] == _shard_name(arm) and descriptor["row_count"] == 9,
                "R6_DIAGNOSTIC_SHARD_PATH_OR_COUNT_CHANGED")
        path = (index_path.parent / descriptor["file"]).resolve()
        require(path.parent == index_path.parent and path.stat().st_size == descriptor["bytes"] <= MAX_REPORT_BYTES
                and sha(path) == descriptor["sha256"], "R6_DIAGNOSTIC_SHARD_SHA_OR_PATH_CHANGED")
        part = read(path)
        require(part.get("schema") == "bgb_en_r6_attention_diagnostics_arm_shard_v1" and part.get("arm") == arm
                and part.get("contract_sha256") == index["contract_sha256"]
                and part.get("logical_object_sha256") == index["logical_object_sha256"] and len(part["rows"]) == 9
                and {(row["arm"], row["partition"], row["seed"]) for row in part["rows"]}
                == {(arm, p, s) for p in PARTITIONS for s in SEEDS}, "R6_DIAGNOSTIC_ARM_SHARD_IDENTITY_CHANGED")
        positions = descriptor["row_positions"]
        require(len(positions) == 9 and all(type(i) is int and 0 <= i < 90 for i in positions)
                and len(set(positions)) == 9, "R6_DIAGNOSTIC_SHARD_ROW_ORDER_CHANGED")
        for position, row in zip(positions, part["rows"]):
            require(rows[position] is None, "R6_DIAGNOSTIC_DUPLICATE_SHARD_ROW_POSITION")
            rows[position] = row
    require(all(row is not None for row in rows), "R6_DIAGNOSTIC_INCOMPLETE_SHARD_ROWS")
    _check_rows(rows)
    result = {key: value for key, value in index.items() if key not in STORAGE_FIELDS}
    result["rows"] = rows
    require(canonical(result) == index["logical_object_sha256"], "R6_DIAGNOSTIC_LOGICAL_OBJECT_SHA_CHANGED")
    return result


def load_diagnostics(index_path):
    return read_diagnostics(index_path)


def parameter_drift(initial, final, trainable_names):
    """Float64 norms only. The isolated L2 gradient excludes risk penalties."""
    groups = {}
    require(len(trainable_names) == len(set(trainable_names)), "R6_DIAGNOSTIC_DUPLICATE_TRAINABLE_PARAMETER")
    for name in trainable_names:
        def array(value):
            return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, dtype=np.float64)
        first, last = array(initial[name]), array(final[name])
        require(first.shape == last.shape and np.isfinite(first).all() and np.isfinite(last).all(),
                "R6_DIAGNOSTIC_PARAMETER_SHAPE_OR_FINITE_CHANGED")
        group = ".".join(name.split(".")[:2])
        value = groups.setdefault(group, dict(trainable_numel=0, parameter_tensors=0,
                                             initial_square=0., final_square=0., drift_square=0.))
        value["trainable_numel"] += int(first.size); value["parameter_tensors"] += 1
        value["initial_square"] += float(np.square(first).sum())
        value["final_square"] += float(np.square(last).sum())
        value["drift_square"] += float(np.square(last - first).sum())
    result = {}
    for name, value in sorted(groups.items()):
        first, last, drift = (math.sqrt(value[key]) for key in ("initial_square", "final_square", "drift_square"))
        result[name] = dict(trainable_numel=value["trainable_numel"], parameter_tensors=value["parameter_tensors"],
            initial_L2=first, final_L2=last, drift_L2=drift, relative_drift=drift / first if first > 0 else None,
            isolated_parameter_L2_gradient_L2=2e-4 * last)
    return result


def activation_norms(output, patch_mask, *, forward_enabled):
    """Existing activations only, no normalization, model call or backward."""
    def array(value):
        return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)

    def summarize_items(values, mask):
        values, mask = np.asarray(values, float), np.asarray(mask, bool)
        require(values.shape == mask.shape and values.ndim == 2 and np.isfinite(values[mask]).all(),
                "R6_DIAGNOSTIC_ACTIVATION_MASK_OR_FINITE_CHANGED")
        return dict(valid_items=int(mask.sum()), invalid_items=int(mask.size - mask.sum()),
                    patient_equal_program_mean=scalar_summary([float(values[i, mask[i]].mean()) if mask[i].any() else None
                                                              for i in range(len(values))]))

    def norms(value, mask):
        value, mask = array(value).astype(np.float64), np.asarray(mask, bool)
        require(value.shape[:-1] == mask.shape and np.isfinite(value[mask]).all(),
                "R6_DIAGNOSTIC_ACTIVATION_SHAPE_OR_FINITE_CHANGED")
        return np.linalg.norm(value, axis=-1)

    def paired(first, delta, mask):
        a, d = norms(first, mask), norms(delta, mask)
        eligible = mask & (a > 0)
        ratios = np.zeros_like(a); np.divide(d, a, out=ratios, where=eligible)
        return dict(base_L2=summarize_items(a, mask), delta_L2=summarize_items(d, mask),
                    delta_over_base_L2=summarize_items(ratios, eligible),
                    valid_zero_base_denominator_items=int((mask & (a == 0)).sum()))

    own_mask, query_mask = array(output["own_program_mask"]).astype(bool), array(output["query_source_mask"]).astype(bool)
    evidence_mask = array(output["patch_base_read"]["program_mask"]).astype(bool) & array(output["patch_cond_read"]["program_mask"]).astype(bool)
    base_q, cond_q = array(output["base_queries"]), array(output["conditioned_queries"])
    values, mask = array(output["patch_values"]).astype(np.float64), np.asarray(patch_mask, bool)
    require(values.shape[:-1] == mask.shape and mask.ndim == 3 and np.isfinite(values[mask]).all(),
            "R6_DIAGNOSTIC_PATCH_ACTIVATION_MASK_CHANGED")
    patch_l2 = np.linalg.norm(values, axis=-1)
    per_patient = []
    for i in range(len(values)):
        per_wsi = [float(patch_l2[i, s, mask[i, s]].mean()) for s in range(mask.shape[1]) if mask[i, s].any()]
        per_patient.append(float(np.mean(per_wsi)) if per_wsi else None)
    kind = output["diagnostics"]["reverse_kind"]
    reverse = None
    if kind != "none":
        first, last = output["reverse_base_read"], output["reverse_cond_read"]
        reverse_mask = array(first["program_mask"]).astype(bool) & array(last["program_mask"]).astype(bool)
        reverse = paired(first["tokens"], output["delta_G"], reverse_mask)
    return dict(G0_L2=summarize_items(norms(output["G0"], own_mask), own_mask),
        query_context_L2=summarize_items(norms(output["query_context_used"], query_mask), query_mask),
        base_query_and_condition_delta=paired(base_q, cond_q - base_q, query_mask),
        Ebase_and_delta_E=paired(output["Ebase"], output["delta_E"], evidence_mask),
        reverse_base_and_delta_G=reverse,
        patch_values_L2=dict(valid_patches=int(mask.sum()), valid_WSI=int(mask.any(-1).sum()),
                             patient_equal_WSI_patch_mean=scalar_summary(per_patient)),
        forward_structure_enabled=bool(forward_enabled), reverse_structure_enabled=kind != "none", reverse_kind=kind,
        reverse_difference_reference=("Read(Econd,own_original_gene_values)-Read(Ebase,own_original_gene_values)" if kind == "gene" else
            "Read(Econd,own_G0_tokens)-Read(Ebase,own_G0_tokens)" if kind == "token" else None),
        reverse_delta_is_not_Read_Econd_minus_G0=True)


def scalar_summary(values):
    values = np.asarray([x for x in values if x is not None], dtype=np.float64)
    require(np.isfinite(values).all(), "R6_DIAGNOSTIC_NONFINITE_SUMMARY")
    return dict(n=int(values.size), mean=float(values.mean()) if values.size else None,
                sd=float(values.std()) if values.size else None)


def entropy(probability):
    p = np.asarray(probability, dtype=np.float64)
    require(np.isfinite(p).all() and (p >= 0).all(), "R6_DIAGNOSTIC_INVALID_PROBABILITY")
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def svd_effective_rank(matrix):
    """exp(H(p)), p=sigma/sum(sigma), raw patch weights, no null/centering."""
    value = np.asarray(matrix, dtype=np.float64)
    require(value.ndim == 2 and np.isfinite(value).all(), "R6_DIAGNOSTIC_BAD_RANK_MATRIX")
    if not value.size:
        return None
    singular = np.linalg.svd(value, compute_uv=False)
    total = singular.sum()
    return float(np.exp(entropy(singular / total))) if total > 0 else None


def top_indices(weights, valid, topk=16):
    w, m = np.asarray(weights), np.asarray(valid, bool)
    require(w.ndim == m.ndim == 1 and w.shape == m.shape and np.isfinite(w[m]).all(), "R6_DIAGNOSTIC_BAD_TOPK_ROW")
    ids = np.flatnonzero(m)
    if not len(ids) or w[ids].sum() <= 0:
        return None
    return ids[np.argsort(-w[ids], kind="stable")[:min(topk, len(ids))]]


def jaccard(first, second):
    if first is None or second is None:
        return None
    a, b = set(map(int, first)), set(map(int, second))
    return len(a & b) / len(a | b) if a | b else None


def patch_statistics(weights, null, patch_mask, valid_rows):
    """Each statistic averages WSIs within patient, then patients equally."""
    w, z, mask, valid = np.asarray(weights, float), np.asarray(null, float), np.asarray(patch_mask, bool), np.asarray(valid_rows, bool)
    require(w.ndim == 4 and w.shape[2] == 4 and z.shape == valid.shape == w.shape[:3]
            and mask.shape == (w.shape[0], w.shape[1], w.shape[3]), "R6_DIAGNOSTIC_PATCH_SHAPE")
    require(np.isfinite(w).all() and np.isfinite(z).all() and (w >= 0).all() and (z >= 0).all(), "R6_DIAGNOSTIC_PATCH_PROBABILITY")
    require(np.all(w * ~mask[:, :, None, :] == 0) and np.allclose(w.sum(-1) + z, 1., atol=2e-6, rtol=0),
            "R6_DIAGNOSTIC_PATCH_PADDING_OR_NORMALIZATION")
    per_program = []
    for k in range(4):
        patient_entropy, patient_neff, patient_null = [], [], []
        empty_mass = 0
        for b in range(len(w)):
            hs, ns, zs = [], [], []
            for s in np.flatnonzero(valid[b, :, k]):
                own = w[b, s, k, mask[b, s]]
                hs.append(entropy(np.r_[own, z[b, s, k]])); zs.append(float(z[b, s, k]))
                if own.sum() > 0:
                    ns.append(float(np.exp(entropy(own / own.sum()))))
                else:
                    empty_mass += 1
            patient_entropy.append(float(np.mean(hs)) if hs else None)
            patient_neff.append(float(np.mean(ns)) if ns else None)
            patient_null.append(float(np.mean(zs)) if zs else None)
        per_program.append(dict(program_index=k, valid_WSI_rows=int(valid[:, :, k].sum()),
            invalid_WSI_rows=int(valid[:, :, k].size - valid[:, :, k].sum()), real_mass_zero_rows=empty_mass,
            entropy_patch_plus_null=scalar_summary(patient_entropy), real_patch_neff_exp_entropy=scalar_summary(patient_neff),
            null=scalar_summary(patient_null)))
    ranks, pair_overlap, pair_correlation = [], {f"{k}_{j}": [] for k in range(4) for j in range(k + 1, 4)}, {f"{k}_{j}": [] for k in range(4) for j in range(k + 1, 4)}
    invalid_rank_wsi, absent_wsi = 0, 0
    for b in range(len(w)):
        local_rank, local_overlap, local_corr = [], {key: [] for key in pair_overlap}, {key: [] for key in pair_overlap}
        for s in range(w.shape[1]):
            ids = np.flatnonzero(mask[b, s]); active = np.flatnonzero(valid[b, s] & (w[b, s].sum(-1) > 0))
            absent_wsi += int(not len(ids))
            if len(ids) and len(active):
                local_rank.append(svd_effective_rank(w[b, s][np.ix_(active, ids)]))
            else:
                invalid_rank_wsi += 1
            for k in range(4):
                for j in range(k + 1, 4):
                    if not valid[b, s, k] or not valid[b, s, j]:
                        continue
                    key = f"{k}_{j}"
                    overlap = jaccard(top_indices(w[b, s, k], mask[b, s]), top_indices(w[b, s, j], mask[b, s]))
                    if overlap is not None:
                        local_overlap[key].append(overlap)
                    a, c = w[b, s, k, ids], w[b, s, j, ids]
                    if len(ids) >= 2 and np.ptp(a) > 0 and np.ptp(c) > 0:
                        local_corr[key].append(float(np.corrcoef(a, c)[0, 1]))
        ranks.append(float(np.mean(local_rank)) if local_rank else None)
        for key in pair_overlap:
            pair_overlap[key].append(float(np.mean(local_overlap[key])) if local_overlap[key] else None)
            pair_correlation[key].append(float(np.mean(local_corr[key])) if local_corr[key] else None)
    return dict(programs=per_program, raw_patch_attention_svd_effective_rank=scalar_summary(ranks),
        invalid_rank_WSI=invalid_rank_wsi, WSI_without_patches=absent_wsi,
        different_program_top16_jaccard={key: scalar_summary(value) for key, value in pair_overlap.items()},
        different_program_patch_weight_Pearson={key: scalar_summary(value) for key, value in pair_correlation.items()})


def paired_top_overlap(first, second, patch_mask, valid_first, valid_second):
    a, b, mask = np.asarray(first), np.asarray(second), np.asarray(patch_mask, bool)
    valid = np.asarray(valid_first, bool) & np.asarray(valid_second, bool)
    result = []
    for k in range(4):
        patient = []
        for i in range(len(a)):
            overlap = [jaccard(top_indices(a[i, s, k], mask[i, s]), top_indices(b[i, s, k], mask[i, s]))
                       for s in np.flatnonzero(valid[i, :, k])]
            overlap = [value for value in overlap if value is not None]
            patient.append(float(np.mean(overlap)) if overlap else None)
        result.append(dict(program_index=k, **scalar_summary(patient)))
    return result


def gene_statistics(readout, edge_program, edge_sign, measured_edge_mask=None):
    if readout is None:
        return None
    weights = np.asarray(readout["weights"], float)
    measured = np.asarray(readout["measured_counts"])
    retained = np.asarray(readout["retained_counts"])
    coverage = np.asarray(readout["coverage"], float)
    components = np.asarray(readout["component_mask"], bool)
    defined = np.asarray(readout["component_defined"], bool)
    p, sign = np.asarray(edge_program), np.asarray(edge_sign)
    require(weights.ndim == 2 and weights.shape[1] == len(p) == len(sign) and np.isfinite(weights).all()
            and (weights >= 0).all(), "R6_DIAGNOSTIC_GENE_WEIGHT_SHAPE")
    if measured_edge_mask is not None:
        measured_edge_mask = np.asarray(measured_edge_mask, bool)
        require(measured_edge_mask.shape == weights.shape and np.all(weights[~measured_edge_mask] == 0),
                "R6_DIAGNOSTIC_UNMEASURED_GENE_RECEIVED_WEIGHT")
    result = []
    for k in range(4):
        for c, direction in enumerate((1, -1)):
            cols = np.flatnonzero((p == k) & (sign == direction))
            require(bool(len(cols)) == bool(defined[k, c]), "R6_DIAGNOSTIC_SIGNED_DEFINITION")
            distributions = weights[:, cols]
            mass = distributions.sum(-1)
            if measured_edge_mask is not None:
                actual_count = measured_edge_mask[:, cols].sum(-1)
                require(np.array_equal(measured[:, k, c], actual_count)
                        and np.all(retained[:, k, c] <= actual_count)
                        and np.allclose(coverage[:, k, c], actual_count / max(1, len(cols)), atol=2e-6, rtol=0),
                        "R6_DIAGNOSTIC_GENE_COVERAGE_NOT_FROM_REAL_MASK")
            require(np.allclose(mass, components[:, k, c].astype(float), atol=2e-6, rtol=0), "R6_DIAGNOSTIC_SIGNED_GENE_NORMALIZATION")
            hs, neff = [], []
            for row, active in zip(distributions, components[:, k, c]):
                if active:
                    probability = row / row.sum(); hs.append(float(np.exp(entropy(probability))))
                    neff.append(float(1. / np.square(probability).sum()))
            result.append(dict(program_index=k, member_sign="up" if direction == 1 else "down", defined_members=len(cols),
                status="DEFINED" if len(cols) else "NO_DEFINED_MEMBERS", valid_patient_components=int(components[:, k, c].sum()),
                invalid_patient_components=int((~components[:, k, c]).sum()),
                measured_count=scalar_summary(measured[:, k, c]) if len(cols) else None,
                retained_count=scalar_summary(retained[:, k, c]) if len(cols) else None,
                measured_coverage=scalar_summary(coverage[:, k, c]) if len(cols) else None,
                separately_normalized_weight_sum=scalar_summary(mass) if len(cols) else None,
                N_eff_exp_entropy=scalar_summary(hs), N_eff_inverse_Simpson=scalar_summary(neff)))
    return result


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def verify_complete_replay_gate(args):
    """JSON and SHA only. Must finish before common.context or torch.load."""
    lock = read(args.private / "targetLOCK.json")
    require(lock.get("schema") == "bgb_en_r6_target_protocol_lock_v1" and lock.get("status") == "LOCKED_BEFORE_TARGET_FITS"
            and canonical({k: v for k, v in lock.items() if k not in ("contract_sha256", "created_UTC")}) == lock["contract_sha256"],
            "R6_DIAGNOSTIC_TARGET_LOCK_REQUIRED")
    require(lock["outer_test_access"] is False and lock["planned_neural_fits"] == 90 and lock["teacher_refits"] == 0,
            "R6_DIAGNOSTIC_TARGET_SCOPE_CHANGED")
    for name, digest in lock["sources"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.suffix in (".py", ".json") and sha(path) == digest,
                "R6_DIAGNOSTIC_PINNED_SOURCE_CHANGED")
    require(sha(args.config) == lock["config_sha256"] and sha(args.source_config) == lock["source_config_sha256"],
            "R6_DIAGNOSTIC_CONFIG_CHANGED")
    require(args.launch_gate is not None and args.launch_gate_sha256 == lock["launch_gate_sha256"] == sha(args.launch_gate)
            and read(args.launch_gate).get("r6_formal_training_allowed") is True, "R6_DIAGNOSTIC_R5_GATE_REQUIRED")
    audit = read(args.private / "REPLAY_AUDIT.json")
    qpath, upath = args.private / "QUALIFICATION.json", args.private / "UPSTREAM_REPLAY_AUDIT.json"
    qualification, upstream = read(qpath), read(upath)
    require(audit.get("status") == "PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY"
            and audit["contract_sha256"] == lock["contract_sha256"]
            and audit["expected_units"] == audit["available_units"] == audit["passed_units"] == 90
            and audit["failed_units"] == 0 and not audit["missing_or_unreplayed_units"]
            and audit["new_optimizer_fits"] == audit["new_preprocessing_fits"] == 0,
            "R6_DIAGNOSTIC_ALL_90_ACTUAL_REPLAYS_REQUIRED")
    require(qualification["status"] == "PASS_R6_TARGET_REAL_DATA_AND_SOURCE_TRANSFER"
            and qualification["contract_sha256"] == upstream["contract_sha256"] == lock["contract_sha256"]
            and audit["qualification_sha256"] == sha(qpath)
            and audit["upstream_replay_audit_sha256"] == qualification["upstream_replay_audit_sha256"] == sha(upath)
            and upstream["status"] == "PASS_R6_39_COEFFICIENTS_3_BETA_PREPROCESSING_REPLAY",
            "R6_DIAGNOSTIC_UPSTREAM_REPLAY_ORIGIN_CHANGED")
    expected = {f"{arm}_p{part}_seed{seed}": (arm, part, seed) for arm in ARMS for part in PARTITIONS for seed in SEEDS}
    require(len(audit["units"]) == 90 and {x["unit"] for x in audit["units"]} == set(expected)
            and not list((args.private / "REPLAY_PRIVATE").glob("*.FAIL.json")), "R6_DIAGNOSTIC_REPLAY_MATRIX_CHANGED")
    require({p.parent.name for p in (args.private / "cells").glob("*/COMPLETE.json")} == set(expected),
            "R6_DIAGNOSTIC_COMPLETE_MATRIX_CHANGED")
    entries = []
    for unit in sorted(audit["units"], key=lambda row: row["unit"]):
        name = unit["unit"]; arm, part, seed = expected[name]
        directory = args.private / "cells" / name
        done = read(directory / "COMPLETE.json"); terminal = read(args.private / "REPLAY_PRIVATE" / (name + ".PASS.json"))
        require(terminal == unit and unit["status"] == "PASS" and unit["contract_sha256"] == lock["contract_sha256"]
                and unit["actual_state_dict_forward_replayed"] and unit["source_frozen_weight_hash_replayed"]
                and unit["saved_query_mapping_recomputed"] and unit["changed_pair_credits"] == 0
                and math.isfinite(unit["maximum_risk_error"]) and unit["maximum_risk_error"] <= 2e-6
                and unit["new_optimizer_updates"] == 0,
                "R6_DIAGNOSTIC_ACTUAL_REPLAY_PROOF_REQUIRED")
        require(not any((directory / name).exists() for name in ("ERROR_RECEIPT.json", "FAILED.json"))
                and (done["arm"], done["partition"], done["seed"]) == (arm, part, seed)
                and done["status"] == "COMPLETE_R6_TARGET_FIT" and done["contract"]["contract_sha256"] == lock["contract_sha256"]
                and unit["completion_sha256"] == sha(directory / "COMPLETE.json")
                and unit["artifact_sha256"] == done["artifact_sha256"] == sha(directory / "model.pt")
                and unit["qualification_sha256"] == sha(qpath) and unit["upstream_replay_audit_sha256"] == sha(upath),
                "R6_DIAGNOSTIC_CHECKPOINT_RECEIPT_CHANGED")
        entries.append((directory, done))
    refs = audit["zero_fit_baseline_reference_replays"]
    ref_expected = {(arm, part, seed) for arm in ("B0", "B1") for part in PARTITIONS for seed in SEEDS}
    require(len(refs) == 18 and {(r["arm"], r["partition"], r["seed"]) for r in refs} == ref_expected,
            "R6_DIAGNOSTIC_18_REFERENCE_REPLAYS_REQUIRED")
    for row in refs:
        path = args.private / "references" / f"{row['arm']}_p{row['partition']}_seed{row['seed']}.json"
        ref = read(path)
        require(row["status"] == "PASS" and row["independent_fixed_anchor_metrics_recomputed"]
                and row["new_optimizer_fits"] == 0 and row["receipt_sha256"] == sha(path)
                and ref["contract_sha256"] == lock["contract_sha256"] and ref["actual_new_optimizer_fits"] == 0
                and (ref["arm"], ref["partition"], ref["seed"]) == (row["arm"], row["partition"], row["seed"]),
                "R6_DIAGNOSTIC_REFERENCE_REPLAY_CHANGED")
    return lock, entries, sha(qpath)


def run(args):
    require(not any((args.output / name).exists() for name in (INDEX_NAME, *(_shard_name(arm) for arm in ARMS))),
            "R6_DIAGNOSTIC_OUTPUT_PRESERVED_NO_OVERWRITE")
    lock, entries, qualification_sha = verify_complete_replay_gate(args)
    # No target loader/model deserialization precedes the complete metadata gate.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from scripts import bgb_en_r6_common as common
    from scripts.bgb_en_r6_replay import validate_entry
    from src.bgb_en.r6_training import saved_state, state_hash
    from src.bgb_en.r6_pathology import source_model_state_sha256
    from src.bgb_en.r6_survival import MeanGeneQueryContext
    common.configure_deterministic_runtime(); torch.set_num_threads(2)
    ctx = common.context(args)
    prepared, origins, unique = {}, {}, set()
    for part in PARTITIONS:
        prepared[part], origins[part] = common.qualified_prepared(args, ctx, part)
        unique.update(map(int, np.asarray(prepared[part]["valid"])[:8]))
    rows = []
    to_array = lambda value: value.detach().cpu().numpy()
    for directory, metadata in entries:
        arm, part, seed = metadata["arm"], metadata["partition"], metadata["seed"]
        local = argparse.Namespace(**(vars(args) | {"partition": part}))
        artifact = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        validate_entry(directory / "COMPLETE.json", metadata, artifact, ctx, args, origins[part], qualification_sha)
        bundle = prepared[part]
        require(np.array_equal(artifact["train"], bundle["train"]) and np.array_equal(artifact["valid"], bundle["valid"]),
                "R6_DIAGNOSTIC_SAVED_VALID_SCOPE_CHANGED")
        model, origin = common.construct(local, ctx, bundle, arm, seed)
        require(origin == metadata["source_origin"] == artifact["model_origin"], "R6_DIAGNOSTIC_MODEL_ORIGIN_CHANGED")
        initial = saved_state(model)
        require(state_hash(initial) == metadata["initialization"]["full_state_sha256"]
                and state_hash(initial, target_only=True) == metadata["initialization"]["target_state_sha256"]
                and artifact["initialization"] == metadata["initialization"], "R6_DIAGNOSTIC_INITIALIZATION_CHANGED")
        model.load_state_dict(artifact["final_state_dict"], strict=True)
        require(source_model_state_sha256(model.core.foundation) == origin["source_origin"]["complete_source_state_sha256"]
                and np.array_equal(model.query_map, artifact["query_map"]), "R6_DIAGNOSTIC_FROZEN_SOURCE_OR_PAIRING_CHANGED")
        model.mean_context = MeanGeneQueryContext(artifact["mean_query"].to(args.device), artifact["mean_query_mask"].to(args.device), len(bundle["train"]))
        model.mean_context.expand(1)  # Shape-only; saved mean was already independently replayed.
        model.eval(); selected = np.asarray(bundle["valid"])[:8]
        idx = torch.as_tensor(selected, device=args.device)
        patch, _ = model.patch_bank.select(idx); patch_mask = to_array(patch["patch_mask"])
        before_state = saved_state(model)
        before = state_hash(before_state)
        drift = parameter_drift(initial, before_state, [name for name, value in model.named_parameters() if value.requires_grad])
        controls, outputs, errors = {}, {}, {}
        with torch.no_grad():
            for control in CONTROLS:
                out = model(idx, control=control)
                saved_risk = np.asarray(artifact["frozen_control_predictions"][control]["risk"])
                require(saved_risk.shape == (len(bundle["valid"]),), "R6_DIAGNOSTIC_SAVED_CONTROL_RISK_SHAPE")
                expected = saved_risk[:len(selected)]
                error = float(np.max(np.abs(to_array(out["risk"]) - expected)))
                require(error <= 2e-6, "R6_DIAGNOSTIC_FIRST8_SAVED_RISK_REPLAY_FAILED")
                errors[control] = error
                reader = out["patch_cond_read"]
                reverse = None
                if out["diagnostics"]["reverse_kind"] == "gene":
                    ep, sign = to_array(model.core.gene_rereader.edge_program), to_array(model.core.gene_rereader.edge_sign)
                    measured_edges = to_array(out["gene_measured_mask"].any(-1)[:, model.core.gene_rereader.edge_gene])
                    reverse = {}
                    for name in ("reverse_base_read", "reverse_cond_read"):
                        source = out[name]
                        reverse[name] = gene_statistics({key: to_array(source[key]) for key in
                            ("weights", "measured_counts", "retained_counts", "coverage", "component_mask", "component_defined")}, ep, sign, measured_edges)
                offset_sd = float(out["offset"].std(unbiased=False))
                controls[control] = dict(patch_attention=patch_statistics(to_array(reader["weights"]), to_array(reader["null"]),
                    patch_mask, to_array(reader["valid_WSI"])), gene_reread=reverse,
                    activation_norms=activation_norms(out, patch_mask, forward_enabled=model.core.forward_enabled),
                    reverse_kind=out["diagnostics"]["reverse_kind"], offset_SD=offset_sd,
                    risk_SD_over_offset_SD=ratio(float(out["risk"].std(unbiased=False)), offset_sd),
                    correction_SD_over_offset_SD=ratio(float(out["delta"].std(unbiased=False)), offset_sd),
                    cross_correction_SD_over_offset_SD=ratio(float(out["cross_delta"].std(unbiased=False)), offset_sd))
                outputs[control] = out
        matched, wrong = outputs["matched_query"], outputs["wrong_query"]
        for control in CONTROLS[1:]:
            require(all(torch.equal(matched[key], outputs[control][key]) for key in
                ("gene_values", "gene_measured_mask", "main_delta", "offset")), "R6_DIAGNOSTIC_QUERY_CONTROL_REPLACED_MAIN_OR_GENES")
            require(torch.equal(matched["patch_values"], outputs[control]["patch_values"]), "R6_DIAGNOSTIC_ORIGINAL_PATCH_VALUES_REPLACED")
        default = outputs["wrong_query" if arm == "B9" else "matched_query"]
        shuffled = outputs["attention_permutation"]["patch_cond_read"]
        require(torch.equal(shuffled["weights"], default["patch_cond_read"]["weights"])
                and torch.equal(shuffled["logits"], default["patch_cond_read"]["logits"]), "R6_DIAGNOSTIC_ASSIGNMENT_CHANGED_ATTENTION")
        assignment = to_array(shuffled["value_assignment"])
        for b in range(len(patch_mask)):
            for s in range(patch_mask.shape[1]):
                ids = np.flatnonzero(patch_mask[b, s])
                require(np.array_equal(np.sort(assignment[b, s, ids]), ids), "R6_DIAGNOSTIC_ASSIGNMENT_NOT_WITHIN_WSI_BIJECTION")
        a, b = matched["patch_cond_read"], wrong["patch_cond_read"]
        tv = .5 * ((a["weights"] - b["weights"]).abs().sum(-1) + (a["null"] - b["null"]).abs())
        tv_valid = a["valid_WSI"] & b["valid_WSI"]
        tv_per_patient = [float(tv[i][tv_valid[i]].mean()) if tv_valid[i].any() else None for i in range(len(selected))]
        offset_sd = float(matched["offset"].std(unbiased=False))
        difference_sd = float((matched["risk"] - wrong["risk"]).std(unbiased=False))
        row = dict(arm=arm, partition=part, seed=seed, fixed_validation_rows=len(selected), controls=controls,
            trainable_parameter_drift=drift,
            first8_saved_control_risk_max_abs_error=errors, matched_wrong_same_program_top16_jaccard=paired_top_overlap(
                to_array(a["weights"]), to_array(b["weights"]), patch_mask, to_array(a["valid_WSI"]), to_array(b["valid_WSI"])),
            matched_wrong_patch_plus_null_TV=scalar_summary(tv_per_patient),
            matched_wrong_risk_difference_SD=difference_sd, matched_wrong_risk_difference_SD_over_offset_SD=ratio(difference_sd, offset_sd),
            query_source_mask_changed_rows=int((matched["query_source_mask"] != wrong["query_source_mask"]).sum()),
            own_gene_patch_values_main_effects_offset_exact=True, assignment_own_value_multiset_preserved=True,
            assignment_weights_and_logits_preserved=True, reverse_gene_attention_not_applicable=controls["matched_query"]["gene_reread"] is None,
            state_unchanged_after_inference=before == state_hash(saved_state(model)))
        require(row["state_unchanged_after_inference"], "R6_DIAGNOSTIC_INFERENCE_CHANGED_STATE")
        rows.append(row)
        print(json.dumps(dict(arm=arm, partition=part, seed=seed, status="PASS_INFERENCE_ONLY_DIAGNOSTIC")), flush=True)
        del model, outputs, artifact, patch, out, matched, wrong, initial, before_state
    require(len(rows) == 90, "R6_DIAGNOSTIC_MATRIX_INCOMPLETE")
    result = dict(schema="bgb_en_r6_attention_diagnostics_v1", status="COMPLETE_90_FIRST8_INFERENCE_DIAGNOSTICS",
        script_sha256=sha(__file__), contract_sha256=lock["contract_sha256"], replay_audit_sha256=sha(args.private / "REPLAY_AUDIT.json"),
        completed_model_cells=90, controls_per_cell=5, diagnostic_forward_calls=450, repeated_patient_evaluations=sum(r["fixed_validation_rows"] for r in rows),
        unique_probed_patients=len(unique), independent_patient_count_is_not_720=True, fixed_first_valid_rows=8,
        cases_selected_using_outcomes=False, outcomes_used_for_new_performance=False, new_optimizer_fits=0, new_backward_calls=0,
        saved_train_mean_reused_after_actual_replay=True, medical_histology_or_causal_gene_truth_claimed=False,
        attention_rank_definition="exp(-sum(p*log(p))), p=sigma/sum(sigma), SVD_of_raw_real_patch_weight_matrix_per_WSI_no_null_no_centering",
        entropy_definition="natural_log_patch_plus_null_entropy_separate_from_real_patch_conditional_entropy",
        aggregation="within_each_patient_equal_valid_WSI_then_equal_patients; null/empty_excluded_counts_reported",
        topk_definition="stable_descending_weights_min(16,valid_patch_count); zero_real_mass_or_invalid_row=None",
        gene_N_eff_definition="up/down_normalized_separately; exp(entropy) and inverse_sum_square; undefined_member_group=None",
        assignment_control="default_trained_query_context; B9_wrong_query; preserve_own_WSI_value_multiset_not_same_index_assignment",
        baseline_pair_credit_zero_change_already_verified=True, new_pair_credit_or_DSS_metric_recomputation=False,
        private_identifiers_coordinates_values_indices_or_weights_exported=False, rows=rows)
    result["parameter_drift_scope"] = ("all named_parameters(requires_grad=True), default de-duplicated shared parameters, "
        "grouped by first two name segments; fresh initial vs loaded final float64 norms; zero-initial relative drift=None; "
        "isolated L2 gradient norm=2e-4*final_L2 from 1e-4*sum(theta^2) only, excluding total/cross delta penalties; "
        "not a total gradient and not scope-matched to first-eight training microbatch pure-Cox gradient; "
        "do not subtract norms or infer optimization causality")
    result["activation_norm_scope"] = ("existing raw target activations only, no added normalization or model calls; "
        "G0/context/Q/E/reverse norms equally average valid programs within patient then patients; "
        "E states were already aggregated equally over valid WSIs by the model; patch norms equally average patches within "
        "WSI, then valid WSIs within patient, then patients; ratios are itemwise delta/base L2 with zero denominators excluded "
        "and reported as None when empty; small amplitudes alone do not identify data scarcity or a unique failure mechanism")
    write_diagnostics(result, args.output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "private", "primary-private", "source-private", "config", "source-config", "launch-gate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--launch-gate-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(dict(status=result["status"], model_cells=90, unique_probed_patients=result["unique_probed_patients"],
                          new_optimizer_fits=0, new_backward_calls=0)))


if __name__ == "__main__":
    main()
