"""Locked R5 accounting, outcome-blind query pairing and finite decisions.

This module does not load measurements, fit teachers, train a neural model or
calculate confidence intervals. Callers keep patient mappings and risks private.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

REFERENCE_ARMS = ("A0", "A1")
MAIN_ARMS = tuple("A" + str(i) for i in range(2, 9))
ALL_MAIN_ARMS = REFERENCE_ARMS + MAIN_ARMS
READOUT_CANDIDATES = ("A4", "A5", "A6", "A7")
MECHANISM_CONTROLS = (
    "WRONG_QUERY", "QUERY_MEAN", "INTERACTION_OFF_RETRAIN",
    "ONEWAY_O2P_ONLY", "ONEWAY_P2O_ONLY",
)
SEEDS = (31, 47, 83)
PARTITIONS = (0, 1, 2)
NUMERATOR_FIELDS = ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den")
UPSTREAM_COMMIT = "ecd273d2036d98e40dd9e997dd45e8de9eba7947"


def require(value: Any, message: str) -> None:
    if not bool(value):
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes(root: str | Path, relative_files: Iterable[str]) -> dict[str, str]:
    root = Path(root).resolve()
    names = list(relative_files)
    require(bool(names) and len(names) == len(set(names)), "Source list is empty or duplicated")
    result = {}
    for name in sorted(names):
        relative = Path(name)
        target = root / relative
        require(not relative.is_absolute() and ".." not in relative.parts
                and target.resolve().is_relative_to(root) and not target.is_symlink()
                and target.is_file(), "Unsafe or absent locked source")
        result[relative.as_posix()] = sha256(target)
    return result


def make_protocol_lock(root: str | Path, main_config: str, mechanism_config: str,
                       source_files: Iterable[str], *,
                       data_fingerprints: Mapping[str, str],
                       prepared_receipt_sha256: Mapping[str, str]) -> dict[str, Any]:
    """Build only public fingerprints; raw data paths/indices are not accepted."""
    root = Path(root)
    main = json.loads((root / main_config).read_text(encoding="utf-8-sig"))
    mechanism = json.loads((root / mechanism_config).read_text(encoding="utf-8-sig"))
    require(main["schema"] == "bgb_en_r5_main_v1"
            and mechanism["schema"] == "bgb_en_r5_mechanism_v1", "Wrong R5 config schema")
    require(main["upstream_commit"] == mechanism["upstream_commit"] == UPSTREAM_COMMIT,
            "R5 upstream commit changed")
    require(main["formal_main_neural_fits"] == 63
            and mechanism["formal_mechanism_neural_fits"] == 45
            and mechanism["maximum_finalists"] == 1, "R5 finite matrix changed")
    require(main["training"]["epochs"] == 80
            and main["prototype"]["neural_seeds"] == list(SEEDS)
            and main["prototype"]["inner_partitions"] == list(PARTITIONS), "R5 schedule changed")
    require(main["training"]["total_delta_L2"] == .1
            and main["training"]["interaction_delta_L2"] == .1, "R5 common penalty changed")
    require(main["training"]["learning_rate"] == .001
            and main["training"]["batch_size"] == 8
            and main["training"]["dropout"] == 0
            and main["readout"]["gamma"] == .1 and main["readout"]["rank"] == 4
            and main["readout"]["temperature"] == 4
            and main["readout"]["sham_rank"] == 75, "R5 optimizer or fixed readout changed")
    require(tuple(row["id"] for row in main["arms"] if row["trainable"]) == MAIN_ARMS
            and tuple(mechanism["controls"]) == MECHANISM_CONTROLS,
            "R5 arm identities changed")
    require(main["primary_endpoint"] == "DSS"
            and not main["prototype"]["outer_test_access"]
            and not any(main["evaluation"][key] for key in ("bootstrap", "confidence_intervals", "p_values")),
            "R5 endpoint, outer access or inference policy changed")
    require(len(prepared_receipt_sha256) == 3, "R5 requires three verified prepared receipts")
    for mapping in (data_fingerprints, prepared_receipt_sha256):
        require(bool(mapping) and all(isinstance(key, str) and key
                and "/" not in key and "\\" not in key
                and isinstance(value, str) and len(value) == 64
                and set(value) <= set("0123456789abcdef")
                for key, value in mapping.items()), "Expected named SHA256 fingerprints only")
    files = set(source_files) | {main_config, mechanism_config}
    result = dict(schema="bgb_en_r5_protocol_lock_v1", status="LOCKED_BEFORE_FORMAL_FITS",
                  upstream_commit=UPSTREAM_COMMIT, main_config=main_config,
                  mechanism_config=mechanism_config, sources=source_hashes(root, sorted(files)),
                  data_fingerprints=dict(data_fingerprints),
                  prepared_receipt_sha256=dict(prepared_receipt_sha256),
                  planned_main_neural_fits=63, planned_mechanism_neural_fits=45,
                  maximum_finalists=1, outer_test_used_for_selection=False,
                  bootstrap=False, confidence_intervals=False, p_values=False)
    result["contract_sha256"] = canonical_sha256(result)
    return result


def write_protocol_lock(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Resume only an identical lock; never silently replace its source pins."""
    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8-sig"))
        compare = {key: item for key, item in previous.items() if key != "created_UTC"}
        require(compare == dict(value), "Existing R5 lock differs; refuse overwrite")
        return previous
    result = dict(value, created_UTC=datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def verify_protocol_lock(lock: Mapping[str, Any], root: str | Path) -> None:
    require(lock.get("status") == "LOCKED_BEFORE_FORMAL_FITS", "R5 lock is not immutable")
    contract = {key: value for key, value in lock.items()
                if key not in ("created_UTC", "contract_sha256")}
    require(canonical_sha256(contract) == lock["contract_sha256"], "R5 contract content changed")
    require(source_hashes(root, lock["sources"]) == lock["sources"], "R5 source bytes changed")


def rna_pattern_labels(measured: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Hash actual pre-imputation finite masks, never a platform proxy."""
    mask = np.asarray(measured)
    require(mask.ndim == 2 and mask.dtype == np.bool_ and min(mask.shape) > 0,
            "RNA finite mask must be a nonempty Boolean matrix")
    packed = np.packbits(mask, axis=1, bitorder="little")
    labels = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in packed])
    unique, counts = np.unique(labels, return_counts=True)
    full = mask.all(1)
    audit = dict(patients=len(mask), genes=mask.shape[1], complete_patients=int(full.sum()),
                 not_fully_measured_patients=int((~full).sum()), distinct_finite_patterns=len(unique),
                 pattern_group_sizes=sorted(counts.tolist(), reverse=True),
                 mask_shape_and_bytes_sha256=hashlib.sha256(
                     str(mask.shape).encode() + packed.tobytes()).hexdigest(),
                 patterns_derived_before_imputation=True, outcome_labels_used=False)
    return labels, audit


def hungarian_query_mismatch(indices: Iterable[int], *, missing_pattern: Iterable[Any],
                             cancer: Iterable[Any], stage: Iterable[Any],
                             platform: Iterable[Any], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Lexicographically minimize total mismatches; keep train/valid separate.

    With base n+1, one higher-priority mismatch costs more than all n rows'
    lower-priority mismatches. The complete random tie-break costs <1/8, so
    it cannot overturn an integer mismatch-cost improvement of at least one.
    """
    fields = [np.asarray(list(x)).astype(str) for x in
              (missing_pattern, cancer, stage, platform)]
    require(len({len(x) for x in fields}) == 1, "Pairing metadata sizes differ")
    source_indices = np.asarray(list(indices))
    require(source_indices.ndim == 1 and source_indices.dtype.kind in "iu",
            "Pairing patient indices must be integers, not silently cast values")
    indices = source_indices.astype(np.int64, copy=False)
    n, total = len(indices), len(fields[0])
    require(n >= 2 and len(np.unique(indices)) == n
            and ((indices >= 0) & (indices < total)).all(), "Invalid pairing partition")
    base = n + 1
    mismatch = [values[indices, None] != values[None, indices] for values in fields]
    integer_cost = sum(x.astype(np.int64) * base**(3 - at) for at, x in enumerate(mismatch))
    cost = integer_cost.astype(np.float64)
    cost += np.random.default_rng(int(seed)).random((n, n)) * (.125 / n)
    np.fill_diagonal(cost, np.inf)
    row, column = linear_sum_assignment(cost)
    require(np.array_equal(row, np.arange(n)) and np.isfinite(cost[row, column]).all(),
            "No valid within-partition query derangement")
    mapping = np.arange(total, dtype=np.int64)
    mapping[indices[row]] = indices[column]
    require(np.all(mapping[indices] != indices) and set(mapping[indices]) == set(indices),
            "Query mapping is not a within-partition fixed-point-free bijection")
    names = ("RNA_finite_pattern", "cancer", "clinical_stage", "platform")
    matches = [~x[row, column] for x in mismatch]
    audit = dict(n=n, fixed_points=0, priority=list(names), integer_base=base,
                 exact_all_fields=int(np.logical_and.reduce(matches).sum()),
                 matched_counts={name: int(value.sum()) for name, value in zip(names, matches)},
                 mismatched_counts={name: int((~value).sum()) for name, value in zip(names, matches)},
                 total_integer_cost=int(integer_cost[row, column].sum()),
                 random_tie_break_total_bound=.125, seed=int(seed),
                 solver="scipy.optimize.linear_sum_assignment", outcomes_accessed=False,
                 partition_bijection=True,
                 mapping_sha256=hashlib.sha256(mapping.tobytes()).hexdigest())
    return mapping, audit


def _metric_numbers(row: Mapping[str, Any]) -> dict[str, float]:
    source = row.get("metrics", row)
    result = {key: float(source[key]) for key in NUMERATOR_FIELDS}
    require(all(np.isfinite(value) and value >= 0 for value in result.values()),
            "Nonfinite or negative concordance components")
    require(result["Harrell_num"] <= result["Harrell_den"] + 1e-8
            and result["Uno_num"] <= result["Uno_den"] + 1e-8, "Concordance numerator exceeds denominator")
    return result


def aggregate_metrics(rows: Iterable[Mapping[str, Any]], *, arms: Iterable[str] = ALL_MAIN_ARMS,
                      seeds: Iterable[int] = SEEDS, partitions: Iterable[int] = PARTITIONS) -> dict[str, Any]:
    """No partial arm mean and no cross-partition comparison of risk values."""
    arms, seeds, partitions = tuple(arms), tuple(seeds), tuple(partitions)
    selected = {}
    for row in rows:
        arm = str(row["arm"])
        if arm not in arms:
            continue
        seed = int(row["seed"])
        part = int(row.get("partition", row.get("inner_partition", -1)))
        require(seed in seeds and part in partitions, "Metric row outside the locked grid")
        key = (arm, seed, part)
        require(key not in selected, "Duplicate fold/seed metric identity")
        selected[key] = row
    summaries = {}
    for arm in arms:
        expected = [(arm, seed, part) for seed in seeds for part in partitions]
        complete = [key for key in expected if key in selected
                    and str(selected[key].get("status", "COMPLETE")).startswith("COMPLETE")]
        entry = dict(status="INCOMPLETE", completed_cells=len(complete), expected_cells=len(expected),
                     Harrell=None, Uno60=None, per_seed=[], missing_or_noncomplete_cells=len(expected) - len(complete))
        if len(complete) == len(expected):
            for seed in seeds:
                records = [_metric_numbers(selected[(arm, seed, part)]) for part in partitions]
                summed = {key: float(sum(row[key] for row in records)) for key in NUMERATOR_FIELDS}
                entry["per_seed"].append(dict(seed=seed, **summed,
                    Harrell=summed["Harrell_num"] / summed["Harrell_den"] if summed["Harrell_den"] else None,
                    Uno60=summed["Uno_num"] / summed["Uno_den"] if summed["Uno_den"] else None))
            if all(row["Harrell"] is not None and row["Uno60"] is not None for row in entry["per_seed"]):
                entry.update(status="COMPLETE", Harrell=float(np.mean([r["Harrell"] for r in entry["per_seed"]])),
                             Uno60=float(np.mean([r["Uno60"] for r in entry["per_seed"]])))
            else:
                entry["status"] = "NO_ELIGIBLE_PAIRS"
            active = [selected[key].get("active_parameters", selected[key].get("parameters")) for key in complete]
            entry["active_parameters"] = max(active) if all(x is not None for x in active) else None
        summaries[arm] = entry
    return dict(status="COMPLETE" if all(x["status"] == "COMPLETE" for x in summaries.values()) else "INCOMPLETE",
                arms=summaries, within_partition_concordance_only=True, cross_partition_risk_comparisons=False,
                seed_repeats_are_not_independent_cohorts=True, inferred_repair_damage_from_C_difference=False)


def paired_comparison(summary: Mapping[str, Any], candidate: str, reference: str,
                      *, epsilon: float = 1e-12) -> dict[str, Any]:
    a, b = summary["arms"][candidate], summary["arms"][reference]
    if a["status"] != "COMPLETE" or b["status"] != "COMPLETE":
        return dict(status="INCOMPLETE", candidate=candidate, reference=reference,
                    delta_Harrell=None, delta_Uno60=None, per_seed=[])
    aa, bb = {r["seed"]: r for r in a["per_seed"]}, {r["seed"]: r for r in b["per_seed"]}
    require(set(aa) == set(bb), "Paired aggregate seed identities differ")
    rows = []
    for seed in sorted(aa):
        require(all(np.isclose(aa[seed][key], bb[seed][key], atol=1e-8, rtol=1e-10)
                    for key in ("Harrell_den", "Uno_den")), "Paired aggregate endpoint denominators differ")
        rows.append(dict(seed=seed, delta_Harrell=aa[seed]["Harrell"] - bb[seed]["Harrell"],
                         delta_Uno60=aa[seed]["Uno60"] - bb[seed]["Uno60"]))
    dh, du = a["Harrell"] - b["Harrell"], a["Uno60"] - b["Uno60"]
    return dict(status="COMPLETE", candidate=candidate, reference=reference, delta_Harrell=dh, delta_Uno60=du,
                per_seed=rows, positive_Harrell_seeds=sum(r["delta_Harrell"] > epsilon for r in rows),
                positive_Uno60_seeds=sum(r["delta_Uno60"] > epsilon for r in rows),
                nonnegative_Harrell_seeds=sum(r["delta_Harrell"] >= -epsilon for r in rows),
                nonnegative_Uno60_seeds=sum(r["delta_Uno60"] >= -epsilon for r in rows),
                weak_Harrell_direction=abs(dh) < .003, weak_Uno60_direction=abs(du) < .003,
                direct_pair_repair_damage_not_inferred=True)


def _complete_nine(entry: Mapping[str, Any]) -> bool:
    return (entry.get("status") == "COMPLETE" and entry.get("completed_cells") == 9
            and entry.get("expected_cells") == 9 and len(entry.get("per_seed", [])) == 3
            and {row.get("seed") for row in entry["per_seed"]} == set(SEEDS))


def select_finalist(summary: Mapping[str, Any], *, epsilon: float = 1e-12) -> dict[str, Any]:
    require(all(_complete_nine(summary["arms"].get(arm, {})) for arm in MAIN_ARMS),
            "All 63 main neural cells must complete before selection")
    ranked, remaining = [], list(READOUT_CANDIDATES)
    while remaining:
        best_h = max(summary["arms"][arm]["Harrell"] for arm in remaining)
        tied = [arm for arm in remaining if summary["arms"][arm]["Harrell"] >= best_h - epsilon]
        best_u = max(summary["arms"][arm]["Uno60"] for arm in tied)
        tied = [arm for arm in tied if summary["arms"][arm]["Uno60"] >= best_u - epsilon]
        chosen = min(tied, key=lambda arm: (
            summary["arms"][arm].get("active_parameters") if summary["arms"][arm].get("active_parameters") is not None else float("inf"), arm))
        ranked.append(chosen)
        remaining.remove(chosen)
    return dict(status="LOCKED_AFTER_COMPLETE_MAIN_BEFORE_MECHANISM", selected_arm=ranked[0],
                ranked_candidates=ranked, maximum_finalists=1, mechanism_new_fits_required=45,
                mechanism_required_even_if_main_negative=True, selection_uses_outer_test=False,
                selection_summary_sha256=canonical_sha256(summary))


def decision_gate(main_summary: Mapping[str, Any], mechanism_summary: Mapping[str, Any],
                  selected_arm: str, *, qualification_passed: bool, replay_passed: bool,
                  epsilon: float = 1e-12) -> dict[str, Any]:
    """Finite directional gate; incomplete controls never authorize formal R6."""
    base = dict(decision="INCONCLUSIVE", R6_formal_start_eligible=False,
                selected_arm=selected_arm, hypotheses={h: "INCONCLUSIVE" for h in ("H5.1", "H5.2", "H5.3")},
                independent_external_confirmation=False, bootstrap=False, confidence_intervals=False, p_values=False)
    if not qualification_passed or not replay_passed:
        return dict(base, reason="Qualification or saved-checkpoint replay incomplete")
    if selected_arm not in READOUT_CANDIDATES or any(
            not _complete_nine(main_summary["arms"].get(arm, {})) for arm in ALL_MAIN_ARMS) or any(
            not _complete_nine(mechanism_summary["arms"].get(arm, {})) for arm in MECHANISM_CONTROLS):
        return dict(base, reason="Required 63 main, baseline references or 45 mechanism cells incomplete")
    comparisons = {ref: paired_comparison(main_summary, selected_arm, ref) for ref in ("A1", "A2", "A3", "A8")}
    mixed = dict(arms={selected_arm: main_summary["arms"][selected_arm], **mechanism_summary["arms"]})
    controls = {ref: paired_comparison(mixed, selected_arm, ref) for ref in MECHANISM_CONTROLS}

    def both_positive(row):
        return row["delta_Harrell"] > epsilon and row["delta_Uno60"] > epsilon

    def not_worse(row):
        return row["delta_Harrell"] >= -epsilon and row["delta_Uno60"] >= -epsilon

    def repeated_positive(row):
        return row["positive_Harrell_seeds"] >= 2 and row["positive_Uno60_seeds"] >= 2

    readout = all(both_positive(comparisons[ref]) for ref in ("A2", "A3"))
    wrong = controls["WRONG_QUERY"]
    pairing = not_worse(wrong) and repeated_positive(wrong) and not_worse(controls["QUERY_MEAN"])
    ablation = all(not_worse(controls[ref]) and (
        controls[ref]["delta_Harrell"] > epsilon or controls[ref]["delta_Uno60"] > epsilon)
        for ref in ("INTERACTION_OFF_RETRAIN", "ONEWAY_O2P_ONLY", "ONEWAY_P2O_ONLY"))
    main_effect = all(both_positive(comparisons[ref]) for ref in ("A1", "A2", "A8"))
    performance = both_positive(comparisons["A1"])
    if readout and pairing and ablation and main_effect:
        decision = "R5_MECHANISM_DIRECTIONAL_GO"
    elif performance:
        decision = "R5_PERFORMANCE_ONLY_NOT_MECHANISM"
    else:
        decision = "R5_READOUT_STOP_R6_ELIGIBLE"
    base.update(decision=decision, R6_formal_start_eligible=decision in (
        "R5_PERFORMANCE_ONLY_NOT_MECHANISM", "R5_READOUT_STOP_R6_ELIGIBLE"),
        hypotheses={"H5.1": "SUPPORTED_DIRECTIONALLY" if readout else "NOT_SUPPORTED",
                    "H5.2": "SUPPORTED_DIRECTIONALLY" if pairing else "NOT_SUPPORTED",
                    "H5.3": "SUPPORTED_DIRECTIONALLY" if main_effect and ablation else "NOT_SUPPORTED"},
        main_comparisons=comparisons, mechanism_comparisons=controls,
        very_small_deltas_remain_weak_directional_evidence=True,
        reason="Complete prelocked development matrices; no new external confirmation")
    return base
