"""Phase22-B: locked external count-thinning / weak-sampling audit.

This runner performs no training and accepts no survival endpoint. Run validation
first; test requires its immutable selection receipt. All rows, arrays, donor
identifiers, failure states and input paths stay in ``--private-out``. Public
JSON contains aggregate metrics and hashes only. A non-converged projection is
retained and cannot silently enter configuration selection or the C gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.phase22_spatial_bridge.partial_measurement_projection import (
    ProjectionConfig, solve_partial_measurement_projection,
)
from src.phase22_spatial_bridge.sampling_simulation import (
    LOCKED_AREA_FRACTIONS, simulate_partial_sampling,
)

SCHEMA = "phase22_sampling_audit_v2"
PARTIAL_MEASUREMENT_POLICY = "full_fixed_dictionary_required_for_B;test_only_partial_sections_ineligible;validation_partial_is_error"
SEEDS = (31, 47, 83)
LAMBDAS = (0.0, 0.1, 1.0, 10.0)
RHOS = (0.25, 0.5, 0.75, 1.0)
METRICS = ("within_tissue_gene_Spearman", "spatially_centered_MSE", "hotspot_recall_q90")
CONTRASTS = (("B3", "B1"), ("B3", "B2"), ("B3", "B4"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def strings(value) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(value).ravel()]


def check_file(path: Path, expected: str, label: str):
    if not path.is_file() or not expected or sha(path) != expected:
        raise ValueError(f"FAIL_DATA_CONTRACT: {label} file/hash mismatch")


def canonical_prior(pi) -> tuple[np.ndarray, float]:
    pi = np.asarray(array(pi), dtype=np.float64)
    if pi.ndim != 2 or min(pi.shape) < 1 or not np.isfinite(pi).all() or np.any(pi <= 0):
        raise ValueError("prior must be finite and strictly positive [position,state]")
    error = float(np.max(np.abs(pi.sum(1) - 1)))
    if error > 1e-5:
        raise ValueError("prior row sums are inconsistent with softmax outputs")
    # Explicitly audit the float32 -> float64 roundoff normalization. No floor,
    # biological smoothing or arbitrary semantic modification is introduced.
    return pi / pi.sum(1, keepdims=True), error


class LocalEvaluator:
    """The same evaluation molecules, gene support and ties for every arm.

    Targets are raw evaluation counts divided by evaluation library exposure.
    B1 is deliberately NOT row-normalized after its per-gene rescaling: that
    would change spatial ranks and would no longer be the declared control.
    Centering is within this tissue/region and gene, never across test donors.
    """
    def __init__(self, counts: np.ndarray, mask: np.ndarray):
        y = np.asarray(counts, dtype=np.float64)
        measured = np.asarray(mask, dtype=bool)
        if y.ndim != 2 or measured.shape != y.shape:
            raise ValueError("evaluation counts and measured mask mismatch")
        if np.any(~np.isfinite(y[measured])) or np.any(y[measured] < 0):
            raise ValueError("invalid measured evaluation counts")
        # Source qualification must select the global common measured vocabulary.
        # Do not silently renormalize a partial dictionary against a differently
        # defined measured-only bulk composition.
        if not measured.all():
            raise ValueError("FAIL_DATA_CONTRACT: partial dictionary measurement requires a new scale contract")
        exposure = y.sum(1)
        self.valid_rows = exposure > 0
        self.target = y[self.valid_rows] / exposure[self.valid_rows, None]
        self.eligible = (np.ptp(self.target, axis=0) > 0) if self.valid_rows.sum() >= 4 else np.zeros(y.shape[1], bool)
        t = self.target[:, self.eligible]
        self.target_centered = t - t.mean(0) if t.size else t
        ranks = rankdata(t, axis=0, method="average") if t.size else t
        self.rank_centered = ranks - ranks.mean(0) if t.size else ranks
        self.rank_norm = np.sqrt(np.square(self.rank_centered).sum(0))
        self.hot_truth = t > np.quantile(t, 0.9, axis=0) if t.size else t.astype(bool)
        self.hot_eligible = self.hot_truth.sum(0) > 0
        self.positions, self.genes = int(self.valid_rows.sum()), int(self.eligible.sum())

    def evaluate(self, prediction) -> dict:
        p = np.asarray(prediction, dtype=np.float64)
        if p.shape != (len(self.valid_rows), len(self.eligible)) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("local prediction must be finite nonnegative and match evaluation rows/genes")
        if not self.genes:
            return dict(zip(METRICS, [None] * 3), eligible_positions=self.positions, eligible_genes=0,
                        status="UNAVAILABLE_INSUFFICIENT_SPATIAL_VARIATION")
        p = p[self.valid_rows][:, self.eligible]
        centered = p - p.mean(0)
        ranks = rankdata(p, axis=0, method="average")
        ranks -= ranks.mean(0)
        denominator = self.rank_norm * np.sqrt(np.square(ranks).sum(0))
        correlations = np.divide((ranks * self.rank_centered).sum(0), denominator,
                                 out=np.zeros_like(denominator), where=denominator > 0)
        hot_pred = p > np.quantile(p, 0.9, axis=0)
        truth = self.hot_truth[:, self.hot_eligible]
        hits = (hot_pred[:, self.hot_eligible] & truth).sum(0) / truth.sum(0)
        return {
            METRICS[0]: float(np.mean(correlations)),
            METRICS[1]: float(np.mean(np.square(centered - self.target_centered))),
            METRICS[2]: float(np.mean(hits)) if len(hits) else None,
            "eligible_positions": self.positions, "eligible_genes": self.genes,
            "constant_prediction_genes_scored_zero_correlation": int(np.sum(denominator == 0)),
            "status": "EVALUATED",
        }


def gene_rescale(mu, bulk, weights, measured_mask):
    mu = np.asarray(mu, dtype=np.float64)
    measured = np.asarray(measured_mask, dtype=bool)
    if not measured.all():
        raise ValueError("FAIL_DATA_CONTRACT: B1 needs the locked common measured dictionary vocabulary")
    mean = np.asarray(weights, dtype=np.float64) @ mu
    if np.any(mean <= 0) or np.any(~np.isfinite(mean)):
        raise ValueError("global rescale has an unidentifiable zero predicted gene")
    scaled = mu * (np.asarray(bulk) / mean)[None, :]
    same = np.array_equal(rankdata(mu, axis=0), rankdata(scaled, axis=0))
    if not same:
        raise ValueError("B1_SPATIAL_RANK_INVARIANCE_FAILED")
    return scaled, {"per_gene_spatial_ranks_bitwise_identical": True,
                    "renormalized_across_genes_after_scaling": False}


def measurement_eligibility_record(entry, measured, *, stage, dictionary_gene_count):
    """A test-only missing-gene exclusion never changes the train vocabulary."""
    mask = np.asarray(measured)
    if (stage not in {"validation", "test"} or dictionary_gene_count < 1 or
        mask.dtype != np.dtype(bool) or mask.shape != (dictionary_gene_count,)):
        raise ValueError("invalid stage or fixed-dictionary measurement mask")
    if entry["split"] != stage:
        raise ValueError("source measurement partition does not match the requested B stage")
    missing = int(np.sum(~mask))
    if missing and stage != "test":
        raise ValueError("FAIL_DATA_CONTRACT: validation requires all fixed-dictionary genes; no zero-imputation or vocabulary reselection")
    return {
        "section_id_private": entry["section_id_private"],
        "donor_id_private": entry["donor_id_private"],
        "study_id_private": entry["study_id"], "assay": entry["assay"],
        "split": entry["split"],
        "status": "INELIGIBLE_PARTIAL_MEASUREMENT" if missing else "ELIGIBLE_FULL_MEASUREMENT",
        "dictionary_gene_count": dictionary_gene_count,
        "measured_dictionary_gene_count": int(mask.sum()),
        "unmeasured_dictionary_gene_count": missing,
        "measured_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "source_count_npz_sha256": entry["npz_sha256"],
        "feature_npz_sha256": entry["features_sha256"],
        "feature_receipt_sha256": entry["feature_receipt_sha256"],
        "missing_values_imputed": False, "dictionary_or_omega_refitted": False,
        "reason": ("partial_measured_composition_not_interchangeable_with_full_dictionary_bulk_scale" if missing else None),
    }


def measurement_eligibility_summary(ledger):
    """Publish counts and assay scope only; excluded IDs/masks stay private."""
    by_assay = {}
    for assay in sorted({r["assay"] for r in ledger}):
        rows = [r for r in ledger if r["assay"] == assay]
        eligible = [r for r in rows if r["status"] == "ELIGIBLE_FULL_MEASUREMENT"]
        excluded = [r for r in rows if r["status"] == "INELIGIBLE_PARTIAL_MEASUREMENT"]
        by_assay[assay] = {
            "source_sections": len(rows), "source_donors": len({r["donor_id_private"] for r in rows}),
            "eligible_sections": len(eligible), "eligible_donors": len({r["donor_id_private"] for r in eligible}),
            "ineligible_sections": len(excluded), "ineligible_donors": len({r["donor_id_private"] for r in excluded}),
            "ineligible_status": "INELIGIBLE_PARTIAL_MEASUREMENT" if excluded else None,
            "unmeasured_dictionary_gene_count_by_ineligible_section": sorted(r["unmeasured_dictionary_gene_count"] for r in excluded),
            "projection_evaluated_in_assay": bool(eligible),
        }
    return {
        "policy": PARTIAL_MEASUREMENT_POLICY, "source_sections": len(ledger),
        "eligible_sections": sum(v["eligible_sections"] for v in by_assay.values()),
        "ineligible_sections": sum(v["ineligible_sections"] for v in by_assay.values()),
        "eligible_assays": [a for a, v in by_assay.items() if v["eligible_sections"]],
        "fully_ineligible_assays": [a for a, v in by_assay.items() if not v["eligible_sections"]],
        "by_assay": by_assay,
        "fixed_train_dictionary_preserved": True, "no_zero_imputation_or_test_gene_reselection": True,
        "gate_generalization_to_ineligible_assays": False,
    }


def donor_assignment(sections: list[dict]) -> dict[str, str | None]:
    """A deterministic partition-local different-donor choice, never a section swap."""
    by_group = {}
    for section in sections:
        e = section["entry"]
        by_group.setdefault((e["study_id"], e["assay"], e["split"]), set()).add(e["donor_id_private"])
    result = {}
    for section in sections:
        e = section["entry"]
        donors = by_group[(e["study_id"], e["assay"], e["split"])]
        alternatives = sorted(d for d in donors if d != e["donor_id_private"])
        alternatives.sort(key=lambda d: digest_json(["phase22_wrong_bulk_v1", e["donor_id_private"], d]))
        result[e["section_id_private"]] = alternatives[0] if alternatives else None
    return result


def mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def donor_values(rows: list[dict], metric: str) -> dict[str, float]:
    # Region/component -> section -> seed -> donor. Areas, spots, seeds and
    # serial sections do not become independent bootstrap participants.
    groups = {}
    for r in rows:
        v = r.get("metrics", {}).get(metric)
        if r.get("numerical_status") != "CONVERGED" or v is None:
            continue
        key = (r["donor_id_private"], r["seed"], r["section_id_private"], r["area_fraction"])
        groups.setdefault(key, []).append(v)
    section = {}
    for (d, seed, sid, _frac), values in groups.items():
        section.setdefault((d, seed, sid), []).append(mean(values))
    seed_values = {}
    for (d, seed, _sid), values in section.items():
        seed_values.setdefault((d, seed), []).append(mean(values))
    donors = {}
    for (d, _seed), values in seed_values.items():
        donors.setdefault(d, []).append(mean(values))
    return {d: mean(values) for d, values in donors.items()}


def paired_rows(rows: list[dict], arm_a: str, arm_b: str):
    def key(r):
        return tuple(r[k] for k in ("seed", "section_id_private", "component_index", "area_fraction"))
    a = {key(r): r for r in rows if r["arm"] == arm_a}
    b = {key(r): r for r in rows if r["arm"] == arm_b}
    common = sorted(set(a) & set(b))
    aa, bb = [], []
    for k in common:
        x, y = a[k], b[k]
        if x["numerical_status"] == y["numerical_status"] == "CONVERGED":
            aa.append(x)
            bb.append(y)
    return aa, bb, len(common)


def bootstrap_contrasts(rows: list[dict], replicates: int = 20000):
    output = {}
    family = len(CONTRASTS) * 2
    for a, b in CONTRASTS:
        aa, bb, expected = paired_rows(rows, a, b)
        metrics = {}
        for metric in METRICS:
            av, bv = donor_values(aa, metric), donor_values(bb, metric)
            donors = sorted(set(av) & set(bv))
            deltas = np.asarray([av[d] - bv[d] for d in donors], dtype=np.float64)
            if not len(deltas):
                metrics[metric] = {"status": "UNAVAILABLE", "donors": 0}
                continue
            rng = np.random.default_rng(22031)
            sampled = deltas[rng.integers(0, len(deltas), size=(replicates, len(deltas)))].mean(1)
            seed_deltas = {}
            for seed in SEEDS:
                x = donor_values([r for r in aa if r["seed"] == seed], metric)
                y = donor_values([r for r in bb if r["seed"] == seed], metric)
                shared = sorted(set(x) & set(y))
                seed_deltas[str(seed)] = mean([x[d] - y[d] for d in shared])
            metrics[metric] = {"status": "EVALUATED", "donors": len(deltas), "point": float(deltas.mean()),
                               "ordinary_95_interval": np.quantile(sampled, [.025, .975]).tolist(),
                               "six_primary_comparison_Bonferroni_interval": (np.quantile(sampled, [.025 / family, 1 - .025 / family]).tolist()
                                                                                 if metric in METRICS[:2] else None),
                               "metric_role": "primary_local_metric" if metric in METRICS[:2] else "supporting_localization_not_in_primary_family",
                               "three_initialization_seed_deltas": seed_deltas,
                               "bootstrap_unit": "donor", "bootstrap_replicates": replicates,
                               "valid_paired_scenario_cells": len(aa), "matched_scenario_cells": expected,
                               "fixed_model_interval_not_full_pipeline_uncertainty": True}
        output[a + "_minus_" + b] = metrics
    return output


def aggregate_arms(rows):
    result = {}
    for arm in ("B0", "B1", "B2", "B3", "B4"):
        selected = [r for r in rows if r["arm"] == arm]
        result[arm] = {"cells": len(selected),
                       "not_converged_cells": sum(r["numerical_status"] not in {"CONVERGED", "UNAVAILABLE_DIFFERENT_DONOR"} for r in selected),
                       "wrong_bulk_unavailable_cells": sum(r["numerical_status"] == "UNAVAILABLE_DIFFERENT_DONOR" for r in selected),
                       "prediction_row_mass_diagnostics": {
                           "mean_of_scenario_mean_mass": mean([r.get("prediction_row_mass", {}).get("mean") for r in selected]),
                           "smallest_scenario_min_mass": min((r["prediction_row_mass"]["min"] for r in selected if "prediction_row_mass" in r), default=None),
                           "largest_scenario_max_mass": max((r["prediction_row_mass"]["max"] for r in selected if "prediction_row_mass" in r), default=None)},
                       "metrics": {m: {"donor_equal_mean": mean(list(donor_values(selected, m).values())),
                                       "donors": len(donor_values(selected, m))} for m in METRICS}}
    return result


def aggregate_by_assay(rows, measurement_ledger=None):
    """Anonymous descriptive strata, with no new selection or gate decision.

    The test assay mix is not a replicated study comparison. In particular, a
    singleton assay can have evaluable B0--B3 predictions but no admissible B4
    donor. Preserve that absence rather than importing a different-assay bulk.
    """
    result = {}
    for assay in sorted({r["assay"] for r in rows}):
        selected = [r for r in rows if r["assay"] == assay]
        wrong = [r for r in selected if r["arm"] == "B4"]
        unavailable = [r for r in wrong if r["numerical_status"] == "UNAVAILABLE_DIFFERENT_DONOR"]
        evaluable = [r for r in wrong if r["numerical_status"] == "CONVERGED"]
        result[assay] = {
            "role": "descriptive_assay_stratum_not_an_additional_primary_test_or_gate",
            "donors": len({r["donor_id_private"] for r in selected}),
            "sections": len({r["section_id_private"] for r in selected}),
            "studies": len({r["study_id_private"] for r in selected}),
            "arms": aggregate_arms(selected),
            "paired_local_contrasts": bootstrap_contrasts(selected),
            "B4_support": {
                "evaluable_donors": len({r["donor_id_private"] for r in evaluable}),
                "evaluable_sections": len({r["section_id_private"] for r in evaluable}),
                "evaluable_cells": len(evaluable),
                "unavailable_donors": len({r["donor_id_private"] for r in unavailable}),
                "unavailable_sections": len({r["section_id_private"] for r in unavailable}),
                "unavailable_cells": len(unavailable),
                "unavailable_reason": "NO_DIFFERENT_DONOR_IN_SAME_STUDY_ASSAY_PARTITION" if unavailable else None,
                "different_assay_bulk_substitution_allowed": False,
            },
            "interval_scope": "fixed_models_and_count_thinning_partition_donor_bootstrap_only",
            "primary_gate_recomputed_in_this_stratum": False,
        }
    if measurement_ledger is not None:
        for assay, support in measurement_eligibility_summary(measurement_ledger)["by_assay"].items():
            if assay not in result:
                result[assay] = {
                    "status": "INELIGIBLE_PARTIAL_MEASUREMENT",
                    "role": "not_evaluated_no_projection_or_wrong_bulk_control",
                    "donors": 0, "sections": 0, "arms": {}, "paired_local_contrasts": {},
                    "B4_support": {"status": "NOT_EVALUATED_INELIGIBLE_MEASUREMENT", "evaluable_cells": 0},
                    "primary_gate_recomputed_in_this_stratum": False,
                }
            result[assay]["measurement_eligibility"] = support
    return result


def source_runner(path: Path):
    spec = importlib.util.spec_from_file_location("phase22_frozen_spatial_runner_for_B", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load Stage A frozen inference adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("load_frozen_model", "infer_arrays"):
        if not callable(getattr(module, name, None)):
            raise ValueError("Stage A runner lacks frozen subset inference API: " + name)
    return module


def configuration(stage, selection, args):
    if stage == "validation":
        configs = [("B0", None, None), ("B1", None, None)]
        configs += [("B2", lam, 1.0) for lam in LAMBDAS]
        configs += [("B3", lam, rho) for lam in LAMBDAS for rho in RHOS]
    else:
        b2, b3 = selection["B2"], selection["B3"]
        configs = [("B0", None, None), ("B1", None, None),
                   ("B2", b2["lambda_b"], 1.0), ("B3", b3["lambda_b"], b3["rho"]),
                   ("B4", b3["lambda_b"], b3["rho"])]
    return configs


def choose(rows, arm):
    pairs = sorted({(r["lambda_b"], r["rho"]) for r in rows if r["arm"] == arm})
    candidates = []
    for lam, rho in pairs:
        selected = [r for r in rows if r["arm"] == arm and r["lambda_b"] == lam and r["rho"] == rho]
        good = bool(selected) and all(r["numerical_status"] == "CONVERGED" for r in selected)
        corr = mean(list(donor_values(selected, METRICS[0]).values())) if good else None
        error = mean(list(donor_values(selected, METRICS[1]).values())) if good else None
        candidates.append({"lambda_b": lam, "rho": rho, "eligible": corr is not None and error is not None,
                           "validation_donor_equal_Spearman": corr, "validation_donor_equal_centered_MSE": error,
                           "cells": len(selected), "not_converged_cells": sum(r["numerical_status"] != "CONVERGED" for r in selected)})
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        raise ValueError("NO_NUMERICALLY_QUALIFIED_VALIDATION_CONFIGURATION: " + arm)
    eligible.sort(key=lambda c: (-c["validation_donor_equal_Spearman"], c["validation_donor_equal_centered_MSE"],
                                 c["lambda_b"], c["rho"]))
    return eligible[0], candidates


def audit_gate(contrasts, rows):
    reasons = []
    for name, metrics in contrasts.items():
        for metric in METRICS[:2]:
            result = metrics[metric]
            if result.get("donors", 0) < 3:
                reasons.append(name + ": fewer_than_three_evaluable_paired_donors")
                continue
            lo, hi = result["six_primary_comparison_Bonferroni_interval"]
            seed_values = list(result["three_initialization_seed_deltas"].values())
            direction = 1 if metric == METRICS[0] else -1
            ci_pass = lo > 0 if direction == 1 else hi < 0
            seeds_pass = all(v is not None and direction * v > 0 for v in seed_values)
            if not ci_pass or not seeds_pass:
                reasons.append(name + ": " + metric + "_robust_positive_increment_not_demonstrated")
    if any(r["numerical_status"] not in {"CONVERGED", "UNAVAILABLE_DIFFERENT_DONOR"} for r in rows):
        reasons.append("some_selected_projection_cells_not_converged")
    return {"status": "PASS_EXTERNAL_LOCAL_MECHANISM_GATE" if not reasons else "NOT_SUPPORTED_UNDER_LOCKED_EXTERNAL_AUDIT",
            "reasons": sorted(set(reasons)), "bulk_mean_error_cannot_pass_gate": True,
            "wrong_bulk_compared_on_paired_available_support_only": True,
            "minimum_paired_donors_per_primary_comparison": 3,
            "primary_family": "three_local_contrasts_times_two_metrics",
            "stage_C_automatically_unlocked": False,
            "stage_C_still_requires": ["TCGA_assay_transform_qualification", "target_image_footprint_qualification", "root_protocol_review"],
            "synthetic_sampling_not_measured_human_overlap": True}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["validation", "test"])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split-lock", type=Path, required=True)
    p.add_argument("--data-qualification", type=Path, required=True)
    p.add_argument("--spatial-private-out", type=Path, required=True)
    p.add_argument("--spatial-runner", type=Path, default=ROOT / "scripts/run_phase22_external_spatial.py")
    p.add_argument("--private-out", type=Path, required=True)
    p.add_argument("--public-report-dir", type=Path, default=ROOT / "reports/phase22_spatial_bridge")
    p.add_argument("--device", default="cuda")
    p.add_argument("--neighbor-radius-um", type=float, default=300.0)
    p.add_argument("--neighbor-cap", type=int, default=16)
    p.add_argument("--count-split-seed", type=int, default=31)
    p.add_argument("--pseudocount", type=float, default=0.5)
    p.add_argument("--max-new-cells", type=int, default=None,
                   help="Bound this invocation only; all unfinished cells remain explicit in the plan. Resume unchanged inputs.")
    p.add_argument("--bootstrap-replicates", type=int, default=20000)
    return p.parse_args()


def main():
    args = parse_args()
    if args.bootstrap_replicates != 20000:
        raise ValueError("formal B audit locks 20,000 donor bootstrap replicates")
    if args.max_new_cells is not None and args.max_new_cells < 0:
        raise ValueError("max-new-cells must be nonnegative")
    args.private_out.mkdir(parents=True, exist_ok=True)
    args.public_report_dir.mkdir(parents=True, exist_ok=True)
    qualification = read_json(args.data_qualification)
    if qualification.get("schema_version") != "phase22_external_st_qualification_v1" or qualification.get("qualification_status") != "PASS_EXTERNAL_ST_DATA_CONTRACT":
        raise ValueError("BLOCKED_EXTERNAL_ST_DATA_QUALIFICATION")
    selection_path = args.spatial_private_out / "SELECTION_PRIVATE.json"
    upstream = read_json(selection_path)
    if upstream.get("status") != "SELECTED_ON_VALIDATION_ONLY" or upstream.get("A_gate") != "PASS_CANDIDATE":
        raise ValueError("BLOCKED_STAGE_A_SELECTION_OR_LOCAL_GATE")
    if tuple(upstream.get("selected_seeds", [])) != SEEDS:
        raise ValueError("Stage A must supply all three locked initialization seeds")
    split_sha = qualification["locked_split_sha"]
    if upstream.get("split_lock_sha256") != split_sha:
        raise ValueError("Stage A and qualification split hashes differ")
    dictionary_path = args.spatial_private_out / "dictionary/dictionary_private.npz"
    dictionary_receipt = args.spatial_private_out / "dictionary/DICT_RECEIPT_PRIVATE.json"
    dr = read_json(dictionary_receipt)
    dictionary_sha = sha(dictionary_path)
    if upstream.get("dictionary_npz_sha256") != dictionary_sha or dr.get("dictionary_npz_sha256") != dictionary_sha or upstream.get("dictionary_receipt_sha256") != sha(dictionary_receipt):
        raise ValueError("dictionary / selected Stage A artifact lineage mismatch")
    runner = source_runner(args.spatial_runner)
    qualified_D, qualified_genes, qualified_receipt, qualified_sha = runner.load_dictionary(args.spatial_private_out)
    if qualified_sha != dictionary_sha or qualified_receipt != dr:
        raise ValueError("Stage A train-only dictionary/error-scale qualification mismatch")
    with np.load(dictionary_path, allow_pickle=False) as n:
        D, genes = np.asarray(n["D"], dtype=np.float64), strings(n["gene_names"])
        u0 = np.asarray(n["u0"] if "u0" in n else dr["u0"], dtype=np.float64)
        omega = np.asarray(n["omega"], dtype=np.float64)
    if not np.array_equal(D, qualified_D) or genes != strings(qualified_genes):
        raise ValueError("B dictionary arrays differ from the qualified Stage A dictionary")
    if D.ndim != 2 or D.shape[1] != len(genes) or D.shape[0] != 32 or len(set(genes)) != len(genes):
        raise ValueError("invalid fixed molecular dictionary shape/vocabulary")
    if np.any(D < 0) or not np.isfinite(D).all() or not np.allclose(D.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("invalid dictionary simplex")
    if u0.shape != (len(D),) or not np.isfinite(u0).all() or np.any(u0 <= 0) or abs(u0.sum()-1) > 1e-12:
        raise ValueError("invalid train-only unobserved-tissue prior")
    if omega.shape != (len(genes),) or not np.isfinite(omega).all() or np.any(omega <= 0):
        raise ValueError("B requires positive finite prelocked train-only discrepancy weights")
    if dr.get("partition_role") not in {"ST-train", "train"} or dr.get("split_lock_sha256") != split_sha:
        raise ValueError("dictionary receipt does not bind train-only donor lock")
    if args.neighbor_radius_um != runner.SCHEDULE["radius_um"] or args.neighbor_cap != runner.SCHEDULE["neighbor_cap"]:
        raise ValueError("B physical sampling neighbourhood must match the locked Stage A schedule")
    arm = upstream["selected_arm"]
    role = "validation" if args.stage == "validation" else "test"
    source_sections, qualified_lock_sha, _donor_roles = runner.qualification_rows(
        args.data_root, args.split_lock, args.data_qualification, roles=(role,), with_features=True)
    if qualified_lock_sha != split_sha:
        raise ValueError("Stage B split file does not match qualified donor lock")
    sections, measurement_ledger = [], []
    for source in source_sections:
        counts, measured = runner.align_genes(source, genes)
        eligibility = measurement_eligibility_record(source.evidence, measured,
            stage=args.stage, dictionary_gene_count=len(genes))
        measurement_ledger.append(eligibility)
        if eligibility["status"] == "INELIGIBLE_PARTIAL_MEASUREMENT":
            continue  # No subset bulk normalization, imputation, or D refit.
        meta = read_json(args.data_root / source.evidence["metadata_path"])
        area = np.asarray(meta["capture_area_um2"], dtype=np.float64)
        if area.ndim > 1 or (area.ndim == 1 and len(area) != len(counts)) or np.any(~np.isfinite(area)) or np.any(area <= 0):
            raise ValueError("FAIL_DATA_CONTRACT: invalid expression-capture measurement area")
        sections.append({"entry": source.evidence, "counts": np.asarray(counts, dtype=np.float64),
                         "coords": source.coords_um, "barcodes": strings(source.barcodes),
                         "measured": np.asarray(measured, dtype=bool), "capture_area": area,
                         "features": source.features})
    if not sections:
        write_json(args.private_out / args.stage / "MEASUREMENT_ELIGIBILITY_PRIVATE.json", measurement_ledger)
        write_json(args.public_report_dir / ("B_" + args.stage.upper() + "_MEASUREMENT_ELIGIBILITY.json"),
                   measurement_eligibility_summary(measurement_ledger))
        raise ValueError("BLOCKED_NO_FULL_MEASUREMENT_B_SECTIONS_IN_LOCKED_ROLE")
    # Bind all feature/model artifacts before testing cache availability. A
    # cached completed row cannot hide a changed checkpoint or feature file.
    feature_cache, feature_evidence, model_shas = {}, {}, {}
    for section in sections:
        sid = section["entry"]["section_id_private"]
        feature_cache[sid] = (section["features"], section["entry"]["features_sha256"],
                              section["entry"]["feature_receipt_sha256"])
        feature_evidence[sid] = {"features_sha256": feature_cache[sid][1], "receipt_sha256": feature_cache[sid][2]}
    for seed in SEEDS:
        model_path = args.spatial_private_out / "fits" / arm / f"seed{seed:03d}" / "model.pt"
        done_path = model_path.with_name("DONE_PRIVATE.json")
        expected = upstream["models"][f"{arm}_seed{seed:03d}"]
        check_file(model_path, expected["model_sha256"], "Stage A selected model")
        check_file(done_path, expected["done_sha256"], "Stage A selected fit receipt")
        done = read_json(done_path)
        if done.get("status") != "COMPLETE" or done.get("model_sha256") != expected["model_sha256"]:
            raise ValueError("Stage A selected fit is incomplete")
        identity = done["identity"]
        if identity.get("split_lock_sha256") != split_sha or identity.get("dictionary_npz_sha256") != dictionary_sha or identity.get("data_qualification_sha256") != sha(args.data_qualification):
            raise ValueError("Stage A model does not bind this source/data/dictionary")
        if identity.get("source_hashes") != runner.source_hashes() or identity.get("source_hashes") != upstream.get("source_hashes"):
            raise ValueError("selected Stage A inference source lineage drift")
        model_shas[seed] = expected["model_sha256"]
    wrong = donor_assignment(sections)
    for section in sections:
        e, n = section["entry"], len(section["counts"])
        section["scenarios"] = simulate_partial_sampling(
            section["counts"], section["coords"], [e["section_id_private"]]*n, [e["donor_id_private"]]*n,
            study_ids=[e["study_id"]]*n, capture_area_um2=section["capture_area"],
            neighbor_radius_um=args.neighbor_radius_um, neighbor_cap=args.neighbor_cap,
            rho_simulation=.5, pseudocount=args.pseudocount, measured_mask=section["measured"],
            condition_probability=.5, random_seed=args.count_split_seed)
    bselection_path = args.private_out / "B_SELECTION_PRIVATE.json"
    bselection = read_json(bselection_path) if args.stage == "test" else None
    cfg = ProjectionConfig()  # 100 steps + objective/KKT gates, no adaptive expansion.
    invariant = {"schema": SCHEMA, "runner_sha256": sha(Path(__file__)),
                 "sampling_simulation_sha256": sha(ROOT / "src/phase22_spatial_bridge/sampling_simulation.py"),
                 "projection_source_sha256": sha(ROOT / "src/phase22_spatial_bridge/partial_measurement_projection.py"),
                 "qualification_sha256": sha(args.data_qualification), "split_lock_sha256": split_sha,
                 "stage_A_selection_sha256": sha(selection_path), "dictionary_file_sha256": dictionary_sha,
                 "dictionary_receipt_sha256": sha(dictionary_receipt), "spatial_runner_sha256": sha(args.spatial_runner),
                 "selected_arm": arm, "seeds": SEEDS, "fractions": LOCKED_AREA_FRACTIONS,
                 "lambdas": LAMBDAS, "rhos": RHOS, "projection": asdict(cfg),
                 "count_split_seed": args.count_split_seed, "condition_probability": .5,
                 "thinning_independence": "Poisson_model_only_not_conditional_on_fixed_observed_counts",
                 "neighbor_radius_um": args.neighbor_radius_um, "neighbor_cap": args.neighbor_cap,
                 "pseudocount": args.pseudocount, "measurement_weights": "declared_capture_area_geometric_only",
                 "omega": dr["omega_source"], "omega_protocol_sha256": dr["omega_calibration"]["protocol_sha256"],
                 "omega_array_sha256": dr["omega_calibration"]["omega_sha256"],
                 "omega_normalization": "1/(sqrt(G)*train_only_sigma)", "oracle_weights_enter_main": False,
                 "B1_row_normalization_after_scaling": False, "subset_context": "ONLY_SUPPLIED_INDICES_REBUILT_PHYSICAL_GRAPH"}
    invariant["local_metric_units"] = "evaluation_raw_count_divided_by_common_gene_evaluation_exposure"
    invariant["B1_output_units"] = "gene_rescaled_relative_abundance_profile_not_row_simplex_composition"
    invariant["rho_model_parameter_is_not_synthetic_area_fraction"] = True
    invariant["synthetic_bulk_unit"] = "one_connected_section_capture_component_not_one_observed_patient_bulk"
    invariant["partial_measurement_policy"] = PARTIAL_MEASUREMENT_POLICY
    # Validation and test have distinct source sections. The cross-stage lock
    # binds the method, upstream selection and all selected models; each stage
    # plan separately binds its own feature receipts and qualified section list.
    invariant["selected_model_sha256"] = model_shas
    invariant_sha = digest_json(invariant)
    if bselection is not None and (bselection.get("status") != "LOCKED_ON_VALIDATION_ONLY" or bselection.get("invariant_sha256") != invariant_sha):
        raise ValueError("test requires a matching completed validation selection; no test-based reselection")
    configs = configuration(args.stage, bselection, args)
    plan = []
    for seed in SEEDS:
        for section in sections:
            for scenario_index, scenario in enumerate(section["scenarios"]):
                for barm, lam, rho in configs:
                    plan.append(dict(seed=seed, section_id_private=section["entry"]["section_id_private"],
                                     scenario_index=scenario_index, arm=barm, lambda_b=lam, rho=rho))
    plan_sha = digest_json({"invariant_sha256": invariant_sha, "stage": args.stage,
                            "feature_evidence_private": feature_evidence,
                            "stage_measurement_eligibility_private": measurement_ledger, "cells": plan})
    stage_root = args.private_out / args.stage
    stage_root.mkdir(parents=True, exist_ok=True)
    plan_path = stage_root / "PLAN_PRIVATE.json"
    new_plan = {"status": "LOCKED_ALL_EXPECTED_CELLS", "invariant": invariant, "invariant_sha256": invariant_sha,
                "plan_sha256": plan_sha, "stage": args.stage, "expected_cells": len(plan),
                "feature_evidence_private": feature_evidence,
                "stage_measurement_eligibility_private": measurement_ledger, "cells_private": plan}
    if plan_path.exists() and read_json(plan_path).get("plan_sha256") != plan_sha:
        raise ValueError("existing B plan differs; use a new private output directory, never overwrite a previous experiment")
    if not plan_path.exists():
        write_json(plan_path, new_plan)
    write_json(stage_root / "MEASUREMENT_ELIGIBILITY_PRIVATE.json", measurement_ledger)
    eligibility_public = measurement_eligibility_summary(measurement_ledger)
    write_json(args.public_report_dir / ("B_" + args.stage.upper() + "_MEASUREMENT_ELIGIBILITY.json"), eligibility_public)
    grouped = {(s["entry"]["section_id_private"], i): (s, scenario)
               for s in sections for i, scenario in enumerate(s["scenarios"])}
    rows, actual_new = [], 0
    current_seed, model = None, None
    prior_cache = {}
    for cell in plan:
        cell_key = digest_json({"plan": plan_sha, "cell": cell})
        row_path = stage_root / "cells" / (cell_key + ".json")
        if row_path.exists():
            row = read_json(row_path)
            if row.get("cell_key") != cell_key or row.get("plan_sha256") != plan_sha:
                raise ValueError("resume row provenance mismatch")
            if "failure_state_npz_private" in row:
                check_file(Path(row["failure_state_npz_private"]), row["failure_state_sha256"], "retained nonconverged projection state")
            rows.append(row)
            continue
        if args.max_new_cells is not None and actual_new >= args.max_new_cells:
            break
        section, scenario = grouped[(cell["section_id_private"], cell["scenario_index"])]
        e = section["entry"]
        prior_key = (cell["seed"], cell["section_id_private"], cell["scenario_index"])
        if prior_key not in prior_cache:
            if current_seed != cell["seed"]:
                checkpoint = args.spatial_private_out / "fits" / arm / f"seed{cell['seed']:03d}" / "model.pt"
                done = read_json(checkpoint.with_name("DONE_PRIVATE.json"))
                checkpoint_sha = sha(checkpoint)
                expected = done.get("checkpoint_sha256", done.get("model_sha256"))
                if expected != checkpoint_sha:
                    raise ValueError("selected Stage A checkpoint hash mismatch")
                model, _modelmeta = runner.load_frozen_model(checkpoint, device=args.device)
                if not np.array_equal(array(_modelmeta["dictionary"]), D) or strings(_modelmeta["gene_names"]) != genes:
                    raise ValueError("Stage A checkpoint dictionary/vocabulary mismatch")
                current_seed = cell["seed"]
                model_shas[current_seed] = checkpoint_sha
                prior_cache.clear()  # bounded resident memory, predictions can be reconstructed.
            sid = cell["section_id_private"]
            features, feature_sha, receipt_sha = feature_cache[sid]
            prediction = runner.infer_arrays(model, features, section["coords"], sid,
                                              indices=scenario.image_position_indices, device=args.device)
            if prediction.get("context_scope") != "ONLY_SUPPLIED_INDICES_REBUILT_PHYSICAL_GRAPH":
                raise ValueError("BLOCKED_SUBDOMAIN_CONTEXT: whole-section pi0 cannot substitute for visible-subset inference")
            pi0, roundoff = canonical_prior(prediction["pi0"])
            if pi0.shape != (len(scenario.image_position_indices), len(D)):
                raise ValueError("frozen prior subset row/state mismatch")
            prior_cache[prior_key] = (pi0, pi0 @ D, LocalEvaluator(scenario.evaluation_image_counts,
                                       scenario.evaluation_measured_mask), roundoff, feature_sha, receipt_sha)
        pi0, mu0, evaluator, roundoff, feature_sha, receipt_sha = prior_cache[prior_key]
        sampling_diagnostics = scenario.aggregate_diagnostics()
        sampling_diagnostics.pop("rho_simulation")
        sampling_diagnostics["sampling_geometry_does_not_use_rho"] = True
        row = {**cell, "cell_key": cell_key, "plan_sha256": plan_sha,
               "donor_id_private": e["donor_id_private"], "study_id_private": e["study_id"], "assay": e["assay"],
               "component_index": scenario.component_index, "area_fraction": scenario.requested_capture_area_fraction,
               "sampling": sampling_diagnostics, "source_count_npz_sha256": e["npz_sha256"],
               "feature_npz_sha256": feature_sha, "feature_receipt_sha256": receipt_sha,
               "source_model_sha256": model_shas[cell["seed"]], "prior_row_sum_roundoff_normalized_max": roundoff,
               "image_context": "ONLY_SUPPLIED_INDICES_REBUILT_PHYSICAL_GRAPH",
               "measurement_weights_kind": "geometric_declared_capture_area", "oracle_weights_used": False}
        bulk, measured = scenario.condition_bulk.composition, scenario.condition_bulk.measured_gene_mask
        if not measured.all():
            raise ValueError("FAIL_DATA_CONTRACT: partial-bulk normalization not interchangeable with full dictionary scale")
        if cell["arm"] == "B0":
            mu, row["numerical_status"] = mu0, "CONVERGED"
            row["solver"] = {"status": "NO_BULK_IDENTITY", "iterations": 0}
        elif cell["arm"] == "B1":
            mu, check = gene_rescale(mu0, bulk, scenario.geometry_measurement_weights, measured)
            row["numerical_status"], row["B1_control"] = "CONVERGED", check
            row["solver"] = {"status": "CLOSED_FORM_GENE_SCALING", "iterations": 0}
        else:
            if cell["arm"] == "B4":
                donor = wrong[e["section_id_private"]]
                if donor is None:
                    row.update(numerical_status="UNAVAILABLE_DIFFERENT_DONOR", metrics={},
                               wrong_bulk_status="NO_DIFFERENT_DONOR_IN_SAME_STUDY_ASSAY_PARTITION")
                    write_json(row_path, row)
                    rows.append(row)
                    actual_new += 1
                    continue
                choices = [s for s in sections if s["entry"]["donor_id_private"] == donor and
                           s["entry"]["study_id"] == e["study_id"] and s["entry"]["assay"] == e["assay"]]
                choices.sort(key=lambda s: digest_json(["wrong_section_v1", e["section_id_private"], s["entry"]["section_id_private"]]))
                receiver_mass = scenario.bulk_capture_area_um2
                donor_scenarios = [sc for sc in choices[0]["scenarios"] if sc.requested_capture_area_fraction == 1.0]
                donor_scenarios.sort(key=lambda sc: (abs(sc.bulk_capture_area_um2-receiver_mass), sc.component_index))
                wrong_scenario = donor_scenarios[0]
                bulk, measured = wrong_scenario.condition_bulk.composition, wrong_scenario.condition_bulk.measured_gene_mask
                row.update(wrong_donor_id_private=donor, wrong_section_id_private=choices[0]["entry"]["section_id_private"],
                           wrong_bulk_status="DIFFERENT_DONOR_SAME_STUDY_ASSAY_PARTITION_CONDITION_COUNTS_ONLY")
            solved = solve_partial_measurement_projection(pi0, D, bulk, u0=u0,
                       measurement_weights=scenario.geometry_measurement_weights, measured_gene_mask=measured,
                       omega=omega, rho=cell["rho"], lambda_b=cell["lambda_b"], tau=1.0, tau_u=1.0, config=cfg)
            mu = solved.pi @ D
            row["numerical_status"], row["solver"] = solved.status, solved.aggregate_diagnostics()
            # Keep final Pi/u and all objective/KKT histories for every numerical
            # failure, without expanding its fixed iteration budget or dropping it.
            if not solved.converged:
                state_path = stage_root / "failed_states" / (cell_key + ".npz")
                state_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(state_path, pi=solved.pi, u=solved.u, pi0=pi0, bulk=bulk,
                                    image_position_indices=scenario.image_position_indices,
                                    objective_history=solved.objective_history, kkt_history=solved.kkt_history,
                                    step_size_history=solved.step_size_history)
                row["failure_state_npz_private"], row["failure_state_sha256"] = str(state_path), sha(state_path)
        row["metrics"] = evaluator.evaluate(mu)
        mass = mu.sum(1)
        row["prediction_output_contract"] = ("gene_rescaled_profile_not_a_row_simplex" if cell["arm"] == "B1"
                                             else "fixed_dictionary_state_mixture_row_composition")
        row["prediction_row_mass"] = {"min": float(mass.min()), "mean": float(mass.mean()),
                                       "max": float(mass.max()), "row_normalized_for_metrics": False}
        row["metric_units"] = invariant["local_metric_units"]
        mean_bulk = scenario.geometry_measurement_weights @ mu
        row["bulk_mean_MSE_auxiliary_only"] = float(np.mean(np.square(mean_bulk[measured] - bulk[measured])))
        write_json(row_path, row)
        rows.append(row)
        actual_new += 1
    complete = len(rows) == len(plan)
    status = "COMPLETE" if complete else "PARTIAL_CELL_BUDGET_REACHED"
    progress = {"schema": SCHEMA, "status": status, "stage": args.stage, "plan_sha256": plan_sha,
                "expected_cells": len(plan), "completed_cells": len(rows), "pending_cells": len(plan)-len(rows),
                "actual_new_cells_this_invocation": actual_new, "formal_model_fits": 0,
                "projection_solver_calls": sum(r["arm"] in {"B2", "B3", "B4"} and r["numerical_status"] != "UNAVAILABLE_DIFFERENT_DONOR" for r in rows),
                "projection_solver_iterations": sum(r.get("solver", {}).get("iterations", 0) for r in rows),
                "solver_max_iterations": 100, "all_failures_retained": True,
                "not_converged_cells": sum(r["numerical_status"] not in {"CONVERGED", "UNAVAILABLE_DIFFERENT_DONOR"} for r in rows),
                "source_measurement_eligibility": eligibility_public,
                "serial_section_real_overlap_evaluated": False, "synthetic_same_view_and_subdomain_only": True}
    write_json(stage_root / "PROGRESS_PRIVATE.json", progress)
    write_json(args.public_report_dir / ("B_" + args.stage.upper() + "_PROGRESS.json"), progress)
    if not complete:
        print(json.dumps(progress))
        return 2
    if args.stage == "validation":
        b2, candidates_b2 = choose(rows, "B2")
        b3, candidates_b3 = choose(rows, "B3")
        lock = {"status": "LOCKED_ON_VALIDATION_ONLY", "schema": SCHEMA,
                "invariant_sha256": invariant_sha, "validation_plan_sha256": plan_sha,
                "validation_cells_sha256": digest_json(rows), "B2": b2, "B3": b3,
                "selection_rule": "max_donor_equal_within_tissue_gene_Spearman_then_min_centered_MSE_then_lower_lambda_then_lower_rho",
                "B2_candidates": candidates_b2, "B3_candidates": candidates_b3,
                "test_predictions_or_test_expression_used_for_selection": False,
                "all_expected_validation_cells_completed_including_failures": True}
        if bselection_path.exists() and read_json(bselection_path) != clean(lock):
            raise ValueError("existing validation selection differs; never overwrite a locked selection")
        write_json(bselection_path, lock)
        write_json(args.public_report_dir / "B_SELECTION_LOCK.json", lock)
        print(json.dumps({"status": lock["status"], "B2": b2, "B3": b3}))
    else:
        contrasts = bootstrap_contrasts(rows)
        summary = {"status": "COMPLETE", "schema": SCHEMA, "progress": progress,
                   "validation_selection_sha256": sha(bselection_path), "arms": aggregate_arms(rows),
                   "paired_local_contrasts": contrasts,
                   "descriptive_assay_strata": aggregate_by_assay(rows, measurement_ledger),
                   "source_measurement_eligibility": eligibility_public,
                   "limitations": ["two_source_studies_not_three_study_confirmation", "only_three_test_donors",
                      "test_assay_domain_shift", "frozen_backbone_pretraining_overlap_unknown",
                      "synthetic_overlap_not_measured_human_overlap", "no_real_serial_section_registration_evaluated",
                      "one_pseudobulk_per_synthetic_section_component_not_real_patient_multislide_bulk",
                      "no_TCGAbulk_scale_qualification_implied",
                      "partial_measurement_test_sections_excluded_from_B_without_refitting_dictionary_or_imputation",
                      "B_gate_does_not_generalize_to_ineligible_assays"],
                   "C_gate": audit_gate(contrasts, rows)}
        summary["C_gate"]["evaluated_assays"] = eligibility_public["eligible_assays"]
        summary["C_gate"]["not_evaluated_assays"] = eligibility_public["fully_ineligible_assays"]
        summary["C_gate"]["generalization_to_ineligible_assays"] = False
        summary["output_contracts"] = {
            "B0_B2_B3_B4": "fixed_dictionary_row_compositions_compared_to_evaluation_count_div_exposure",
            "B1": "gene_rescaled_non_simplex_relative_abundance_profile_no_row_renormalization",
            "B1_rank_invariance": "exact_average_rank_equality_checked_per_gene_per_scenario",
            "centered_MSE": "within_region_gene_centered_relative_abundances_same_evaluation_exposure_target",
            "bulk_mean_MSE": "auxiliary_only_not_selection_or_mechanism_gate"}
        write_json(args.public_report_dir / "B_TEST_SUMMARY.json", summary)
        write_json(args.private_out / "B_TEST_DONE_PRIVATE.json", dict(summary, invariant=invariant))
        print(json.dumps({"status": "COMPLETE", "C_gate": summary["C_gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
