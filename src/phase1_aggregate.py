from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_utils import censoring_km, harrell_c, uno_c


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs/pathology_context_phase1"
SEED = 20260723


META = {
    "B0": (False, "none", False, "ridge_cox"),
    "B1": (False, "none", False, "stochastic_cox"),
    "B2": (False, "none", True, "stochastic_cox"),
    "P0": (True, "spatial_block_0.40", True, "ridge_cox_probe"),
    "P1": (True, "spatial_block_0.40", True, "stochastic_cox"),
    "P2": (True, "spatial_block_0.40", True, "stochastic_cox+rank"),
    "P3": (True, "spatial_block_0.40", True, "stochastic_cox+ipcw_rank"),
    "P0_random040": (True, "random_0.40", True, "ridge_cox_probe"),
    "P0_spatial025": (True, "spatial_block_0.25", True, "ridge_cox_probe"),
    "P0_nocoords": (True, "spatial_block_0.40", False, "ridge_cox_probe"),
    "P2_permuted": (True, "spatial_block_0.40", True, "stochastic_cox+rank_permuted_labels"),
    "B1_probe": (False, "none", False, "ridge_cox_probe"),
    "B2_probe": (False, "none", True, "ridge_cox_probe"),
    "P1_probe": (True, "spatial_block_0.40", True, "ridge_cox_probe"),
    "P2_probe": (True, "spatial_block_0.40", True, "ridge_cox_probe"),
}


def consolidate_parts() -> list[tuple[str, int, Path]]:
    records = []
    base = PHASE / "oof_predictions/fold_parts"
    for method_dir in sorted(base.glob("*")):
        if not method_dir.is_dir():
            continue
        for seed in sorted({int(p.stem.split("_seed_")[-1]) for p in method_dir.glob("fold_*_seed_*.csv")}):
            parts = sorted(method_dir.glob(f"fold_*_seed_{seed}.csv"))
            if len(parts) != 5:
                continue
            frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
            if len(frame) != 487 or frame.patient_id.nunique() != 487:
                raise RuntimeError(f"{method_dir.name} seed {seed}: invalid OOF cardinality {len(frame)}")
            frame = frame.sort_values("patient_id").reset_index(drop=True)
            path = PHASE / "oof_predictions" / f"{method_dir.name}_seed_{seed}.csv"
            frame.to_csv(path, index=False)
            records.append((method_dir.name, seed, path))
    b0 = PHASE / "oof_predictions" / f"B0_seed_{SEED}.csv"
    if b0.exists():
        records.append(("B0", SEED, b0))
    return records


def ipcw_auc(frame: pd.DataFrame, horizon: float, manifest: pd.DataFrame) -> float:
    case_scores, case_weights, control_scores, control_weights = [], [], [], []
    for fold in range(1, 6):
        test = frame.loc[frame.outer_fold == fold]
        train = manifest.loc[manifest.outer_fold != fold]
        km = censoring_km(train.dss_months.to_numpy(float), train.dss_status.to_numpy(int))
        cases = test.loc[(test.dss_status == 1) & (test.dss_months <= horizon)]
        controls = test.loc[test.dss_months > horizon]
        if len(cases):
            g = km(cases.dss_months.to_numpy(float), left_limit=True)
            valid = g > 1e-8
            case_scores.extend(cases.oof_risk.to_numpy(float)[valid])
            case_weights.extend((1.0 / g[valid]).tolist())
        if len(controls):
            gh = float(np.asarray(km(np.asarray([horizon]), left_limit=False))[0])
            if gh > 1e-8:
                control_scores.extend(controls.oof_risk.to_numpy(float))
                control_weights.extend(np.full(len(controls), 1.0 / gh).tolist())
    if len(case_scores) < 5 or len(control_scores) < 10:
        return float("nan")
    cs = np.asarray(case_scores)[:, None]
    ns = np.asarray(control_scores)[None, :]
    weights = np.asarray(case_weights)[:, None] * np.asarray(control_weights)[None, :]
    wins = (cs > ns).astype(float) + 0.5 * (cs == ns)
    return float(np.sum(weights * wins) / np.sum(weights))


def metrics(frame: pd.DataFrame) -> tuple[float, float]:
    return (
        harrell_c(frame.dss_months, frame.dss_status, frame.oof_risk),
        uno_c(frame.dss_months, frame.dss_status, frame.oof_risk, frame.censor_survival_outer_train),
    )


def bootstrap(frame: pd.DataFrame, baseline: pd.DataFrame, n: int = 1000):
    joined = frame.merge(
        baseline[["patient_id", "oof_risk"]].rename(columns={"oof_risk": "b0_risk"}),
        on="patient_id",
        validate="one_to_one",
    )
    rng = np.random.default_rng(20260723)
    h, u, dh, du = [], [], [], []
    for _ in range(n):
        sample = joined.iloc[rng.integers(0, len(joined), len(joined))].reset_index(drop=True)
        hm = harrell_c(sample.dss_months, sample.dss_status, sample.oof_risk)
        um = uno_c(sample.dss_months, sample.dss_status, sample.oof_risk, sample.censor_survival_outer_train)
        hb = harrell_c(sample.dss_months, sample.dss_status, sample.b0_risk)
        ub = uno_c(sample.dss_months, sample.dss_status, sample.b0_risk, sample.censor_survival_outer_train)
        h.append(hm); u.append(um); dh.append(hm - hb); du.append(um - ub)
    q = lambda x: np.nanpercentile(np.asarray(x), [2.5, 97.5]).tolist()
    return q(h), q(u), q(dh), q(du)


def resource_summary(model: str, seed: int):
    values = []
    for p in (PHASE / "fold_metrics" / model).glob(f"fold_*_seed_{seed}.json"):
        values.append(json.loads(p.read_text(encoding="utf-8")))
    if not values:
        return np.nan, np.nan, np.nan
    params = np.asarray([v.get("n_parameters", v.get("n_parameters_encoder", np.nan)) for v in values], dtype=float)
    times = np.asarray([v.get("training_time_seconds", np.nan) for v in values], dtype=float)
    memories = np.asarray([v.get("peak_gpu_memory_bytes", np.nan) for v in values], dtype=float)
    finite_params = params[np.isfinite(params)]
    finite_memory = memories[np.isfinite(memories)]
    return (
        float(np.max(finite_params)) if finite_params.size else np.nan,
        float(np.nansum(times)) if np.isfinite(times).any() else np.nan,
        float(np.max(finite_memory)) if finite_memory.size else np.nan,
    )


def main():
    records = consolidate_parts()
    manifest = pd.read_csv(PHASE / "audit/patient_manifest_used.csv")
    baseline = pd.read_csv(PHASE / f"oof_predictions/B0_seed_{SEED}.csv").sort_values("patient_id")
    b0_fold = {fold: metrics(baseline.loc[baseline.outer_fold == fold])[0] for fold in range(1, 6)}
    b0_h, b0_u = metrics(baseline)
    rows = []
    stat_dir = PHASE / "statistical_tests"
    stat_dir.mkdir(parents=True, exist_ok=True)
    for model, seed, path in sorted(set(records)):
        frame = pd.read_csv(path).sort_values("patient_id").reset_index(drop=True)
        h, u = metrics(frame)
        h_ci, u_ci, dh_ci, du_ci = bootstrap(frame, baseline)
        fold_h = [metrics(frame.loc[frame.outer_fold == fold])[0] for fold in range(1, 6)]
        params, seconds, memory = resource_summary(model, seed)
        ssl, mask, coords, loss = META.get(model, (model.startswith("P"), "derived", True, "ridge_cox_probe" if model.endswith("_probe") else "unknown"))
        row = {
            "model": model, "ssl_pretrained": ssl, "mask_type": mask, "use_coords": coords,
            "survival_loss": loss, "seed": seed, "n_patients": len(frame),
            "n_events": int(frame.dss_status.sum()), "oof_harrell_c": h, "oof_uno_c": u,
            "auc_12m": ipcw_auc(frame, 12.0, manifest), "auc_24m": ipcw_auc(frame, 24.0, manifest),
            "auc_36m": ipcw_auc(frame, 36.0, manifest), "ibs": np.nan,
            "ibs_na_reason": "Fold-wise calibrated survival functions were not retained; risk-only Cox discrimination cannot identify IBS.",
            "delta_harrell_vs_B0": h - b0_h, "delta_uno_vs_B0": u - b0_u,
            "bootstrap_ci_low_harrell": dh_ci[0], "bootstrap_ci_high_harrell": dh_ci[1],
            "bootstrap_ci_low_uno": du_ci[0], "bootstrap_ci_high_uno": du_ci[1],
            "metric_ci_low_harrell": h_ci[0], "metric_ci_high_harrell": h_ci[1],
            "metric_ci_low_uno": u_ci[0], "metric_ci_high_uno": u_ci[1],
            "fold_harrell_mean": float(np.nanmean(fold_h)), "fold_harrell_std": float(np.nanstd(fold_h, ddof=1)),
            "n_folds_better_than_B0": int(sum(x > b0_fold[i + 1] for i, x in enumerate(fold_h))),
            "parameter_count": params, "training_time": seconds, "peak_gpu_memory": memory,
        }
        rows.append(row)
        pd.DataFrame({"bootstrap_harrell_ci": h_ci, "bootstrap_uno_ci": u_ci,
                      "paired_delta_harrell_ci": dh_ci, "paired_delta_uno_ci": du_ci},
                     index=["low", "high"]).to_csv(stat_dir / f"{model}_seed_{seed}_bootstrap_1000.csv")
    result = pd.DataFrame(rows).sort_values(["seed", "oof_harrell_c"], ascending=[True, False])
    result.to_csv(PHASE / "results_master.csv", index=False)
    print(result[["model", "seed", "oof_harrell_c", "oof_uno_c", "delta_harrell_vs_B0", "delta_uno_vs_B0"]].to_string(index=False))


if __name__ == "__main__":
    main()
