from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

import phase1_neural
from phase1_b0 import compute_patient_means, tune
from phase1_utils import censoring_km, fit_cox, harrell_c, scaler_pca, uno_c


METHOD_FILES = {
    "B1_ABMIL": "B1",
    "P0_nocoords": "P0_nocoords",
    "P0_coords": "P0",
}


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping")
    return payload


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def assign_confirmatory_folds(patients: pd.DataFrame, n_folds: int, outer_seed: int) -> pd.DataFrame:
    frame = patients.drop(columns=["outer_fold"], errors="ignore").copy()
    events = pd.to_numeric(frame["dss_status"], errors="raise").to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=outer_seed)
    folds = np.zeros(len(frame), dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(events)), events), start=1):
        folds[test_idx] = fold
    if not np.all(folds > 0):
        raise RuntimeError("Confirmatory outer split assignment is incomplete")
    frame["outer_fold"] = folds
    return frame


def absolute_data_path(data_root: Path, value: object) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        path = data_root / path
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Referenced local Phase P1 artifact is missing: {resolved}")
    return str(resolved)


def prepare_runtime_manifests(
    repo_root: Path,
    data_root: Path,
    n_folds: int,
    outer_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_audit = data_root / "outputs" / "pathology_context_phase1" / "audit"
    patient_path = require_file(source_audit / "patient_manifest_used.csv", "Phase P1 patient manifest")
    slide_path = require_file(source_audit / "slide_manifest_used.csv", "Phase P1 slide manifest")
    patients = pd.read_csv(patient_path)
    slides = pd.read_csv(slide_path)
    required_patient = {"patient_id", "dss_status", "dss_months"}
    required_slide = {"patient_id", "slide_id", "bag_path", "coords_path", "n_patches"}
    if not required_patient.issubset(patients.columns):
        raise ValueError(f"Patient manifest missing columns: {sorted(required_patient - set(patients.columns))}")
    if not required_slide.issubset(slides.columns):
        raise ValueError(f"Slide manifest missing columns: {sorted(required_slide - set(slides.columns))}")
    patients["patient_id"] = patients["patient_id"].astype(str)
    slides["patient_id"] = slides["patient_id"].astype(str)
    patients["dss_status"] = pd.to_numeric(patients["dss_status"], errors="raise").astype(int)
    patients["dss_months"] = pd.to_numeric(patients["dss_months"], errors="raise").astype(float)
    if patients["patient_id"].duplicated().any():
        raise ValueError("Duplicated patients in Phase P1 patient manifest")
    if set(patients["patient_id"]) != set(slides["patient_id"]):
        raise ValueError("Patient and slide manifests contain different patient sets")
    if not patients["dss_status"].isin([0, 1]).all():
        raise ValueError("DSS status must be binary")
    if not np.isfinite(patients["dss_months"]).all() or not (patients["dss_months"] > 0).all():
        raise ValueError("DSS duration must be finite and positive")

    patients = assign_confirmatory_folds(
        patients.sort_values("patient_id").reset_index(drop=True), n_folds, outer_seed
    )
    slides = slides.reset_index(drop=True).copy()
    slides["bag_path"] = [absolute_data_path(data_root, value) for value in slides["bag_path"]]
    slides["coords_path"] = [absolute_data_path(data_root, value) for value in slides["coords_path"]]

    runtime_audit = repo_root / "outputs" / "pathology_context_phase1" / "audit"
    runtime_audit.mkdir(parents=True, exist_ok=True)
    patients.to_csv(runtime_audit / "patient_manifest_used.csv", index=False)
    slides.to_csv(runtime_audit / "slide_manifest_used.csv", index=False)
    run_state = {
        "stage": "phase1r_confirmatory",
        "status": "RUNNING",
        "completed_jobs": [],
        "pending_jobs": [],
        "updated_at": None,
    }
    (repo_root / "outputs" / "pathology_context_phase1" / "run_state.json").write_text(
        json.dumps(run_state, indent=2), encoding="utf-8"
    )
    return patients, slides


def evaluate_b0(
    repo_root: Path,
    patients: pd.DataFrame,
    slides: pd.DataFrame,
    model_seed: int,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    features = compute_patient_means(repo_root, patients, slides)
    times = patients["dss_months"].to_numpy(dtype=float)
    events = patients["dss_status"].to_numpy(dtype=int)
    fold_ids = patients["outer_fold"].to_numpy(dtype=int)
    risk = np.full(len(patients), np.nan, dtype=float)
    g_values = np.full(len(patients), np.nan, dtype=float)
    rows = []
    for fold in range(1, n_folds + 1):
        test = fold_ids == fold
        train = ~test
        n_components, alpha, inner_c = tune(
            features[train], times[train], events[train], model_seed + fold
        )
        x_train, x_test, _, pca = scaler_pca(
            features[train], features[test], n_components, model_seed + fold
        )
        beta = fit_cox(times[train], events[train], x_train, alpha)
        risk[test] = x_test @ beta
        censor = censoring_km(times[train], events[train])
        g_values[test] = censor(times[test], left_limit=True)
        rows.append(
            {
                "method": "B0_mean",
                "fold": fold,
                "events_test": int(events[test].sum()),
                "pca_components": int(n_components),
                "alpha": float(alpha),
                "inner_harrell_c": float(inner_c),
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "test_harrell_c": float(harrell_c(times[test], events[test], risk[test])),
                "test_uno_c": float(uno_c(times[test], events[test], risk[test], g_values[test])),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    return risk, g_values, rows


def run_neural_confirmatory(repo_root: Path, model_seed: int, n_folds: int) -> None:
    phase1_neural.SEED = model_seed
    for fold in range(1, n_folds + 1):
        print(f"\n=== neural confirmatory fold {fold}/{n_folds} ===", flush=True)
        phase1_neural.train_survival(repo_root, "B1", fold, model_seed)
        phase1_neural.train_ssl(repo_root, "spatial040_nocoords", fold, model_seed)
        phase1_neural.run_probe(repo_root, "spatial040_nocoords", fold, model_seed)
        phase1_neural.train_ssl(repo_root, "spatial040", fold, model_seed)
        phase1_neural.run_probe(repo_root, "spatial040", fold, model_seed)


def load_neural_oof(
    repo_root: Path,
    patients: pd.DataFrame,
    method_label: str,
    phase1_method: str,
    model_seed: int,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    phase = repo_root / "outputs" / "pathology_context_phase1"
    risk = np.full(len(patients), np.nan, dtype=float)
    g_values = np.full(len(patients), np.nan, dtype=float)
    patient_lookup = {
        pid: index for index, pid in enumerate(patients["patient_id"].astype(str))
    }
    rows = []
    for fold in range(1, n_folds + 1):
        part_path = (
            phase
            / "oof_predictions"
            / "fold_parts"
            / phase1_method
            / f"fold_{fold}_seed_{model_seed}.csv"
        )
        metric_path = (
            phase
            / "fold_metrics"
            / phase1_method
            / f"fold_{fold}_seed_{model_seed}.json"
        )
        part = pd.read_csv(require_file(part_path, f"{method_label} fold prediction"))
        metric = json.loads(
            require_file(metric_path, f"{method_label} fold metrics").read_text(encoding="utf-8")
        )
        for record in part.to_dict("records"):
            index = patient_lookup[str(record["patient_id"])]
            risk[index] = float(record["oof_risk"])
            g_values[index] = float(record["censor_survival_outer_train"])
        rows.append(
            {
                "method": method_label,
                "fold": fold,
                "events_test": int(part["dss_status"].sum()),
                "test_harrell_c": float(metric["test_harrell_c"]),
                "test_uno_c": float(metric["test_uno_c"]),
                "inner_harrell_c": float(
                    metric.get("inner_harrell_c", metric.get("inner_val_harrell_c", np.nan))
                ),
                "best_lr": metric.get("best_lr", ""),
                "best_epoch": metric.get("best_epoch", ""),
                "alpha": metric.get("alpha", ""),
                "best_ssl_val_loss": metric.get("best_ssl_val_loss", ""),
            }
        )
    if not np.isfinite(risk).all() or not np.isfinite(g_values).all():
        raise RuntimeError(f"Incomplete OOF output for {method_label}")
    return risk, g_values, rows


def paired_bootstrap(
    times: np.ndarray,
    events: np.ndarray,
    baseline_risk: np.ndarray,
    method_risk: np.ndarray,
    g_values: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    delta_h = []
    delta_u = []
    n = len(times)
    for _ in range(replicates):
        indices = rng.integers(0, n, size=n)
        h0 = harrell_c(times[indices], events[indices], baseline_risk[indices])
        h1 = harrell_c(times[indices], events[indices], method_risk[indices])
        u0 = uno_c(times[indices], events[indices], baseline_risk[indices], g_values[indices])
        u1 = uno_c(times[indices], events[indices], method_risk[indices], g_values[indices])
        if np.isfinite(h0) and np.isfinite(h1):
            delta_h.append(h1 - h0)
        if np.isfinite(u0) and np.isfinite(u1):
            delta_u.append(u1 - u0)
    return {
        "delta_harrell_ci95": [float(x) for x in np.quantile(delta_h, [0.025, 0.975])],
        "delta_uno_ci95": [float(x) for x in np.quantile(delta_u, [0.025, 0.975])],
        "valid_harrell_bootstraps": len(delta_h),
        "valid_uno_bootstraps": len(delta_u),
    }


def summarize(
    patients: pd.DataFrame,
    risks: dict[str, np.ndarray],
    g_values: np.ndarray,
    fold_rows: list[dict],
    bootstrap_replicates: int,
    outer_seed: int,
    min_positive_folds: int,
) -> tuple[dict, dict]:
    times = patients["dss_months"].to_numpy(dtype=float)
    events = patients["dss_status"].to_numpy(dtype=int)
    fold_frame = pd.DataFrame(fold_rows)
    methods = ["B0_mean", "B1_ABMIL", "P0_nocoords", "P0_coords"]
    results: dict[str, dict] = {}
    for method in methods:
        method_rows = fold_frame.loc[fold_frame["method"] == method]
        results[method] = {
            "oof_harrell_c": float(harrell_c(times, events, risks[method])),
            "oof_uno_c": float(uno_c(times, events, risks[method], g_values)),
            "fold_mean_harrell_c": float(method_rows["test_harrell_c"].mean()),
            "fold_std_harrell_c": float(method_rows["test_harrell_c"].std(ddof=1)),
            "fold_mean_uno_c": float(method_rows["test_uno_c"].mean()),
            "fold_std_uno_c": float(method_rows["test_uno_c"].std(ddof=1)),
        }
    baseline = results["B0_mean"]
    baseline_rows = fold_frame.loc[fold_frame["method"] == "B0_mean"].set_index("fold")
    for offset, method in enumerate(methods[1:], start=1):
        method_rows = fold_frame.loc[fold_frame["method"] == method].set_index("fold")
        results[method]["delta_harrell_vs_b0"] = (
            results[method]["oof_harrell_c"] - baseline["oof_harrell_c"]
        )
        results[method]["delta_uno_vs_b0"] = (
            results[method]["oof_uno_c"] - baseline["oof_uno_c"]
        )
        results[method]["positive_harrell_folds_vs_b0"] = int(
            (method_rows["test_harrell_c"] > baseline_rows["test_harrell_c"]).sum()
        )
        results[method]["positive_uno_folds_vs_b0"] = int(
            (method_rows["test_uno_c"] > baseline_rows["test_uno_c"]).sum()
        )
        results[method].update(
            paired_bootstrap(
                times,
                events,
                risks["B0_mean"],
                risks[method],
                g_values,
                bootstrap_replicates,
                outer_seed + offset * 101,
            )
        )

    def passes(method: str) -> bool:
        item = results[method]
        return bool(
            item["delta_harrell_vs_b0"] > 0
            and item["delta_uno_vs_b0"] > 0
            and item["positive_harrell_folds_vs_b0"] >= min_positive_folds
            and item["positive_uno_folds_vs_b0"] >= min_positive_folds
        )

    if passes("P0_nocoords"):
        selected = "P0_nocoords_frozen_probe"
        decision_code = "PASS_P0_NOCOORDS"
    elif passes("B1_ABMIL"):
        selected = "B1_ABMIL"
        decision_code = "FALLBACK_B1"
    else:
        selected = "B0_mean"
        decision_code = "FALLBACK_B0"
    decision = {
        "decision": decision_code,
        "selected_pathology_branch_for_phase2": selected,
        "min_positive_folds_per_metric": min_positive_folds,
        "p0_nocoords_beats_p0_coords_on_both_oof_metrics": bool(
            results["P0_nocoords"]["oof_harrell_c"] > results["P0_coords"]["oof_harrell_c"]
            and results["P0_nocoords"]["oof_uno_c"] > results["P0_coords"]["oof_uno_c"]
        ),
        "scope_note": (
            "Split-replication on the same TCGA cohort; not independent external validation."
        ),
    }
    return results, decision


def write_metrics(run_dir: Path, results: dict[str, dict]) -> None:
    lines = ["metric,value"]
    for method, values in results.items():
        lines.append(f"{method}.oof_harrell_c,{values['oof_harrell_c']}")
        lines.append(f"{method}.oof_uno_c,{values['oof_uno_c']}")
    (run_dir / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cleanup_runtime_outputs(repo_root: Path) -> None:
    phase = repo_root / "outputs" / "pathology_context_phase1"
    for relative in [
        "checkpoints",
        "cache",
        "embeddings",
        "fold_metrics",
        "oof_predictions",
        "run_state.json",
    ]:
        path = phase / relative
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)
    for name in ["patient_manifest_used.csv", "slide_manifest_used.csv"]:
        (phase / "audit" / name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    repo_root = Path(__file__).resolve().parents[1]
    data_root_value = os.environ.get("DIFFUSION_DATA_ROOT")
    run_dir_value = os.environ.get("DIFFUSION_RUN_DIR")
    if not data_root_value or not run_dir_value:
        raise RuntimeError("DIFFUSION_DATA_ROOT and DIFFUSION_RUN_DIR must be set")
    data_root = Path(data_root_value).resolve()
    run_dir = Path(run_dir_value).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    outer_seed = int(config["split"]["outer_seed"])
    n_folds = int(config["split"]["n_folds"])
    model_seed = int(config["reproducibility"]["model_seed"])
    bootstrap_replicates = int(config["statistics"]["bootstrap_replicates"])
    min_positive_folds = int(config["decision"]["min_positive_folds_per_metric"])
    started = time.perf_counter()

    cleanup_runtime_outputs(repo_root)
    patients, slides = prepare_runtime_manifests(repo_root, data_root, n_folds, outer_seed)
    print(
        json.dumps(
            {
                "phase": "phase1r-confirmatory",
                "n_patients": len(patients),
                "n_events": int(patients["dss_status"].sum()),
                "outer_seed": outer_seed,
                "model_seed": model_seed,
                "device": str(phase1_neural.DEVICE),
            }
        ),
        flush=True,
    )

    b0_risk, b0_g, b0_rows = evaluate_b0(
        repo_root, patients, slides, model_seed, n_folds
    )
    run_neural_confirmatory(repo_root, model_seed, n_folds)
    risks = {"B0_mean": b0_risk}
    fold_rows = list(b0_rows)
    for label, phase1_method in METHOD_FILES.items():
        method_risk, method_g, method_rows = load_neural_oof(
            repo_root, patients, label, phase1_method, model_seed, n_folds
        )
        if not np.allclose(b0_g, method_g, rtol=0, atol=1e-12, equal_nan=True):
            raise RuntimeError(f"Fold-wise censoring weights disagree for {label}")
        risks[label] = method_risk
        fold_rows.extend(method_rows)

    results, decision = summarize(
        patients,
        risks,
        b0_g,
        fold_rows,
        bootstrap_replicates,
        outer_seed,
        min_positive_folds,
    )
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(run_dir / "fold_metrics.csv", index=False)
    summary = {
        "phase": "phase1r-confirmatory",
        "n_patients": len(patients),
        "n_events": int(patients["dss_status"].sum()),
        "outer_seed": outer_seed,
        "model_seed": model_seed,
        "elapsed_seconds": float(time.perf_counter() - started),
        "methods": results,
        "decision": decision,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_metrics(run_dir, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    cleanup_runtime_outputs(repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
