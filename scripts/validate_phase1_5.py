"""Fail-closed integrity check for a completed Phase 1.5 runtime tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_AUDIT = {
    "cohort_summary.csv",
    "patient_slide_mapping.csv",
    "endpoint_summary.csv",
    "feature_inventory.csv",
    "split_audit.csv",
    "leakage_audit.md",
    "data_audit_report.md",
}
REQUIRED_METHODS = {
    "B0_mean",
    "B0b_mean_std",
    "B1_ABMIL",
    "B2_GatedABMIL",
    "B4_TransMIL",
    "B6_PANTHER_GMM",
    "B8_EAGLE_spatial",
    "B9a_ABMIL_Cox",
    "B9b_ABMIL_CoxRank",
}


def validate(root: Path) -> dict[str, object]:
    out = root / "outputs" / "phase1_5_benchmark_audit"
    state = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE":
        raise RuntimeError(f"run_state status is not COMPLETE: {state.get('status')!r}")
    missing = sorted(name for name in REQUIRED_AUDIT if not (out / "data_audit" / name).exists())
    if missing:
        raise RuntimeError(f"missing audit artifacts: {missing}")
    split = pd.read_csv(out / "splits" / "fixed_patient_folds.csv")
    if split.patient_id.duplicated().any() or len(split) != 487 or split.fold.nunique() != 5:
        raise RuntimeError("fixed split is not unique 5-fold patient-level coverage")
    oof = pd.read_csv(out / "oof_predictions" / "all_methods.csv")
    methods = set(oof.method.astype(str))
    if methods != REQUIRED_METHODS:
        raise RuntimeError(f"unexpected OOF method set: {sorted(methods)}")
    duplicate = oof.duplicated(["method", "seed", "fold", "patient_id"]).sum()
    if duplicate:
        raise RuntimeError(f"duplicate OOF rows: {duplicate}")
    counts = oof.groupby(["method", "seed", "fold"]).patient_id.nunique()
    if len(counts) != len(REQUIRED_METHODS) * 3 * 5 or not (counts.isin([97, 98]).all()):
        raise RuntimeError("OOF does not contain 3 seeds x 5 complete folds per method")
    bootstrap = pd.read_csv(out / "statistical_tests" / "bootstrap_comparisons.csv")
    if len(bootstrap) == 0 or bootstrap.n_bootstrap.min() < 2000:
        raise RuntimeError("paired bootstrap has fewer than 2000 replicates")
    return {
        "status": "PASS",
        "patients": len(split),
        "events": int(split.dss_event.sum()),
        "oof_rows": len(oof),
        "methods": sorted(methods),
        "bootstrap_rows": len(bootstrap),
        "bootstrap_replicates": int(bootstrap.n_bootstrap.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
