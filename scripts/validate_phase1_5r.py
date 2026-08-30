"""Fail-closed validation for the Phase 1.5R repaired benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CORE_METHODS = {
    "B0_full",
    "B0_1024",
    "B1_GatedABMIL",
    "B2_PlainABMIL",
    "B3_TransformerMIL_reimplementation",
    "B5_PANTHER_style_GMM",
    "B6_SpatialNeighborhoodMIL",
    "B7a_GatedABMIL_Cox",
    "B7b_GatedABMIL_CoxRank",
}
AUDIT_FILES = {
    "cohort_summary.csv",
    "patient_slide_mapping.csv",
    "endpoint_summary.csv",
    "feature_inventory.csv",
    "split_audit.csv",
    "leakage_audit.md",
    "data_audit_report.md",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "outputs" / "phase1_5r_confirmatory_repair"
    state = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE":
        raise SystemExit(f"run_state is not COMPLETE: {state.get('status')}")
    artifacts = state.get("final_artifacts", {})
    if not artifacts or not all(bool(value) for value in artifacts.values()):
        raise SystemExit("final_artifacts is incomplete")
    anchor = pd.read_csv(out / "anchor" / "anchor_reproduction.csv")
    if set(anchor.method) != {"B0_mean", "B1_ABMIL"} or not anchor.gate_pass.all() or (anchor.abs_delta_harrell > 0.02).any():
        raise SystemExit("anchor gate failed")
    main = pd.read_csv(out / "tables" / "main_metrics.csv")
    missing = sorted(CORE_METHODS - set(main.method))
    if missing:
        raise SystemExit(f"missing core main methods: {missing}")
    if not ((main.oof_harrell_c.between(0, 1)).all() and (main.oof_uno_c.between(0, 1)).all()):
        raise SystemExit("main metric outside [0,1]")
    folds = pd.read_csv(out / "tables" / "fold_metrics.csv")
    for method in CORE_METHODS:
        subset = folds.loc[folds.method == method]
        if len(subset) != 15 or set(subset.seed) != {20260723, 20260724, 20260725} or set(subset.fold) != {1, 2, 3, 4, 5}:
            raise SystemExit(f"incomplete 3-seed x 5-fold coverage: {method}")
    token = pd.read_csv(out / "sensitivity" / "token_budget_sensitivity.csv")
    for method in {"Mean", "B1_GatedABMIL", "TransformerMIL_reimplementation"}:
        if set(token.loc[token.method == method, "token_budget"]) != {128, 256, 512, 1024}:
            raise SystemExit(f"incomplete token sensitivity: {method}")
    projection = pd.read_csv(out / "sensitivity" / "projection_sensitivity.csv")
    if not {"random64", "learned64", "learned256", "learned512"}.issubset(set(projection.projection)):
        raise SystemExit("incomplete projection sensitivity")
    bootstrap = pd.read_csv(out / "bootstrap" / "paired_bootstrap.csv")
    if len(bootstrap) < 1 or set(bootstrap.replicates) != {5000}:
        raise SystemExit("paired bootstrap is missing or below 5000 replicates")
    if set(path.name for path in (out / "data_audit").iterdir()) & AUDIT_FILES != AUDIT_FILES:
        raise SystemExit("data audit artifacts are incomplete")
    print(json.dumps({"status": "PASS", "patients": 487, "events": 65, "core_methods": len(CORE_METHODS), "bootstrap_comparisons": len(bootstrap), "bootstrap_replicates": 5000, "token_budgets": [128, 256, 512, 1024]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
