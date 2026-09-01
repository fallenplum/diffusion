"""Strict post-run audit for the Phase1.5R2 functional Cox repair.

This validator is intentionally read-only apart from the optional JSON audit
record.  It checks the hard gates and the required result matrix without
opening raw TCGA files or treating historical OOF as a fresh training input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SEEDS = {20260723, 20260724, 20260725}
FOLDS = {1, 2, 3, 4, 5}
NEURAL = {
    "B1_GatedABMIL",
    "B2_PlainABMIL",
    "B3_TransformerMIL_reimplementation",
    "B6_SpatialNeighborhoodMIL",
    "B7a_GatedABMIL_Cox",
    "B7b_GatedABMIL_CoxRank",
}
TOKEN_METHODS = {
    "Token_B1_K128",
    "Token_B1_K256",
    "Token_B1_K512",
    "Token_Transformer_K128",
    "Token_Transformer_K256",
    "Token_Transformer_K512",
}
PROJECTIONS = {"Projection_random64", "Projection_learned64", "Projection_learned512"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(ok: bool, message: str, failures: list[str]) -> None:
    if not ok:
        failures.append(message)


def _job_keys(path: Path) -> set[tuple[int, int]]:
    frame = pd.read_json(path) if path.suffix == ".json" else pd.DataFrame()
    return {(int(row.seed), int(row.fold)) for row in frame.itertuples(index=False)}


def validate(root: Path) -> dict[str, Any]:
    out = root / "outputs" / "phase1_5r2_functional_repair"
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    state_path = out / "run_state.json"
    _check(state_path.exists(), "run_state.json missing", failures)
    if not state_path.exists():
        return {"status": False, "checks": checks, "failures": failures}
    state = _load(state_path)
    checks["run_complete"] = state.get("status") == "COMPLETE" and state.get("stage") == "complete"
    _check(checks["run_complete"], f"run_state is {state.get('status')}/{state.get('stage')}", failures)

    functional_path = out / "unit_tests" / "functional_summary.json"
    _check(functional_path.exists(), "functional_summary.json missing", failures)
    functional = _load(functional_path) if functional_path.exists() else {}
    required_functional = {
        "cox_autograd_match",
        "cox_finite_difference_match",
        "cox_one_step_descent",
        "rank_autograd_match",
        "rank_finite_difference_match",
        "combined_gradient_match",
        "two_pass_equivalence",
    }
    checks["functional_gate"] = bool(functional.get("status")) and all(bool(functional.get(k)) for k in required_functional)
    _check(checks["functional_gate"], "functional gate is incomplete or failed", failures)

    anchor_gate_path = out / "anchor" / "fresh_anchor_gate.json"
    anchor_manifest_path = out / "anchor" / "fresh_anchor_manifest.json"
    anchor_table_path = out / "anchor" / "fresh_anchor_table.csv"
    _check(anchor_gate_path.exists(), "fresh_anchor_gate.json missing", failures)
    _check(anchor_manifest_path.exists(), "fresh_anchor_manifest.json missing", failures)
    _check(anchor_table_path.exists(), "fresh_anchor_table.csv missing", failures)
    anchor_gate = _load(anchor_gate_path) if anchor_gate_path.exists() else {}
    anchor_manifest = _load(anchor_manifest_path) if anchor_manifest_path.exists() else {}
    checks["fresh_anchor"] = (
        anchor_gate.get("status") == "PASS"
        and float(anchor_gate.get("abs_delta_harrell", 999.0)) <= 0.02
        and bool(anchor_manifest.get("historical_oof_read_for_training") is False)
        and anchor_manifest.get("implementation_module") == "src/phase1_neural.py"
    )
    _check(checks["fresh_anchor"], "fresh anchor gate/manifest failed", failures)
    if anchor_table_path.exists():
        anchor = pd.read_csv(anchor_table_path)
        _check(len(anchor) == 1 and bool(anchor.iloc[0].get("harrell_gate", False)), "fresh anchor table is not a single passing row", failures)

    main_path = out / "tables" / "main_metrics.csv"
    _check(main_path.exists(), "main_metrics.csv missing", failures)
    main = pd.read_csv(main_path) if main_path.exists() else pd.DataFrame()
    required_main = {"B0_full", "B0_1024", "B5_PANTHER_style_GMM", *NEURAL}
    present_main = set(main.get("method", pd.Series(dtype=str)).astype(str))
    checks["main_methods"] = required_main.issubset(present_main)
    _check(checks["main_methods"], f"main methods missing: {sorted(required_main - present_main)}", failures)

    job_dir = out / "tables" / "job_metrics"
    fold_risk_warnings: list[str] = []
    for method in sorted(NEURAL):
        files = list((job_dir / method).glob("*.json"))
        keys: set[tuple[int, int]] = set()
        for file in files:
            payload = _load(file)
            if "seed" in payload and "fold" in payload:
                keys.add((int(payload["seed"]), int(payload["fold"])))
            for field in ("test_harrell_c", "test_uno_c", "risk_direction_warning", "initial_train_cox", "best_train_cox", "final_train_cox"):
                _check(field in payload, f"{method}/{file.name}: missing {field}", failures)
            if bool(payload.get("risk_direction_warning", False)):
                # Keep the preregistered per-fold warning visible.  An
                # isolated test-fold inversion is a model-level diagnostic,
                # not evidence that the corrected Cox implementation ascended.
                fold_risk_warnings.append(f"{method}/{file.name}")
        _check(keys == {(s, f) for s in SEEDS for f in FOLDS}, f"{method}: expected 15 seed/fold jobs, found {len(keys)}", failures)
    checks["neural_matrix"] = not any(f.startswith("B") and "expected 15" in f for f in failures)

    token_path = out / "sensitivity" / "token_budget_sensitivity.csv"
    _check(token_path.exists(), "token_budget_sensitivity.csv missing", failures)
    token = pd.read_csv(token_path) if token_path.exists() else pd.DataFrame()
    checks["token_sensitivity"] = bool(not token.empty and {"B1_GatedABMIL", "TransformerMIL_reimplementation"}.issubset(set(token.get("method", []))) and set(token.get("token_budget", [])) == {128, 256, 512, 1024})
    _check(checks["token_sensitivity"], "token sensitivity does not contain both repaired curves at K=128/256/512/1024", failures)
    for method in sorted(TOKEN_METHODS):
        files = list((job_dir / method).glob("*.json"))
        _check(len(files) == 15, f"{method}: expected 15 jobs, found {len(files)}", failures)

    projection_path = out / "sensitivity" / "projection_sensitivity.csv"
    _check(projection_path.exists(), "projection_sensitivity.csv missing", failures)
    projection = pd.read_csv(projection_path) if projection_path.exists() else pd.DataFrame()
    checks["projection_sensitivity"] = bool(not projection.empty and {"random64", "learned64", "learned256", "learned512"}.issubset(set(projection.get("projection", []))))
    _check(checks["projection_sensitivity"], "projection sensitivity is incomplete", failures)
    for method in sorted(PROJECTIONS):
        _check(len(list((job_dir / method).glob("*.json"))) == 15, f"{method}: expected 15 jobs", failures)

    bootstrap_path = out / "bootstrap" / "paired_bootstrap.csv"
    _check(bootstrap_path.exists(), "paired_bootstrap.csv missing", failures)
    bootstrap = pd.read_csv(bootstrap_path) if bootstrap_path.exists() else pd.DataFrame()
    checks["bootstrap"] = bool(not bootstrap.empty and int(bootstrap.get("replicates", pd.Series([0])).max()) == 5000)
    _check(checks["bootstrap"], "paired bootstrap does not record 5000 replicates", failures)
    for method in sorted(NEURAL):
        for baseline in ("B0_full", "B0_1024", "B1_GatedABMIL"):
            if method == baseline:
                continue
            _check(bool(((bootstrap.get("method", pd.Series(dtype=str)) == method) & (bootstrap.get("baseline", pd.Series(dtype=str)) == baseline)).any()), f"bootstrap missing {method} vs {baseline}", failures)

    risk_path = out / "tables" / "risk_direction_sanity.csv"
    _check(risk_path.exists(), "risk_direction_sanity.csv missing", failures)
    global_inversions: list[str] = []
    if risk_path.exists():
        risk = pd.read_csv(risk_path)
        required_risk = {"method", "harrell_risk", "harrell_neg_risk", "risk_direction_warning"}
        has_columns = required_risk.issubset(set(risk.columns))
        _check(not risk.empty and has_columns, "risk direction sanity table is empty or malformed", failures)
        if has_columns and not risk.empty:
            for method, frame in risk.groupby("method"):
                h = pd.to_numeric(frame["harrell_risk"], errors="coerce").mean()
                h_neg = pd.to_numeric(frame["harrell_neg_risk"], errors="coerce").mean()
                if pd.notna(h) and pd.notna(h_neg) and h < 0.5 and h_neg > h + 0.05:
                    global_inversions.append(str(method))
            if global_inversions:
                _check(False, f"global risk-direction inversion: {sorted(global_inversions)}", failures)
            warning_count = int(risk["risk_direction_warning"].astype(bool).sum())
            if warning_count:
                warnings.append(
                    f"{warning_count} isolated fold/method risk-direction warnings retained in risk_direction_sanity.csv"
                )
    if fold_risk_warnings:
        warnings.append(f"{len(fold_risk_warnings)} neural job metrics carry isolated risk-direction warnings")
    checks["risk_direction_sanity"] = not global_inversions and risk_path.exists()

    history_count = len(list((out / "fold_metrics").glob("**/*.csv")))
    checks["training_history"] = history_count >= 15
    _check(checks["training_history"], f"training history coverage too small: {history_count} files", failures)
    reuse_path = out / "configs" / "unaffected_reuse_provenance.json"
    reuse = _load(reuse_path) if reuse_path.exists() else {}
    checks["reuse_provenance"] = reuse.get("source_experiment") == "phase1_5r_confirmatory_repair" and set(reuse.get("reused_methods", [])) >= {"B0_full", "B0_1024", "B5_PANTHER_style_GMM"}
    _check(checks["reuse_provenance"], "unaffected-result reuse provenance missing", failures)

    result = {"status": not failures, "checks": checks, "failures": failures, "warnings": warnings, "state_status": state.get("status"), "decision": state.get("decision")}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true", help="write audit JSON under the experiment output")
    args = parser.parse_args()
    result = validate(args.root.resolve())
    if args.write:
        target = args.root.resolve() / "outputs" / "phase1_5r2_functional_repair" / "phase1_5r2_validation.json"
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
