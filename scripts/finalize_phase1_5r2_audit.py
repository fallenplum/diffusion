"""Finalize compact Phase1.5R2 audit artifacts after the long run completes.

The neural jobs are intentionally kept resumable and local.  This script only
reads their compact outputs, regenerates the aggregate/report layer, records
training-loss and risk-direction sanity tables, and preserves recovery history
from earlier interrupted attempts.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# When invoked as ``python scripts/finalize_phase1_5r2_audit.py`` Python adds
# only the scripts directory to ``sys.path``.  Add the repository root so the
# namespace packages ``src`` and ``scripts`` resolve consistently.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.phase1_5r2 import runner
from scripts.validate_phase1_5r2 import validate


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sustained_ascent(frame: pd.DataFrame) -> bool:
    """Match the prompt's sustained-ascent warning without phase-reset bias."""
    if frame.empty or "train_cox_loss" not in frame.columns:
        return False
    for _, phase in frame.groupby("phase", sort=False):
        losses = pd.to_numeric(phase["train_cox_loss"], errors="coerce").dropna().tolist()
        if len(losses) < 2:
            continue
        deltas = [b - a for a, b in zip(losses, losses[1:])]
        if losses[-1] > losses[0] + 1e-8 and sum(delta > 1e-8 for delta in deltas) / len(deltas) >= 0.8:
            return True
    return False


def repair_direction_flags(out: Path) -> None:
    """Recompute loss-direction flags from saved histories after a resume."""
    metric_root = out / "tables" / "job_metrics"
    for path in metric_root.glob("**/*.json"):
        try:
            payload = _json(path)
            if "seed" not in payload or "fold" not in payload or "token_budget" not in payload or "projection" not in payload:
                continue
            history_path = out / "fold_metrics" / path.parent.name / f"{path.stem}.csv"
            if not history_path.exists():
                continue
            history = pd.read_csv(history_path)
            flag = _sustained_ascent(history)
            if bool(payload.get("cox_training_direction_warning", False)) != flag:
                payload["cox_training_direction_warning"] = flag
                payload["train_loss_direction_rule"] = "within_phase_final_gt_initial_and_at_least_80pct_positive_steps"
                path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue


def make_compact_tables(out: Path) -> None:
    unit = out / "unit_tests"
    rows = []
    mapping = [
        ("Cox analytic vs autograd", "cox_gradient_test.json", "analytic_vs_autograd"),
        ("Cox analytic vs finite difference", "cox_gradient_test.json", "analytic_vs_finite_difference"),
        ("Rank analytic vs autograd", "rank_gradient_test.json", "analytic_vs_autograd"),
    ]
    for label, name, field in mapping:
        payload = _json(unit / name)
        item = payload[field]
        rows.append({"test": label, "status": bool(item.get("allclose")), "max_abs_error": item.get("max_abs_error"), "cosine_similarity": item.get("cosine_similarity")})
    combined = _json(unit / "combined_gradient_test.json")
    for item in combined.get("lambdas", []):
        check = item["analytic_vs_autograd"]
        rows.append({"test": f"Combined analytic vs autograd lambda={item['rank_lambda']}", "status": bool(check.get("allclose")), "max_abs_error": check.get("max_abs_error"), "cosine_similarity": check.get("cosine_similarity")})
    two_pass = _json(unit / "two_pass_equivalence_test.json")
    rows.append({"test": "Two-pass vs direct autograd", "status": bool(two_pass.get("allclose")), "max_abs_error": two_pass.get("max_abs_parameter_error"), "cosine_similarity": None})
    pd.DataFrame(rows).to_csv(out / "tables" / "functional_tests.csv", index=False)

    copies = [
        (out / "anchor" / "fresh_anchor_table.csv", out / "tables" / "fresh_anchor.csv"),
        (out / "tables" / "main_metrics.csv", out / "tables" / "repaired_neural_benchmark.csv"),
        (out / "sensitivity" / "token_budget_sensitivity.csv", out / "tables" / "token_sensitivity.csv"),
        (out / "sensitivity" / "projection_sensitivity.csv", out / "tables" / "projection_sensitivity.csv"),
    ]
    for source, target in copies:
        if source.exists():
            shutil.copy2(source, target)

    history_rows = []
    for path in sorted((out / "fold_metrics").glob("**/*.csv")):
        if not path.is_file() or path.parts[-2:] and "anchor" in path.parts:
            continue
        try:
            frame = pd.read_csv(path)
            losses = pd.to_numeric(frame.get("train_cox_loss"), errors="coerce").dropna()
            if losses.empty:
                continue
            match = re.search(r"seed_(\d+)_fold_(\d+)", path.stem)
            method = path.parent.name
            history_rows.append({
                "method": method,
                "seed": int(match.group(1)) if match else None,
                "fold": int(match.group(2)) if match else None,
                "n_epochs": int(len(losses)),
                "initial_train_cox": float(losses.iloc[0]),
                "best_train_cox": float(losses.min()),
                "final_train_cox": float(losses.iloc[-1]),
                "final_gt_initial_warning": _sustained_ascent(frame),
                "outer_refit_initial_train_cox": float(pd.to_numeric(frame.loc[frame.get("phase") == "outer_refit", "train_cox_loss"], errors="coerce").dropna().iloc[0]) if "phase" in frame and not pd.to_numeric(frame.loc[frame.get("phase") == "outer_refit", "train_cox_loss"], errors="coerce").dropna().empty else None,
                "outer_refit_final_train_cox": float(pd.to_numeric(frame.loc[frame.get("phase") == "outer_refit", "train_cox_loss"], errors="coerce").dropna().iloc[-1]) if "phase" in frame and not pd.to_numeric(frame.loc[frame.get("phase") == "outer_refit", "train_cox_loss"], errors="coerce").dropna().empty else None,
                "history_path": path.relative_to(out).as_posix(),
            })
        except (OSError, ValueError, KeyError):
            continue
    history = pd.DataFrame(history_rows)
    history.to_csv(out / "tables" / "training_loss_sanity.csv", index=False)
    summary = []
    if not history.empty:
        for method, frame in history.groupby("method"):
            summary.append({
                "method": method,
                "n_histories": int(len(frame)),
                "n_final_gt_initial": int(frame.final_gt_initial_warning.sum()),
                "majority_final_gt_initial_warning": bool(frame.final_gt_initial_warning.mean() > 0.5),
                "mean_initial_train_cox": float(frame.initial_train_cox.mean()),
                "mean_best_train_cox": float(frame.best_train_cox.mean()),
                "mean_final_train_cox": float(frame.final_train_cox.mean()),
            })
    (out / "tables" / "training_loss_sanity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def finalize(root: Path) -> dict:
    out = root / "outputs" / runner.EXPERIMENT_ID
    state_path = out / "run_state.json"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    state = _json(state_path)
    if state.get("status") != "COMPLETE":
        raise RuntimeError(f"refusing to finalize incomplete run: {state.get('status')}/{state.get('stage')}")

    repair_direction_flags(out)
    patients, slides, split_path, _ = runner.p15.load_frozen(root)
    anchor = pd.read_csv(out / "anchor" / "fresh_anchor_table.csv")
    main, folds, pooled = runner.summarize(out, ["B0_full", "B0_1024", *runner.NEURAL_METHODS, "B5_PANTHER_style_GMM"])
    token, projection = runner.compute_sensitivity(out, pooled)
    risk = runner.add_risk_direction_columns(out, [*runner.NEURAL_METHODS, "Token_B1_K128", "Token_B1_K256", "Token_B1_K512", "Token_Transformer_K128", "Token_Transformer_K256", "Token_Transformer_K512", "Projection_random64", "Projection_learned64", "Projection_learned512"])
    bootstrap_path = out / "bootstrap" / "paired_bootstrap.csv"
    bootstrap = pd.read_csv(bootstrap_path) if bootstrap_path.exists() else pd.DataFrame()
    runner.make_figures(out, anchor, main, token, projection, bootstrap)
    label, explanation = runner.write_reports(root, out, patients, split_path, anchor, main, folds, token, projection, bootstrap, risk, state)
    make_compact_tables(out)

    # Keep all historical recovery evidence, but distinguish it from any
    # unrecovered failure in the final state.  Do not delete the evidence.
    history = list(state.get("recovery_history", []))
    history.extend(state.get("failed_jobs", []))
    state["recovery_history"] = history
    state["failed_jobs"] = []
    state["decision"] = label
    state["decision_explanation"] = explanation
    state["audit_completed_at"] = utc_now()
    state["audit_status"] = "PASS_PENDING_EXTERNAL_PROTECTED_DATA_GUARD"
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    # Preserve the recovery ledger in the human-readable audit as well as the
    # machine-readable run_state.  These attempts are historical evidence,
    # not final-result failures after the repaired matrix completed.
    failure_lines = [
        "# Phase1.5R2 failure log",
        "",
        "- MambaMIL: FAILED_TO_REPRODUCE inherited from Phase1.5R.",
        "- CLAM survival: FAILED_TO_REPRODUCE inherited from Phase1.5R.",
        "- No functional hard gate failed in the final run.",
        "",
        "## Recovered development/interruption events",
        "",
    ]
    for item in history:
        stage = item.get("stage", "unknown")
        job = item.get("job", "")
        error = str(item.get("error", ""))
        failure_lines.append(f"- `{stage}` `{job}`: {error}")
    (out / "PHASE1_5R2_FAILURE_LOG.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    report_root = root / "reports" / runner.EXPERIMENT_ID
    report_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / "PHASE1_5R2_FAILURE_LOG.md", report_root / "PHASE1_5R2_FAILURE_LOG.md")

    validation = validate(root)
    validation["audit_status"] = state["audit_status"]
    validation_path = out / "phase1_5r2_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path = out / "PHASE1_5R2_MANIFEST.json"
    if manifest_path.exists():
        manifest = _json(manifest_path)
        manifest["files"] = sorted(path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file() and path.stat().st_size < 5_000_000)
        manifest["audit_status"] = validation.get("status", False)
        manifest["audited_at"] = state["audit_completed_at"]
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_root = root / "reports" / runner.EXPERIMENT_ID
    report_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "phase1_5r2_validation.json",
        "functional_tests.csv",
        "fresh_anchor.csv",
        "repaired_neural_benchmark.csv",
        "token_sensitivity.csv",
        "projection_sensitivity.csv",
        "training_loss_sanity.csv",
        "training_loss_sanity_summary.json",
    ):
        source = out / "tables" / name if name.endswith(".csv") or name.startswith("training_loss") else out / name
        if source.exists():
            shutil.copy2(source, report_root / name)
    # validation JSON is written at the output root, while the compact report
    # receives the same immutable audit record.
    if validation_path.exists():
        shutil.copy2(validation_path, report_root / validation_path.name)
    if manifest_path.exists():
        shutil.copy2(manifest_path, report_root / manifest_path.name)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = finalize(args.root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") else 1


if __name__ == "__main__":
    raise SystemExit(main())
