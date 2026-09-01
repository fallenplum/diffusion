"""Resumable Phase 1.5R2 functional-repair experiment runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import phase1_5r_confirmatory as p15
import phase1_5_benchmark as benchmark
from .cox_gradients import cox_risk_coefficients
from .fresh_anchor import run_fresh_anchor
from .functional_tests import run_all


EXPERIMENT_ID = "phase1_5r2_functional_repair"
SEEDS = (20260723, 20260724, 20260725)
FOLDS = (1, 2, 3, 4, 5)
MAIN_TOKENS = 1024
TOKEN_GRID = (128, 256, 512, 1024)
PROJECTION_GRID = ("random64", "learned64", "learned256", "learned512")
BOOTSTRAPS = 5000
NEURAL_METHODS = (
    "B1_GatedABMIL",
    "B2_PlainABMIL",
    "B3_TransformerMIL_reimplementation",
    "B6_SpatialNeighborhoodMIL",
    "B7a_GatedABMIL_Cox",
    "B7b_GatedABMIL_CoxRank",
)
UNANFFECTED_METHODS = ("B0_full", "B0_1024", "B5_PANTHER_style_GMM")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def root_output(root: Path) -> Path:
    return root / "outputs" / EXPERIMENT_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def acquire_lock(out: Path) -> Path:
    lock = out / "run.lock"
    out.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()}\nstarted={utc_now()}\n".encode("utf-8"))
        os.close(fd)
        return lock
    except FileExistsError:
        try:
            text = lock.read_text(encoding="utf-8")
            pid = int(next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith("pid=")))
            os.kill(pid, 0)
        except (FileNotFoundError, ValueError, StopIteration, OSError):
            lock.unlink(missing_ok=True)
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()}\nstarted={utc_now()}\n".encode("utf-8"))
            os.close(fd)
            return lock
        raise RuntimeError(f"Phase1.5R2 is already running; lock={lock}")


def mark(state: dict[str, Any], out: Path, *, stage: str | None = None, job: str | None = None, complete: bool = False) -> None:
    if stage is not None:
        state["stage"] = stage
    state["current_job"] = None if complete else job
    if state.get("status") != "COMPLETE":
        state["status"] = "RUNNING"
    if complete and job and job not in state["completed_jobs"]:
        state["completed_jobs"].append(job)
    state["updated_at"] = utc_now()
    atomic_json(out / "run_state.json", state)


def load_state(out: Path, resume: bool) -> dict[str, Any]:
    path = out / "run_state.json"
    if resume and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    state = {
        "experiment_id": EXPERIMENT_ID,
        "status": "RUNNING",
        "stage": "initializing",
        "current_job": None,
        "completed_jobs": [],
        "failed_jobs": [],
        "reused_jobs": [],
        "start_time": utc_now(),
        "updated_at": utc_now(),
    }
    atomic_json(path, state)
    return state


def _copy_tree_files(source: Path, target: Path, names: list[str] | None = None) -> None:
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if names is not None and item.name not in names:
            continue
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def reuse_unaffected(root: Path, out: Path, state: dict[str, Any]) -> None:
    """Copy only non-neural Phase1.5R runtime artifacts; never rerun them."""

    old = root / "outputs" / "phase1_5r_confirmatory_repair"
    if not old.exists():
        raise FileNotFoundError(f"Phase1.5R output is required for unaffected reuse: {old}")
    for directory in ("data_audit", "splits"):
        _copy_tree_files(old / directory, out / directory)
    provenance = {
        "source_experiment": "phase1_5r_confirmatory_repair",
        "source_output": str(old),
        "reused_methods": list(UNANFFECTED_METHODS),
        "reused_sensitivity": ["Mean@128", "Mean@256", "Mean@512", "Mean@1024"],
        "reason": "These paths do not use the repaired neural Cox analytic gradient and are explicitly not rerun by the Phase1.5R2 prompt.",
        "copied_at": utc_now(),
    }
    atomic_json(out / "configs" / "unaffected_reuse_provenance.json", provenance)
    for method in (*UNANFFECTED_METHODS, "Mean@128", "Mean@256", "Mean@512", "Mean@1024"):
        for directory in ("oof_predictions", "tables/job_metrics"):
            source = old / directory / method
            target = out / directory / method
            _copy_tree_files(source, target)
        marker = f"{method}:reused_from_phase1_5R_non_neural_unaffected_results"
        if marker not in state["reused_jobs"]:
            state["reused_jobs"].append(marker)


def write_protocol(out: Path, root: Path, patients: pd.DataFrame, split_path: Path) -> None:
    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "repair": "Cox analytic gradient sign; fresh original Phase1R B1 anchor",
        "raw_feature_dim": 1536,
        "main_tokens": MAIN_TOKENS,
        "token_grid": list(TOKEN_GRID),
        "projection_grid": list(PROJECTION_GRID),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "eval_sampling_repeats": 5,
        "train_sampling": "stochastic_without_replacement_sorted_indices",
        "patient_aggregation": "slide_first_then_mean",
        "uno_censoring": "outer_train_censoring_km",
        "bootstrap_replicates": BOOTSTRAPS,
        "split_source": str(split_path.relative_to(root).as_posix()),
        "split_hash": p15.p15.canonical_split_hash(patients),
        "cohort": {"patients": len(patients), "events": int(patients.dss_event.sum()), "slides": 494},
        "unaffected_results": "reused_from_phase1_5R_non_neural_unaffected_results",
    }
    atomic_json(out / "configs" / "resolved_config.json", protocol)
    atomic_json(out / "configs" / "protocol.json", protocol)


def run_functional_gate(out: Path, state: dict[str, Any]) -> dict[str, Any]:
    unit_dir = out / "unit_tests"
    result = run_all(unit_dir)
    if not result["status"]:
        atomic_text(out / "PHASE1_5R2_FAILURE_LOG.md", "# Phase1.5R2 failure log\n\nSTOP_PHASE1_5R2: functional gate failed.\n")
        raise RuntimeError("STOP_PHASE1_5R2: functional tests failed")
    state["functional_gate"] = result
    mark(state, out, stage="functional_gate_passed", complete=True, job="functional_tests")
    return result


def run_repaired_neural_suite(root: Path, out: Path, state: dict[str, Any], patients: pd.DataFrame, slides: pd.DataFrame, device: torch.device) -> None:
    # The existing implementation is reused for sampling, architectures,
    # exact two-pass plumbing and metrics; only its corrected coefficient
    # function is installed before this stage.
    p15.cox_risk_coefficients = cox_risk_coefficients
    store = p15.SlideStore(root, slides)
    for method in NEURAL_METHODS:
        rank = method == "B7b_GatedABMIL_CoxRank"
        for seed in SEEDS:
            for fold in FOLDS:
                p15.neural_job(root, out, state, store, patients, method, seed, fold, device, MAIN_TOKENS, "learned256", rank, method)


def run_token_sensitivity(root: Path, out: Path, state: dict[str, Any], patients: pd.DataFrame, slides: pd.DataFrame, device: torch.device) -> None:
    p15.cox_risk_coefficients = cox_risk_coefficients
    store = p15.SlideStore(root, slides)
    for budget in (128, 256, 512):
        for method, output_method in (("B1_GatedABMIL", f"Token_B1_K{budget}"), ("B3_TransformerMIL_reimplementation", f"Token_Transformer_K{budget}")):
            for seed in SEEDS:
                for fold in FOLDS:
                    p15.neural_job(root, out, state, store, patients, method, seed, fold, device, budget, "learned256", False, output_method)


def run_projection_sensitivity(root: Path, out: Path, state: dict[str, Any], patients: pd.DataFrame, slides: pd.DataFrame, device: torch.device) -> None:
    p15.cox_risk_coefficients = cox_risk_coefficients
    store = p15.SlideStore(root, slides)
    for projection in ("random64", "learned64", "learned512"):
        output_method = f"Projection_{projection}"
        for seed in SEEDS:
            for fold in FOLDS:
                p15.neural_job(root, out, state, store, patients, "B1_GatedABMIL", seed, fold, device, MAIN_TOKENS, projection, False, output_method)


def add_risk_direction_columns(out: Path, methods: list[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        directory = out / "tables" / "job_metrics" / method
        for path in sorted(directory.glob("*.json")) if directory.exists() else []:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "method": method,
                "seed": payload.get("seed"),
                "fold": payload.get("fold"),
                "harrell_risk": payload.get("test_harrell_c"),
                "harrell_neg_risk": payload.get("risk_direction_harrell_negated"),
                "uno_risk": payload.get("test_uno_c"),
                "uno_neg_risk": payload.get("risk_direction_uno_negated"),
                "risk_direction_warning": payload.get("risk_direction_warning", False),
            })
    result = pd.DataFrame(rows)
    atomic_csv(result, out / "tables" / "risk_direction_sanity.csv")
    return result


def copy_old_bugged_summary(root: Path, out: Path) -> dict[str, Any]:
    old = root / "outputs" / "phase1_5r_confirmatory_repair"
    payload: dict[str, Any] = {"source": str(old), "status": "missing"}
    path = old / "tables" / "main_metrics.csv"
    if path.exists():
        old_main = pd.read_csv(path)
        rows = old_main.loc[old_main.method.isin(["B1_GatedABMIL", "B2_PlainABMIL", "B3_TransformerMIL_reimplementation", "B6_SpatialNeighborhoodMIL", "B7a_GatedABMIL_Cox", "B7b_GatedABMIL_CoxRank"])]
        atomic_csv(rows, out / "tables" / "phase1_5r_bugged_neural_metrics.csv")
        payload["status"] = "copied_for_provenance"
        payload["methods"] = rows.method.tolist()
    atomic_json(out / "configs" / "phase1_5r_bugged_provenance.json", payload)
    return payload


def compute_sensitivity(out: Path, pooled: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ``summarize`` intentionally receives only the main benchmark methods.
    # Load the separately trained token/projection OOF rows here so the
    # sensitivity tables are complete on both a fresh run and a resume.
    sensitivity_methods = {
        "Mean@128",
        "Mean@256",
        "Mean@512",
        "Mean@1024",
        "Token_B1_K128",
        "Token_B1_K256",
        "Token_B1_K512",
        "B1_GatedABMIL",
        "Token_Transformer_K128",
        "Token_Transformer_K256",
        "Token_Transformer_K512",
        "B3_TransformerMIL_reimplementation",
        "Projection_random64",
        "Projection_learned64",
        "Projection_learned512",
    }
    for method in sensitivity_methods:
        if method not in pooled:
            frame = p15.pooled_risk_frame(out, method)
            if not frame.empty:
                pooled[method] = frame
    token_rows: list[dict[str, Any]] = []
    for method, label in [("Mean@128", "Mean"), ("Mean@256", "Mean"), ("Mean@512", "Mean"), ("Mean@1024", "Mean"), ("Token_B1_K128", "B1_GatedABMIL"), ("Token_B1_K256", "B1_GatedABMIL"), ("Token_B1_K512", "B1_GatedABMIL"), ("B1_GatedABMIL", "B1_GatedABMIL"), ("Token_Transformer_K128", "TransformerMIL_reimplementation"), ("Token_Transformer_K256", "TransformerMIL_reimplementation"), ("Token_Transformer_K512", "TransformerMIL_reimplementation"), ("B3_TransformerMIL_reimplementation", "TransformerMIL_reimplementation")]:
        frame = pooled.get(method, pd.DataFrame())
        if frame.empty:
            continue
        if method.startswith("Mean@"):
            token = int(method.split("@", 1)[1])
        elif method.startswith("Token_"):
            token = int(method.split("K", 1)[1])
        else:
            # The main B1/Transformer rows are evaluated at the fixed
            # 1024-token budget.  They do not contain a ``K###`` suffix, so
            # parsing them as token-sensitivity labels raises an IndexError.
            token = MAIN_TOKENS
        token_rows.append({"method": label, "token_budget": token, "oof_harrell_c": float(p15.harrell_c(frame.dss_time, frame.dss_event, frame.risk_score)), "oof_uno_c": float(p15.uno_c(frame.dss_time, frame.dss_event, frame.risk_score, frame.censor_survival_outer_train)), "n_seeds": int(frame.seed.nunique()) if "seed" in frame else 3})
    token = pd.DataFrame(token_rows).drop_duplicates(["method", "token_budget"], keep="last").sort_values(["method", "token_budget"]) if token_rows else pd.DataFrame()
    projection_rows: list[dict[str, Any]] = []
    for projection, method, dim in (("random64", "Projection_random64", 64), ("learned64", "Projection_learned64", 64), ("learned256", "B1_GatedABMIL", 256), ("learned512", "Projection_learned512", 512)):
        frame = pooled.get(method, pd.DataFrame())
        if frame.empty:
            continue
        projection_rows.append({"projection": projection, "feature_dim": dim, "token_budget": MAIN_TOKENS, "oof_harrell_c": float(p15.harrell_c(frame.dss_time, frame.dss_event, frame.risk_score)), "oof_uno_c": float(p15.uno_c(frame.dss_time, frame.dss_event, frame.risk_score, frame.censor_survival_outer_train)), "n_seeds": int(frame.seed.nunique()) if "seed" in frame else 3})
    projection = pd.DataFrame(projection_rows)
    atomic_csv(token, out / "sensitivity" / "token_budget_sensitivity.csv")
    atomic_csv(projection, out / "sensitivity" / "projection_sensitivity.csv")
    return token, projection


def summarize(out: Path, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    main, folds, pooled = p15.summarize_methods(out, methods)
    return main, folds, pooled


def make_figures(out: Path, anchor: pd.DataFrame, main: pd.DataFrame, token: pd.DataFrame, projection: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    # Fig 1: sign repair (legacy ascent versus corrected descent).
    cox_json = json.loads((out / "unit_tests" / "cox_gradient_test.json").read_text(encoding="utf-8"))
    plt.figure(figsize=(7, 4))
    labels = ["legacy wrong", "corrected"]
    values = [cox_json["legacy_wrong_cosine_similarity"], cox_json["corrected_cosine_similarity"]]
    plt.bar(labels, values, color=["#b44", "#2878b5"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("cosine(analytic, autograd)")
    plt.title("Cox gradient direction repair")
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig1_cox_gradient_direction.png", dpi=180)
    plt.close()

    historical = float(anchor.historical_harrell.iloc[0])
    fresh = float(anchor.fresh_harrell.iloc[0])
    plt.figure(figsize=(6, 4))
    plt.bar(["Phase1R historical", "Phase1R fresh"], [historical, fresh], color=["#777", "#2c7"])
    plt.ylabel("Harrell C")
    plt.ylim(0.45, 0.8)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig2_fresh_anchor.png", dpi=180)
    plt.close()

    bug_path = out / "tables" / "phase1_5r_bugged_neural_metrics.csv"
    if bug_path.exists() and not main.empty:
        bug = pd.read_csv(bug_path).set_index("method")
        repaired = main.set_index("method")
        names = [name for name in repaired.index if name in bug.index]
        x = np.arange(len(names))
        width = 0.35
        plt.figure(figsize=(10, 4.5))
        plt.bar(x - width / 2, bug.loc[names, "oof_harrell_c"], width, label="Phase1.5R bugged")
        plt.bar(x + width / 2, repaired.loc[names, "oof_harrell_c"], width, label="Phase1.5R2 repaired")
        plt.xticks(x, names, rotation=25, ha="right", fontsize=8)
        plt.ylabel("Harrell C")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "Fig3_bugged_vs_repaired.png", dpi=180)
        plt.close()

    plt.figure(figsize=(8, 4.5))
    if not token.empty:
        for method, frame in token.groupby("method"):
            frame = frame.sort_values("token_budget")
            plt.plot(frame.token_budget, frame.oof_harrell_c, marker="o", label=f"{method} Harrell")
            plt.plot(frame.token_budget, frame.oof_uno_c, marker="x", linestyle="--", label=f"{method} Uno")
    plt.xticks(list(TOKEN_GRID))
    plt.ylabel("OOF C-index")
    plt.xlabel("Token budget K")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig4_repaired_token_sensitivity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    if not main.empty:
        frame = main.sort_values("oof_harrell_c")
        pos = np.arange(len(frame))
        plt.barh(pos - 0.18, frame.oof_harrell_c, 0.36, label="Harrell")
        plt.barh(pos + 0.18, frame.oof_uno_c, 0.36, label="Uno")
        plt.yticks(pos, frame.method, fontsize=8)
    plt.xlabel("OOF C-index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig5_repaired_benchmark.png", dpi=180)
    plt.close()


def decision(anchor: pd.DataFrame, main: pd.DataFrame, bootstrap: pd.DataFrame) -> tuple[str, str]:
    if anchor.empty or not bool(anchor.harrell_gate.all()):
        return "STOP_PHASE1_5R2 ANCHOR_REPRODUCTION_FAILED", "Fresh original Phase1R B1 did not reproduce the historical Harrell anchor within ±0.02."
    values = {row.method: float(row.oof_harrell_c) for row in main.itertuples(index=False)}
    full = values.get("B0_full", float("nan"))
    mean = values.get("B0_1024", float("nan"))
    b1 = values.get("B1_GatedABMIL", float("nan"))
    advanced = [name for name in NEURAL_METHODS if name != "B7a_GatedABMIL_Cox"]
    strongest_gain = bootstrap.loc[bootstrap.baseline == "B0_full"] if not bootstrap.empty else pd.DataFrame()
    # The two-metric gate must hold on the same comparison row.  Taking the
    # maximum Harrell lower bound and maximum Uno lower bound independently
    # could combine two different targets and create an oracle-style gain.
    stable = bool(
        not strongest_gain.empty
        and (
            (strongest_gain.delta_harrell_ci_low > 0)
            & (strongest_gain.delta_uno_ci_low > 0)
        ).any()
    )
    if np.isfinite(full) and np.isfinite(mean) and np.isfinite(b1) and full >= b1 and mean >= b1 and all(values.get(name, -np.inf) <= full for name in advanced) and not stable:
        return "SIMPLE_POOLING_SUFFICIENT_CONFIRMED", "Functional gates and fresh anchor passed; Full Mean and Mean@1024 are not weaker than repaired neural methods, and no strongest-vs-Full-Mean two-metric bootstrap gain is stable."
    if np.isfinite(b1) and np.isfinite(full) and b1 > full:
        return "MIL_SIGNAL_RESTORED", "Repaired GatedABMIL exceeds Full Mean on pooled Harrell; the old simple-pooling conclusion is not retained."
    rank = bootstrap.loc[(bootstrap.method == "B7b_GatedABMIL_CoxRank") & (bootstrap.baseline == "B1_GatedABMIL")]
    if not rank.empty and rank.delta_harrell_ci_low.iloc[0] > 0 and rank.delta_uno_ci_low.iloc[0] > 0:
        return "SURVIVAL_SUPERVISION_BOTTLENECK", "Cox+Rank has a stable two-metric gain over GatedABMIL Cox-only while architecture differences remain limited."
    return "MIXED_EVIDENCE", "The repaired evidence does not satisfy a stable simple-pooling confirmation or a stable MIL/supervision gain; retain the ambiguity and stop architecture expansion."


def write_reports(root: Path, out: Path, patients: pd.DataFrame, split_path: Path, anchor: pd.DataFrame, main: pd.DataFrame, folds: pd.DataFrame, token: pd.DataFrame, projection: pd.DataFrame, bootstrap: pd.DataFrame, risk_direction: pd.DataFrame, state: dict[str, Any]) -> tuple[str, str]:
    label, explanation = decision(anchor, main, bootstrap)
    functional = state.get("functional_gate", {})
    main_lines = ["| Method | Harrell | Uno | Δ vs Full Mean | Δ vs B1 |", "|---|---:|---:|---:|---:|"]
    full = float(main.loc[main.method == "B0_full", "oof_harrell_c"].iloc[0]) if (main.method == "B0_full").any() else float("nan")
    b1 = float(main.loc[main.method == "B1_GatedABMIL", "oof_harrell_c"].iloc[0]) if (main.method == "B1_GatedABMIL").any() else float("nan")
    for row in main.sort_values("oof_harrell_c").itertuples(index=False):
        main_lines.append(f"| {row.method} | {row.oof_harrell_c:.4f} | {row.oof_uno_c:.4f} | {row.oof_harrell_c - full:.4f} | {row.oof_harrell_c - b1:.4f} |")
    anchor_lines = ["| Method | Historical Harrell | Fresh Harrell | Abs Δ | Gate |", "|---|---:|---:|---:|---|"]
    for row in anchor.itertuples(index=False):
        anchor_lines.append(f"| {row.method} | {row.historical_harrell:.6f} | {row.fresh_harrell:.6f} | {row.abs_delta_harrell:.6f} | {row.harrell_gate} |")
    token_lines = ["| Method | K | Harrell | Uno |", "|---|---:|---:|---:|"]
    for row in token.sort_values(["method", "token_budget"]).itertuples(index=False) if not token.empty else []:
        token_lines.append(f"| {row.method} | {int(row.token_budget)} | {row.oof_harrell_c:.4f} | {row.oof_uno_c:.4f} |")
    projection_lines = ["| Projection | Dim | Harrell | Uno |", "|---|---:|---:|---:|"]
    for row in projection.itertuples(index=False) if not projection.empty else []:
        projection_lines.append(f"| {row.projection} | {int(row.feature_dim)} | {row.oof_harrell_c:.4f} | {row.oof_uno_c:.4f} |")
    warning_count = int(risk_direction.risk_direction_warning.sum()) if not risk_direction.empty else 0
    result_text = f"""# Phase1.5R2 Functional Cox Repair Results

## Decision

**{label}**

{explanation}

This is the functional repair stage only. Phase2, tri-modal, pan-cancer and morphology-level risk attribution were not entered.

## 1. Functional gates

```json
{json.dumps(functional, indent=2, ensure_ascii=False)}
```

The pre-repair coefficient had the opposite direction (legacy cosine is recorded in `unit_tests/cox_gradient_test.json`); corrected analytic, autograd, finite-difference, ranking, combined λ={{0.05,0.2}} and two-pass tests all passed before TCGA training.

## 2. Fresh Phase1R B1 anchor

{chr(10).join(anchor_lines)}

The fresh anchor directly calls `src/phase1_neural.py`'s original sampled risk-set implementation with seed 20260723, frozen Phase1R folds, raw 1536D, learned 256D projection and K=1024. Historical OOF was used only as a metric reference; the manifest records no historical OOF training read.

## 3. Repaired neural benchmark

{chr(10).join(main_lines)}

The Full Mean, Mean@1024 and PANTHER-style GMM rows are explicitly reused from Phase1.5R because they are not affected by the neural Cox analytic-gradient bug. The old Phase1.5R bugged neural rows remain in `tables/phase1_5r_bugged_neural_metrics.csv` for provenance.

Risk-direction warnings: **{warning_count}** fold/method records. They are retained as model-level diagnostics; a global inversion would be a hard integrity stop, and no warning is silently hidden or sign-flipped.

## 4. Token sensitivity

{chr(10).join(token_lines)}

## 5. Projection sensitivity

{chr(10).join(projection_lines)}

## 6. Statistical comparison and required scope answers

- Paired bootstrap uses 5000 replicates and outer-training censoring KM values held fixed per patient/fold; see `bootstrap/paired_bootstrap.csv`.
- PlainABMIL, TransformerMIL reimplementation, SpatialNeighborhoodMIL and Cox+Rank are all rerun with the corrected gradient.
- MambaMIL and CLAM retain `FAILED_TO_REPRODUCE`; dependency expansion was outside this functional repair.
- Any Full Mean advantage, fold/seed variability and Harrell/Uno conflict are reported rather than converted into an architectural claim.
- No morphology-level credit assignment, new pathology architecture, pan-cancer benchmark or tri-modal experiment is authorized by this stage.

## 7. Provenance and data boundary

- `Phase1R historical`, `Phase1.5`, `Phase1.5R bugged` and `Phase1.5R2 repaired` are kept distinct.
- Raw TCGA/H5/NPY/NPZ files, checkpoints, patient-level OOF and attention remain local and ignored by Git.
- The tracked report contains only compact aggregate tables, functional evidence, audit/provenance metadata and figures.
"""
    atomic_text(out / "PHASE1_5R2_RESULTS.md", result_text)
    atomic_text(out / "PHASE1_5R2_DECISION.md", f"# Phase1.5R2 decision\n\n## {label}\n\n{explanation}\n\nScope stopped after the repaired benchmark and audit.\n")
    repro = {
        "experiment_id": EXPERIMENT_ID,
        "git_revision": benchmark.repo_git_revision(root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "raw_feature_dim": 1536,
        "main_token_budget": MAIN_TOKENS,
        "token_grid": list(TOKEN_GRID),
        "projection_grid": list(PROJECTION_GRID),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "bootstrap_replicates": BOOTSTRAPS,
        "split_hash": p15.p15.canonical_split_hash(patients),
        "uno_censoring": "outer-train censoring KM per fold",
        "cox_gradient": "correct d negative partial log-likelihood / d risk",
        "unaffected_results": "reused_from_phase1_5R_non_neural_unaffected_results",
    }
    atomic_json(out / "PHASE1_5R2_REPRODUCIBILITY.json", repro)
    atomic_text(out / "PHASE1_5R2_REPRODUCIBILITY.md", "# Phase1.5R2 reproducibility\n\nThe adjacent JSON records the corrected gradient, fresh-anchor boundary, frozen split, raw feature budget, neural rerun grids and 5000-replicate bootstrap protocol.\n")
    atomic_json(out / "PHASE1_5R2_MANIFEST.json", {"experiment_id": EXPERIMENT_ID, "created_at": utc_now(), "decision": label, "files": sorted(path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file() and path.stat().st_size < 5_000_000), "raw_data_committed": False})
    atomic_text(out / "PHASE1_5R2_FAILURE_LOG.md", "# Phase1.5R2 failure log\n\n- MambaMIL: FAILED_TO_REPRODUCE inherited from Phase1.5R.\n- CLAM survival: FAILED_TO_REPRODUCE inherited from Phase1.5R.\n- No functional hard gate failed.\n")

    report_root = root / "reports" / EXPERIMENT_ID
    report_root.mkdir(parents=True, exist_ok=True)
    compact = ["PHASE1_5R2_RESULTS.md", "PHASE1_5R2_DECISION.md", "PHASE1_5R2_REPRODUCIBILITY.md", "PHASE1_5R2_REPRODUCIBILITY.json", "PHASE1_5R2_FAILURE_LOG.md", "PHASE1_5R2_MANIFEST.json"]
    for name in compact:
        shutil.copy2(out / name, report_root / name)
    for name, source in [("main_metrics.csv", out / "tables" / "main_metrics.csv"), ("fold_metrics.csv", out / "tables" / "fold_metrics.csv"), ("token_budget_sensitivity.csv", out / "sensitivity" / "token_budget_sensitivity.csv"), ("projection_sensitivity.csv", out / "sensitivity" / "projection_sensitivity.csv"), ("paired_bootstrap.csv", out / "bootstrap" / "paired_bootstrap.csv"), ("risk_direction_sanity.csv", out / "tables" / "risk_direction_sanity.csv"), ("fresh_anchor_table.csv", out / "anchor" / "fresh_anchor_table.csv"), ("functional_summary.json", out / "unit_tests" / "functional_summary.json")]:
        if source.exists():
            shutil.copy2(source, report_root / name)
    figure_root = report_root / "figures"
    figure_root.mkdir(exist_ok=True)
    for path in (out / "figures").glob("*.png"):
        shutil.copy2(path, figure_root / path.name)
    return label, explanation


def run(root: Path, resume: bool = False) -> None:
    root = root.resolve()
    out = root_output(root)
    lock = acquire_lock(out)
    try:
        state = load_state(out, resume)
        if state.get("status") != "COMPLETE":
            state["status"] = "RUNNING"
            state["updated_at"] = utc_now()
            atomic_json(out / "run_state.json", state)
        for name in ("unit_tests", "anchor", "configs", "logs", "checkpoints", "oof_predictions", "fold_metrics", "tables", "embeddings", "attention", "figures", "bootstrap", "sensitivity"):
            (out / name).mkdir(parents=True, exist_ok=True)
        if state.get("status") == "COMPLETE" and resume:
            print(json.dumps({"status": "COMPLETE", "decision": state.get("decision"), "message": "existing complete run preserved"}, ensure_ascii=False), flush=True)
            return

        # Cohort/frozen split checks only; the full data audit is inherited,
        # exactly as required by the Phase1.5R2 prompt.
        patients, slides, split_path, _ = p15.load_frozen(root)
        if state.get("stage") in {"initializing", "functional_gate_passed"} or not (out / "configs" / "unaffected_reuse_provenance.json").exists():
            reuse_unaffected(root, out, state)
            write_protocol(out, root, patients, split_path)
            copy_old_bugged_summary(root, out)
            atomic_json(out / "configs" / "data_audit_reuse.json", {"status": "REUSED_NOT_RERUN", "source": "outputs/phase1_5r_confirmatory_repair/data_audit", "cohort_only_check": {"patients": len(patients), "events": int(patients.dss_event.sum()), "slides": len(slides)}})

        if not state.get("functional_gate", {}).get("status"):
            run_functional_gate(out, state)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gate_path = out / "anchor" / "fresh_anchor_gate.json"
        gate_status = json.loads(gate_path.read_text(encoding="utf-8")).get("status") if gate_path.exists() else None
        # A gate file is not evidence of a passing gate.  In particular, a
        # resumed process must never advance from a recorded STOP status into
        # neural jobs.  Preserve any interrupted pre-gate job as an explicit
        # audit entry before restarting the anchor stage.
        if gate_status != "PASS" and state.get("current_job") and state.get("stage") == "main_neural":
            state.setdefault("failed_jobs", []).append({"stage": "main_neural", "job": state.get("current_job"), "error": "INTERRUPTED_BEFORE_FRESH_ANCHOR_GATE; no neural result promoted", "time": utc_now()})
            state["current_job"] = None
            state["stage"] = "fresh_anchor"
            atomic_json(out / "run_state.json", state)
        if gate_status != "PASS":
            mark(state, out, stage="fresh_anchor", job="B1_Phase1R_Fresh")
            anchor, anchor_manifest = run_fresh_anchor(root, out, patients, slides, device, resume=resume)
            state["anchor_gate"] = json.loads((out / "anchor" / "fresh_anchor_gate.json").read_text(encoding="utf-8"))
            mark(state, out, stage="fresh_anchor_passed", complete=True, job="B1_Phase1R_Fresh")
        else:
            anchor = pd.read_csv(out / "anchor" / "fresh_anchor_table.csv")

        mark(state, out, stage="repaired_main_neural")
        run_repaired_neural_suite(root, out, state, patients, slides, device)
        mark(state, out, stage="repaired_token_sensitivity")
        run_token_sensitivity(root, out, state, patients, slides, device)
        mark(state, out, stage="repaired_projection_sensitivity")
        run_projection_sensitivity(root, out, state, patients, slides, device)

        methods = ["B0_full", "B0_1024", *NEURAL_METHODS, "B5_PANTHER_style_GMM"]
        main, folds, pooled = summarize(out, methods)
        token, projection = compute_sensitivity(out, pooled)
        risk_direction = add_risk_direction_columns(out, [*NEURAL_METHODS, "Token_B1_K128", "Token_B1_K256", "Token_B1_K512", "Token_Transformer_K128", "Token_Transformer_K256", "Token_Transformer_K512", "Projection_random64", "Projection_learned64", "Projection_learned512"])
        mark(state, out, stage="paired_bootstrap")
        bootstrap = p15.run_bootstrap_suite(out, main, pooled)
        mark(state, out, stage="finalize")
        make_figures(out, anchor, main, token, projection, bootstrap)
        label, explanation = write_reports(root, out, patients, split_path, anchor, main, folds, token, projection, bootstrap, risk_direction, state)
        state.update({"status": "COMPLETE", "stage": "complete", "current_job": None, "end_time": utc_now(), "decision": label, "decision_explanation": explanation, "final_artifacts": {"functional_tests": True, "fresh_anchor": True, "repaired_main_neural": True, "token_sensitivity": True, "projection_sensitivity": True, "bootstrap": True, "risk_direction_sanity": True, "reports": True, "figures": True, "data_audit_reuse_record": True}})
        atomic_json(out / "run_state.json", state)
        print(json.dumps({"status": state["status"], "decision": label, "completed_jobs": len(state["completed_jobs"]), "failed_jobs": len(state["failed_jobs"]), "reused_jobs": len(state["reused_jobs"]), "device": str(device)}, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        state = locals().get("state")
        if isinstance(state, dict):
            state.setdefault("failed_jobs", []).append({"stage": state.get("stage"), "error": f"{type(exc).__name__}: {exc}", "time": utc_now()})
            state["status"] = "FAILED"
            state["updated_at"] = utc_now()
            atomic_json(out / "run_state.json", state)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.5R2 functional Cox repair")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.root, resume=args.resume)


if __name__ == "__main__":
    main()
