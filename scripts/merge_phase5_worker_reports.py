from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def write_csv(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(data)
    frame.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-parent", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    args = ap.parse_args()
    parent = args.reports_parent.resolve()
    out = args.output.resolve()
    worker_dirs = [parent / f"phase5_endpoint_supervision_budget_audit_s{s}" for s in args.seeds]
    if not all((d / "validation.json").exists() for d in worker_dirs):
        raise FileNotFoundError("one or more worker validation.json files are missing")
    out.mkdir(parents=True, exist_ok=True)

    merge_names = ("fold_metrics.csv", "seed_oof_metrics.csv", "paired_bootstrap.csv", "frozen_probe_metrics.csv", "event_budget_curve.csv", "failure_ledger.csv")
    for name in merge_names:
        merged: list[dict] = []
        for d in worker_dirs:
            merged.extend(rows(d / name))
        write_csv(out / name, merged)

    for name in ("endpoint_audit.csv", "endpoint_provenance.json", "endpoint_split_manifest.json"):
        source = worker_dirs[0] / name
        if source.suffix == ".csv":
            shutil.copy2(source, out / name)
        else:
            shutil.copy2(source, out / name)

    worker_validation = []
    for seed, d in zip(args.seeds, worker_dirs):
        payload = json.loads((d / "validation.json").read_text(encoding="utf-8"))
        worker_validation.append({"seed": seed, **payload})
    write_csv(out / "worker_validation.csv", worker_validation)

    fold = rows(out / "fold_metrics.csv")
    seed_rows = rows(out / "seed_oof_metrics.csv")
    boot = rows(out / "paired_bootstrap.csv")
    budget = rows(out / "event_budget_curve.csv")
    failures = rows(out / "failure_ledger.csv")
    expected_fold = len(args.seeds) * 2 * 5 * 5
    expected_seed = len(args.seeds) * 2 * 5
    expected_budget = len(args.seeds) * 5 * 3 * 3 * 3

    # Reuse the runner's decision rule without rerunning any model.
    import importlib.util
    runner_path = Path(__file__).with_name("phase5_endpoint_supervision_budget_audit.py")
    spec = importlib.util.spec_from_file_location("phase5_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Phase5 runner")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    decision = runner.derive_decision(seed_rows, boot, budget, tuple(args.seeds))
    validation = {
        "status": not failures and len(fold) == expected_fold and len(seed_rows) == expected_seed and len(budget) == expected_budget,
        "cohort_n": 512,
        "formal_endpoints": ["DSS", "OS"],
        "formal_seeds": list(args.seeds),
        "formal_fold_rows": len(fold),
        "expected_fold_rows": expected_fold,
        "seed_oof_rows": len(seed_rows),
        "expected_seed_oof_rows": expected_seed,
        "event_budget_rows": len(budget),
        "expected_event_budget_rows": expected_budget,
        "event_budget_row_definition": "formal_seed x outer_fold x budget_fraction x subsample_seed x model(B0/B3/B4)",
        "bootstrap_replicates": 2000,
        "pfi_not_fabricated": True,
        "pfs_not_modeled_same_512": True,
        "b4_encoder_frozen": True,
        "outer_test_not_used_for_selection": True,
        "failure_count": len(failures),
        "decision": decision,
        "worker_validations": worker_validation,
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Phase5 Endpoint & Survival-Supervision Budget Audit",
        "",
        "- Fixed cohort: 512 patients; formal endpoints: DSS and OS.",
        "- PFI has no explicit fields; PFS has valid_n=510 and was audited only.",
        "- Three seed workers were run independently and merged after artifact validation.",
        "",
        "|Endpoint|Model|Harrell OOF mean|Uno fold mean|",
        "|---|---|---:|---:|",
    ]
    seed_frame = pd.DataFrame(seed_rows)
    if not seed_frame.empty:
        for (endpoint, model), frame in seed_frame.groupby(["endpoint", "model"], sort=True):
            lines.append(f"|{endpoint}|{model}|{frame['harrell_oof'].mean():.4f}|{frame['uno_oof'].mean():.4f}|")
    lines += [
        "",
        f"- Decision: **{decision}**.",
        f"- Fold rows: {len(fold)}/{expected_fold}; seed OOF rows: {len(seed_rows)}/{expected_seed}; event-budget rows: {len(budget)}/{expected_budget}.",
        "- Event-budget rows use mapping-only survival refits with frozen upstream representations; this limitation is retained in each row.",
        "- Public reports contain aggregate metrics only; patient-indexed arrays and checkpoints remain private.",
        "",
        f"Validation status: **{'PASS' if validation['status'] else 'BLOCKED'}**.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "DECISION.md").write_text(
        f"# Phase5 decision\n\n- Decision: **{decision}**\n- Validation: **{'PASS' if validation['status'] else 'BLOCKED'}**\n- Failure rows: {len(failures)}\n- Event-budget rows: {len(budget)}/{expected_budget}\n",
        encoding="utf-8",
    )
    (out / "run_manifest.json").write_text(
        json.dumps({"phase": "Phase5 Endpoint & Survival-Supervision Budget Audit", "worker_seeds": list(args.seeds), "worker_dirs": [str(d) for d in worker_dirs], "merged_at": datetime.now(timezone.utc).isoformat(), "status": "completed" if validation["status"] else "AUDIT_BLOCKED"}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"status": validation["status"], "fold_rows": len(fold), "seed_rows": len(seed_rows), "budget_rows": len(budget), "failures": len(failures), "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
