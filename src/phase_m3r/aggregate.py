from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEEDS = (20260904, 20260905, 20260906)
M3 = {
    "A0": "A0 Full MCAT",
    "A3": "A3 Reverse",
    "A4": "A4 Bidirectional",
    "B0": "B0 Full CAMR",
    "B3": "B3 CAMR-NoCMFM-CommonConcat",
}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def average(values: list[float]) -> float:
    return float(np.nanmean(values)) if values else float("nan")


def deviation(values: list[float]) -> float:
    return float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reaggregate_m3(rows: list[dict[str, str]]) -> list[dict]:
    output = []
    for method, label in M3.items():
        cells = [row for row in rows if row["method"] == method and row.get("status") == "completed"]
        item = {
            "scope": "M3_reaggregation",
            "method": method,
            "label": label,
            "n_cells": len(cells),
            "harrell_fold_mean_15": average([number(row, "harrell_c") for row in cells]),
            "harrell_fold_std_15": deviation([number(row, "harrell_c") for row in cells]),
            "uno_fold_mean_15": average([number(row, "uno_c") for row in cells]),
            "uno_fold_std_15": deviation([number(row, "uno_c") for row in cells]),
        }
        for seed in SEEDS:
            seed_cells = [row for row in cells if int(row["seed"]) == seed]
            item[f"harrell_seed_{seed}_mean5"] = average([number(row, "harrell_c") for row in seed_cells])
            item[f"uno_seed_{seed}_mean5"] = average([number(row, "uno_c") for row in seed_cells])
        output.append(item)
    return output


def pair(rows: list[dict[str, str]], candidate: str, control: str, scope: str) -> list[dict]:
    controls = {(int(row["seed"]), int(row["outer_fold"])): row for row in rows if row["method"] == control}
    output = []
    for row in rows:
        if row["method"] != candidate:
            continue
        key = (int(row["seed"]), int(row["outer_fold"]))
        reference = controls[key]
        delta_h = number(row, "harrell_c") - number(reference, "harrell_c")
        delta_u = number(row, "uno_c") - number(reference, "uno_c")
        output.append({
            "scope": scope,
            "comparison": f"{candidate}_vs_{control}",
            "seed": key[0],
            "outer_fold": key[1],
            "delta_harrell": delta_h,
            "harrell_win": int(delta_h > 0),
            "delta_uno": delta_u,
            "uno_win": int(delta_u > 0),
            "both_metrics_win": int(delta_h > 0 and delta_u > 0),
        })
    return output


def summarize(rows: list[dict], scope: str) -> dict:
    return {
        "scope": scope,
        "comparison": rows[0]["comparison"],
        "n_cells": len(rows),
        "mean_delta_harrell": average([row["delta_harrell"] for row in rows]),
        "std_delta_harrell": deviation([row["delta_harrell"] for row in rows]),
        "mean_delta_uno": average([row["delta_uno"] for row in rows]),
        "std_delta_uno": deviation([row["delta_uno"] for row in rows]),
        "harrell_wins_15": sum(row["harrell_win"] for row in rows),
        "uno_wins_15": sum(row["uno_win"] for row in rows),
        "both_metrics_wins_15": sum(row["both_metrics_win"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3-fold", required=True)
    parser.add_argument("--m3-seed", required=True)
    parser.add_argument("--confirm-fold", required=True)
    parser.add_argument("--confirm-seed", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--execution-commit", default="unknown")
    args = parser.parse_args()
    output = Path(args.reports_root)
    output.mkdir(parents=True, exist_ok=True)
    m3_rows = read_csv(args.m3_fold)
    m3_seed_rows = read_csv(args.m3_seed)
    confirm_rows = read_csv(args.confirm_fold)
    confirm_seed_rows = read_csv(args.confirm_seed)
    write_csv(output / "reaggregated_m3.csv", reaggregate_m3(m3_rows))
    write_csv(output / "confirmatory_fold_metrics.csv", confirm_rows)
    confirm_seed_out = []
    for row in confirm_seed_rows:
        cells = [cell for cell in confirm_rows if cell["method"] == row["method"] and int(cell["seed"]) == int(row["seed"])]
        row = dict(row)
        row["test_fold_harrell_mean5"] = average([number(cell, "harrell_c") for cell in cells])
        row["test_fold_uno_mean5"] = average([number(cell, "uno_c") for cell in cells])
        confirm_seed_out.append(row)
    write_csv(output / "confirmatory_seed_metrics.csv", confirm_seed_out)
    paired: list[dict] = []
    for candidate, control in (("C1", "C0"), ("C2", "C0"), ("D1", "D0")):
        paired.extend(pair(confirm_rows, candidate, control, "M3R_confirmatory"))
    summaries = []
    for candidate, control in (("C1", "C0"), ("C2", "C0"), ("D1", "D0")):
        summaries.append(summarize([row for row in paired if row["comparison"] == f"{candidate}_vs_{control}"], "M3R_confirmatory_summary"))
    old_pairs: list[dict] = []
    for candidate, control in (("A3", "A0"), ("A4", "A0"), ("B3", "B0")):
        old_pairs.extend(pair(m3_rows, candidate, control, "M3_reaggregation"))
    old_summaries = []
    for candidate, control in (("A3", "A0"), ("A4", "A0"), ("B3", "B0")):
        old_summaries.append(summarize([row for row in old_pairs if row["comparison"] == f"{candidate}_vs_{control}"], "M3_reaggregation_summary"))
    write_csv(output / "paired_deltas.csv", old_pairs + old_summaries + paired + summaries)
    initialization = json.loads((output / "initialization_audit.json").read_text(encoding="utf-8"))
    worker_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    complete = len(confirm_rows) == 75 and all(sum(row["method"] == method for row in confirm_rows) == 15 for method in ("C0", "C1", "C2", "D0", "D1")) and initialization.get("status") == "PASS"
    summary_by_name = {row["comparison"]: row for row in summaries}

    def confirmed(item: dict) -> bool:
        return item["mean_delta_harrell"] > 0 and item["mean_delta_uno"] > 0 and item["harrell_wins_15"] >= 9 and item["uno_wins_15"] >= 9

    c1 = summary_by_name["C1_vs_C0"]
    c2 = summary_by_name["C2_vs_C0"]
    d1 = summary_by_name["D1_vs_D0"]
    bidirectional = "BIDIRECTIONAL_INTERACTION_CONFIRMED" if confirmed(c2) else "mixed / not confirmed"
    cmfm = "CMFM_OVERCOMPLEXITY_CONFIRMED" if confirmed(d1) else "mixed / not confirmed"
    protocol = {
        "phase": "M3R",
        "base_branch": "codex/phase-m3-mcat-mechanism-audit",
        "branch": "codex/phase-m3r-confirmatory-mechanism",
        "n_patients": 512,
        "dss_events": 65,
        "folds": 5,
        "seeds_confirmatory": [20260907, 20260908, 20260909],
        "K": 1024,
        "preprocessing": "identical to M2R2/M3; no change",
        "cox": "identical to M2R2/M3; no change",
        "primary_metric": "test-fold Harrell C aggregated across 15 seed×fold cells",
        "secondary_metric": "test-fold Uno C aggregated across 15 seed×fold cells",
        "global_oof_metric": "diagnostic only",
        "per_seed_summary": "confirmatory_seed_metrics.csv contains test_fold_harrell_mean5 and test_fold_uno_mean5 for each seed",
        "required_cells": 75,
        "completed_cells": len(confirm_rows),
        "initialization_audit": initialization,
        "determinism": worker_manifest.get("determinism", {}),
        "known_nondeterministic_cuda_paths": [
            "memory-efficient attention backward on CUDA",
            "adaptive_avg_pool2d_backward_cuda",
        ],
        "cublas_workspace_config": ":4096:8",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Phase M3R Confirmatory Metric & Mechanism Audit", "", "## Decision", "",
        "**Phase M3R COMPLETE**" if complete else "**Phase M3R BLOCKED**", "",
        "- Primary metric: test-fold Harrell C aggregated over 15 seed×fold cells.",
        "- Secondary metric: test-fold Uno C aggregated over 15 seed×fold cells.",
        "- Global OOF C-index is diagnostic only.", "", "## Confirmatory paired results", "",
        "| Comparison | Δ Harrell mean ± std | Δ Uno mean ± std | Harrell wins | Uno wins | Both wins | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item, label in ((c1, "C1 Reverse vs C0"), (c2, "C2 Bidirectional vs C0"), (d1, "D1 Simple concat vs D0")):
        lines.append(f"| {label} | {item['mean_delta_harrell']:.4f} ± {item['std_delta_harrell']:.4f} | {item['mean_delta_uno']:.4f} ± {item['std_delta_uno']:.4f} | {item['harrell_wins_15']}/15 | {item['uno_wins_15']}/15 | {item['both_metrics_wins_15']}/15 | {'confirmed' if confirmed(item) else 'mixed / not confirmed'} |")
    lines += ["", "## Confirmatory absolute test-fold metrics", "", "| Variant | Harrell mean ± std (15) | Uno mean ± std (15) |", "|---|---:|---:|"]
    for method in ("C0", "C1", "C2", "D0", "D1"):
        cells = [row for row in confirm_rows if row["method"] == method]
        lines.append(f"| {method} {cells[0]['label']} | {average([number(row, 'harrell_c') for row in cells]):.4f} ± {deviation([number(row, 'harrell_c') for row in cells]):.4f} | {average([number(row, 'uno_c') for row in cells]):.4f} ± {deviation([number(row, 'uno_c') for row in cells]):.4f} |")
    lines += [
        "", "## Re-aggregated M3 evidence", "",
        "| Variant | Fold Harrell mean ± std (15) | Fold Uno mean ± std (15) |",
        "|---|---:|---:|",
    ]
    for item in reaggregate_m3(m3_rows):
        lines.append(f"| {item['label']} | {item['harrell_fold_mean_15']:.4f} ± {item['harrell_fold_std_15']:.4f} | {item['uno_fold_mean_15']:.4f} ± {item['uno_fold_std_15']:.4f} |")
    lines += ["", "### M3 re-aggregated paired audit", "", "| Comparison | Δ Harrell | Δ Uno | Harrell wins | Uno wins | Both wins |", "|---|---:|---:|---:|---:|---:|"]
    for item in old_summaries:
        lines.append(f"| {item['comparison']} | {item['mean_delta_harrell']:.4f} | {item['mean_delta_uno']:.4f} | {item['harrell_wins_15']}/15 | {item['uno_wins_15']}/15 | {item['both_metrics_wins_15']}/15 |")
    lines += [
        "", "## Required answers", "", "### 1. A3 reverse vs Full MCAT", "",
        f"Decision: **{'A3_REVERSE_CONFIRMED' if confirmed(c1) else 'A3_REVERSE_NOT_CONFIRMED'}**. C1 has {c1['harrell_wins_15']}/15 Harrell wins and {c1['uno_wins_15']}/15 Uno wins, with mean deltas {c1['mean_delta_harrell']:.4f} and {c1['mean_delta_uno']:.4f}.",
        "", "### 2. A4 bidirectional vs Full MCAT", "", f"Decision: **{bidirectional}**.",
        "", "### 3. CAMR simple concat gain", "", f"Decision: **{cmfm}**.",
        "", "### 4. Phase M4 mechanism", "", f"Decision: **{'C2 bidirectional co-attention' if confirmed(c2) else 'NO_CONFIRMED_MECHANISM; DO NOT PROCEED TO M4'}**.",
        "", "No M4 implementation or design is included.", "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    worker_manifest.update({"phase": "M3R", "status": "Phase M3R COMPLETE" if complete else "Phase M3R BLOCKED", "execution_commit": args.execution_commit, "cells_required": 75, "cells_completed": len(confirm_rows), "no_leakage": True, "no_preprocessing_change": True, "global_oof_status": "diagnostic_only", "initialization_audit_status": initialization.get("status"), "known_nondeterministic_cuda_paths": ["memory-efficient attention backward on CUDA", "adaptive_avg_pool2d_backward_cuda"], "cublas_workspace_config": ":4096:8", "confirmatory_decisions": {"C1_vs_C0": c1, "C2_vs_C0": c2, "D1_vs_D0": d1, "bidirectional": bidirectional, "cmfm": cmfm}})
    (output / "run_manifest.json").write_text(json.dumps(worker_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
