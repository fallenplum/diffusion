from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.phase_m2.common import atomic_csv, atomic_json, harrell_c, uno_c_index
from src.phase_m2.data import load_raw_inputs, preprocess_fold
from src.phase_m3r.runner import PathTokenProvider
from src.phase_m8p0.runner import crossfit_anchor, fit_residual


SEEDS = (20261012, 20261013, 20261014, 20261015, 20261016)
VARIANTS = ("SP", "SO", "SPO")


def git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _metric(raw, test_idx: np.ndarray, train_idx: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    h = harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], pred)
    u, num, den = uno_c_index(raw.dss_time[test_idx], raw.dss_event[test_idx], pred, raw.dss_time[train_idx], raw.dss_event[train_idx])
    return {"harrell_c": float(h), "uno_c": float(u), "uno_numerator": float(num), "uno_denominator": float(den)}


def run(args: argparse.Namespace) -> None:
    repo = Path(args.repo_root)
    raw = load_raw_inputs(Path(args.data_root), repo)
    runtime = Path(args.runtime)
    workers = Path(args.source_workers)
    out = Path(args.worker)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    det = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
    pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r")
    seed = int(args.seed)
    fold_rows: list[dict] = []
    profiles: list[dict] = []
    crossfit_rows: list[dict] = []
    diagnostics: list[dict] = []
    for fold in range(1, 6):
        arrays = preprocess_fold(raw, pool, fold)
        provider = PathTokenProvider(pool, det, train_seed=seed, k=1024)
        test_idx = np.flatnonzero(raw.outer_folds == fold)
        train_idx = np.flatnonzero(raw.outer_folds != fold)
        source_path = workers / f"seed_{seed}" / "cells" / f"C0-Full_{seed}_{fold}.npy"
        anchor_test = np.asarray(np.load(source_path), dtype=np.float64)
        if anchor_test.shape != (len(test_idx),) or not np.isfinite(anchor_test).all():
            raise RuntimeError(f"invalid frozen C0-Full prediction: {source_path}")
        m = _metric(raw, test_idx, train_idx, anchor_test)
        s0 = {"variant": "S0", "seed": seed, "outer_fold": fold, "n_test": len(test_idx), "events_test": int(raw.dss_event[test_idx].sum()), **m, "params": 0, "status": "frozen_exact_c0"}
        fold_rows.append(s0)
        np.save(out / "cells" / f"S0_{seed}_{fold}.npy", anchor_test)
        atomic_json(out / "cells" / f"S0_{seed}_{fold}.json", s0)
        # One identical clinical cross-fit is shared by SP, SO and SPO.
        cf, test_offset, audit = crossfit_anchor(arrays.clinical, raw.dss_time, raw.dss_event, train_idx, test_idx, seed, fold, device)
        if not np.isfinite(cf).all() or not np.isfinite(test_offset).all():
            raise RuntimeError(f"non-finite Clinical cross-fit offset at seed={seed}, fold={fold}")
        crossfit_rows.extend(audit)
        for variant in VARIANTS:
            internal_variant = {"SP": "RP", "SO": "RO", "SPO": "RPO"}[variant]
            pred, profile, _, drows = fit_residual(
                internal_variant,
                raw,
                arrays,
                provider,
                fold,
                seed,
                device,
                args,
                cf,
                test_offset,
                audit,
            )
            profile["variant"] = variant
            for row in drows:
                row["variant"] = variant
            metric = _metric(raw, test_idx, train_idx, pred)
            row = {"variant": variant, "seed": seed, "outer_fold": fold, "n_test": len(test_idx), "events_test": int(raw.dss_event[test_idx].sum()), **metric, "params": int(profile["params"]), "best_epoch": int(profile["best_epoch"]), "status": "completed"}
            fold_rows.append(row)
            profiles.append(profile)
            diagnostics.extend(drows)
            np.save(out / "cells" / f"{variant}_{seed}_{fold}.npy", pred)
            atomic_json(out / "cells" / f"{variant}_{seed}_{fold}.json", row)
        del arrays, provider, cf, test_offset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_csv(out / "fold_metrics.csv", fold_rows)
    atomic_csv(out / "compute_profile.csv", profiles)
    atomic_csv(out / "clinical_crossfit_audit.csv", crossfit_rows)
    atomic_csv(out / "residual_diagnostics.csv", diagnostics)
    atomic_json(out / "worker_manifest.json", {"phase": "M8-P1", "seed": seed, "variants": ["S0", *VARIANTS], "cells_completed": len(fold_rows), "cells_expected": 20, "execution_sha": git_sha(repo), "status": "complete", "device": str(device)})
    print(json.dumps({"seed": seed, "cells": len(fold_rows), "device": str(device)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--source-workers", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())

