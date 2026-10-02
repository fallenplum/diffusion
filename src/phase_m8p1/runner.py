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

from .clinical import Config, REPAIR_VERSION, nested_crossfit
from .residual import fit_residual


SEEDS = (20261012, 20261013, 20261014, 20261015, 20261016)
VARIANTS = ("SP", "SO", "SPO")


def git_sha(repo):
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def metric(raw, test, train, pred):
    h = harrell_c(raw.dss_time[test], raw.dss_event[test], pred)
    u, num, den = uno_c_index(raw.dss_time[test], raw.dss_event[test], pred, raw.dss_time[train], raw.dss_event[train])
    return {"harrell_c": float(h), "uno_c": float(u), "uno_numerator": float(num), "uno_denominator": float(den)}


def run(args):
    repo = Path(args.repo_root)
    raw = load_raw_inputs(Path(args.data_root), repo)
    runtime = Path(args.runtime)
    source = Path(args.source_workers)
    out = Path(args.worker)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    deterministic = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
    train_pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r")
    seed = int(args.seed)
    cfg = Config(max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size)
    fold_rows = []
    profiles = []
    crossfit_audit = []
    diagnostics = []
    source_hashes = {}
    for fold in range(1, 6):
        arrays = preprocess_fold(raw, train_pool, fold)
        provider = PathTokenProvider(train_pool, deterministic, train_seed=seed, k=1024)
        test = np.flatnonzero(raw.outer_folds == fold)
        outer_train = np.flatnonzero(raw.outer_folds != fold)
        path = source / f"seed_{seed}" / "cells" / f"C0-Full_{seed}_{fold}.npy"
        anchor_test = np.asarray(np.load(path), dtype=np.float64)
        source_hashes[f"{seed}/{fold}"] = subprocess.check_output(["sha256sum", str(path)], text=True).split()[0]
        if anchor_test.shape != (len(test),) or not np.isfinite(anchor_test).all():
            raise RuntimeError(f"invalid C0-Full outer-test prediction: {path}")
        exact_test_offsets = np.full(len(raw.patient_ids), np.nan, dtype=np.float64)
        exact_test_offsets[test] = anchor_test
        s0 = {"variant": "S0", "seed": seed, "outer_fold": fold, "n_test": len(test), "events_test": int(raw.dss_event[test].sum()), **metric(raw, test, outer_train, anchor_test), "params": 0, "status": "frozen_exact_c0"}
        fold_rows.append(s0)
        np.save(out / "cells" / f"S0_{seed}_{fold}.npy", anchor_test)
        # One nested, label-safe Clinical cross-fit is shared by all residual variants.
        crossfit_offset, rows, _lineage = nested_crossfit(raw, outer_train, test, seed, fold, device, cfg, REPAIR_VERSION)
        crossfit_audit.extend(rows)
        for variant in VARIANTS:
            pred, profile, diag = fit_residual(variant, raw, arrays, provider, outer_train, test, fold, seed, device, cfg, crossfit_offset, exact_test_offsets)
            profile["clinical_crossfit_version"] = REPAIR_VERSION
            row = {"variant": variant, "seed": seed, "outer_fold": fold, "n_test": len(test), "events_test": int(raw.dss_event[test].sum()), **metric(raw, test, outer_train, pred), "params": profile["params"], "best_epoch": profile["best_epoch"], "status": "completed"}
            fold_rows.append(row); profiles.append(profile); diagnostics.extend(diag)
            np.save(out / "cells" / f"{variant}_{seed}_{fold}.npy", pred)
            atomic_json(out / "cells" / f"{variant}_{seed}_{fold}.json", row)
        del arrays, provider, crossfit_offset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_csv(out / "fold_metrics.csv", fold_rows)
    atomic_csv(out / "compute_profile.csv", profiles)
    atomic_csv(out / "clinical_crossfit_audit.csv", crossfit_audit)
    atomic_csv(out / "residual_diagnostics.csv", diagnostics)
    atomic_json(out / "worker_manifest.json", {"phase": "M8-P1R", "repair_version": REPAIR_VERSION, "seed": seed, "variants": ["S0", *VARIANTS], "cells_completed": len(fold_rows), "cells_expected": 20, "execution_sha": git_sha(repo), "source_c0_sha256": source_hashes, "status": "complete", "device": str(device)})
    print(json.dumps({"seed": seed, "cells": len(fold_rows), "device": str(device)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True); p.add_argument("--data-root", required=True); p.add_argument("--runtime", required=True); p.add_argument("--source-workers", required=True); p.add_argument("--worker", required=True); p.add_argument("--seed", required=True); p.add_argument("--max-epochs", type=int, default=25); p.add_argument("--patience", type=int, default=5); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--cpu", action="store_true")
    run(p.parse_args())

