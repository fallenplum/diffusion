from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase1_neural import (
    DEVICE,
    BagStore,
    SurvivalModel,
    evaluate_head,
    load_ssl_encoder,
    patient_split,
    train_survival_candidate,
)
from phase1_utils import censoring_km, harrell_c, set_seed, uno_c, update_run_state


SEED = 20260723


def permute_outcomes(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    pairs = out[["dss_months", "dss_status"]].to_numpy()[rng.permutation(len(out))]
    out[["dss_months", "dss_status"]] = pairs
    return out


def run(root: Path, fold: int, seed: int):
    phase = root / "outputs/pathology_context_phase1"
    method = "P2_permuted"
    part = phase / "oof_predictions/fold_parts" / method / f"fold_{fold}_seed_{seed}.csv"
    if part.exists():
        print(f"[skip] {part}", flush=True)
        return
    set_seed(seed + fold)
    patients = pd.read_csv(phase / "audit/patient_manifest_used.csv")
    slides = pd.read_csv(phase / "audit/slide_manifest_used.csv")
    inner_train, inner_val, outer_test = patient_split(patients, fold, seed)
    rng = np.random.default_rng(seed + 900000 + fold)
    perm_train = permute_outcomes(inner_train, rng)
    perm_val = permute_outcomes(inner_val, rng)
    pretrained, config = load_ssl_encoder(root, "spatial040", fold, seed)
    base = SurvivalModel(config)
    base.encoder.load_state_dict(pretrained.state_dict())
    initial = copy.deepcopy(base.state_dict())
    store = BagStore(root, slides)
    candidates, history = [], []
    started = time.perf_counter()
    for lr in [1e-4, 3e-4]:
        model = SurvivalModel(config).to(DEVICE)
        model.load_state_dict(initial)
        val_c, epoch, state, rows = train_survival_candidate(
            model, store, perm_train, perm_val, "cox_rank", lr, fold, seed + 700000
        )
        candidates.append((val_c, -lr, lr, epoch, state))
        history.extend(rows)
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
    val_c, _, lr, epoch, state = candidates[0]
    model = SurvivalModel(config).to(DEVICE)
    model.load_state_dict(state)
    z, risk = evaluate_head(model, store, outer_test)
    outer_train = patients.loc[patients.outer_fold != fold]
    km = censoring_km(outer_train.dss_months.to_numpy(float), outer_train.dss_status.to_numpy(int))
    g = km(outer_test.dss_months.to_numpy(float), left_limit=True)
    result = {
        "model": method, "source": "P2", "seed": seed, "outer_fold": fold,
        "permutation": "(dss_months,dss_status) pairs permuted independently inside inner_train and inner_val",
        "best_lr": lr, "best_epoch": epoch, "permuted_inner_val_harrell_c": val_c,
        "true_test_harrell_c": harrell_c(outer_test.dss_months, outer_test.dss_status, risk),
        "true_test_uno_c": uno_c(outer_test.dss_months, outer_test.dss_status, risk, g),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "training_time_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if DEVICE.type == "cuda" else 0,
    }
    part.parent.mkdir(parents=True, exist_ok=True)
    pred = outer_test[["patient_id", "outer_fold", "dss_months", "dss_status"]].copy()
    pred["seed"] = seed; pred["oof_risk"] = risk; pred["censor_survival_outer_train"] = g
    pred.to_csv(part, index=False)
    metric_dir = phase / "fold_metrics" / method
    metric_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(metric_dir / f"fold_{fold}_seed_{seed}_training.csv", index=False)
    (metric_dir / f"fold_{fold}_seed_{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    ckpt = phase / "checkpoints/survival" / method / f"fold_{fold}" / f"seed_{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": {k: v.cpu() for k, v in state.items()}, "encoder_config": asdict(config), "result": result}, ckpt)
    update_run_state(root, f"{method}/fold_{fold}/seed_{seed}", "stage_5_sanity")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args.root.resolve(), args.fold, args.seed)


if __name__ == "__main__":
    main()
