from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from phase1_neural import (
    DEVICE,
    BagStore,
    EncoderConfig,
    SurvivalModel,
    deterministic_embeddings,
)
from phase1_utils import censoring_km, fit_cox, harrell_c, scaler_pca, set_seed, uno_c, update_run_state


SEED = 20260723


def load_survival(root: Path, model_name: str, fold: int, seed: int):
    path = (
        root
        / "outputs/pathology_context_phase1/checkpoints/survival"
        / model_name
        / f"fold_{fold}"
        / f"seed_{seed}.pt"
    )
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    config = EncoderConfig(**bundle["encoder_config"])
    model = SurvivalModel(config)
    model.load_state_dict(bundle["model_state"])
    return model.to(DEVICE).eval(), bundle


def fit_common_probe(root: Path, model_name: str, fold: int, seed: int):
    phase = root / "outputs/pathology_context_phase1"
    out_name = f"{model_name}_probe"
    pred_path = phase / "oof_predictions/fold_parts" / out_name / f"fold_{fold}_seed_{seed}.csv"
    if pred_path.exists():
        print(f"[skip] {pred_path}", flush=True)
        return

    set_seed(seed + fold)
    patients = pd.read_csv(phase / "audit/patient_manifest_used.csv")
    slides = pd.read_csv(phase / "audit/slide_manifest_used.csv")
    train = patients.loc[patients.outer_fold != fold].copy().reset_index(drop=True)
    test = patients.loc[patients.outer_fold == fold].copy().reset_index(drop=True)
    model, bundle = load_survival(root, model_name, fold, seed)
    store = BagStore(root, slides)
    x_train = deterministic_embeddings(model.encoder, store, train.patient_id.astype(str))
    x_test = deterministic_embeddings(model.encoder, store, test.patient_id.astype(str))

    times = train.dss_months.to_numpy(float)
    events = train.dss_status.to_numpy(int)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold)
    candidates = []
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        scores = []
        for ti, vi in inner.split(x_train, events):
            xt, xv, _, _ = scaler_pca(x_train[ti], x_train[vi], 32, seed + fold)
            beta = fit_cox(times[ti], events[ti], xt, alpha)
            scores.append(harrell_c(times[vi], events[vi], xv @ beta))
        candidates.append((float(np.nanmean(scores)), -alpha, alpha))
    candidates.sort(reverse=True)
    inner_c, _, alpha = candidates[0]
    xt, xv, scaler, pca = scaler_pca(x_train, x_test, 32, seed + fold)
    beta = fit_cox(times, events, xt, alpha)
    train_risk = xt @ beta
    test_risk = xv @ beta
    censor = censoring_km(times, events)
    g = censor(test.dss_months.to_numpy(float), left_limit=True)

    result = {
        "model": out_name,
        "source_representation": model_name,
        "outer_fold": fold,
        "seed": seed,
        "alpha": alpha,
        "inner_harrell_c": inner_c,
        "test_harrell_c": harrell_c(test.dss_months, test.dss_status, test_risk),
        "test_uno_c": uno_c(test.dss_months, test.dss_status, test_risk, g),
        "n_parameters_encoder": sum(p.numel() for p in model.encoder.parameters()),
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred = test[["patient_id", "outer_fold", "dss_months", "dss_status"]].copy()
    pred["seed"] = seed
    pred["oof_risk"] = test_risk
    pred["censor_survival_outer_train"] = g
    pred.to_csv(pred_path, index=False)

    emb_dir = phase / "embeddings" / out_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb = pred.copy()
    emb["embedding"] = [json.dumps(row.tolist()) for row in x_test]
    emb.to_csv(emb_dir / f"fold_{fold}_seed_{seed}.csv", index=False)

    train_dir = phase / "calibration/train_risk" / out_name
    train_dir.mkdir(parents=True, exist_ok=True)
    train_out = train[["patient_id", "outer_fold", "dss_months", "dss_status"]].copy()
    train_out["seed"] = seed
    train_out["risk"] = train_risk
    train_out.to_csv(train_dir / f"fold_{fold}_seed_{seed}.csv", index=False)

    metric_dir = phase / "fold_metrics" / out_name
    metric_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / f"fold_{fold}_seed_{seed}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    update_run_state(root, f"{out_name}/fold_{fold}/seed_{seed}", "stage_4_representation_probes")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", required=True, choices=["B1", "B2", "P1", "P2", "P3"])
    parser.add_argument("--fold", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    fit_common_probe(args.root.resolve(), args.model, args.fold, args.seed)


if __name__ == "__main__":
    main()
