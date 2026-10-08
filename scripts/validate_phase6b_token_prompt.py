"""Independently reconstruct every seed's OOF and verify repaired controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase6b_token_prompt_titan import TOKEN_METHODS, ENDPOINTS, metrics, normalize_label, sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--clinical-targets", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    done = json.loads((out / "DONE.json").read_text())
    assert done["status"] == "PASS"
    assert done["config_sha256"] == sha256(out / "config.json")
    config = json.loads((out / "config.json").read_text())
    assert config["inner_folds"] == 0 and config["schedule"]["max_epochs"] == 3
    gate = json.loads((out / "encoder_audit.json").read_text())
    assert gate["status"] == "PASS" and gate["text_transformer_trainable_parameters"] == 0
    assert 0 <= gate["no_prompt_official_max_abs_error"] <= 2e-4
    assert gate["prompt_grad_norm"] > 0 and gate["shifted_mask_different_positions"] > 0
    ids = np.load(args.cache / "patient_representations.npz", allow_pickle=False)["patient_ids"].astype(str)
    folds = pd.read_csv(args.folds).set_index("patient_id").loc[ids, "fold"].to_numpy()
    targets = pd.read_csv(args.clinical_targets, dtype=str, usecols=["patient_id", *ENDPOINTS]).set_index("patient_id").loc[ids]
    rows = pd.read_csv(out / "metrics.csv").set_index(["seed", "endpoint", "method"])
    original = pd.read_csv(args.baseline_source / "metrics.csv").set_index(["seed", "endpoint", "method"])
    pairs = pd.read_csv(out / "paired_comparisons_by_seed.csv")
    max_error = 0.0
    baseline_cells = 0
    checked_files = 0
    fit_audits = 0
    methods = (*TOKEN_METHODS, "S4_wrong_patient_rna_token_prompt")
    for seed in config["seeds"]:
        for endpoint in ENDPOINTS:
            normalized = [normalize_label(v, endpoint) for v in targets[endpoint]]
            classes = sorted({v for v in normalized if v is not None})
            mapping = {v: index for index, v in enumerate(classes)}
            y = np.array([mapping[v] if v is not None else -1 for v in normalized])
            valid = y >= 0
            for method in methods:
                p = np.full((len(ids), len(classes)), np.nan)
                visited = np.zeros(len(ids), dtype=int)
                for fold in range(1, 6):
                    test = np.flatnonzero((folds == fold) & valid)
                    train = np.flatnonzero((folds != fold) & valid)
                    name = f"{endpoint}__{method}__seed{seed}__fold{fold}.npz"
                    with np.load(out / "oof" / name, allow_pickle=False) as cell:
                        assert np.array_equal(cell["patient_ids"].astype(str), ids[test]), name
                        assert np.array_equal(cell["y_true"], y[test]), name
                        assert np.array_equal(cell["classes"], classes), name
                        assert int(cell["selected_epoch"]) == 3, name
                        probabilities = cell["probabilities"]
                        assert probabilities.shape == (len(test), len(classes)), name
                        assert np.isfinite(probabilities).all() and (probabilities >= 0).all(), name
                        assert np.allclose(probabilities.sum(1), 1, atol=1e-5), name
                        p[test] = probabilities
                    visited[test] += 1
                    checked_files += 1
                    if method in {TOKEN_METHODS[0], TOKEN_METHODS[3]}:
                        assert sha256(out / "oof" / name) == sha256(args.baseline_source / "oof" / name), name
                        baseline_cells += 1
                    if method in {TOKEN_METHODS[1], TOKEN_METHODS[2]}:
                        audit = json.loads((out / "audits" / name.replace(".npz", ".json")).read_text())
                        assert audit["fit_n"] == len(train) and audit["selected_dim"] == 512, name
                        from hashlib import sha256 as digest
                        assert audit["fit_indices_sha256"] == digest(train.astype(np.int64).tobytes()).hexdigest(), name
                        assert [row["epoch"] for row in audit["training_history"]] == [1, 2, 3], name
                        assert np.isfinite([row["mean_batch_train_ce"] for row in audit["training_history"]]).all(), name
                        if method == TOKEN_METHODS[2]:
                            assert audit["wrong_patient_fixed_points"] == 0, name
                        fit_audits += 1
                assert (visited[valid] == 1).all() and (visited[~valid] == 0).all()
                recomputed = metrics(y[valid], p[valid], len(classes))
                stored = rows.loc[seed, endpoint, method]
                for metric, value in recomputed.items():
                    max_error = max(max_error, abs(value - stored[metric]))
                if method in {TOKEN_METHODS[0], TOKEN_METHODS[3]}:
                    # v1's binary AUROC was missing; check valid original metrics.
                    for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
                        assert abs(stored[metric] - original.loc[(seed, endpoint, method), metric]) < 1e-10
            for row in pairs[(pairs.seed == seed) & (pairs.endpoint == endpoint)].itertuples(index=False):
                right = {"S2-S1": methods[1], "S2-S3": methods[3], "S2-S4": methods[4]}[row.comparison]
                expected = rows.loc[(seed, endpoint, methods[2]), "balanced_accuracy"] - rows.loc[(seed, endpoint, right), "balanced_accuracy"]
                assert abs(expected - row.delta_balanced_accuracy) < 1e-10
                assert row.bootstrap_scheme == "class_stratified_paired_patient"
    assert checked_files == 300 and baseline_cells == 120 and fit_audits == 120
    assert max_error <= 1e-10
    report = {"status": "PASS", "oof_files": checked_files, "patient_count": len(ids),
              "valid_reused_baseline_cells": baseline_cells, "fold_local_fit_audits": fit_audits,
              "max_recomputed_metric_error": max_error,
              "OOF_patient_exactly_once_per_seed_endpoint_control": True,
              "S4_predictions_persisted": True, "wrong_patient_fixed_points": 0,
              "survival_columns_read": [], "encoder_gate": gate}
    (out / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
