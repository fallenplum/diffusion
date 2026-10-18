"""Frozen per-fold CONCH-PC accounting; no fitting, selection or bootstrap.

Every PC number refers to that fold's own train-fitted PCA basis. Equal PC
numbers from different folds are NOT identified semantic directions. Removing
a column here is an inference accounting intervention, never a refitted arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from audit_phase25_a_visual import SEEDS, FOLDS, require, digest, summary, counts, transitions
from validate_phase11_independent import ipcw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--light-root", type=Path, default=Path("/root/phase16r_light_rna_private"))
    parser.add_argument("--output", type=Path, default=ROOT/"reports/phase25_a/CONCH_PC_ATTRIBUTION.json")
    args = parser.parse_args()
    require(not args.output.exists(), "append-only PC audit refuses existing output")
    torch.set_num_threads(2)
    time, event, folds = np.full(512, np.nan), np.full(512, np.nan), np.zeros(512, int)
    qualified = []
    for seed in SEEDS:
        for fold in FOLDS:
            name = f"seed{seed}_fold{fold}"
            artifact = args.light_root/"artifacts"/(name+".pt")
            receipt = json.loads((args.light_root/"cells"/(name+".json")).read_text())
            require(receipt["status"] == "COMPLETE" and digest(artifact) == receipt["sha256"], "artifact receipt failed")
            z = torch.load(artifact, map_location="cpu", weights_only=False)
            require(z["source"] == receipt["source"] and z["seed"] == seed and z["fold"] == fold, "artifact identity changed")
            bankpath = Path(z["outer_feature_path"])
            require(digest(bankpath) == z["outer_feature_sha256"], "PC feature bank byte hash changed")
            tr, te = np.asarray(z["part"]["outer_train"]), np.asarray(z["part"]["outer_test"])
            if seed == SEEDS[0]:
                require(not np.isfinite(time[te]).any(), "overlapping OOF patient fold")
                time[te], event[te], folds[te] = z["test_time"], z["test_event"], fold
            else:
                require(np.array_equal(time[te], z["test_time"]) and np.array_equal(event[te], z["test_event"])
                        and (folds[te] == fold).all(), "cross-seed patient/endpoint mismatch")
            qualified.append((seed, fold, artifact, bankpath, tr, te))
    require(np.isfinite(time).all() and np.isin(event, [0, 1]).all() and (folds > 0).all(), "incomplete inherited OOF coverage")
    rows, concentration, basis = [], [], []
    reference_by_fold = {}
    max_replay = 0.
    for seed, fold, artifact, bankpath, tr, te in qualified:
        z = torch.load(artifact, map_location="cpu", weights_only=False)
        bank = torch.load(bankpath, map_location="cpu", weights_only=False)
        require(np.array_equal(tr, np.flatnonzero(folds != fold)) and np.array_equal(bank["fit"], tr), "non-fit-local PC bank")
        ncs = bank["x"]["CS"].shape[1]
        design = np.asarray(bank["x"]["LOCAL"], float)
        model = z["models"]["LOCAL__CV3"]
        w = np.asarray(model["w"], float)
        require(design.shape[1] == len(w) == ncs+32, "PC width changed")
        full, contributions = design[te]@w, design[te, ncs:]*w[ncs:]
        error = float(np.max(np.abs(full-model["risk"])))
        require(error <= 1e-10, "original full risk replay failed")
        max_replay = max(max_replay, error)
        weights = ipcw(time[tr], event[tr], time[te], event[te], 60)
        original = counts(time[te], event[te], full, weights)
        total_sd_energy = float(np.sum(np.var(contributions, axis=0)))
        require(total_sd_energy > 0, "all CONCH coefficients/columns are constant")
        energies = np.var(contributions, axis=0)/total_sd_energy
        order = np.argsort(-energies, kind="stable")
        concentration.append({"seed": seed, "fold": fold,
            "SD_squared_energy_sum": total_sd_energy,
            "actual_heldout_total_CONCH_risk_variance": float(contributions.sum(1).var()),
            "top1_SD_squared_energy_fraction": float(energies[order[:1]].sum()),
            "top3_SD_squared_energy_fraction": float(energies[order[:3]].sum()),
            "top5_SD_squared_energy_fraction": float(energies[order[:5]].sum()),
            "top10_SD_squared_energy_fraction": float(energies[order[:10]].sum()),
            "effective_component_count_inverse_energy_Herfindahl": float(1/np.sum(energies*energies)),
            "largest_SD_PC_numbers_in_this_fold": (order[:5]+1).tolist()})
        projector = np.asarray(bank["state"]["visual"]["LOCAL"]["components"], float)
        basis_hash = hashlib_array(projector)
        basis.append({"seed": seed, "fold": fold, "PCA_basis_sha256": basis_hash,
                      "PC_count": 32, "original_CONCH_dimension": int(projector.shape[1]),
                      "feature_bank_sha256": digest(bankpath), "artifact_sha256": digest(artifact)})
        fold_record = {"risk": full.copy(), "contributions": contributions.copy(), "basis": basis_hash}
        if fold in reference_by_fold:
            old = reference_by_fold[fold]
            require(old["basis"] == basis_hash and np.array_equal(old["risk"], full)
                    and np.array_equal(old["contributions"], contributions), "expected deterministic across-seed identity failed")
        else:
            reference_by_fold[fold] = fold_record
        for pc in range(32):
            removed = full-contributions[:, pc]
            current = counts(time[te], event[te], removed, weights)
            moved = transitions(time[te], event[te], removed, full, weights)
            row = {"seed": seed, "fold": fold, "fold_local_PC_number": pc+1,
                   "heldout_risk_contribution": summary(contributions[:, pc]),
                   "within_fold_SD_squared_energy_fraction": float(energies[pc]),
                   "removal_minus_full": {}}
            for metric in ("Harrell", "Uno60"):
                before, after = original[metric], current[metric]
                repair, damage, denominator = moved[metric]
                delta = after[0]/after[1]-before[0]/before[1]
                require(abs(delta-(repair-damage)/denominator) <= 2e-15, "PC repair/damage does not match delta")
                row["removal_minus_full"][metric] = {
                    "full_metric": float(before[0]/before[1]), "removed_metric": float(after[0]/after[1]),
                    "delta": float(delta), "repaired_pair_credit_mass": float(repair/denominator),
                    "damaged_pair_credit_mass": float(damage/denominator)}
            rows.append(row)
    unique = [r for r in rows if r["seed"] == SEEDS[0]]
    aggregate = {"unique_fold_PC_observations": len(unique), "identical_predictions_across_three_seeds": True,
                 "weighting": "five_folds_equal;three_seed_identical_records_not_independent",
                 "concentration": {key: summary([r[key] for r in concentration if r["seed"] == SEEDS[0]])
                    for key in ("top1_SD_squared_energy_fraction", "top3_SD_squared_energy_fraction",
                                "top5_SD_squared_energy_fraction", "top10_SD_squared_energy_fraction",
                                "effective_component_count_inverse_energy_Herfindahl")},
                 "removal_delta_descriptive_distribution": {metric: summary(
                      [r["removal_minus_full"][metric]["delta"] for r in unique]) for metric in ("Harrell", "Uno60")}}
    record = {"status": "PASS", "schema": "phase25_a_fold_local_CONCH_PC_frozen_accounting_v1",
        "script_sha256": digest(__file__), "cells": 15, "PC_interventions": len(rows),
        "unique_fold_PC_interventions": len(unique), "CONCH_full_risk_replay_max_error": max_replay,
        "new_training_fits": 0, "new_PCA_fits": 0, "new_neural_forward_calls": 0,
        "bootstrap_performed": False, "new_confirmatory_intervals": False,
        "PC_selection_or_refit_for_Phase25_B": False,
        "direction": "delta=metric_after_frozen_PC_removal_minus_full;negative_means_removal_harmed_ranking",
        "basis_warning": "PC_numbers_are_fold_local;equal_numbers_across_folds_are_not_same_medical_or_feature_direction",
        "variance_warning": "sum_of_individual_heldout_component_SD_squared_ignores_covariance;not_total_CONCH_risk_variance_decomposition",
        "privacy": "only_fold_seed_PC_aggregate_statistics;no_patient_arrays_or_loading_vectors_published",
        "aggregate": aggregate, "fold_local_basis_audit": basis,
        "concentration_by_cell": concentration, "PC_contributions_by_cell": rows}
    text = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n"
    require(len(text.encode()) < 2*1024**2, "public PC summary too large")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "PASS", "cells": 15, "PC_interventions": len(rows),
                      "risk_replay_error": max_replay, "report_bytes": args.output.stat().st_size}), flush=True)


def hashlib_array(values):
    import hashlib
    x = np.ascontiguousarray(values)
    return hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode()+x.tobytes()).hexdigest()


if __name__ == "__main__":
    main()
