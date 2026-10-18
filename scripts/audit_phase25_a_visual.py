"""Append-only CPU checkpoint accounting for existing CONCH32 and C4 models.

No fitting, patch encoding, neural prediction, teacher construction or model
selection. Existing protected OOF endpoints are read only for descriptive
within-fold ranking accounting. Public output contains aggregate cells only.
Frozen component removal is not an estimate of retraining necessity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from validate_phase11_independent import pairs, ipcw

SEEDS, FOLDS = (31, 47, 83), (1, 2, 3, 4, 5)


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def summary(values):
    x = np.asarray(values, float).reshape(-1)
    require(len(x) > 0 and np.isfinite(x).all(), "nonfinite/empty anonymous summary")
    return {"count": len(x), "mean": float(x.mean()), "population_SD": float(x.std()),
            "RMS": float(np.sqrt(np.mean(x*x))), "minimum": float(x.min()), "maximum": float(x.max())}


def credits(time, event, risk):
    i, j = pairs(time, event)
    d = np.asarray(risk, float)[i]-np.asarray(risk, float)[j]
    return i, np.where(np.abs(d) <= 1e-8, .5, (d > 0).astype(float))


def counts(time, event, risk, weight):
    i, c = credits(time, event, risk)
    return {"Harrell": [float(c.sum()), len(c)],
            "Uno60": [float(np.dot(weight[i], c)), float(weight[i].sum())]}


def transitions(time, event, newer, older, weight):
    i, a = credits(time, event, newer)
    ii, b = credits(time, event, older)
    require(np.array_equal(i, ii), "component comparison pairs differ")
    d = a-b
    return {"Harrell": [float(d.clip(min=0).sum()), float((-d).clip(min=0).sum()), len(d)],
            "Uno60": [float(np.dot(weight[i], d.clip(min=0))),
                      float(np.dot(weight[i], (-d).clip(min=0))), float(weight[i].sum())]}


def add_counts(accumulator, name, current):
    target = accumulator.setdefault(name, {"Harrell": np.zeros(2), "Uno60": np.zeros(2)})
    for metric, values in current.items():
        target[metric] += values


def add_transitions(accumulator, name, current):
    target = accumulator.setdefault(name, {"Harrell": np.zeros(3), "Uno60": np.zeros(3)})
    for metric, values in current.items():
        target[metric] += values


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def existing_evidence():
    report = ROOT/"reports/phase16r_light_baseline_rna_compact"
    chosen_models = {"CS__CV3", "LOCAL__CV3", "GLOBAL_MATCHED__CV3", "GLOBAL_ALL__CV3",
                     "WRONG_LOCAL__CV3", "CS__FIXED1", "LOCAL__FIXED1", "LOCAL_BLOCK",
                     "CS_AT_LOCAL_LAMBDA", "LOCAL_AT_CS_LAMBDA", "C4_REF"}
    summary_rows = [r for r in read_csv(report/"SUMMARY.csv")
                    if r["endpoint"] == "DSS" and r["model"] in chosen_models]
    comparisons = [r for r in read_csv(report/"PAIRED_COMPARISONS.csv")
                   if r["endpoint"] == "DSS" and r["family"] in {"clinical_primary", "clinical_mechanism"}]
    transition_rows = [r for r in read_csv(report/"PAIR_TRANSITIONS.csv")
                       if r["endpoint"] == "DSS" and r["family"] in {"clinical_primary", "clinical_mechanism"}]
    fixed = read_csv(ROOT/"reports/phase12_c4_confirmation/FIXED_MODEL_SENSITIVITY_SUMMARY.csv")
    return {"historical_CONCH_summary": summary_rows,
            "historical_confirmatory_and_mechanism_intervals": comparisons,
            "historical_repair_damage_credit_mass": transition_rows,
            "historical_C4_fixed_RNA_sensitivity": fixed,
            "source_files_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in (
                report/"SUMMARY.csv", report/"PAIRED_COMPARISONS.csv", report/"PAIR_TRANSITIONS.csv",
                ROOT/"reports/phase12_c4_confirmation/FIXED_MODEL_SENSITIVITY_SUMMARY.csv")}}


def audit(light_root, c4_root, phase12_root):
    receipts, all_time, all_event, all_fold = [], np.full(512, np.nan), np.full(512, np.nan), np.zeros(512, int)
    # Verify the full original C4 checkpoint/OOF/preprocessing before reading
    # component tensors. Each label vector is inherited from an existing OOF.
    for seed in SEEDS:
        for fold in FOLDS:
            stem = f"C4_seed{seed}_fold{fold}"
            cellpath = c4_root/"formal/cells"/(stem+".json")
            cell = json.loads(cellpath.read_text())
            ck, oof = c4_root/"formal/checkpoints"/(stem+".pt"), c4_root/"formal/oof"/(stem+".npz")
            manifest = c4_root/"formal/manifests"/(stem+".json")
            require(cell["status"] == "COMPLETE" and cell["arm"] == "C4" and
                    cell["seed"] == seed and cell["fold"] == fold and not cell["outer_test_selection"],
                    "original C4 cell identity changed")
            require(digest(ck) == cell["checkpoint_sha256"] and digest(oof) == cell["prediction_sha256"]
                    and digest(manifest) == cell["manifest_sha256"], "original C4 byte receipt failed")
            pp = c4_root/"preprocessing"/f"fold{fold}_outer.npz"
            pm = json.loads(pp.with_suffix(".json").read_text())
            require(digest(pp) == pm["npz_sha256"], "original C4 preprocessing byte receipt failed")
            with np.load(oof) as z:
                ids, times, events = z["test_indices"], z["time"], z["event"]
                if seed == SEEDS[0]:
                    require(not np.isfinite(all_time[ids]).any(), "C4 OOF folds overlap")
                    all_time[ids], all_event[ids], all_fold[ids] = times, events, fold
                else:
                    require(np.array_equal(all_time[ids], times) and np.array_equal(all_event[ids], events)
                            and (all_fold[ids] == fold).all(), "cross-seed original OOF identity mismatch")
            reused = phase12_root/"formal/oof"/f"R0_128_seed{seed}_fold{fold}.npz"
            with np.load(reused) as a, np.load(oof) as b:
                require(all(np.array_equal(a[k], b[k]) for k in ("test_indices", "risk", "time", "event")),
                        "R0_128 does not exactly reuse original C4 OOF")
            receipts.append((seed, fold, cell, ck, oof, pp, manifest))
    require(np.isfinite(all_time).all() and np.isin(all_event, [0, 1]).all() and (all_fold > 0).all(),
            "original OOF is not complete 512 patient coverage")
    cells, aggregate_metrics, aggregate_pairs = [], {}, {}
    max_conch_replay, checkpoint_coverage = 0., {}
    for seed, fold, c4_cell, checkpoint, prediction, preprocessing, manifestpath in receipts:
        stem = f"seed{seed}_fold{fold}"
        artifact = light_root/"artifacts"/(stem+".pt")
        receipt = json.loads((light_root/"cells"/(stem+".json")).read_text())
        require(receipt["status"] == "COMPLETE" and digest(artifact) == receipt["sha256"],
                "light baseline checkpoint receipt failed")
        saved = torch.load(artifact, map_location="cpu", weights_only=False)
        require(saved["source"] == receipt["source"] and saved["seed"] == seed and saved["fold"] == fold,
                "light baseline checkpoint identity changed")
        bankpath = Path(saved["outer_feature_path"])
        require(digest(bankpath) == saved["outer_feature_sha256"], "light baseline feature bank mutated")
        bank = torch.load(bankpath, map_location="cpu", weights_only=False)
        train, test = (np.asarray(saved["part"][k], int) for k in ("outer_train", "outer_test"))
        require(np.array_equal(test, np.flatnonzero(all_fold == fold)) and
                np.array_equal(train, np.flatnonzero(all_fold != fold)) and np.array_equal(bank["fit"], train),
                "light baseline and C4 patient folds differ")
        require(np.array_equal(saved["test_time"], all_time[test]) and np.array_equal(saved["test_event"], all_event[test]),
                "light baseline and C4 endpoint vectors differ")
        time, event = all_time[test], all_event[test]
        weight = ipcw(all_time[train], all_event[train], time, event, 60)
        xcs, xall = np.asarray(bank["x"]["CS"], float), np.asarray(bank["x"]["LOCAL"], float)
        ncs = xcs.shape[1]
        require(xall.shape[1] == ncs+32 and np.array_equal(xcs, xall[:, :ncs]), "CONCH branch columns changed")
        local, base = saved["models"]["LOCAL__CV3"], saved["models"]["CS__CV3"]
        lw, bw = np.asarray(local["w"], float), np.asarray(base["w"], float)
        full, frozen_cs, conch = xall[test]@lw, xcs[test]@lw[:ncs], xall[test, ncs:]@lw[ncs:]
        error = float(np.max(np.abs(full-local["risk"])))
        require(error <= 1e-10, "CONCH32 raw-feature risk replay failed")
        max_conch_replay = max(max_conch_replay, error)
        original_cs = xcs[test]@bw
        require(np.max(np.abs(original_cs-base["risk"])) <= 1e-10 and
                np.max(np.abs(full-(frozen_cs+conch))) <= 1e-12, "linear component accounting failed")
        # Existing C4 state saves BOTH the Ridge-initialized and jointly
        # optimized anchor. No neural patch replay is repeated for this audit.
        archive = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(archive["arm"] == "C4" and archive["seed"] == seed and
                archive["epochs"] == c4_cell["selected_epoch"], "C4 checkpoint architecture changed")
        with np.load(preprocessing) as pp:
            c4_x = np.concatenate((pp["f_C"], pp["f_S"]), 1).astype(np.float32)
        after_w, initial_w = archive["state"]["anchor.weight"], archive["initial_state"]["anchor.weight"]
        require(after_w.shape == initial_w.shape == (1, c4_x.shape[1]), "C4 anchor state missing/dimension changed")
        after_anchor = torch.nn.functional.linear(torch.as_tensor(c4_x[test]), after_w).flatten().double().numpy()
        initial_anchor = torch.nn.functional.linear(torch.as_tensor(c4_x[test]), initial_w).flatten().double().numpy()
        with np.load(prediction) as p:
            c4_full = np.asarray(p["risk"], float)
        c4_residual = c4_full-after_anchor
        risk = {"CONCH_FULL": full, "CONCH_FROZEN_CS_ONLY": frozen_cs,
                "CS_RETRAINED_REFERENCE": original_cs,
                "C4_FULL": c4_full, "C4_OPTIMIZED_ANCHOR_ONLY": after_anchor,
                "C4_INITIAL_RIDGE_ANCHOR": initial_anchor}
        comparisons = (("CONCH_FULL", "CONCH_FROZEN_CS_ONLY"),
                       ("CONCH_FROZEN_CS_ONLY", "CS_RETRAINED_REFERENCE"),
                       ("CONCH_FULL", "CS_RETRAINED_REFERENCE"),
                       ("C4_FULL", "C4_OPTIMIZED_ANCHOR_ONLY"),
                       ("C4_OPTIMIZED_ANCHOR_ONLY", "C4_INITIAL_RIDGE_ANCHOR"),
                       ("C4_FULL", "C4_INITIAL_RIDGE_ANCHOR"))
        for name, current in risk.items():
            add_counts(aggregate_metrics.setdefault(seed, {}), name, counts(time, event, current, weight))
        for newer, older in comparisons:
            add_transitions(aggregate_pairs.setdefault(seed, {}), newer+"__MINUS__"+older,
                            transitions(time, event, risk[newer], risk[older], weight))
        cells.append({"seed": seed, "fold": fold, "patients": len(test),
                      "CONCH_model": {"CS_lambda": base["lambda"], "LOCAL_lambda": local["lambda"],
                        "frozen_CS_component": summary(frozen_cs), "CONCH_component": summary(conch),
                        "CS_reoptimization_change": summary(frozen_cs-original_cs),
                        "CONCH_coefficient_L2": float(np.linalg.norm(lw[ncs:])),
                        "CS_coefficient_change_L2": float(np.linalg.norm(lw[:ncs]-bw)),
                        "exact_raw_feature_risk_replay_max_error": error},
                      "C4_model": {"selected_epoch": archive["epochs"],
                        "initial_anchor_component": summary(initial_anchor),
                        "optimized_anchor_component": summary(after_anchor),
                        "anchor_optimization_change": summary(after_anchor-initial_anchor),
                        "deep_residual_from_saved_total_minus_current_anchor": summary(c4_residual),
                        "anchor_coefficient_change_L2": float((after_w-initial_w).double().norm()),
                        "full_neural_replay_repeated": False,
                        "CPU_float32_linear_risk_accounting": True},
                      "file_hashes": {"CONCH_artifact": digest(artifact), "CONCH_feature_bank": digest(bankpath),
                                      "C4_checkpoint": digest(checkpoint), "C4_prediction": digest(prediction),
                                      "C4_preprocessing": digest(preprocessing), "C4_manifest": digest(manifestpath)}})
    seed_metrics, seed_pairs = [], []
    for seed in SEEDS:
        for name, record in aggregate_metrics[seed].items():
            seed_metrics.append({"seed": seed, "model": name,
                **{metric: float(v[0]/v[1]) for metric, v in record.items()}})
        for name, record in aggregate_pairs[seed].items():
            for metric, (repair, damage, denominator) in record.items():
                seed_pairs.append({"seed": seed, "comparison": name, "metric": metric,
                                   "repaired_pair_credit_mass": float(repair/denominator),
                                   "damaged_pair_credit_mass": float(damage/denominator),
                                   "net_pair_credit_change": float((repair-damage)/denominator)})
    names = sorted({row["model"] for row in seed_metrics})
    model_summary = [{"model": name, **{metric+"_mean": float(np.mean([
        row[metric] for row in seed_metrics if row["model"] == name])) for metric in ("Harrell", "Uno60")}}
        for name in names]
    pair_summary = []
    for name in sorted({r["comparison"] for r in seed_pairs}):
        for metric in ("Harrell", "Uno60"):
            rows = [r for r in seed_pairs if r["comparison"] == name and r["metric"] == metric]
            pair_summary.append({"comparison": name, "metric": metric,
                **{key: float(np.mean([row[key] for row in rows])) for key in
                   ("repaired_pair_credit_mass", "damaged_pair_credit_mass", "net_pair_credit_change")},
                "positive_seeds": sum(r["net_pair_credit_change"] > 0 for r in rows)})
    for name, folder, pattern in (
        ("CONCH_light_artifacts", light_root/"artifacts", "seed*_fold*.pt"),
        ("CONCH_light_feature_banks", light_root/"features", "*.pt"),
        ("original_C4_checkpoints", c4_root/"formal/checkpoints", "C4_seed*_fold*.pt"),
        ("original_C4_OOF", c4_root/"formal/oof", "C4_seed*_fold*.npz"),
        ("C4_real128_OOF_aliases", phase12_root/"formal/oof", "R0_128_seed*_fold*.npz"),
        ("C4_Null128_checkpoints", phase12_root/"formal/checkpoints", "N_128_seed*_fold*.pt"),
        ("C4_Wrong128_checkpoints", phase12_root/"formal/checkpoints", "W_128_seed*_fold*.pt"),
        ("C4_NoAnchor128_checkpoints", phase12_root/"formal/checkpoints", "J_128_seed*_fold*.pt")):
        checkpoint_coverage[name] = len(list(folder.glob(pattern)))
    return {"status": "PASS", "cells_checked": len(cells), "new_training_fits": 0,
            "new_patch_or_neural_forward_calls": 0,
            "existing_protected_OOF_endpoints_read_for_descriptive_pairs": True,
            "no_patient_arrays_saved_or_published": True,
            "all_original_C4_OOF_exactly_equal_Phase12_R0_128": True,
            "CONCH_raw_feature_risk_replay_max_error": max_conch_replay,
            "checkpoint_coverage": checkpoint_coverage,
            "scope": "posthoc_frozen_prediction_accounting;not_retrained_necessity_or_causal_attribution",
            "metric_protocol": "within_fold_num_den_summed_then_three_seed_mean;no_cross_fold_risk_ranking_or_risk_ensemble",
            "bootstrap_or_new_confirmatory_interval": False,
            "C4_deep_residual_scope": "all_deep_paths_together;RNA_main_effect_and_RNA_patch_retrieval_not_separately_identified",
            "CONCH_CS_scope": "CS_contains_Clinical_and_S32;not_Clinical_only",
            "cells": cells, "seed_metrics": seed_metrics, "model_summary": model_summary,
            "paired_component_credit_summary": pair_summary, "existing_evidence": existing_evidence()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--light-root", type=Path, default=Path("/root/phase16r_light_rna_private"))
    parser.add_argument("--c4-root", type=Path, default=Path("/root/phase11_semantic_anchor_private"))
    parser.add_argument("--phase12-root", type=Path, default=Path("/root/phase12_c4_private"))
    parser.add_argument("--output", type=Path, default=ROOT/"reports/phase25_a/VISUAL_ATTRIBUTION.json")
    args = parser.parse_args()
    require(not args.output.exists(), "append-only audit refuses to overwrite an existing report")
    torch.set_num_threads(2)
    record = audit(args.light_root, args.c4_root, args.phase12_root)
    record.update(schema="phase25_a_existing_visual_checkpoint_accounting_v1", script_sha256=digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n"
    require(len(text.encode()) < 2*1024**2, "aggregate output exceeds publication size budget")
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"status": record["status"], "cells": record["cells_checked"],
                      "CONCH_replay_error": record["CONCH_raw_feature_risk_replay_max_error"],
                      "report_bytes": args.output.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
