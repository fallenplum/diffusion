"""Read-only CDEI-G1 history/risk replay and streamed local-message diagnosis.

No fitting or endpoint method is called. The inherited Data constructor does
temporarily load/copy protected endpoints; that limitation is explicitly
reported. Only its endpoint-free feature_view plus frozen anchor SD survives
the constructor. No patient arrays, tokens, coordinates, or weights are saved.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.phase23_cdei.data import Data
from src.phase23_cdei.warmup import _seed
from src.phase24_cdei_g1.model import G1Model
from src.phase24_cdei_g1 import train as training
from src.phase24_cdei_g1.runtime import (ARMS, PRIVATE, atomic_json, cfg, sha, source, verify_done)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


class Moments:
    """Numerically stable scalar aggregates; retain no observation vectors."""
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0
        self.lo, self.hi = math.inf, -math.inf

    def add(self, values):
        x = torch.as_tensor(values).detach().double().reshape(-1)
        if not x.numel():
            return
        require(torch.isfinite(x).all(), "nonfinite diagnostic scalar")
        n, mean = x.numel(), float(x.mean())
        m2, lo, hi = float(x.var(unbiased=False))*n, float(x.min()), float(x.max())
        if not self.n:
            self.n, self.mean, self.m2 = n, mean, m2
        else:
            count, difference = self.n+n, mean-self.mean
            self.m2 += m2 + difference*difference*self.n*n/count
            self.mean += difference*n/count
            self.n = count
        self.lo, self.hi = min(self.lo, lo), max(self.hi, hi)

    def result(self):
        if not self.n:
            return {"count": 0, "defined": False, "reason": "NO_VALID_OBSERVATIONS"}
        variance = max(self.m2/self.n, 0.0)
        result = {"count": int(self.n), "mean": self.mean,
                  "population_SD": math.sqrt(variance),
                  "RMS": math.sqrt(variance+self.mean*self.mean),
                  "minimum": self.lo, "maximum": self.hi,
                  "maximum_absolute": max(abs(self.lo), abs(self.hi))}
        require(all(math.isfinite(v) for v in result.values()), "diagnostic moment overflow")
        return result


class Stream:
    def __init__(self):
        self.values = {}

    def add(self, name, value):
        self.values.setdefault(name, Moments()).add(value)

    def result(self):
        return {name: metric.result() for name, metric in sorted(self.values.items())}


def summary(values):
    aggregate = Moments()
    aggregate.add(values)
    return aggregate.result()


def array(value, name):
    value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    require(value.ndim == 1 and value.size > 0 and np.isfinite(value).all(), "invalid " + name)
    return np.asarray(value, np.float64)


def history_summary(record):
    rows, audit = record["history"], record["training_audit"]
    require(len(rows) == audit["actual_epochs"] == audit["optimizer_steps"] == 80,
            "checkpoint did not complete the locked 80 updates")
    require([row["epoch"] for row in rows] == list(range(1, 81)), "epoch history has gaps")
    require(audit["fixed_final_epoch_no_validation_selection"] and audit["heldout_endpoint_reads"] == 0
            and audit["anchor_unchanged"] and audit["reference_reencoded_each_forward"]
            and not audit["reference_gradient_detached"], "training isolation/reference receipt changed")
    train = np.array([row["training_loss"] for row in rows], np.float64)
    fitted = np.array([row["fit_loss"] for row in rows], np.float64)
    require(np.isfinite(train).all() and np.isfinite(fitted).all(), "nonfinite saved objective")
    groups = tuple(rows[0]["gradients"])
    require(all(tuple(row["gradients"]) == groups for row in rows), "gradient groups changed")
    gradients = {}
    for group in groups:
        values = np.array([row["gradients"][group] for row in rows], np.float64)
        gradients[group] = {**summary(values), "first": float(values[0]), "last": float(values[-1]),
                            "zero_epochs": int((values == 0).sum())}
    return {"actual_epochs": 80, "optimizer_steps": 80, "fixed_final_epoch": True,
            "training_loss_first": float(train[0]), "training_loss_last": float(train[-1]),
            "fit_loss_first_updated": float(fitted[0]), "fit_loss_last_updated": float(fitted[-1]),
            "anchor_only_fit_loss": float(audit["anchor_only_fit_loss"]),
            "fit_loss_improvement_over_anchor": float(audit["anchor_only_fit_loss"]-fitted[-1]),
            "last10_fit_loss_improvement": float(fitted[-11]-fitted[-1]),
            "last10_relative_fit_loss_improvement": float((fitted[-11]-fitted[-1])/max(abs(fitted[-11]), 1e-12)),
            "last10_fit_loss_slope": float(np.polyfit(np.arange(10), fitted[-10:], 1)[0]),
            "gradient_groups": gradients,
            "global_gradient_norm": summary([row["grad_norm"] for row in rows]),
            "fit_correction_SD": summary([row["fit_correction_SD"] for row in rows]),
            "epoch_seconds": summary([row["epoch_seconds"] for row in rows]),
            "parameter_counts": dict(record["parameter_counts"]),
            "kernel_parameters_changed": audit["initial_backbone_sha256"] != audit["final_backbone_sha256"],
            "generalization_or_global_convergence_established": False,
            "scope": "training_objective_trends_only;fixed_budget_is_not_a_convergence_proof"}


def _blocked_endpoint_or_fit(*args, **kwargs):
    raise RuntimeError("endpoint access/fitting is forbidden in the mechanism diagnostic")


def safe_features(fold, role, seed, device):
    """Destroy the legacy endpoint-bearing wrapper after extracting its view."""
    loaded = Data(fold, role, seed, device=device)
    safe = SimpleNamespace(**vars(loaded.feature_view), anchor_sd_fit=float(loaded.anchor_sd_fit))
    audit = {"legacy_loader_temporarily_loads_protected_endpoints": bool(
                    loaded.audit["legacy_loader_temporarily_loads_protected_endpoints"]),
             "fit_patient_count": len(safe.fit), "heldout_patient_count": len(safe.valid)}
    del loaded
    require(not any(hasattr(safe, key) for key in ("_fit_supervision", "_selection_supervision", "ep", "d",
                                                   "fit_labels", "selection_labels")),
            "endpoint-bearing feature object escaped into diagnosis")
    return safe, audit


def restore_model(record, device):
    model = G1Model(int(record["clinical_dim"]), record["membership"], arm=record["_arm"], tau=1.0)
    model.load_state_dict(record["model_state"], strict=True)
    model.to(device).eval()
    return model


def check_replay(model, ds, record):
    ref = training.reference_raw(ds)
    result = {}
    for partition, key in (("fit", "fit_risk"), ("valid", "risk")):
        reproduced, correction, _ = training._forward(model, ds, np.asarray(getattr(ds, partition)), ref)
        original = array(record[key], "stored risk")
        actual = array(reproduced, "replayed risk")
        require(actual.shape == original.shape, "risk replay patient shape changed")
        error = float(np.max(np.abs(actual-original)))
        require(error <= 1e-7, "stored risk replay exceeded 1e-7 absolute error")
        result[partition] = {"patients": len(actual), "maximum_absolute_replay_error": error,
                             "total_correction": summary(correction),
                             "correction_over_fit_anchor_SD": summary(correction/ds.anchor_sd_fit)}
    if model.arm == "LOCAL":
        require(set(record["sensitivities"]) == {"OFF", "NO_C", "WRONG_C", "WRONG_O"},
                "saved LOCAL interventions differ")
        matched = array(record["risk"], "LOCAL matched risk")
        result["frozen_heldout_interventions"] = {}
        for condition in ("OFF", "NO_C", "WRONG_C", "WRONG_O"):
            risk, _, _ = training._forward(model, ds, np.asarray(ds.valid), ref, intervention=condition)
            expected = array(record["sensitivities"][condition], "stored frozen intervention")
            actual = array(risk, "replayed frozen intervention")
            error = float(np.max(np.abs(actual-expected)))
            require(error <= 1e-7, "frozen intervention replay exceeded 1e-7")
            result["frozen_heldout_interventions"][condition] = {
                "maximum_absolute_replay_error": error,
                "matched_minus_intervened_risk": summary(matched-actual),
                "matched_minus_intervened_over_fit_anchor_SD": summary((matched-actual)/ds.anchor_sd_fit)}
    return result


def support(ds, ids):
    # CUDA has no int64 addmm path. Boolean program-count sums are at most
    # 512 and therefore exactly represented in float32; only >0 is used.
    member = ds.ft["membership"].to(torch.float32)
    r = (ds.ft["RM"][ids].to(torch.float32) @ member.T) > 0
    c = (ds.ft["CM"][ids].to(torch.float32) @ member.T) > 0
    return torch.cat((r, c, c), -1)


def add_local_metrics(stream, model, output, patch_mask):
    d = output["details"]
    valid = d["absolute_valid_pairs"]
    rows = valid.any(-1)
    tokens = valid.any(-2)
    difference, modulation, weights = d["D"], d["signed_modulation"], d["weighted_weights"]
    stream.add("valid_edge_D", difference[valid])
    stream.add("valid_edge_D_over_tau", difference[valid]/model.tau)
    stream.add("valid_edge_tanh_D", modulation[valid])
    stream.add("valid_edge_tanh_abs_ge_0_95_indicator", (modulation[valid].abs() >= .95).float())
    stream.add("valid_edge_tanh_abs_ge_0_99_indicator", (modulation[valid].abs() >= .99).float())
    stream.add("valid_edge_matched_A", d["A"][valid])
    stream.add("valid_edge_signed_weight", weights[valid])
    stream.add("valid_edge_context_score", d["context_scores"][valid])
    stream.add("valid_edge_current_reference_score", d["null"][valid])
    stream.add("valid_reference_slot_donor_count", d["reference_counts"][tokens])
    positive, negative = weights.clamp_min(0).sum(-1), -weights.clamp_max(0).sum(-1)
    absolute_mass = positive+negative
    stream.add("valid_patch_signed_weight_mass", weights.sum(-1)[rows])
    stream.add("valid_patch_positive_weight_mass", positive[rows])
    stream.add("valid_patch_negative_weight_mass", negative[rows])
    stream.add("valid_patch_absolute_weight_mass", absolute_mass[rows])
    nonzero = rows & (absolute_mass > 0)
    stream.add("valid_patch_signed_over_absolute_mass", (weights.sum(-1)/absolute_mass.clamp_min(1e-12))[nonzero])
    stream.add("valid_patch_local_message_L2", d["weightedmessages"].norm(dim=-1)[rows])
    stream.add("valid_context_token_value_L2", d["context_omics_values"].norm(dim=-1)[tokens])
    stream.add("valid_context_token_embedding_L2", d["context_omics_tokens"].norm(dim=-1)[tokens])
    stream.add("valid_patch_eligible_slot_count", valid.sum(-1)[patch_mask])
    stream.add("WSI_extra_message_L2", output["extra"].norm(dim=-1))
    stream.add("WSI_relative_feature_L2", output["relative"].norm(dim=-1))
    h_relative = model.relative_linear(output["relative"])
    h_extra = model.extra_linear(output["extra"])
    relative_logit = model.output(model.activation(h_relative)).squeeze(-1)
    full_logit = model.output(model.activation(h_relative+h_extra)).squeeze(-1)
    require(torch.equal(output["risk_delta_unit"], torch.tanh(full_logit)),
            "same-head extra contribution reconstruction changed")
    stream.add("WSI_relative_head_hidden_L2", h_relative.norm(dim=-1))
    stream.add("WSI_extra_head_hidden_L2", h_extra.norm(dim=-1))
    stream.add("WSI_extra_over_relative_hidden_L2", h_extra.norm(dim=-1)/h_relative.norm(dim=-1).clamp_min(1e-12))
    stream.add("WSI_nonlinear_extra_contribution_before_final_tanh", full_logit-relative_logit)
    stream.add("WSI_extra_contribution_unit_risk", torch.tanh(full_logit)-torch.tanh(relative_logit))
    stream.add("WSI_total_unit_risk_correction", output["risk_delta_unit"])


def add_sensitivity_metrics(stream, baseline, intervened, anchor_sd):
    b, x = baseline["details"], intervened["details"]
    require(torch.equal(b["A"], x["A"]) and torch.equal(baseline["relative"], intervened["relative"]),
            "frozen intervention altered matched attention or relative features")
    require(torch.equal(b["reference_counts"], x["reference_counts"])
            and torch.equal(b["reference_tokens"], x["reference_tokens"]),
            "frozen intervention changed paired reference composition")
    valid = b["absolute_valid_pairs"] & x["absolute_valid_pairs"]
    rows = valid.any(-1)
    stream.add("paired_valid_edge_D_change", (b["D"]-x["D"])[valid])
    stream.add("paired_valid_edge_tanh_change", (b["signed_modulation"]-x["signed_modulation"])[valid])
    stream.add("paired_valid_edge_signed_weight_change", (b["weighted_weights"]-x["weighted_weights"])[valid])
    stream.add("paired_valid_patch_local_message_change_L2",
               (b["weightedmessages"]-x["weightedmessages"]).norm(dim=-1)[rows])
    stream.add("paired_WSI_extra_change_L2", (baseline["extra"]-intervened["extra"]).norm(dim=-1))
    change = baseline["risk_delta_unit"]-intervened["risk_delta_unit"]
    stream.add("paired_WSI_unit_risk_change", change)
    stream.add("paired_WSI_scaled_risk_change", change*anchor_sd)


def local_stream_diagnosis(model, ds):
    """Sixteen-WSI streams; heldout and fit paired controls use the same plan."""
    ref = training.reference_raw(ds)
    result = {}
    for partition in ("fit", "valid"):
        selected = np.flatnonzero(np.isin(ds.owner, np.asarray(getattr(ds, partition))))
        conditions = {name: Stream() for name in ("MATCHED", "NO_C", "WRONG_C", "WRONG_O")}
        sensitivities = {name: Stream() for name in ("NO_C", "WRONG_C", "WRONG_O")}
        for start in range(0, len(selected), 16):
            slides = selected[start:start+16]
            owners = np.asarray(ds.owner)[slides]
            donors_o, donors_c = np.asarray(ds.wrong_o)[owners], np.asarray(ds.wrong_c)[owners]
            common = support(ds, owners) & support(ds, donors_o)
            plan = np.stack((owners, donors_o), -1)
            ft = ds.ft
            args = (ds.visual[slides], ds.mask[slides], ft["C"][owners],
                    *(ft[k][owners] for k in ("R", "G", "L", "RM", "CM")), owners, ref)
            shared = {"reference_exclusions": plan, "absolute_token_mask": common, "return_details": True}
            baseline = model(*args, **shared)
            add_local_metrics(conditions["MATCHED"], model, baseline, ds.mask[slides])
            variants = {
                "NO_C": {"absolute_clinical": ft["C"][np.array(ds.fit, copy=True)].mean(0).expand(len(owners), -1)},
                "WRONG_C": {"absolute_clinical": ft["C"][donors_c]},
                "WRONG_O": {"absolute_omics": tuple(ft[k][donors_o] for k in ("R", "G", "L", "RM", "CM")),
                            "omics_source_indices": donors_o}}
            for name, options in variants.items():
                modified = model(*args, **shared, **options)
                add_local_metrics(conditions[name], model, modified, ds.mask[slides])
                add_sensitivity_metrics(sensitivities[name], baseline, modified, ds.anchor_sd_fit)
                del modified
            del baseline
        result[partition] = {
            "WSI_count": len(selected), "patients": len(getattr(ds, partition)), "stream_WSI_batch_size": 16,
            "reference_exclusions": "recipient_plus_wrong_omics_donor_union_for_all_four_conditions",
            "shared_absolute_support": "recipient_intersect_wrong_omics_donor;matched_A_not_renormalized",
            "same_as_stored_prediction_reference_plan": partition == "valid",
            "fit_pairing_reference_caveat": "FIT_mechanism_pairs_exclude_two_patients;stored_FIT_risk_replay_uses_original_LOO",
            "weighting": "edge_patch_token_or_WSI_weighted_as_named;not_patient_equal_in_mechanism_summaries",
            "conditions": {name: stream.result() for name, stream in conditions.items()},
            "matched_minus_intervened": {name: stream.result() for name, stream in sensitivities.items()},
            "relative_and_A_unchanged_under_all_interventions": True,
            "paired_reference_tokens_and_counts_identical": True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, default=PRIVATE)
    parser.add_argument("--role", choices=("inner", "outer"), default="inner")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/phase24_cdei_g1/MECHANISM_AUDIT.json")
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite an existing diagnostic report")
    identity, config = source(), cfg()
    matrix = [(int(seed), int(fold)) for fold in config["folds"] for seed in config["seeds"]]
    require(len(matrix) == len(set(matrix)) == 15, "mechanism audit requires all 15 locked cells")
    cells = []
    # All byte hashes and receipts are checked before loading ANY checkpoint.
    paths = []
    for seed, fold in matrix:
        directory = args.private_root/args.role/f"seed{seed}_fold{fold}"
        verify_done(directory, identity, args.role, seed, fold)
        paths.append((seed, fold, directory))
    Data.fit_labels = _blocked_endpoint_or_fit
    Data.selection_labels = _blocked_endpoint_or_fit
    training.train_arm = _blocked_endpoint_or_fit
    training.run_cell = _blocked_endpoint_or_fit
    with torch.no_grad():
        for seed, fold, directory in paths:
            _seed(seed, fold)
            saved = torch.load(directory/"checkpoint.pt", map_location="cpu", weights_only=False)
            require(all(saved.get(key) == val for key, val in
                        (("source", identity), ("role", args.role), ("seed", seed), ("fold", fold))),
                    "checkpoint identity mismatch")
            require(set(saved["arms"]) == set(ARMS), "checkpoint arms changed")
            ds, loader_audit = safe_features(fold, args.role, seed, args.device)
            for partition in ("fit", "valid"):
                np.testing.assert_array_equal(saved[partition], getattr(ds, partition))
                current = array(ds.ft["ANCHOR"][np.array(getattr(ds, partition), copy=True)], "frozen anchor")
                require(np.array_equal(current, array(saved["anchor_"+partition], "stored frozen anchor")),
                        "frozen patient anchor changed")
            summaries = {}
            initial = []
            for arm in ARMS:
                record = saved["arms"][arm]
                summaries[arm] = {"training": history_summary(record)}
                initial.append(record["training_audit"]["initial_backbone_sha256"])
                record["_arm"] = arm  # In-memory discriminator only; checkpoint bytes remain read-only.
                model = restore_model(record, args.device)
                require(sum(p.numel() for p in model.parameters()) == record["parameter_counts"]["total"],
                        "restored parameter budget changed")
                summaries[arm]["patient_risk_replay"] = check_replay(model, ds, record)
                if arm == "LOCAL":
                    summaries[arm]["local_mechanism"] = local_stream_diagnosis(model, ds)
                del model
            require(len(set(initial)) == 1, "arm initial compatibility backbones were not identical")
            require(summaries["LOCAL"]["training"]["parameter_counts"] ==
                    summaries["CTRL"]["training"]["parameter_counts"], "LOCAL/CTRL budget mismatch")
            cells.append({"seed": seed, "fold": fold, "role": args.role,
                          "feature_loader_audit": loader_audit,
                          "fit_anchor_population_SD": ds.anchor_sd_fit, "arms": summaries,
                          "all_common_initial_backbones_identical": True})
            print(f"DIAGNOSED seed={seed} fold={fold}", flush=True)
            del saved, ds
            gc.collect()
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    require(source() == identity, "formal source changed during the diagnostic")
    for seed, fold, directory in paths:
        verify_done(directory, identity, args.role, seed, fold)
    record = {"schema": "phase24_cdei_g1_endpoint_unused_streamed_mechanism_v1", "status": "PASS",
              "source": identity, "diagnostic_script_sha256": sha(__file__), "role": args.role,
              "cells_completed": len(cells), "trained_arms_replayed": len(cells)*4,
              "new_predictor_fits": 0, "optimizer_steps": 0, "endpoint_method_calls": 0,
              "endpoint_methods_and_training_entrypoints_guarded_to_raise": True,
              "legacy_loader_temporarily_loads_and_copies_protected_endpoints": True,
              "diagnosis_uses_endpoint_free_feature_view_only": True,
              "endpoint_bytes_never_loaded": False,
              "risk_replay_absolute_tolerance": 1e-7, "all_patient_risks_replayed": True,
              "all_frozen_LOCAL_heldout_interventions_replayed": True,
              "public_patient_arrays_or_weights": False,
              "mechanism_metrics_do_not_establish_prognostic_increment": True,
              "cells": cells}
    atomic_json(args.output, record)
    require(args.output.stat().st_size <= 2_500_000, "diagnostic report exceeded aggregate-only size budget")
    print("MECHANISM_AUDIT_PASS", len(cells), args.output.name, flush=True)


if __name__ == "__main__":
    main()
