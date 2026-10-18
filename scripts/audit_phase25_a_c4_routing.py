"""Frozen original C4: isolate RNA attention-query and Patch-value pathways.

Only rp (RNA -> Patch) is overridden at runtime. Its recipient residual query,
feedforward, Patch K/V, all other branches and all model weights stay fixed.
No fitting, bootstrap, checkpoint rewrite or outcome-based permutation occurs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import types

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))

import numpy as np
import torch
from src.phase11_semantic.models import CrossAttention, SemanticModel
from src.phase11_semantic.data import PatchCache
from src.phase23_cdei.data import raw_rna_missing_patterns, partition_derangement
from audit_phase25_a_visual import (SEEDS, FOLDS, require, digest, summary, counts, transitions,
                                    add_counts, add_transitions)
from validate_phase11_independent import ipcw

REPLAY_ATOL = 1e-5  # Locked before opening any intervention outcomes.
CONDITIONS = ("MATCHED", "VALUE_OFF", "QUERY_MEAN", "QUERY_WRONG")


class RPOverride:
    def __init__(self, module):
        self.module, self.original = module, module.forward
        self.mode, self.query_override = "MATCHED", None
        self.calls, self.max_residual_query_error = 0, 0.0

    def forward(self, module, query, key_value, mask=None, return_attention=False):
        self.calls += 1
        # The residual and FF always receive the unmodified recipient query.
        if self.mode == "VALUE_OFF":
            h = query
            if mask is not None:
                require(mask.any(-1).all(), "all Patch keys masked")
            weights = query.new_zeros((*query.shape[:2], key_value.shape[1]))
        else:
            source = query if self.mode == "MATCHED" else self.query_override
            require(source is not None and source.shape == query.shape and source.device == query.device,
                    "attention-only override query shape/device mismatch")
            q = module.qkv(module.norm(source))[..., :module.d]
            _, k, v = module.qkv(module.norm(key_value)).chunk(3, -1)
            score = q @ k.transpose(-1, -2) / module.d**.5
            if mask is not None:
                require(mask.any(-1).all(), "all Patch keys masked")
                score = score.masked_fill(~mask[:, None, :], -torch.inf)
            weights = score.softmax(-1)
            h = query + weights @ v
        h = h + module.ff(h)
        return (h, weights) if return_attention else h

    def install(self):
        self.module.forward = types.MethodType(self.forward, self.module)

    def restore(self):
        self.module.forward = self.original


def synthetic_qualification():
    """Small exact formula/mask checks; synthetic input, no survival fitting."""
    torch.manual_seed(250101)
    module = CrossAttention(8).eval()
    own, keys = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, False, True, False], [True, False, True, True, False]])
    with torch.no_grad():
        before = module(own, keys, mask, True)
        state = {k: v.clone() for k, v in module.state_dict().items()}
        audit = RPOverride(module); audit.install()
        matched = module(own, keys, mask, True)
        require(all(torch.equal(a, b) for a, b in zip(before, matched)), "synthetic matched-wrapper parity failed")
        audit.mode = "VALUE_OFF"
        off = module(own, keys, mask)
        require(torch.equal(off, own+module.ff(own)), "VALUE_OFF changed residual/FF formula")
        for name, override in (("QUERY_MEAN", own.mean(0, keepdim=True).expand_as(own)),
                               ("QUERY_WRONG", own.flip(0))):
            audit.mode, audit.query_override = name, override
            output, attention = module(own, keys, mask, True)
            q = module.qkv(module.norm(override))[..., :module.d]
            _, k, v = module.qkv(module.norm(keys)).chunk(3, -1)
            expected = (q @ k.transpose(-1, -2) / module.d**.5).masked_fill(~mask[:, None], -torch.inf).softmax(-1)
            h = own+expected@v
            require(torch.equal(output, h+module.ff(h)) and torch.equal(attention, expected), "query-only exact formula failed")
            require(not attention.masked_select(~mask[:, None].expand_as(attention)).any(), "query override changed Patch mask")
        audit.restore()
        require(all(torch.equal(v, state[k]) for k, v in module.state_dict().items()), "synthetic override changed parameters")
    return {"status": "PASS", "matched_attention_output_exact_parity": True,
            "VALUE_OFF_residual_and_FF_exact": True, "query_only_q_KV_and_mask_exact": True,
            "weights_unchanged": True, "synthetic_new_training_steps": 0}


def encode_rt(model, rna, indices, device):
    result = []
    for start in range(0, len(indices), 64):
        ids = indices[start:start+64]
        result.append(model.rna(torch.tensor(rna[ids], device=device), "msa")[1])
    return torch.cat(result)


def predict(model, override, f, cache, test, mode, fit_rt_mean, heldout_rt, donor_position, device):
    result = []
    for start in range(0, len(test), 64):
        ids = test[start:start+64]
        c, s, r = (torch.tensor(f[k][ids], device=device) for k in ("C", "S", "R"))
        patches, mask = cache.batch(ids, device)
        active_slides = mask.any(-1).reshape(-1).nonzero().squeeze(1)
        owners = torch.div(active_slides, patches.shape[1], rounding_mode="floor")
        if override is not None:
            override.mode = mode
            if mode == "QUERY_MEAN":
                override.query_override = fit_rt_mean[None].expand(len(owners), -1, -1)
            elif mode == "QUERY_WRONG":
                source = torch.as_tensor(donor_position[start:start+len(ids)], device=device)
                override.query_override = heldout_rt[source][owners]
            else:
                override.query_override = None
        result.append(model(c, s, r, patches, mask).detach().cpu().numpy())
    return np.concatenate(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c4-root", type=Path, default=Path("/root/phase11_semantic_anchor_private"))
    p.add_argument("--dataset", type=Path, default=Path("/root/phase7a_foundation_headroom_v1/private_dataset.npz"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT/"reports/phase25_a/C4_ROUTING_ATTRIBUTION.json")
    a = p.parse_args()
    qualification = synthetic_qualification()
    if a.synthetic_only:
        print(json.dumps(qualification)); return
    require(not a.output.exists(), "append-only routing audit refuses an existing output")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache_root = a.c4_root/"cache"
    cache_receipt = json.loads((cache_root/"data_manifest.json").read_text())
    require(digest(cache_root/"patch1024.npy") == cache_receipt["patch_sha256"] and
            digest(cache_root/"private_slide_manifest.json") == cache_receipt["private_slide_manifest_sha256"],
            "original C4 Patch/cache-manifest hash changed")
    cache = PatchCache(cache_root, 128)
    # Select only features needed for an outcome-blind wrong-query permutation.
    # DSS vectors come exclusively from already saved protected C4 OOF files.
    with np.load(a.dataset, allow_pickle=True) as d:
        cancer = d["cancer_type"].astype(str)
        patterns = raw_rna_missing_patterns(d["rna_genes"])
    times, events, folds = np.full(512, np.nan), np.full(512, np.nan), np.zeros(512, int)
    paths = []
    for seed in SEEDS:
        for fold in FOLDS:
            name = f"C4_seed{seed}_fold{fold}"
            cp, op, mp = (a.c4_root/"formal"/folder/(name+suffix) for folder, suffix in
                          (("checkpoints", ".pt"), ("oof", ".npz"), ("manifests", ".json")))
            cell = json.loads((a.c4_root/"formal/cells"/(name+".json")).read_text())
            require(cell["status"] == "COMPLETE" and not cell["outer_test_selection"]
                    and digest(cp) == cell["checkpoint_sha256"] and digest(op) == cell["prediction_sha256"]
                    and digest(mp) == cell["manifest_sha256"], "original C4 receipt/hash changed")
            with np.load(op) as z:
                ids = z["test_indices"]
                if seed == SEEDS[0]:
                    require(not np.isfinite(times[ids]).any(), "OOF folds overlap")
                    times[ids], events[ids], folds[ids] = z["time"], z["event"], fold
                else:
                    require(np.array_equal(times[ids], z["time"]) and np.array_equal(events[ids], z["event"])
                            and (folds[ids] == fold).all(), "OOF identity differs across seeds")
            paths.append((seed, fold, cp, op, mp, cell))
    require(np.isfinite(times).all() and np.isin(events, [0, 1]).all(), "incomplete inherited OOF")
    all_counts, all_transitions, cells = {}, {}, []
    with torch.no_grad():
        for seed, fold, cp, op, mp, cell in paths:
            test, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
            manifest = json.loads(mp.read_text())
            require(np.array_equal(manifest["outer_train"], fit) and np.array_equal(manifest["outer_test"], test), "C4 split mismatch")
            pp = a.c4_root/"preprocessing"/f"fold{fold}_outer.npz"
            pm = json.loads(pp.with_suffix(".json").read_text())
            require(digest(pp) == pm["npz_sha256"], "C4 preprocessing hash changed")
            with np.load(pp) as z:
                f = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
            ck = torch.load(cp, map_location="cpu", weights_only=False)
            model = SemanticModel("C4", ck["clinical_mask"], ck["rna_membership"], 64).to(a.device).eval()
            model.load_state_dict(ck["state"], strict=True)
            initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # Native historical forward first. No intervention result is used
            # if this pre-locked numerical check fails.
            native = predict(model, None, f, cache, test, "MATCHED", None, None, None, a.device)
            with np.load(op) as old:
                original = old["risk"].copy()
            error = float(np.max(np.abs(native-original)))
            require(error <= REPLAY_ATOL, "C4 native checkpoint replay exceeds locked 1e-5; no interventions evaluated")
            fit_rt_mean = encode_rt(model, f["R"], fit, a.device).mean(0)
            heldout_rt = encode_rt(model, f["R"], test, a.device)
            mapping, pairing = partition_derangement(cancer, patterns, test, 250101+100*fold+seed)
            position = {int(patient): j for j, patient in enumerate(test)}
            donor_position = np.array([position[int(v)] for v in mapping[test]], int)
            require(not (mapping[test] == test).any() and set(mapping[test]) == set(test), "query donor crosses heldout partition")
            override = RPOverride(model.rp); override.install()
            risks = {"MATCHED": native}
            for mode in CONDITIONS[1:]:
                risks[mode] = predict(model, override, f, cache, test, mode, fit_rt_mean,
                                      heldout_rt, donor_position, a.device)
            override.restore()
            require(all(torch.equal(v.detach().cpu(), initial[k]) for k, v in model.state_dict().items()), "frozen C4 parameters changed")
            weights = ipcw(times[fit], events[fit], times[test], events[test], 60)
            records = {}
            for mode, risk in risks.items():
                require(np.isfinite(risk).all(), "nonfinite frozen routing risk")
                add_counts(all_counts.setdefault(seed, {}), mode, counts(times[test], events[test], risk, weights))
                if mode != "MATCHED":
                    add_transitions(all_transitions.setdefault(seed, {}), mode,
                                    transitions(times[test], events[test], risk, native, weights))
                    records[mode] = {"intervention_minus_matched_risk": summary(risk-native)}
            public_pairing = {k: pairing[k] for k in (
                "patients", "primary_cancer_pattern_strata", "primary_singleton_patients", "missing_pattern_strata",
                "pattern_singleton_patients", "identity_count", "missing_pattern_mismatch_count", "relaxed_cancer_match_count")}
            cells.append({"seed": seed, "fold": fold, "patients": len(test),
                          "native_risk_replay_maximum_absolute_error": error,
                          "frozen_interventions": records, "query_wrong_pairing": public_pairing,
                          "weights_unchanged": True,
                          "fit_RT_mean_shape": list(fit_rt_mean.shape), "fit_RT_mean_patient_count": len(fit),
                          "checkpoint_sha256": digest(cp), "OOF_sha256": digest(op)})
            print(f"C4_ROUTING_CELL_PASS seed={seed} fold={fold} replay={error:.9g}", flush=True)
            del model, initial, heldout_rt, fit_rt_mean, override
            if str(a.device).startswith("cuda"):
                torch.cuda.empty_cache()
    seed_metrics, pair_rows = [], []
    for seed in SEEDS:
        for mode, metric in all_counts[seed].items():
            seed_metrics.append({"seed": seed, "condition": mode,
                                 **{k: float(v[0]/v[1]) for k, v in metric.items()}})
        for mode, metric in all_transitions[seed].items():
            for name, (repair, damage, den) in metric.items():
                pair_rows.append({"seed": seed, "condition": mode, "metric": name,
                                  "intervention_minus_matched_delta": float((repair-damage)/den),
                                  "repaired_pair_credit_mass": float(repair/den),
                                  "damaged_pair_credit_mass": float(damage/den)})
    model_summary = [{"condition": mode, **{name+"_mean": float(np.mean([
        v[name] for v in seed_metrics if v["condition"] == mode])) for name in ("Harrell", "Uno60")}}
        for mode in CONDITIONS]
    pair_summary = []
    for mode in CONDITIONS[1:]:
        for name in ("Harrell", "Uno60"):
            rows = [r for r in pair_rows if r["condition"] == mode and r["metric"] == name]
            pair_summary.append({"condition": mode, "metric": name,
                **{key: float(np.mean([r[key] for r in rows])) for key in
                   ("intervention_minus_matched_delta", "repaired_pair_credit_mass", "damaged_pair_credit_mass")},
                "positive_seeds": sum(r["intervention_minus_matched_delta"] > 0 for r in rows)})
    record = {"status": "PASS", "schema": "phase25_a_C4_attention_query_only_frozen_routing_v1",
        "script_sha256": digest(__file__), "synthetic_qualification": qualification,
        "cells": 15, "conditions_per_cell": 4, "new_training_fits": 0, "new_bootstrap_draws": 0,
        "new_hypothesis_or_PC_selection": False, "all_model_weights_unchanged": True,
        "replay_tolerance_locked_before_run": REPLAY_ATOL,
        "native_risk_replay_maximum_absolute_error": max(c["native_risk_replay_maximum_absolute_error"] for c in cells),
        "scope": "fixed_original_C4_heldout_inference_only;not_retrained_necessity;no_new_confirmatory_intervals",
        "formulas": {"VALUE_OFF": "h=RT_recipient;out=h+FF(h)",
            "QUERY_MEAN": "q=Q(LN(mean_fit_RT_per_slot));h=RT_recipient+softmax(q K_patch^T/sqrt(d)) V_patch;out=h+FF(h)",
            "QUERY_WRONG": "q=Q(LN(RT_wrong_heldout_patient));h=RT_recipient+softmax(q K_patch^T/sqrt(d)) V_patch;out=h+FF(h)"},
        "all_other_CP_anchor_heads_and_recipient_query_residual_unchanged": True,
        "fit_RT_mean": "8x64_current_frozen_RNA_encoder_output_patient_equal_slot_mean_on_outer_train_only",
        "wrong_pairing": "heldout_partition_zero_self_cancer_x_full_raw_RNA_missing_pattern;no_DSS_Stage_Age_Sex",
        "pairing_seed_formula": "250101+100*fold+model_seed",
        "metric_protocol": "within_fold_num_den_then_each_seed_then_three_seed_mean;no_risk_ensemble",
        "comparison_direction": "intervention_minus_matched;negative_delta_means_loss_after_intervention",
        "existing_protected_OOF_endpoints_used_for_metrics_only": True,
        "no_patient_risks_features_weights_or_maps_published": True,
        "cell_summary": cells, "seed_metrics": seed_metrics,
        "model_summary": model_summary, "paired_credit_summary": pair_summary}
    text = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n"
    require(len(text.encode()) < 2*1024**2, "public routing summary too large")
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "PASS", "cells": 15,
                      "replay_maximum_absolute_error": record["native_risk_replay_maximum_absolute_error"],
                      "report_bytes": a.output.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
