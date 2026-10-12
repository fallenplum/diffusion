"""Phase10-P0 SCPN-Surv runner. Private H5, checkpoints and OOF stay local."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.phase8_v0.dataset import load_private, write_json
from src.phase10_p0.data import load_cohort
from src.phase10_p0.regions import build_region_cohort, fit_prototypes
from src.phase10_p0.train import fit_mean_control, fit_neural, prepare_fold

ARMS = ["P0", "P1", "P2", "P3", "P4", "P3-noTopo"]


def _cfg(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest_audit(manifest, root, ids):
    patients, audit = load_cohort(manifest, root, ids)
    return patients, audit


def audit(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    d = load_private(Path(args.data_parent))
    _, result = _manifest_audit(args.manifest, args.features, d["patient_ids"])
    result.update({"data_parent": str(args.data_parent), "manifest": str(args.manifest),
                   "feature_root": str(args.features), "patient_id_hash": ""})
    result["patient_id_hash"] = hashlib.sha256("\n".join(d["patient_ids"].astype(str)).encode()).hexdigest()
    write_json(out / "data_alignment_audit.json", result)
    write_json(out / "source_registry.json", {"patients": len(d["patient_ids"]), "DSS_events": int(d["DSS_event"].sum()),
                                                "slides_expected": 519, "encoder_required": "conch_v15",
                                                "feature_shape_required": "[N,768]", "coordinates_required": True,
                                                "labels_read_for_audit": False})
    print(json.dumps({k: result[k] for k in ("status", "patients", "slides", "patches")}))
    return 0


def _plot_maps(regions, report_dir):
    import matplotlib.pyplot as plt
    maps = Path(report_dir) / "region_maps"; maps.mkdir(parents=True, exist_ok=True)
    candidates = []
    for patient_id, slides in regions.items():
        for filename, region in slides:
            candidates.append((filename, region))
    for index, (filename, region) in enumerate(candidates[:2], start=1):
        xy = region.get("centroids")
        if xy is None:
            continue
        sizes = np.asarray(region["patch_count"], float)
        plt.figure(figsize=(7, 5), dpi=150)
        plt.scatter(xy[:, 0], xy[:, 1], c=np.log1p(sizes), s=4, cmap="viridis", linewidths=0)
        plt.title(f"SCPN region coordinates (slide_{index:02d})")
        plt.xlabel("normalized x"); plt.ylabel("normalized y"); plt.colorbar(label="log1p patches")
        plt.tight_layout(); plt.savefig(maps / f"slide_{index:02d}_regions.png"); plt.close()


def _save_cell(out, arm, seed, fold, te, risk, base, info):
    out = Path(out); (out / "oof").mkdir(parents=True, exist_ok=True); (out / "cells").mkdir(exist_ok=True)
    np.savez_compressed(out / "oof" / f"DSS_seed{seed}_fold{fold}_{arm}.npz",
                        test_indices=te, risk=np.asarray(risk), base_risk=np.asarray(base), seed=seed, fold=fold, arm=arm)
    public = {"status": "PASS", "arm": arm, "seed": seed, "fold": fold, "test_n": int(len(te)),
              "test_indices_private": True, **info}
    write_json(out / "cells" / f"DSS_seed{seed}_fold{fold}_{arm}.json", public)


def run(args, smoke=False):
    cfg = _cfg(args.config); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    d = load_private(Path(args.data_parent)); patients, data_audit = _manifest_audit(args.manifest, args.features, d["patient_ids"])
    if data_audit.get("status") != "PASS":
        write_json(out / "BLOCKED.json", data_audit); return 2
    seed = cfg["seed"]; folds = [1] if smoke else ([args.fold] if args.fold else cfg["folds"])
    all_rows = []; region_rows = []; compute_rows = []; public_maps = None
    for fold in folds:
        ctx, ctx_audit = prepare_fold(d, cfg, fold)
        tr, te, it, iv = ctx["tr"], ctx["te"], ctx["it"], ctx["iv"]
        fold_dir = out / "private_fold_artifacts" / f"seed{seed}_fold{fold}"; fold_dir.mkdir(parents=True, exist_ok=True)
        write_json(fold_dir / "context_audit.json", ctx_audit)
        base_test = ctx["outer_base"][te]
        _save_cell(out, "P0", seed, fold, te, base_test, base_test, {"parameters": 0, "selected_epoch": 0, "outer_test_used_for_selection": False})
        p1_risk, p1_delta, p1_info = fit_mean_control(patients, d, ctx, cfg)
        _save_cell(out, "P1", seed, fold, te, p1_risk, base_test, p1_info)
        centers_i, proto_i = fit_prototypes(patients, d["patient_ids"][it], seed + fold, cfg["prototype_k"], cfg["prototype_samples_per_slide"])
        regions_i, audit_i = build_region_cohort(patients, centers_i, d["patient_ids"], cfg["max_regions"], cfg["irregular_radius_multiplier"])
        centers_o, proto_o = fit_prototypes(patients, d["patient_ids"][tr], seed + 1000 + fold, cfg["prototype_k"], cfg["prototype_samples_per_slide"])
        regions_o, audit_o = build_region_cohort(patients, centers_o, d["patient_ids"], cfg["max_regions"], cfg["irregular_radius_multiplier"])
        if public_maps is None: public_maps = regions_o
        region_rows.extend([{"seed": seed, "fold": fold, "split": "inner", **row} for row in audit_i])
        region_rows.extend([{"seed": seed, "fold": fold, "split": "outer", **row} for row in audit_o])
        write_json(fold_dir / "prototype_audit.json", {"inner": proto_i, "outer": proto_o,
                    "inner_centers_sha256": hashlib.sha256(centers_i.tobytes()).hexdigest(),
                    "outer_centers_sha256": hashlib.sha256(centers_o.tobytes()).hexdigest()})
        for arm in ["P2", "P3", "P4", "P3-noTopo"]:
            # Identical weights and dropout stream for the topology/query controls.
            init_arm = "P3" if arm in ["P3", "P4", "P3-noTopo"] else arm
            inner_regions = None if arm == "P2" else regions_i
            outer_regions = None if arm == "P2" else regions_o
            inner_model, inner_info = fit_neural(arm, patients, inner_regions, d, it, iv, ctx["inner_C"], ctx["inner_S"], ctx["inner_cf"][it], ctx["inner_base"][iv], cfg, seed + fold * 100 + sum(map(ord, init_arm)), fixed_epochs=None if not smoke else cfg["smoke_epochs"], checkpoint=fold_dir / f"{arm}_inner.pt", batch_size=cfg["batch_size"])
            selected = int(inner_info["selected_epoch"])
            outer_model, outer_info = fit_neural(arm, patients, outer_regions, d, tr, np.asarray([], dtype=int), ctx["outer_C"], ctx["outer_S"], ctx["outer_cf"][tr], np.asarray([], dtype=float), cfg, seed + fold * 1000 + sum(map(ord, init_arm)), fixed_epochs=selected, checkpoint=fold_dir / f"{arm}_outer.pt", batch_size=cfg["batch_size"])
            from src.phase10_p0.train import predict
            delta = predict(outer_model, arm, patients, outer_regions, d, te, ctx["outer_C"], ctx["outer_S"], cfg["batch_size"])
            risk = base_test + delta
            info = {"selected_epoch": selected, "inner_parameters": inner_info["parameters"], "outer_parameters": outer_info["parameters"],
                    "inner_training_seconds": inner_info["training_seconds"], "outer_training_seconds": outer_info["training_seconds"],
                    "peak_gpu_bytes": max(inner_info["peak_gpu_bytes"], outer_info["peak_gpu_bytes"]),
                    "inner_trace": inner_info["trace"], "outer_trace": outer_info["trace"],
                    "outer_test_used_for_selection": False, "clinical_anchor_reused": True}
            _save_cell(out, arm, seed, fold, te, risk, base_test, info)
            del inner_model, outer_model
            compute_rows.append({"seed": seed, "fold": fold, "arm": arm, "selected_epoch": selected,
                                 "parameters": outer_info["parameters"], "peak_gpu_bytes": max(inner_info["peak_gpu_bytes"], outer_info["peak_gpu_bytes"])})
        del regions_i, regions_o, centers_i, centers_o
    if public_maps is not None and not smoke: _plot_maps(public_maps, out)
    import pandas as pd
    pd.DataFrame(region_rows).to_csv(out / "region_statistics.csv", index=False)
    pd.DataFrame(compute_rows).to_csv(out / "compute_profile.csv", index=False)
    write_json(out / "architecture_audit.json", {"status": "PASS", "arms": ARMS, "encoder": "CONCH v1.5 [N,768] frozen",
              "p2_projection": 128, "p3_graph_layers": 1, "p3_noTopo_same_region_tokens": True, "p4_only_query_conditioned": True,
              "risk": "outer Clinical+S32 anchor + residual delta", "full_risk_set_cox": True, "test_labels_used_for_selection": False})
    write_json(out / "leakage_audit.json", {"status": "PASS", "prototype_outcome_blind": True, "inner_outer_prototypes_separate": True,
              "train_fold_only_preprocessing": True, "crossfit_anchor_offsets": True, "outer_test_used_for_selection": False,
              "conch_patch_row_alignment_with_uni2": False})
    if smoke: write_json(out / "smoke_DONE.json", {"status": "PASS", "seed": seed, "folds": folds, "arms": ARMS, "excluded_from_formal": True})
    else: write_json(out / "DONE.json", {"status": "PASS", "seed": seed, "folds": folds, "arms": ARMS, "cells": len(folds) * len(ARMS), "patient_oof_private": True})
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "smoke", "formal"], required=True)
    parser.add_argument("--fold", type=int, choices=[1,2,3,4,5], default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/phase10_p0.json"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/phase6_conch_v15_download_audit/cohort_manifest.csv"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "audit": return audit(args)
    return run(args, smoke=args.mode == "smoke")


if __name__ == "__main__":
    raise SystemExit(main())
