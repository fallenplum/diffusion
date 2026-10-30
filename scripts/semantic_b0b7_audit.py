"""P0 read-only canonical-input audit; no training and no publication of rows."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("audit output already exists; retain historical receipts")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True, type=Path)
    p.add_argument("--legacy-root", required=True, type=Path)
    p.add_argument("--titan-cache", required=True, type=Path)
    p.add_argument("--conch-model", required=True, type=Path)
    p.add_argument("--titan-model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location("frozen_old_bgb_data", args.legacy_root / "src/bgb_en/data.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    assets = json.loads(args.assets.read_text())
    raw = old.load_raw(assets, include_endpoints=True)  # P0 audit ONLY, never a P1 data structure.
    files = {}
    for name in ("cohort", "endpoints", "visual_sources", "gmt", "legacy_groups"):
        path = Path(assets[name])
        files[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    caches = {}
    for name in ("patch_regions", "patch_grid"):
        cache = old.read_patch_cache(assets[name], verify_hash=True)
        owner, mask = np.asarray(cache["owner"]), np.asarray(cache["mask"])
        assert len(owner) == 519 and set(owner) == set(range(512))
        assert np.all(np.bincount(owner, weights=mask.sum(1), minlength=512) <= 512)
        caches[name] = {"status": "PASS_EXACT_RECEIPT_ALL_ARRAY_SHA256", "ready_sha256": sha(Path(assets[name]) / "READY.json"),
                        "slides": len(owner), "patients": len(np.unique(owner)), "patches": int(mask.sum()),
                        "selector": cache["audit"]["selector"], "patient_budget": 512,
                        "slide_aggregation": "within_slide_then_equal_patient_slide_mean",
                        "coordinate_step_counts": cache["audit"]["coordinate_step_counts"]}
    inv_path = Path(assets["conch_inventory"])
    inventory = json.loads(inv_path.read_text())
    rows = inventory["chosen_conch"]

    def source_file(row):
        path = Path(row["path"])
        if not path.is_file():
            return "MISSING"
        before = path.stat()
        if before.st_size != row["bytes"]:
            return "SIZE_MISMATCH"
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return "CHANGED_DURING_AUDIT"
        return "PASS" if digest == row["sha256"] else "SHA_MISMATCH"

    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(pool.map(source_file, rows))
    assert len(rows) == 519 and all(s == "PASS" for s in states), {s: states.count(s) for s in set(states)}
    assert caches["patch_regions"]["selector"] == "REGIONS" and caches["patch_grid"]["selector"] == "GRID"
    models = {}
    for name, root in (("CONCH_Z", args.conch_model), ("TITAN", args.titan_model)):
        info = json.loads((root / "DOWNLOAD_COMPLETE.json").read_text())
        for entry in info["files"]:
            filename = entry.get("file", entry.get("filename"))
            assert filename and sha(root / filename) == entry["sha256"], (name, filename)
        models[name] = {"status": "PASS_PINNED_ALL_FILES_SHA256", "revision": info.get("revision"),
                        "weights_sha256": sha(root / "model.safetensors"), "checked_files": len(info["files"])}
    with np.load(args.titan_cache, allow_pickle=False) as z:
        assert np.array_equal(z["patient_ids"].astype(str), raw["patient_ids"])
        assert z["native"].shape == z["semantic"].shape == (512, 768)
        assert z["concept_cosine"].shape == (512, 32) and z["text_prototypes"].shape == (32, 768)
        assert all(np.isfinite(z[k]).all() for k in ("native", "semantic", "concept_cosine", "text_prototypes"))
        native_error = float(np.max(np.abs(z["native"] - raw["titan"])))
        score_error = float(np.max(np.abs(z["concept_cosine"] - raw["S32"])))
        cosine_error = float(np.max(np.abs(z["semantic"] @ z["text_prototypes"].T - z["concept_cosine"])))
        assert max(native_error, score_error, cosine_error) < 3e-6
    # Preserve exact identities privately; output only counts and hashes.
    missing = (~raw["rna_measured"]).sum(1)
    fold = [{"fold": int(k), "n": int(sum(raw["folds"] == k)),
             "DSS_events": int(raw["e"][raw["folds"] == k].sum())} for k in range(1, 6)]
    stage = raw["clinical_raw"]["AJCC_PATHOLOGIC_TUMOR_STAGE"]
    stages = [{"category": str(s), "n": int(sum(stage == s)),
               "DSS_events": int(raw["e"][stage == s].sum())} for s in np.unique(stage)]
    report = {"schema": "semantic_b0b7_P0_data_v1", "status": "PASS_DATA_BYTE_INTEGRITY_ONLY",
              "clinical_time_qualification": "historical_baseline_fields_only_no_new_variables_no_individual_field_timestamps_available",
              "system_disk_clone_completion": "NOT_INFERRED_FROM_DIRECTORY_EXISTENCE",
              "required_data_files": "all_current_bytes_equal_preexisting_pinned_SHA256",
              "cohort": {"patients": 512, "DSS_events": 65, "RNA_shape": list(raw["rna"].shape),
                         "CNA_shape": list(raw["cna"].shape), "RNA_missing_count_histogram":
                         {str(int(k)): int(sum(missing == k)) for k in np.unique(missing)},
                         "CNA_missing_entries": int((~raw["cna_measured"]).sum()), "folds": fold,
                         "stage_distribution": stages, "time_unit": "months",
                         "RNA_transform": "already_log2_x_plus_1_do_not_log_again",
                         "missingness_platform": "proxy_only_not_verified_sequencing_platform"},
              "source_hashes": files, "original_CONCH_H5_verified": len(states),
              "CONCH_inventory_sha256": sha(inv_path), "patch_caches": caches, "models": models,
              "TITAN": {"cache_sha256": sha(args.titan_cache), "native_max_error": native_error,
                        "S32_max_error": score_error, "semantic_cosine_max_error": cosine_error,
                        "official_forward_reverification": "PENDING_SEPARATE_OFFICIAL_INTERFACE_GATE"},
              "audit_execution": {"seconds": time.monotonic() - started, "fits": 0,
                                  "optimizer_updates": 0, "model_forwards": 0,
                                  "endpoints_read": "P0_only_separate_from_P1"},
              "G0_overall": "NOT_YET_PASS_MODEL_ALIGNMENT_FAIRNESS_REAL_GRADIENT_ANCHOR_AUDITS_PENDING"}
    dump(args.output, report)
    print(json.dumps({"status": report["status"], "h5_checked": len(states),
                      "seconds": report["audit_execution"]["seconds"], "output_sha256": sha(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
