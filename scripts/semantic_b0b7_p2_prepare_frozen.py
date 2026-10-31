"""Prepare FM-only canonical images for P2; never load endpoints or fit statistics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.semantic_b0b7_prepare import (file_sha, read_json, legacy_data_module,
    pack_patient_patches, verify_model_manifest)
from src.semantic_b0b7.data import array_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "legacy-root", "prior-input", "conch-model", "titan-cache", "output-newdir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--prior-input-sha256", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.output_newdir.exists():
        raise ValueError("REFUSE_OVERWRITE")
    if file_sha(args.prior_input) != args.prior_input_sha256:
        raise ValueError("PARENT_FROZEN_INPUT_CHANGED")
    start = time.monotonic()
    assets = read_json(args.assets)
    old, _ = legacy_data_module(args.legacy_root)
    raw = old.load_raw(assets, include_endpoints=False)
    if raw["audit"]["endpoint_read"]:
        raise ValueError("OUTCOME_ACCESS")
    parent = torch.load(args.prior_input, map_location="cpu", weights_only=False)
    cache = old.read_patch_cache(assets["patch_regions"], verify_hash=True)
    raw_patch, mask, slides, audit = pack_patient_patches(cache, np.arange(512))
    del cache
    proof = verify_model_manifest(args.conch_model, "MahmoodLab/CONCH-Z")
    paths = [args.conch_model / name for name in proof["files_sha256"]
             if name.endswith((".bin", ".safetensors"))]
    if len(paths) != 1:
        raise ValueError("UNEXPECTED_OFFICIAL_WEIGHT_SHARDS")
    key = "visual.proj_contrast"
    if paths[0].suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(paths[0]), framework="pt", device="cpu") as handle:
            keys = [k for k in handle.keys() if k.endswith(key)]
            if keys != [key]:
                raise ValueError("UNEXPECTED_OFFICIAL_PROJECTION_KEY")
            projection = handle.get_tensor(key).float()
    else:
        state = torch.load(paths[0], map_location="cpu", weights_only=True)
        if key not in state:
            raise ValueError("OFFICIAL_PROJECTION_KEY_MISSING")
        projection = state[key].float().clone()
        del state
    if projection.shape != (768, 768) or not torch.isfinite(projection).all():
        raise ValueError("OFFICIAL_PROJECTION_INVALID")
    # Exact official CONCH-Z project_image_features formula, pinned by parent
    # real official-model qualification and full selected-row equality below.
    aligned = np.zeros_like(raw_patch)
    weight = projection.cuda()
    with torch.inference_mode():
        for i in range(512):
            x = torch.from_numpy(raw_patch[i, mask[i]]).cuda()
            y = torch.nn.functional.normalize(x @ weight, dim=-1)
            aligned[i, mask[i]] = y.cpu().numpy()
    rows = parent["global_rows"].numpy()
    errors = {"raw": float(np.max(np.abs(raw_patch[rows] - parent["image"]["patch_values"].numpy()))),
              "aligned": float(np.max(np.abs(aligned[rows] - parent["image"]["patch_aligned"].numpy())))}
    if max(errors.values()) > 2e-6:
        raise ValueError("PARENT_OFFICIAL_PROJECTION_REPLAY_MISMATCH")
    if not np.array_equal(mask[rows], parent["image"]["patch_mask"].numpy()) or not np.array_equal(slides[rows], parent["image"]["patch_slide"].numpy()):
        raise ValueError("PARENT_PATCH_REGISTRY_MISMATCH")
    with np.load(args.titan_cache, allow_pickle=False) as z:
        if not np.array_equal(z["patient_ids"].astype(str), raw["patient_ids"]):
            raise ValueError("TITAN_PATIENT_IDENTITY_CHANGED")
        image = {"patch_values": torch.from_numpy(raw_patch), "patch_aligned": torch.from_numpy(aligned),
                 "patch_mask": torch.from_numpy(mask), "patch_slide": torch.from_numpy(slides),
                 "titan_native": torch.from_numpy(np.array(z["native"], np.float32)),
                 "titan_aligned": torch.from_numpy(np.array(z["semantic"], np.float32)),
                 "titan_scores": torch.from_numpy(np.array(z["concept_cosine"], np.float32))}
    for name in ("titan_native", "titan_aligned", "titan_scores"):
        error = float((image[name][rows] - parent["image"][name]).abs().max())
        errors[name] = error
        if error != 0:
            raise ValueError("PARENT_TITAN_REPLAY_CHANGED")
    payload = {"image": image, "global_rows": torch.arange(512),
               "program_names": parent["program_names"]}
    for name in ("conch_prototypes", "titan_prototypes", "relation_prior", "shuffled_relation_prior"):
        payload[name] = parent[name].clone()
    args.output_newdir.mkdir(parents=True)
    out = args.output_newdir / "FROZEN_IMAGES.private.pt"
    with out.open("xb") as handle:
        torch.save(payload, handle)
    report = {"schema": "semantic_p2_frozen_images_v1", "status": "COMPLETE_FM_ONLY_NO_LEARNED_STATISTICS",
              "patients": 512, "neural_fits": 0, "optimizer_updates": 0, "outcomes_read": False,
              "source_sha256": file_sha(__file__), "parent_input_sha256": args.prior_input_sha256,
              "cohort_sha256": raw["audit"]["cohort_sha256"], "official_model_manifest": proof,
              "projection_key": key, "projection_sha256": array_sha(projection.numpy()),
              "parent_actual_selected_rows_max_errors": errors, "patch_audit": audit,
              "titan_cache_sha256": file_sha(args.titan_cache),
              "frozen_input_sha256": file_sha(out), "frozen_input_bytes": out.stat().st_size,
              "elapsed_seconds": time.monotonic() - start}
    (args.output_newdir / "COMPLETE.private.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
