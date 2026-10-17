"""Frozen official UNI2-h inference on verified HEST 224px RGB patch arrays.

Run on the authorized GPU host after copying a finite E-disk subset. Source
images, features, barcodes and weights remain private. No endpoints are read.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import timm

WEIGHT_SHA = "6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e"
MODEL_REVISION = "d517a8dd47902dd7c308b3c36f63bce47e7b9a43"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_model(weights, device):
    if sha256(weights) != WEIGHT_SHA:
        raise ValueError("official model weight SHA mismatch")
    # Exact manual-loading configuration in the official UNI2-h model card.
    model = timm.create_model(
        "vit_giant_patch14_224", pretrained=False, img_size=224,
        patch_size=14, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
        no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True,
    )
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False).to(device)
    return model


def prepare_images(array, device):
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[1:] != (224, 224, 3):
        raise ValueError("official HEST RGB uint8 [N,224,224,3] required")
    # Resize(224) is an identity for the qualified square images. ToTensor and
    # ImageNet Normalize below follow the official manual-loading recipe.
    x = torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2)
    x = x.to(device=device, dtype=torch.float32).div_(255.)
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return (x - mean) / std


def decode_strings(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v)
                       for v in np.asarray(values).reshape(-1)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--asset-plan", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--sections", nargs="+")
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.batch_size < 1:
        raise RuntimeError("authorized CUDA inference host and positive batch size required")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable; explicitly lock FP32 instead")
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock = json.loads(args.split_lock.read_text(encoding="utf-8"))
    plan = json.loads(args.asset_plan.read_text(encoding="utf-8"))
    allowed = {row["section_id_private"] for row in lock["rows"]}
    sections = args.sections or sorted(allowed)
    if not set(sections).issubset(allowed):
        raise ValueError("section outside frozen source/donor lock")
    patches = {Path(x["path"]).stem: x for x in plan["files"] if x["path"].startswith("patches/")}
    model = build_model(args.weights, device)
    source_sha = sha256(__file__)
    feature_dir = args.data_root / "features/uni2-h"
    if not args.profile_only:
        feature_dir.mkdir(parents=True, exist_ok=True)
    parity = None
    for section in sections:
        item = patches[section]
        source = args.data_root / "raw/hest" / item["path"]
        if source.stat().st_size != item["bytes"] or sha256(source) != item["sha256"]:
            raise ValueError("patch source does not match pinned repository LFS SHA")
        with h5py.File(source, "r") as h:
            if set(["img", "coords", "barcode"]) - set(h):
                raise ValueError("patch H5 missing paired keys")
            n = len(h["img"])
            if n < 2 or h["img"].shape[1:] != (224, 224, 3):
                raise ValueError("unqualified RGB patch shape")
            barcode = decode_strings(h["barcode"][:])
            coords_px_top_left = h["coords"][:].astype(np.float64)
            if len(np.unique(barcode)) != n or coords_px_top_left.shape != (n, 2):
                raise ValueError("duplicate barcodes or malformed coordinates")
            counts_path = args.data_root / "processed/sections" / (section + ".npz")
            qualification_path = counts_path.with_suffix(".json")
            qualified = json.loads(qualification_path.read_text(encoding="utf-8"))
            if (qualified.get("qualification_status") != "PASS_SECTION_DATA_CONTRACT"
                    or qualified.get("section_id_private") != section
                    or qualified.get("coordinate_unit") != "um"
                    or qualified.get("npz_sha256") != sha256(counts_path)
                    or qualified["cache_signature"].get("locked_split_sha") != sha256(args.split_lock)
                    or qualified["source_files"]["patches"].get("sha256") != item["sha256"]):
                raise ValueError("individual spatial/image qualification receipt mismatch")
            footprint = qualified["audit"]["image_footprint"]
            actual_view = float(footprint["actual_native_crop_view_um"])
            source_mpp = float(qualified["audit"]["coordinates"]["mpp_x"])
            if (not np.isfinite(actual_view) or not np.isfinite(source_mpp) or source_mpp <= 0
                    or abs(actual_view - 112.) > .5 * source_mpp + 1e-6
                    or footprint["target_pixels"] != 224):
                raise ValueError("unqualified physical image footprint")
            with np.load(counts_path, allow_pickle=False) as paired:
                if not np.array_equal(paired["barcodes"].astype(str), barcode):
                    raise ValueError("qualified count rows must match exact image barcode order")
                coords_um = paired["coords_um"].astype(np.float64)
            if coords_um.shape != (n, 2) or not np.isfinite(coords_um).all():
                raise ValueError("missing qualified expression-center coordinates in micrometers")
            metadata = {"schema_version": "phase22_uni2_feature_section_v1",
                        "status": "RUNNING", "section_id_private": section,
                        "model": "UNI2-h", "model_revision": MODEL_REVISION,
                        "model_sha256": WEIGHT_SHA, "source_patch_sha256": item["sha256"],
                        "counts_npz_sha256": sha256(counts_path),
                        "qualification_metadata_sha256": sha256(qualification_path),
                        "split_lock_sha256": sha256(args.split_lock), "extractor_sha256": source_sha,
                        "precision": args.precision, "batch_size": args.batch_size,
                        "feature_dimension": 1536, "image_view_um_nominal": 112.,
                        "image_mpp_nominal": .5, "image_pixels": 224,
                        "actual_native_crop_view_um": actual_view,
                        "exact_112um": footprint["exact_112um"],
                        "torch_version": torch.__version__, "timm_version": timm.__version__,
                        "normalization": "official_RGB_uint8_div255_ImageNet_mean_std; no_resize_for_224square",
                        "frozen": all(not p.requires_grad for p in model.parameters())}
            target = feature_dir / (section + ".npz")
            receipt = feature_dir / (section + ".json")
            if target.exists() and receipt.exists() and not args.profile_only:
                old = json.loads(receipt.read_text())
                keys = ["model_sha256", "source_patch_sha256", "counts_npz_sha256", "qualification_metadata_sha256", "split_lock_sha256", "extractor_sha256", "precision"]
                if old.get("status") != "COMPLETE" or any(old.get(k) != metadata[k] for k in keys) or old.get("features_sha256") != sha256(target):
                    raise ValueError("stale or mismatched feature cache; never silently overwrite")
                print("FEATURE_CACHE_VERIFIED", section, flush=True)
                continue
            x = prepare_images(h["img"][:min(n, 8)], device)
            if parity is None:
                with torch.inference_mode():
                    reference = model(x).float()
                    if not torch.isfinite(reference).all() or (reference.norm(dim=1) <= 0).any():
                        raise ValueError("invalid FP32 reference features")
                    if args.precision == "bf16":
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            reduced = model(x).float()
                        if not torch.isfinite(reduced).all() or (reduced.norm(dim=1) <= 0).any():
                            raise ValueError("invalid BF16 features")
                        cosine = torch.nn.functional.cosine_similarity(reference, reduced, dim=1)
                        rel = (reference - reduced).norm(dim=1) / reference.norm(dim=1)
                        if not torch.isfinite(cosine).all() or not torch.isfinite(rel).all():
                            raise ValueError("nonfinite precision parity statistics")
                        parity = {"status": "PASS_BF16_FP32_PARITY", "n": len(x),
                                  "cosine_min": float(cosine.min()), "cosine_mean": float(cosine.mean()),
                                  "relative_L2_max": float(rel.max()), "fp32_reference_same_frozen_model": True}
                        del reduced
                    else:
                        parity = {"status": "FP32_NO_REDUCED_PRECISION_GATE", "n": len(x),
                                  "fp32_reference_same_frozen_model": True}
                parity["reference_source_patch_sha256"] = item["sha256"]
                parity["reference_scope"] = "first_section_first_fixed_eight_patches_run_level_not_per_section"
                if args.precision == "bf16" and (parity["cosine_min"] < .995 or parity["cosine_mean"] < .999 or parity["relative_L2_max"] > .05):
                    raise ValueError("prelocked BF16/FP32 parity gate failed")
                print("PRECISION_PARITY", json.dumps(parity), flush=True)
                del reference, x
            metadata["precision_parity"] = parity
            if args.profile_only:
                # Timing/profile is held only in memory, never a formal fit/artifact.
                with torch.inference_mode():
                    x = prepare_images(h["img"][:min(n, args.batch_size)], device)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize(); started = time.monotonic()
                    with torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext():
                        out = model(x)
                    torch.cuda.synchronize()
                    print("PROFILE_ONLY_NOT_SAVED", len(x), time.monotonic() - started,
                          torch.cuda.max_memory_allocated(), bool(torch.isfinite(out).all()), flush=True)
                return
            started = time.monotonic()
            features = np.empty((n, 1536), dtype=np.float32)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for start in range(0, n, args.batch_size):
                    batch = prepare_images(h["img"][start:start + args.batch_size], device)
                    with torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext():
                        out = model(batch).float()
                    if out.shape != (len(batch), 1536) or not torch.isfinite(out).all():
                        raise ValueError("nonfinite or wrong-dimensional UNI2 feature")
                    features[start:start + len(batch)] = out.cpu().numpy()
                    if start % (args.batch_size * 10) == 0:
                        print("EXTRACT_PROGRESS", section, start, n, flush=True)
            norms = np.linalg.norm(features, axis=1)
            mean_variance = float(features.var(axis=0).mean())
            if (norms <= 0).any() or not np.isfinite(mean_variance) or mean_variance <= 0:
                raise ValueError("zero-norm or collapsed frozen feature cache")
            # Independent qualifier supplies expression centers, not a guessed
            # coordinate derived from the top-left crop alone.
            temporary = target.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, features=features, barcodes=barcode,
                                coords_px_top_left=coords_px_top_left, coords_um=coords_um)
            temporary.replace(target)
            metadata.update(status="COMPLETE", n_patches=n, features_sha256=sha256(target),
                            seconds=time.monotonic() - started,
                            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                            feature_norm_min=float(norms.min()), feature_norm_max=float(norms.max()),
                            mean_dimension_variance=mean_variance)
            write_json(receipt, metadata)
            print("FEATURE_SECTION_COMPLETE", section, n, metadata["seconds"], metadata["peak_cuda_bytes"], flush=True)


if __name__ == "__main__":
    main()
