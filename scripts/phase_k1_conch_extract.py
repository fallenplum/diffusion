"""Extract frozen CONCH patch embeddings at the exact UNI2-h H5 coordinates.

This script deliberately keeps raw SVS files in a staging directory only while
one slide is processed.  It writes only derived embeddings and audit metadata.
It never writes credentials or raw images to the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CONCEPT_TEXTS = [
    ("K1", "Interferon and cytotoxic immune activation within an inflamed colorectal tumour microenvironment."),
    ("K2", "Canonical epithelial identity with WNT beta catenin and MYC driven transcriptional activity."),
    ("K3", "Tumour metabolic state spanning glycolysis, oxidative phosphorylation, lipid handling, and peroxisomal metabolism."),
    ("K4", "Mesenchymal and stromal remodelling driven by TGF beta and tumour stroma interaction."),
    ("K5", "Tumour proliferation and cell cycle activity spanning E2F, mitotic spindle, G2M, DNA repair, and MYC programmes."),
    ("K6", "Hypoxic tumour stress and vascular adaptation, including angiogenic and reactive oxygen responses."),
    ("K7", "Invasion and extracellular matrix remodelling associated with EMT, cell adhesion, coagulation, and tissue interface change."),
    ("K8", "Inflammatory and potentially immune suppressive tumour microenvironment signalling through NF kB, IL6 STAT3, and inflammatory response programmes."),
]


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def md5_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def h5_for_slide(data_root: Path, file_name: str) -> Path | None:
    stem = Path(file_name).stem
    candidates = list(data_root.glob(f"**/{stem}.h5"))
    return candidates[0] if len(candidates) == 1 else None


def download_s3(row: dict[str, str], staging: Path) -> Path:
    file_id = row["file_id"]
    file_name = row["file_name"]
    target = staging / file_name
    partial = target.with_suffix(target.suffix + ".partial")
    # The path-style endpoint is materially faster from the remote host than
    # the virtual-host endpoint, while referring to the same public object.
    url = f"https://s3.amazonaws.com/tcga-2-open/{file_id}/{file_name}"
    if target.exists() and target.stat().st_size == int(row["file_size"]) and md5_file(target).lower() == row["md5sum"].lower():
        return target
    if partial.exists():
        partial.unlink()
    expected_size = int(row["file_size"])
    # Parallel range requests reduce the effect of the high-latency route to
    # the public bucket.  Parts are written directly to disk, so memory use is
    # bounded even for multi-gigabyte slides.
    import requests

    workers = min(32, max(8, expected_size // (8 * 1024 * 1024)))
    chunk = (expected_size + workers - 1) // workers
    parts = [staging / f".{file_name}.part{i}" for i in range(workers)]

    def fetch(i: int) -> None:
        start = i * chunk
        end = min(expected_size - 1, (i + 1) * chunk - 1)
        if start > end:
            parts[i].write_bytes(b"")
            return
        for attempt in range(4):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 60)) as r:
                    r.raise_for_status()
                    with parts[i].open("wb") as f:
                        for block in r.iter_content(1024 * 1024):
                            f.write(block)
                if parts[i].stat().st_size == end - start + 1:
                    return
            except Exception:
                if attempt == 3:
                    raise
        raise RuntimeError(f"failed range {start}-{end} for {file_name}")

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, range(workers)))
        with partial.open("wb") as out_f:
            for p in parts:
                with p.open("rb") as in_f:
                    while True:
                        b = in_f.read(8 * 1024 * 1024)
                        if not b:
                            break
                        out_f.write(b)
    finally:
        for p in parts:
            p.unlink(missing_ok=True)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {file_name}: {partial.stat().st_size} != {expected_size}")
    observed = md5_file(partial)
    if observed.lower() != row["md5sum"].lower():
        raise RuntimeError(f"md5 mismatch for {file_name}: {observed} != {row['md5sum']}")
    partial.replace(target)
    return target


def ensure_text_embeddings(model: Any, out: Path, device: str, checkpoint: Path) -> None:
    import torch
    import torch.nn.functional as F
    from conch.open_clip_custom import get_tokenizer

    target = out / "concept_text_embeddings.pt"
    if target.exists():
        return
    tokenizer = get_tokenizer()
    texts = [x[1] for x in CONCEPT_TEXTS]
    ids = tokenizer(
        texts,
        max_length=127,
        add_special_tokens=True,
        return_token_type_ids=False,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"]
    ids = F.pad(ids, (0, 1), value=tokenizer.pad_token_id).to(device)
    with torch.inference_mode():
        emb = model.encode_text(ids, normalize=True).detach().cpu().float()
    torch.save(
        {
            "concept_ids": [x[0] for x in CONCEPT_TEXTS],
            "texts": texts,
            "embeddings": emb,
            "encoder": "MahmoodLab/CONCH conch_ViT-B-16",
            "checkpoint_sha256": sha256_file(checkpoint),
            "normalization": "L2",
        },
        target,
    )


def extract_slide(model: Any, preprocess: Any, row: dict[str, str], h5_path: Path, svs: Path, out: Path, device: str, batch_size: int, max_patches: int | None) -> dict[str, Any]:
    import torch
    from openslide import OpenSlide

    slide_id = Path(row["file_name"]).stem
    done = out / f"{slide_id}.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    with h5py.File(h5_path, "r") as f:
        coords = np.asarray(f["coords"][0], dtype=np.int64)
        attrs = {str(k): (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else str(v)) for k, v in f["coords"].attrs.items()}
    if max_patches is not None:
        coords = coords[:max_patches]
    if int(attrs.get("patch_level", 0)) != 0 or int(attrs.get("patch_size_level0", 512)) != 512:
        raise RuntimeError(f"unexpected H5 coordinate protocol for {slide_id}: {attrs}")
    with OpenSlide(str(svs)) as slide:
        width, height = slide.dimensions
        if np.any(coords[:, 0] < 0) or np.any(coords[:, 1] < 0) or np.any(coords[:, 0] + 512 > width) or np.any(coords[:, 1] + 512 > height):
            raise RuntimeError(f"coordinates out of level-0 bounds for {slide_id}: dimensions={slide.dimensions}")
        emb_path = out / f"{slide_id}.embeddings.f16.npy"
        arr = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(len(coords), 512))
        for start in range(0, len(coords), batch_size):
            batch = coords[start : start + batch_size]
            images = [preprocess(slide.read_region((int(x), int(y)), 0, (512, 512)).convert("RGB")) for x, y in batch]
            x = torch.stack(images).to(device)
            with torch.inference_mode():
                z = model.encode_image(x, proj_contrast=True, normalize=True).float().cpu().numpy().astype(np.float16)
            arr[start : start + len(batch)] = z
        arr.flush()
    np.save(out / f"{slide_id}.coords.i64.npy", coords)
    rec = {
        "status": "DONE",
        "slide_id": slide_id,
        "patient_id": row.get("case_submitter_id", ""),
        "file_id": row["file_id"],
        "file_name": row["file_name"],
        "h5_path": str(h5_path),
        "n_patches": int(len(coords)),
        "embedding_shape": [int(len(coords)), 512],
        "embedding_dtype": "float16",
        "coords_protocol": attrs,
        "svs_size": int(row["file_size"]),
        "svs_md5": row["md5sum"],
    }
    done.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-slides", type=int, default=None)
    ap.add_argument("--max-patches", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.staging.mkdir(parents=True, exist_ok=True)
    import torch
    from conch.open_clip_custom import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=str(args.checkpoint), device=args.device)
    model.eval()
    ensure_text_embeddings(model, args.out, args.device, args.checkpoint)
    rows = read_manifest(args.manifest)
    records = []
    for row in rows[: args.max_slides]:
        h5_path = h5_for_slide(args.data_root, row["file_name"])
        if h5_path is None:
            continue
        slide_path = None
        try:
            slide_path = download_s3(row, args.staging)
            records.append(extract_slide(model, preprocess, row, h5_path, slide_path, args.out, args.device, args.batch_size, args.max_patches))
        finally:
            if slide_path is not None:
                slide_path.unlink(missing_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DONE", "slides": len(records), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
