"""Frozen official TITAN inference worker for its qualified legacy environment.

The parent performs CONCH inference in a different compatible environment.
This worker receives only a fixed patient scope, frozen feature cache, original
text and local model files.  It imports no endpoints and performs no training.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.semantic_b0b7_prepare import (check_titan_scores, file_sha, load_official,
    official_text, read_json, require, verify_model_manifest, write_new_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--slide-cache", type=Path)
    parser.add_argument("--output-newdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    require(not args.output_newdir.exists(), "REFUSE_TO_OVERWRITE_TITAN_WORKER_OUTPUT")
    require(str(args.device).startswith("cuda") and torch.cuda.is_available(), "TITAN_WORKER_REQUIRES_REAL_GPU")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    vocabulary = read_json(args.concepts)
    require(len(vocabulary["concepts"]) == 32 and len(vocabulary["templates"]) == 2,
            "TITAN_WORKER_FROZEN_VOCABULARY_SHAPE")
    with np.load(args.scope, allow_pickle=False) as scope:
        require(set(scope.files) == {"patient_rows", "owners"}, "TITAN_WORKER_SCOPE_SCHEMA")
        patients = np.asarray(scope["patient_rows"])
        owners = np.asarray(scope["owners"])
    require(patients.shape == (409,) and patients.dtype.kind in "iu"
            and len(np.unique(patients)) == 409 and (patients >= 0).all() and (patients < 512).all(),
            "TITAN_WORKER_PATIENT_SCOPE")
    require(owners.shape == (519,) and owners.dtype.kind in "iu" and set(owners) == set(range(512)),
            "TITAN_WORKER_SLIDE_OWNER_SCOPE")
    with np.load(args.cache, allow_pickle=False) as cache:
        require(cache["concept_ids"].astype(str).tolist() == [x["id"] for x in vocabulary["concepts"]],
                "TITAN_WORKER_CONCEPT_ORDER")
        native = np.array(cache["native"], dtype=np.float32)
        semantic = np.array(cache["semantic"], dtype=np.float32)
        old_text = np.array(cache["text_prototypes"], dtype=np.float32)
        scores = np.array(cache["concept_cosine"], dtype=np.float32)
    require(native.shape == semantic.shape == (512, 768) and old_text.shape == (32, 768)
            and scores.shape == (512, 32), "TITAN_WORKER_CACHE_SHAPE")
    proof = verify_model_manifest(args.model, "MahmoodLab/TITAN")
    sources = {"worker_source_sha256": file_sha(__file__),
        "prepare_helper_source_sha256": file_sha(ROOT / "scripts/semantic_b0b7_prepare.py"),
        "concept_file_sha256": file_sha(args.concepts), "scope_sha256": file_sha(args.scope),
        "patient_cache_sha256": file_sha(args.cache),
        "inventory_sha256": file_sha(args.inventory) if args.inventory else None,
        "model_manifest_sha256": proof["manifest_sha256"]}
    args.output_newdir.mkdir(parents=True, exist_ok=False)
    model = load_official(args.model, "MahmoodLab/TITAN", args.device,
                          args.output_newdir / "official_dynamic_modules")
    text, text_audit = official_text(model, vocabulary, "titan", args.device)
    inventory = read_json(args.inventory) if args.inventory else None
    score_audit = check_titan_scores(model, native, semantic, text, old_text, scores,
        owners, patients, args.device, args.slide_cache, inventory)
    del model
    torch.cuda.empty_cache()
    text_path = args.output_newdir / "text32.private.npy"
    with text_path.open("xb") as handle:
        np.save(handle, text, allow_pickle=False)
    audit = {"schema": "semantic_b0b7_isolated_TITAN_audit_v1",
        "status": "EXACT_OFFICIAL_LOAD_AND_FROZEN_NUMERICAL_GATES_COMPLETE",
        "official_loading_exact": True, "sources": sources, "model": proof,
        "text": text_audit, "zero_shot": score_audit,
        "patients_in_scope": 409, "patient_expression_or_clinical_inputs_received": False,
        "new_neural_fits": 0, "optimizer_updates": 0,
        "runtime": {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "timm", "numpy", "safetensors")},
        "elapsed_seconds": time.monotonic() - started}
    audit_path = args.output_newdir / "AUDIT.private.json"
    write_new_json(audit_path, audit)
    receipt = {"schema": "semantic_b0b7_isolated_TITAN_receipt_v1", "status": "COMPLETE_FROZEN_TITAN_ONLY",
        "sources": sources, "text_sha256": file_sha(text_path), "audit_sha256": file_sha(audit_path),
        "text_shape": list(text.shape), "new_neural_fits": 0, "optimizer_updates": 0}
    write_new_json(args.output_newdir / "COMPLETE.private.json", receipt)
    print(json.dumps({"status": receipt["status"], "text_sha256": receipt["text_sha256"],
        "real_slide_embeddings_checked": score_audit["real_slide_embeddings_checked"],
        "official_zero_shot_max_abs_error": score_audit["official_zero_shot_max_abs_error"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
