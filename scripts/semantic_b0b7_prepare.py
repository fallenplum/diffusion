"""Outcome-free P1 inputs: pinned official text/projections and train-only preprocessing.

Only frozen inference and preprocessing occur here; no neural/survival fitting.
Original files remain read-only. Output must be a new private directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic_b0b7.data import (ProgramTransform, array_sha, clinical_design,
                                    pretext_split, read_membership)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify_model_manifest(directory, expected_repo):
    directory = Path(directory).resolve()
    manifest_path = directory / "DOWNLOAD_COMPLETE.json"
    manifest = read_json(manifest_path)
    require(manifest.get("repo") == expected_repo, "OFFICIAL_REPOSITORY_CHANGED")
    entries = manifest.get("files")
    require(isinstance(entries, list) and bool(entries), "MISSING_MODEL_MANIFEST_ENTRIES")
    verified = {}
    for entry in entries:
        name = entry.get("file", entry.get("filename"))
        require(isinstance(name, str) and name not in verified, "INVALID_MODEL_MANIFEST_PATH")
        target = (directory / name).resolve()
        require(target.is_relative_to(directory) and target.is_file(), "MODEL_PATH_ESCAPE_OR_MISSING")
        require(target.stat().st_size == int(entry["bytes"]) and file_sha(target) == entry["sha256"],
                "OFFICIAL_MODEL_FILE_HASH_CHANGED")
        verified[name] = entry["sha256"]
    require("config.json" in verified and any(p.endswith((".safetensors", ".bin")) for p in verified),
            "MANIFEST_MUST_BIND_CONFIG_AND_WEIGHTS")
    config = read_json(directory / "config.json")
    for value in config.get("auto_map", {}).values():
        if isinstance(value, str):
            require(value.split(".")[0] + ".py" in verified, "AUTO_MODEL_SOURCE_NOT_HASH_BOUND")
    return {"repo": expected_repo, "revision": manifest.get("revision"),
            "manifest_sha256": file_sha(manifest_path), "files_sha256": verified}


def load_official(directory, repo, device, code_cache):
    from transformers import AutoModel, PreTrainedTokenizerFast
    from transformers import dynamic_module_utils
    original = PreTrainedTokenizerFast.from_pretrained

    # Original folders share basename "model", but their tokenizers differ.
    # Distinct private aliases avoid Python remote-code namespace collisions.
    # Small verified files are copied exactly; large weights are read through
    # local symlinks.  No original file, weight or model arithmetic changes.
    code_cache = Path(code_cache).resolve()
    code_cache.mkdir(parents=True, exist_ok=True)
    directory = Path(directory).resolve()
    alias = code_cache.parent / (repo.split("/")[-1].lower().replace("-", "_") + "_pinned")
    alias.mkdir(parents=True, exist_ok=False)
    for entry in read_json(directory / "DOWNLOAD_COMPLETE.json")["files"]:
        name = entry.get("file", entry.get("filename"))
        source = directory / name
        target = alias / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_size > 8 * 1024**2:
            require(source.suffix in {".safetensors", ".bin", ".pt"}, "UNEXPECTED_LARGE_MODEL_SIDE_FILE")
            target.symlink_to(source.resolve())
        else:
            shutil.copyfile(source, target)
        require(file_sha(target) == entry["sha256"], "OFFICIAL_IO_ALIAS_CHANGED_BYTES")
    dynamic_name = "transformers_modules"
    package = code_cache / dynamic_name / alias.name
    package.mkdir(parents=True, exist_ok=False)
    for path in (code_cache, code_cache / dynamic_name, package):
        (path / "__init__.py").touch(exist_ok=True)
    for source in alias.glob("*.py"):
        shutil.copyfile(source, package / source.name)
    if str(code_cache) not in sys.path:
        sys.path.insert(0, str(code_cache))

    def local_tokenizer(name, *args, **kwargs):
        if str(name) in {"MahmoodLab/TITAN", "MahmoodLab/CONCH-Z", repo}:
            name = str(alias)
        kwargs["local_files_only"] = True
        return original(name, *args, **kwargs)

    # Official TITAN hard-codes its tokenizer repository; only I/O is redirected.
    with patch.object(dynamic_module_utils, "HF_MODULES_CACHE", str(code_cache)):
        with patch.object(PreTrainedTokenizerFast, "from_pretrained", side_effect=local_tokenizer):
            model, loading = AutoModel.from_pretrained(str(alias), trust_remote_code=True,
                local_files_only=True, output_loading_info=True, torch_dtype=torch.float32)
            if repo == "MahmoodLab/CONCH-Z":
                _ = model.tokenizer
    require(not any(loading.get(k) for k in
            ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            "NONEXACT_OFFICIAL_WEIGHT_LOADING")
    model.eval().requires_grad_(False).to(device)
    require(not any(p.requires_grad for p in model.parameters()), "FOUNDATION_MODEL_NOT_FROZEN")
    return model


def official_text(model, vocabulary, kind, device):
    prompts = [t.replace("CLASSNAME", c["text"]) for c in vocabulary["concepts"]
               for t in vocabulary["templates"]]
    tokenizer = model.tokenizer if kind == "conch" else model.text_encoder.tokenizer
    with torch.inference_mode():
        per_template = model.encode_text(tokenizer(prompts).to(device), normalize=True).float()
        manual = F.normalize(per_template.reshape(32, 2, -1).mean(1), dim=-1)
        official = model.zero_shot_classifier([[x["text"]] for x in vocabulary["concepts"]],
            vocabulary["templates"], device=device).float().T
    require(tuple(manual.shape) == (32, 768) and torch.isfinite(manual).all(), "TEXT_PROTOTYPE_SHAPE")
    error = float((manual - official).abs().max())
    require(error <= 2e-6, "OFFICIAL_TEMPLATE_ENSEMBLE_MISMATCH")
    return official.cpu().numpy(), {"concepts": 32, "templates": 2,
        "official_classifier_max_abs_error": error, "precision": "float32",
        "template_rule": "normalize encoded templates, mean two templates, normalize mean"}


def coarse_stage(values):
    result = []
    for value in np.asarray(values).astype(str):
        value = re.sub(r"^STAGE\s*", "", value.strip().upper())
        match = re.fullmatch(r"(IV|III|II|I)(?:[ABC])?", value)
        result.append(match.group(1) if match else "UNKNOWN")
    return np.asarray(result, dtype="U7")


def group_text(conch, titan, *, seed=103031):
    geometry = np.concatenate([conch, titan], axis=1).astype(np.float64) / np.sqrt(2.)
    labels = KMeans(n_clusters=8, random_state=seed, n_init=20).fit_predict(geometry)
    order = sorted(np.unique(labels), key=lambda k: int(np.flatnonzero(labels == k)[0]))
    labels = np.asarray([order.index(x) for x in labels], np.int64)
    prototypes = []
    for original in (conch, titan):
        means = np.stack([original[labels == k].mean(0) for k in range(8)])
        prototypes.append(means / np.linalg.norm(means, axis=1, keepdims=True))
    return labels, prototypes[0].astype(np.float32), prototypes[1].astype(np.float32)


def shuffle_relation_support(prior, *, seed=103037, swaps=100):
    """Preserve binary degrees and row weight multisets, not weighted columns."""
    value = np.asarray(prior, np.float32).copy()
    original = value.copy()
    rng = np.random.default_rng(seed)
    successful = 0
    for _ in range(max(1000, swaps * 100)):
        edges = np.argwhere(value > 0)
        if len(edges) < 2:
            break
        (r1, c1), (r2, c2) = edges[rng.choice(len(edges), 2, replace=False)]
        if r1 == r2 or c1 == c2 or value[r1, c2] > 0 or value[r2, c1] > 0:
            continue
        w1, w2 = value[r1, c1], value[r2, c2]
        value[r1, c1] = value[r2, c2] = 0
        value[r1, c2], value[r2, c1] = w1, w2
        successful += 1
        if successful >= swaps and not np.array_equal(value, original):
            break
    changed = not np.array_equal(value, original)
    require(np.array_equal((value > 0).sum(0), (original > 0).sum(0))
            and np.array_equal((value > 0).sum(1), (original > 0).sum(1)), "RELATION_DEGREES_CHANGED")
    require(all(np.array_equal(np.sort(a[a > 0]), np.sort(b[b > 0])) for a, b in zip(value, original)),
            "RELATION_ROW_WEIGHT_MULTISET_CHANGED")
    return value, {"status": "AVAILABLE" if changed else "UNAVAILABLE_NO_VALID_SUPPORT_SWAP",
        "seed": seed, "successful_swaps": successful, "binary_row_degrees_preserved": True,
        "binary_column_degrees_preserved": True, "row_weight_multisets_preserved": True,
        "weighted_column_sums_claimed_preserved": False,
        "weighted_column_sum_max_change": float(np.max(np.abs(value.sum(0) - original.sum(0)))),
        "changed_entries": int((value != original).sum()), "only_relation_matrix_changed": True}


def pack_patient_patches(cache, patients, *, budget=512):
    require(cache["audit"]["selector"] == "REGIONS" and cache["audit"]["budget_per_patient"] == budget,
            "WRONG_FROZEN_PATCH_RECIPE")
    owners = np.asarray(cache["owner"], np.int64)
    require(len(owners) == 519 and set(owners) == set(range(512)), "PATCH_SLIDE_PATIENT_COVERAGE")
    features = np.zeros((len(patients), budget, 768), np.float32)
    mask = np.zeros((len(patients), budget), bool)
    slide_ids = np.full((len(patients), budget), -1, np.int64)
    counts, slide_counts = [], []
    for local, patient in enumerate(patients):
        offset = 0
        slides = np.flatnonzero(owners == patient)
        require(len(slides) > 0, "PATIENT_WITHOUT_REAL_SLIDE")
        for local_slide, slide in enumerate(slides):
            valid = np.asarray(cache["mask"][slide], bool)
            x = np.asarray(cache["features"][slide][valid], np.float32)
            require(len(x) > 0 and offset + len(x) <= budget, "PATIENT_PATCH_BUDGET_OR_EMPTY_SLIDE")
            require(np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1, atol=2e-6, rtol=0),
                    "LEGACY_PATCH_L2_CONTRACT_CHANGED")
            features[local, offset:offset + len(x)] = x
            mask[local, offset:offset + len(x)] = True
            slide_ids[local, offset:offset + len(x)] = local_slide
            offset += len(x)
        counts.append(offset)
        slide_counts.append(len(slides))
    return features, mask, slide_ids, {"patients": len(patients), "patch_budget_max": budget,
        "selected_patches": int(mask.sum()), "minimum_patches": min(counts), "maximum_patches": max(counts),
        "patients_with_multiple_slides": sum(x > 1 for x in slide_counts),
        "slide_aggregation": "pool within actual slide then equal valid-slide patient weighting",
        "patch_values": "legacy CONCH raw-feature coordinates, per-patch L2; not yet text-aligned"}


def project_patches(model, features, mask, slide_ids, device):
    aligned = np.zeros_like(features)
    xflat, aflat = features.reshape(-1, 768), aligned.reshape(-1, 768)
    valid, sample = np.flatnonzero(mask.ravel()), []
    checked_slides = 0
    for patient in range(min(40, len(features))):
        for slide in np.unique(slide_ids[patient][mask[patient]]):
            ix = np.flatnonzero(mask[patient] & (slide_ids[patient] == slide))[:32]
            sample.append(features[patient, ix])
            checked_slides += 1
    with torch.inference_mode():
        x = torch.from_numpy(np.concatenate(sample)).to(device)
        official = model.project_image_features(x)
        literal = F.normalize(x @ model.visual.proj_contrast, dim=-1)
        error = float((official - literal).abs().max())
        require(error <= 1e-6, "OFFICIAL_CONCH_PROJECTION_MISMATCH")
        for start in range(0, len(valid), 8192):
            rows = valid[start:start + 8192]
            value = model.project_image_features(torch.from_numpy(xflat[rows]).to(device)).float().cpu().numpy()
            require(np.isfinite(value).all(), "NONFINITE_PROJECTED_CONCH_PATCH")
            aflat[rows] = value
    require((aligned[~mask] == 0).all(), "PADDED_PROJECTION_NOT_ZERO")
    return aligned, {"real_slides_checked": checked_slides, "real_patches_checked": sum(len(x) for x in sample),
        "official_vs_literal_max_abs_error": error, "all_selected_patches_officially_projected": len(valid),
        "raw_CONCH_text_cosine_used": False}


def check_titan_scores(model, native, semantic, text, old_text, old_scores, owners,
                       patient_rows, device, slide_directory=None, inventory=None):
    projection = model.vision_encoder.proj.detach().float()
    classifier = torch.from_numpy(text.T.copy()).to(device)
    rows = []
    cache_kind = "single_slide_patient_native_cache"
    if slide_directory is not None and Path(slide_directory).is_dir() and inventory is not None:
        chosen = sorted(inventory["chosen_conch"], key=lambda r: (int(r["patient_index"]), r["stem"]))
        allowed = set(int(v) for v in patient_rows)
        for item in chosen:
            if int(item["patient_index"]) not in allowed:
                continue
            path = Path(slide_directory) / (item["stem"] + ".npz")
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as data:
                require(str(data["source_sha256"].item()) == item["sha256"], "TITAN_SLIDE_SOURCE_CHANGED")
                z = np.asarray(data["native"], np.float32).reshape(-1)
            require(z.shape == (768,) and np.isfinite(z).all(), "TITAN_NATIVE_SLIDE_SHAPE")
            rows.append(z)
        if rows:
            cache_kind = "original_per_slide_native_cache"
    if not rows:
        for patient in patient_rows:
            if int((owners == patient).sum()) == 1:
                rows.append(native[int(patient)])
            if len(rows) >= 40:
                break
    require(len(rows) >= 32, "INSUFFICIENT_REAL_SLIDE_ZERO_SHOT_QUALIFICATION")
    z = torch.from_numpy(np.stack(rows)).to(device)
    with torch.inference_mode():
        literal = (F.normalize(z @ projection, dim=-1) @ classifier).softmax(-1)
        official = model.zero_shot(z, classifier).reshape(len(rows), 32)
    error = float((literal - official).abs().max())
    require(error <= 2e-6, "OFFICIAL_TITAN_ZERO_SHOT_MISMATCH")
    old_error = float(np.max(np.abs(semantic[patient_rows] @ old_text.T - old_scores[patient_rows])))
    require(old_error <= 2e-6, "TITAN_CACHED_COSINE_REPLAY_CHANGED")
    old_new_text_difference = float(np.max(np.abs(text - old_text)))
    # Old official inference was fp16; this new official equality probe is fp32.
    require(old_new_text_difference <= 1e-3, "TITAN_TEXT_CACHE_IDENTITY_OR_PRECISION_MISMATCH")
    return {"qualification_cache": cache_kind, "real_slide_embeddings_checked": len(rows),
        "official_zero_shot_max_abs_error": error, "historical_cosine_replay_max_error": old_error,
        "current_fp32_vs_historical_text_max_abs_difference": old_new_text_difference,
        "historical_text_precision_tolerance": 0.001, "new_raw_WSI_inference_executed": False,
        "native_vs_semantic_patient_aggregation_conflated": False,
        "clinical_or_molecular_inputs_used_in_zero_shot_check": False}


def legacy_data_module(directory):
    path = Path(directory) / "src/bgb_en/data.py"
    spec = importlib.util.spec_from_file_location("semantic_b0b7_readonly_legacy_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def run_titan_worker(args, owners, patients):
    """Run unmodified official TITAN in its separately qualified interpreter."""
    require(args.titan_python.is_file(), "TITAN_WORKER_INTERPRETER_MISSING")
    worker = ROOT / "scripts/semantic_b0b7_titan_worker.py"
    require(worker.is_file(), "TITAN_WORKER_SOURCE_MISSING")
    scope = args.output_newdir / "TITAN_WORKER_SCOPE.private.npz"
    with scope.open("xb") as handle:
        np.savez(handle, patient_rows=np.asarray(patients, np.int64), owners=np.asarray(owners, np.int64))
    output = args.output_newdir / "titan_isolated_worker"
    command = [str(args.titan_python), str(worker), "--model", str(args.titan_model),
        "--cache", str(args.titan_cache), "--scope", str(scope), "--concepts", str(args.concepts),
        "--slide-cache", str(args.titan_slide_cache or args.titan_cache.parent / "slides"),
        "--output-newdir", str(output), "--device", str(args.device)]
    assets = read_json(args.assets)
    inventory = Path(assets["conch_inventory"]) if assets.get("conch_inventory") else None
    if inventory is not None:
        command += ["--inventory", str(inventory)]
    expected = {"worker_source_sha256": file_sha(worker), "prepare_helper_source_sha256": file_sha(__file__),
        "concept_file_sha256": file_sha(args.concepts), "scope_sha256": file_sha(scope),
        "patient_cache_sha256": file_sha(args.titan_cache),
        "inventory_sha256": file_sha(inventory) if inventory is not None else None,
        "model_manifest_sha256": file_sha(args.titan_model / "DOWNLOAD_COMPLETE.json")}
    with (args.output_newdir / "TITAN_WORKER.private.log").open("x", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    require(process.returncode == 0, "TITAN_ISOLATED_WORKER_FAILED_SEE_PRIVATE_LOG")
    receipt = read_json(output / "COMPLETE.private.json")
    require(receipt.get("status") == "COMPLETE_FROZEN_TITAN_ONLY" and receipt.get("sources") == expected,
            "TITAN_WORKER_RECEIPT_OR_SOURCE_BINDING_CHANGED")
    text_path, audit_path = output / "text32.private.npy", output / "AUDIT.private.json"
    require(file_sha(text_path) == receipt["text_sha256"] and file_sha(audit_path) == receipt["audit_sha256"],
            "TITAN_WORKER_RETURNED_BYTES_CHANGED")
    text, audit = np.load(text_path, allow_pickle=False), read_json(audit_path)
    require(text.shape == (32, 768) and text.dtype == np.float32 and np.isfinite(text).all(),
            "TITAN_WORKER_RETURNED_TEXT_INVALID")
    require(np.allclose(np.linalg.norm(text, axis=1), 1., atol=2e-6, rtol=0),
            "TITAN_WORKER_RETURNED_TEXT_NOT_UNIT_NORMALIZED")
    require(audit["official_loading_exact"] is True and audit["sources"] == expected
            and audit["new_neural_fits"] == 0 and audit["optimizer_updates"] == 0,
            "TITAN_WORKER_EXACT_LOADING_OR_NO_TRAINING_GATE")
    require(audit["text"]["official_classifier_max_abs_error"] <= 2e-6
            and audit["zero_shot"]["official_zero_shot_max_abs_error"] <= 2e-6
            and audit["zero_shot"]["real_slide_embeddings_checked"] >= 32,
            "TITAN_WORKER_RETURNED_NUMERICAL_GATE_FAILED")
    binding = {"mode": "isolated_official_TITAN_environment", "sources": expected,
        "receipt_sha256": file_sha(output / "COMPLETE.private.json"), "runtime": audit["runtime"],
        "worker_subprocess_returncode": process.returncode, "official_math_or_weight_edits": False}
    return text, audit["text"], audit["zero_shot"], binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--conch-model", type=Path, required=True)
    parser.add_argument("--titan-model", type=Path, required=True)
    parser.add_argument("--titan-cache", type=Path, required=True)
    parser.add_argument("--titan-slide-cache", type=Path)
    parser.add_argument("--titan-python", type=Path,
                        help="Separate qualified TITAN interpreter; keeps official code and weights unchanged")
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output-newdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    require(not args.output_newdir.exists(), "REFUSE_TO_OVERWRITE_PRIVATE_PREPARATION")
    require(str(args.device).startswith("cuda") and torch.cuda.is_available(), "REAL_GPU_QUALIFICATION_REQUIRED")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    assets, vocabulary, prior = read_json(args.assets), read_json(args.concepts), read_json(args.prior)
    require(len(vocabulary["concepts"]) == 32 and len(vocabulary["templates"]) == 2,
            "FROZEN_32_CONCEPT_TWO_TEMPLATE_CONTRACT")
    concept_ids = [x["id"] for x in vocabulary["concepts"]]
    require(prior["concept_ids"] == concept_ids and prior["matrix_shape"] == [32, 50], "PRIOR_CONCEPT_ORDER_CHANGED")
    require(file_sha(args.concepts) == prior["source_files"]["frozen_concepts"]["sha256"],
            "FROZEN_CONCEPT_TEXT_OR_TEMPLATE_BYTES_CHANGED")
    old, legacy_file = legacy_data_module(args.legacy_root)
    raw = old.load_raw(assets, include_endpoints=False)
    require(not raw["audit"]["endpoint_read"] and not any(k in raw for k in ("t", "e", "pfi_t", "pfi_e")),
            "OUTCOME_ARRAY_ENTERED_PRETEXT_READER")
    train_global, valid_global, sealed = pretext_split(raw["folds"], outer_fold=1, inner_partition=0)
    patients = np.flatnonzero(np.asarray(raw["folds"]) != 1)
    require(len(patients) == 409 and len(sealed) == 103 and
            np.array_equal(np.sort(np.r_[train_global, valid_global]), patients), "P1_PATIENT_SCOPE_CHANGED")
    global_to_local = np.full(512, -1, np.int64)
    global_to_local[patients] = np.arange(len(patients))
    train, valid = global_to_local[train_global], global_to_local[valid_global]
    require(not np.intersect1d(patients, sealed).size, "SEALED_OUTER_ROWS_IN_PRETEXT_CACHE")
    cache = old.read_patch_cache(assets["patch_regions"], verify_hash=True)
    features, mask, patch_slide, patch_audit = pack_patient_patches(cache, patients)
    gmt_path = Path(assets["gmt"])
    if assets.get("gmt_sha256"):
        require(file_sha(gmt_path) == assets["gmt_sha256"], "FROZEN_GMT_CHANGED")
    gmt = gmt_path.read_text(encoding="utf-8-sig")
    names, rna_membership = read_membership(gmt, raw["rna_symbols"])
    cna_names, cna_membership = read_membership(gmt, raw["cna_symbols"])
    require(names == cna_names and set(names) == set(prior["program_names"]), "RNA_CNA_PRIOR_PROGRAM_NAMES_DIFFER")
    columns = [prior["program_names"].index(name) for name in names]
    relation32 = np.asarray(prior["allowed_mask"], np.float32)[:, columns]
    require(relation32.shape == (32, 50) and int(relation32.sum()) == 15, "CANDIDATE_PRIOR_CHANGED")
    # Original measured masks are restored BEFORE any train statistic.
    rna = np.where(np.asarray(raw["rna_measured"])[patients], np.asarray(raw["rna"])[patients], np.nan)
    cna = np.where(np.asarray(raw["cna_measured"])[patients], np.asarray(raw["cna"])[patients], np.nan)
    transform = ProgramTransform.fit(rna, cna, train, rna_membership, cna_membership,
                                     min_coverage=0.5, min_genes=5)
    molecular = transform.transform(rna, cna)
    clinical_raw = {key: np.asarray(value)[patients].astype(str) for key, value in raw["clinical_raw"].items()}
    clinical, clinical_state = clinical_design(clinical_raw, train)
    with np.load(args.titan_cache, allow_pickle=False) as data:
        require(np.array_equal(data["patient_ids"].astype(str), raw["patient_ids"]), "TITAN_PATIENT_ORDER_CHANGED")
        require(data["concept_ids"].astype(str).tolist() == concept_ids, "TITAN_CONCEPT_ORDER_CHANGED")
        native = np.array(data["native"], dtype=np.float32)
        aligned_titan = np.array(data["semantic"], dtype=np.float32)
        old_text = np.array(data["text_prototypes"], dtype=np.float32)
        scores = np.array(data["concept_cosine"], dtype=np.float32)
    require(native.shape == aligned_titan.shape == (512, 768) and scores.shape == (512, 32), "TITAN_CACHE_SHAPE_CHANGED")
    require(np.allclose(scores[patients], raw["S32"][patients], atol=2e-6, rtol=0), "SHARED_S32_INPUT_CHANGED")
    require(np.allclose(native[patients], raw["titan"][patients], atol=2e-6, rtol=0), "SHARED_NATIVE_TITAN_INPUT_CHANGED")
    conch_proof = verify_model_manifest(args.conch_model, "MahmoodLab/CONCH-Z")
    titan_proof = verify_model_manifest(args.titan_model, "MahmoodLab/TITAN")
    args.output_newdir.mkdir(parents=True, exist_ok=False)
    conch = load_official(args.conch_model, "MahmoodLab/CONCH-Z", args.device,
                          args.output_newdir / "official_dynamic_modules")
    conch_text, conch_text_audit = official_text(conch, vocabulary, "conch", args.device)
    projected, conch_projection_audit = project_patches(conch, features, mask, patch_slide, args.device)
    del conch
    torch.cuda.empty_cache()
    if args.titan_python is not None:
        titan_text, titan_text_audit, titan_score_audit, titan_environment = run_titan_worker(
            args, np.asarray(cache["owner"]), patients)
    else:
        titan = load_official(args.titan_model, "MahmoodLab/TITAN", args.device,
                              args.output_newdir / "official_dynamic_modules")
        titan_text, titan_text_audit = official_text(titan, vocabulary, "titan", args.device)
        inventory = read_json(assets["conch_inventory"]) if assets.get("conch_inventory") else None
        slide_dir = args.titan_slide_cache or args.titan_cache.parent / "slides"
        titan_score_audit = check_titan_scores(titan, native, aligned_titan, titan_text, old_text, scores,
            np.asarray(cache["owner"]), patients, args.device, slide_dir, inventory)
        del titan
        torch.cuda.empty_cache()
        titan_environment = {"mode": "same_parent_environment", "official_math_or_weight_edits": False}
    groups, conch_groups, titan_groups = group_text(conch_text, titan_text)
    relation = np.stack([relation32[groups == k].mean(0) for k in range(8)])
    shuffled, shuffle_audit = shuffle_relation_support(relation)
    require(shuffle_audit["status"] == "AVAILABLE", "RELATION_NEGATIVE_CONTROL_NOT_CONSTRUCTIBLE")
    sources = {"preparation_source_sha256": file_sha(__file__),
        "semantic_data_source_sha256": file_sha(ROOT / "src/semantic_b0b7/data.py"),
        "legacy_data_source_sha256": file_sha(legacy_file), "assets_config_sha256": file_sha(args.assets),
        "concept_file_sha256": file_sha(args.concepts), "prior_file_sha256": file_sha(args.prior),
        "GMT_sha256": file_sha(gmt_path), "titan_patient_cache_sha256": file_sha(args.titan_cache),
        "patch_READY_sha256": file_sha(Path(assets["patch_regions"]) / "READY.json"),
        "canonical_covariate_container_sha256": raw["audit"]["cohort_sha256"]}
    if args.titan_python is not None:
        sources["titan_worker_source_sha256"] = file_sha(ROOT / "scripts/semantic_b0b7_titan_worker.py")
    provenance = {"schema": "semantic_b0b7_P1_inputs_v1", "status": "FROZEN_INPUTS_PREPARED_NO_MODEL_TRAINING",
        "sources": sources, "patients": 409, "outer_fold": 1, "inner_partition": 0,
        "train_global_rows_sha256": array_sha(train_global), "valid_global_rows_sha256": array_sha(valid_global),
        "sealed_outer_rows_sha256": array_sha(sealed), "train_local_rows_sha256": array_sha(train),
        "valid_local_rows_sha256": array_sha(valid), "train_n": len(train), "valid_n": len(valid),
        "outcome_arrays_in_payload": False, "outcome_arrays_loaded": [],
        "global_covariate_container_read": True, "sealed_outer_input_rows_stored_in_payload": 0,
        "preprocessing_fit_scope": "only the outcome-blind current inner-training rows",
        "patch": patch_audit, "conch_model": conch_proof, "titan_model": titan_proof,
        "conch_text": conch_text_audit, "titan_text": titan_text_audit,
        "conch_projection": conch_projection_audit, "titan_zero_shot": titan_score_audit,
        "titan_environment": titan_environment,
        "text_grouping": {"algorithm": "KMeans", "clusters": 8, "random_state": 103031, "n_init": 20,
            "geometry": "concatenate unit CONCH-Z and TITAN prototypes, divide by sqrt(2)",
            "group_labels_by_concept_order": groups.tolist(), "group_sizes": np.bincount(groups, minlength=8).tolist(),
            "patient_data_used": False},
        "relation_aggregation": "mean original32 candidate rows within fixed text group; columns joined by exact name",
        "relation_shuffle": shuffle_audit, "molecular_fit_hash": molecular["fit_hash"],
        "rna_log_transform_applied": False,
        "CNA_channels": "gain/loss split before observed-only train scale; dosage not pathway activity",
        "stage_pairing": "coarse I/II/III/IV/UNKNOWN; model retains original six fields",
        "new_neural_fits": 0, "optimizer_updates": 0, "elapsed_seconds": time.monotonic() - started,
        "official_model_io": "distinct private conch_z_pinned/titan_pinned aliases; exact small-file copies, symlinks to verified existing weights, isolated official source-copy module cache; no mathematical edits",
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": str(args.device)}}
    image = {"patch_values": torch.from_numpy(features), "patch_aligned": torch.from_numpy(projected),
        "patch_mask": torch.from_numpy(mask), "patch_slide": torch.from_numpy(patch_slide),
        "titan_native": torch.from_numpy(native[patients].copy()),
        "titan_aligned": torch.from_numpy(aligned_titan[patients].copy()),
        "titan_scores": torch.from_numpy(scores[patients].copy()),
        "clinical": torch.from_numpy(clinical.astype(np.float32))}
    payload = {"image": image, "omics": {"values": torch.from_numpy(molecular["values"]),
            "measured": torch.from_numpy(molecular["measured"])},
        "target": torch.from_numpy(molecular["values"][..., 0].copy()),
        "target_mask": torch.from_numpy(molecular["measured"][..., 0].copy()),
        "coverage": torch.from_numpy(molecular["coverage"]),
        "train_indices": torch.from_numpy(train), "valid_indices": torch.from_numpy(valid),
        "global_rows": torch.from_numpy(patients),
        "metadata": {"patient": patients.copy(),
            "stage": coarse_stage(clinical_raw["AJCC_PATHOLOGIC_TUMOR_STAGE"]),
            "cohort": np.asarray(raw["cancer"])[patients].astype(str),
            "missing_pattern": np.asarray(raw["platform"])[patients].astype(np.int64)},
        "clinical_raw": clinical_raw, "conch_prototypes": torch.from_numpy(conch_groups),
        "titan_prototypes": torch.from_numpy(titan_groups), "relation_prior": torch.from_numpy(relation),
        "shuffled_relation_prior": torch.from_numpy(shuffled), "program_names": list(names),
        "provenance": provenance}
    require(not {"time", "event", "t", "e", "DSS", "OS", "PFI", "risk", "offset"}.intersection(payload),
            "PROHIBITED_OUTCOME_OR_RISK_FIELD_IN_CACHE")
    for name, value in image.items():
        require(len(value) == 409 and (not value.is_floating_point() or torch.isfinite(value).all()),
                "INVALID_IMAGE_CACHE_TENSOR_" + name)
    destination = args.output_newdir / "P1_INPUTS.private.pt"
    with destination.open("xb") as handle:
        torch.save(payload, handle)
    preprocessing = args.output_newdir / "P1_PREPROCESSING.private.pt"
    with preprocessing.open("xb") as handle:
        torch.save({"program_transform": transform, "clinical_state": clinical_state,
                    "train_global_rows": train_global, "valid_global_rows": valid_global,
                    "train_local_rows": train, "valid_local_rows": valid}, handle)
    receipt = {"schema": "semantic_b0b7_P1_preparation_receipt_v1", "status": "COMPLETE_FROZEN_INPUTS_ONLY",
        "payload_sha256": file_sha(destination), "payload_bytes": destination.stat().st_size,
        "preprocessing_sha256": file_sha(preprocessing), "sources": sources,
        "patients": 409, "train_n": len(train), "valid_n": len(valid), "sealed_outer_rows_in_cache": 0,
        "new_neural_fits": 0, "optimizer_updates": 0, "conch_projection": conch_projection_audit,
        "titan_zero_shot": titan_score_audit, "text_grouping": provenance["text_grouping"],
        "titan_environment": titan_environment,
        "relation_shuffle": shuffle_audit, "elapsed_seconds": time.monotonic() - started}
    write_new_json(args.output_newdir / "COMPLETE.private.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "payload_sha256", "payload_bytes", "patients",
        "train_n", "valid_n", "sealed_outer_rows_in_cache", "new_neural_fits", "elapsed_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
