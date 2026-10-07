"""Frozen TITAN slide/text inference, outcome-blind geometry, and clinical audit.

Weights, features, patient labels, embeddings and OOF predictions stay local.
Use the official contrastive vision projection, not native TITAN cosine scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import time

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms <= 1e-12):
        raise ValueError("nonfinite or zero embedding")
    return x / norms


def spectrum(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0)
    eig = np.linalg.svd(x, full_matrices=False, compute_uv=False) ** 2
    if eig.sum() <= 1e-20:
        return {"effective_rank": 0.0, "first_component_fraction": 1.0, "stable_rank": 0.0, "total_variance": 0.0}
    prob = eig / eig.sum()
    pos = prob > 1e-15
    return {"effective_rank": float(np.exp(-np.sum(prob[pos] * np.log(prob[pos])))),
            "first_component_fraction": float(prob[0]), "stable_rank": float(eig.sum() / eig[0]),
            "total_variance": float(eig.sum() / (len(x) - 1))}


def normalize_label(value, endpoint):
    if pd.isna(value):
        return None
    value = re.sub(r"\s+", "", str(value).upper())
    if endpoint == "Stage":
        match = re.fullmatch(r"(?:STAGE)?(IV|III|II|I)[ABC]?", value)
        return match.group(1) if match else None
    allowed = {"T": "1234", "N": "012", "M": "01"}[endpoint]
    match = re.fullmatch(endpoint + "([" + allowed + r"])[ABC]?", value)
    return endpoint + match.group(1) if match else None


def load_protocol(args):
    config_path = ROOT / "configs/phase6a_titan_semantic_preflight.json"
    concepts_path = ROOT / "configs/phase6a_titan_crc_morphology_concepts.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vocab = json.loads(concepts_path.read_text(encoding="utf-8"))
    if not 24 <= len(vocab["concepts"]) <= 32:
        raise ValueError("concept vocabulary size out of range")
    if any("CLASSNAME" not in t for t in vocab["templates"]):
        raise ValueError("official templates require CLASSNAME placeholder")
    forbidden = re.compile(r"hallmark|gene.set|survival|\bdss\b|\brna\b|\btnm\b|\bstage\b", re.I)
    if any(forbidden.search(c["text"]) for c in vocab["concepts"]):
        raise ValueError("forbidden molecular/outcome/target prompt")
    if len({c["id"] for c in vocab["concepts"]}) != len(vocab["concepts"]):
        raise ValueError("duplicate concept IDs")
    manifest = pd.read_csv(ROOT / "reports/phase6_conch_v15_download_audit/cohort_manifest.csv", dtype=str)
    ids = sorted(manifest.patient_id.unique())
    matched = manifest.loc[manifest.status == "EXACT_SLIDE_MATCH"].sort_values(["patient_id", "filename"])
    if len(ids) != config["data"]["patients"] or len(matched) != config["data"]["available_slides"]:
        raise ValueError("cohort size changed")
    if set(ids) != set(matched.patient_id):
        raise ValueError("patient without any available CONCH slide")
    lock = {"protocol_sha256": sha256(config_path), "vocabulary_sha256": sha256(concepts_path),
            "script_sha256": sha256(__file__), "input_manifest_sha256": sha256(ROOT / "reports/phase6_conch_v15_download_audit/cohort_manifest.csv"),
            "memory_helper_sha256": sha256(ROOT / "scripts/phase6a_titan_memory.py"),
            "model_revision": config["model"]["revision"]}
    args.cache.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    lock_path = args.cache / "protocol_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        if (args.cache / "patient_representations.npz").exists() or any((args.cache / "slides").glob("*.npz")):
            raise RuntimeError("code/config/manifest changed: use a new cache; never mix revisions")
        # Loader repairs before the first successful embedding can refresh a
        # setup-only lock. Never reuse numerical output across code revisions.
    write_json(lock_path, lock)
    write_json(args.report / "protocol_lock.json", lock)
    return config, vocab, matched, ids


def clinical_preflight(args, config, matched, ids):
    # Load ONLY the explicitly permitted columns; survival outcomes are not inputs.
    cols = config["labels"]["allowed_columns"]
    clinical = pd.read_csv(args.clinical, sep="\t", comment="#", usecols=cols, dtype=str)
    clinical.PATIENT_ID = clinical.PATIENT_ID.str.upper()
    if clinical.PATIENT_ID.duplicated().any():
        raise ValueError("duplicate clinical patient ID")
    missing = sorted(set(ids) - set(clinical.PATIENT_ID))
    if missing:
        raise ValueError(f"missing clinical rows: {len(missing)}")
    clinical = clinical.set_index("PATIENT_ID").loc[ids]
    mapping = {"T": "PATH_T_STAGE", "N": "PATH_N_STAGE", "M": "PATH_M_STAGE", "Stage": "AJCC_PATHOLOGIC_TUMOR_STAGE"}
    rows = {"patient_id": ids}
    counts = {}
    for endpoint, col in mapping.items():
        labels = [normalize_label(v, endpoint) for v in clinical[col]]
        rows[endpoint] = labels
        counts[endpoint] = {"valid": sum(v is not None for v in labels), "excluded": sum(v is None for v in labels),
                            "classes": dict(Counter(v for v in labels if v is not None)),
                            "source_value_counts": clinical[col].fillna("MISSING").value_counts().to_dict()}
    pd.DataFrame(rows).to_csv(args.cache / "clinical_targets.csv", index=False)
    available = sum((args.features / p).is_file() for p in matched.delivery_relative_path)
    if available != len(matched) and not args.streamed_inputs:
        raise RuntimeError(f"only {available}/{len(matched)} H5 files on requested drive")
    result = {"status": "PASS", "patients": len(ids), "available_slides": len(matched),
              "files_present_at_preflight": available, "streamed_inputs": args.streamed_inputs,
              "original_missing_slide_count": config["data"]["original_slides"] - len(matched),
              "slides_per_patient": dict(Counter(Counter(matched.patient_id).values())),
              "clinical_targets": counts, "clinical_columns_loaded": cols, "survival_columns_loaded": [],
              "historical_cohort_membership": "Frozen existing 512-patient cohort; membership is not reselected in Phase6A",
              "inputs_transferred_temporarily": args.streamed_inputs, "permanent_feature_copy_on_server": False}
    write_json(args.report / "data_preflight.json", result)
    print("DATA_PREFLIGHT", json.dumps({k: result[k] for k in ["patients", "available_slides", "clinical_targets"]}), flush=True)


def load_slide(path, row):
    deadline = time.monotonic() + 3600
    while not path.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("streamed input has not arrived within 1 hour")
        time.sleep(0.5)
    if path.stat().st_size != int(row.conch_size_bytes) or sha256(path) != row.conch_sha256:
        raise ValueError("CONCH source checksum changed")
    with h5py.File(path, "r") as h5:
        features = h5["features"][:]
        coords = h5["coords"][:]
        level0_size = int(h5["coords"].attrs["patch_size_level0"])
        if features.ndim != 2 or features.shape[1] != 768 or coords.shape != (len(features), 2):
            raise ValueError("unexpected patch or coordinate shape")
        if not np.isfinite(features).all() or (coords < 0).any() or len(np.unique(coords, axis=0)) != len(coords):
            raise ValueError("nonfinite patches or invalid coordinates")
        if int(h5["coords"].attrs["patch_size"]) != 512 or int(h5["coords"].attrs["target_magnification"]) != 20:
            raise ValueError("incorrect physical patch size")
        grid = (coords - coords.min(axis=0)) // level0_size
        if len(np.unique(grid, axis=0)) != len(coords):
            raise ValueError("coordinate cells collide under official TITAN grid conversion")
    return features, coords, level0_size


def infer(args, config, vocab, matched, ids):
    import torch
    from transformers import AutoModel, PreTrainedTokenizerFast
    from unittest.mock import patch
    from phase6a_titan_memory import optimize_alibi

    proof = json.loads((args.model / "DOWNLOAD_COMPLETE.json").read_text())
    if proof["revision"] != config["model"]["revision"] or not proof["weight_sha256_verified"]:
        raise RuntimeError("missing verified pinned weights")
    for file in proof["files"]:
        if sha256(args.model / file["filename"]) != file["sha256"]:
            raise RuntimeError("model file hash differs from verified download")
    # Transformers 4.46 local dynamic loader omits second-level relative imports.
    # Prepopulate its package with identical reviewed source files from the same
    # verified snapshot. This changes loading I/O, not official model operations.
    module_root = Path(os.environ["HF_MODULES_CACHE"])
    package = module_root / "transformers_modules" / args.model.name
    package.mkdir(parents=True, exist_ok=True)
    for directory in [module_root, module_root / "transformers_modules", package]:
        (directory / "__init__.py").touch(exist_ok=True)
    for source in args.model.glob("*.py"):
        shutil.copyfile(source, package / source.name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently change inference device")
    # Official tokenizer hard-codes HF repo ID. Route that one loader to the same
    # local pinned tokenizer; tokenization and all official model code stay intact.
    original_loader = PreTrainedTokenizerFast.from_pretrained

    def local_tokenizer(name, *a, **kw):
        if name == config["model"]["repo"]:
            name = str(args.model)
            kw["local_files_only"] = True
        return original_loader(name, *a, **kw)

    with patch.object(PreTrainedTokenizerFast, "from_pretrained", side_effect=local_tokenizer):
        model, loading = AutoModel.from_pretrained(str(args.model), trust_remote_code=True,
                                                 local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"nonexact model state load: {loading}")
    model.eval().requires_grad_(False).to("cuda")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    memory_proof = optimize_alibi(model) if args.allocation_mode == "chunked" else {"mode": "unmodified_official", "attention_allocation_modified": False}
    env = {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "transformers", "timm", "einops", "einops-exts", "huggingface-hub", "numpy", "scikit-learn"]}
    write_json(args.report / "runtime.json", {"versions": env, "gpu": torch.cuda.get_device_name(0),
               "parameter_count": sum(p.numel() for p in model.parameters()), "loading_info": loading,
               "official_torch_deviation": f"Uses existing CUDA PyTorch {env['torch']} rather than documented 2.0.1; no architecture/weights changes",
               "tokenizer_io_override": "Official fixed tokenizer files from same pinned local snapshot",
               "precision": "Official CUDA fp16 autocast for encoders; fp32 semantic projection and geometry",
               "model_download": proof})
    write_json(args.report / "allocation_equivalence.json", memory_proof)
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        prototypes = model.zero_shot_classifier([[c["text"]] for c in vocab["concepts"]], vocab["templates"], device="cuda")
    text = unit(prototypes.float().cpu().numpy().T)
    # Text encoding is complete and remains frozen; release its GPU residency
    # before large slide attention. zero_shot uses only the visual projection.
    model.text_encoder.to("cpu")
    del prototypes
    torch.cuda.empty_cache()
    if text.shape[0] != len(vocab["concepts"]):
        raise RuntimeError("incorrect prototype dimensions")
    projection = model.vision_encoder.proj.detach().float().cpu().numpy()
    cache = args.cache / "slides"
    cache.mkdir(exist_ok=True)
    slide_rows, native, semantic, patient_order = [], [], [], []
    start = time.monotonic()
    records = list(matched.itertuples(index=False))
    # Prefetch only one complete next bag: avoids competing quadratic ALiBi RAM
    # allocations while overlapping disk I/O with CUDA inference.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(load_slide, args.features / records[0].delivery_relative_path, records[0])
        for i, row in enumerate(records):
            features, coords, level0_size = future.result()
            if i + 1 < len(records):
                following = records[i + 1]
                future = pool.submit(load_slide, args.features / following.delivery_relative_path, following)
            destination = cache / (Path(row.filename).stem + ".npz")
            if destination.exists():
                stored = np.load(destination, allow_pickle=False)
                z = np.asarray(stored["native"], dtype=np.float32)
                if str(stored["source_sha256"].item()) != row.conch_sha256:
                    raise RuntimeError("stale slide cache")
                inference_seconds = float(stored["inference_seconds"].item())
                peak = float(stored["peak_cuda_gib"].item())
            else:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.monotonic()
                try:
                    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
                        ztensor = model.encode_slide_from_patch_features(torch.from_numpy(features).to("cuda"),
                                                                        torch.from_numpy(coords).to("cuda"), level0_size)
                    z = ztensor.detach().float().cpu().numpy().reshape(-1)
                    del ztensor
                except Exception as exc:
                    write_json(args.report / "inference_failure.json", {"slide_index": i, "patient_count_completed": len(set(patient_order)),
                               "exception_type": type(exc).__name__, "message": str(exc), "patch_count": len(features),
                               "no_patient_dropped": True})
                    raise
                inference_seconds = time.monotonic() - t0
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                unit(z)
                np.savez_compressed(destination, native=z, source_sha256=np.array(row.conch_sha256),
                                    inference_seconds=np.array(inference_seconds), peak_cuda_gib=np.array(peak))
            if z.ndim != 1 or z.shape[0] != projection.shape[0]:
                raise RuntimeError("native dimension/projection mismatch")
            zp = unit(z @ projection)
            cos = zp @ text.T
            # Official zero_shot normalizes native @ proj then softmaxes cosine.
            # Match EVERY slide; never report softmax probabilities as cosine.
            with torch.inference_mode():
                official = model.zero_shot(torch.from_numpy(z[None]).to("cuda"), torch.from_numpy(text.T.copy()).to("cuda")).float().cpu().numpy().reshape(-1)
            exp = np.exp(cos - cos.max())
            diff = float(np.max(np.abs(official - exp / exp.sum())))
            if diff > 2e-6:
                raise RuntimeError(f"official semantic scoring mismatch: {diff}")
            native.append(z)
            semantic.append(zp)
            patient_order.append(row.patient_id)
            slide_rows.append({"slide_index": i, "patches": len(features), "native_norm": float(np.linalg.norm(z)),
                               "projected_norm": float(np.linalg.norm(z @ projection)), "semantic_norm": float(np.linalg.norm(zp)),
                               "inference_seconds": inference_seconds, "peak_cuda_gib": peak, "official_softmax_max_error": diff})
            if (i + 1) % 10 == 0 or i + 1 == len(records):
                print(f"TITAN_SLIDES {i + 1}/{len(records)} patients={len(set(patient_order))} wall_s={time.monotonic()-start:.1f} last_s={inference_seconds:.2f} peak_GiB={peak:.2f}", flush=True)
            if args.streamed_inputs:
                source_path = (args.features / row.delivery_relative_path).resolve()
                source_path.relative_to(args.features.resolve())
                source_path.unlink()
    native, semantic = np.stack(native), np.stack(semantic)
    pn, ps = [], []
    for patient in ids:
        use = np.array(patient_order) == patient
        pn.append(unit(native[use].mean(axis=0)))
        ps.append(unit(semantic[use].mean(axis=0)))
    pn, ps = np.stack(pn), np.stack(ps)
    scores = ps @ text.T
    if scores.shape != (len(ids), len(vocab["concepts"])) or not np.isfinite(scores).all():
        raise RuntimeError("invalid patient cosine scores")
    np.savez_compressed(args.cache / "patient_representations.npz", patient_ids=np.array(ids),
                        native=pn, semantic=ps, concept_cosine=scores, text_prototypes=text,
                        concept_ids=np.array([c["id"] for c in vocab["concepts"]]))
    pd.DataFrame(slide_rows).to_csv(args.report / "slide_inference_diagnostics.csv", index=False)
    write_json(args.report / "embedding_audit.json", {"status": "PASS", "successful_slides": len(records), "patients": len(ids),
               "native_dim": native.shape[1], "semantic_dim": semantic.shape[1], "finite": True,
               "native_slide_norm_range": [float(np.linalg.norm(native, axis=1).min()), float(np.linalg.norm(native, axis=1).max())],
               "patient_native_spectrum": spectrum(pn), "patient_semantic_spectrum": spectrum(ps),
               "patient_norm_range": [float(np.linalg.norm(ps, axis=1).min()), float(np.linalg.norm(ps, axis=1).max())],
               "official_zero_shot_max_error": max(r["official_softmax_max_error"] for r in slide_rows),
               "wall_seconds": round(time.monotonic()-start, 2), "peak_cuda_gib": max(r["peak_cuda_gib"] for r in slide_rows),
               "patient_aggregation": config["aggregation"], "survival_columns_loaded": []})


def geometry(args, config, vocab):
    bundle = np.load(args.cache / "patient_representations.npz", allow_pickle=False)
    scores = bundle["concept_cosine"]
    sd = scores.std(axis=0, ddof=1)
    primary = spectrum(scores)
    standardized = (scores - scores.mean(axis=0)) / np.maximum(sd, 1e-12)
    corr = np.corrcoef(standardized.T)
    np.nan_to_num(corr, copy=False)
    stop = primary["effective_rank"] <= config["collapse_gate"]["stop_if_effective_rank_leq"] or float(sd.max()) < 1e-8
    rows = [{"concept_id": c["id"], "name": c["name"], "mean": float(scores[:, j].mean()),
             "std": float(sd[j]), "min": float(scores[:, j].min()), "max": float(scores[:, j].max())}
            for j, c in enumerate(vocab["concepts"])]
    pd.DataFrame(rows).to_csv(args.report / "concept_variation.csv", index=False)
    labels = [c["id"] for c in vocab["concepts"]]
    pd.DataFrame(corr, index=labels, columns=labels).to_csv(args.report / "concept_correlation.csv")
    pd.DataFrame(bundle["text_prototypes"] @ bundle["text_prototypes"].T, index=labels, columns=labels).to_csv(args.report / "text_prototype_cosine.csv")
    result = {"status": "STOP_COLLAPSE" if stop else "PASS_NONCOLLAPSED", "raw_centered_cosine_spectrum": primary,
              "correlation_spectrum": spectrum(standardized), "std_range": [float(sd.min()), float(sd.max())],
              "median_abs_concept_correlation": float(np.median(np.abs(corr[np.triu_indices(len(corr), 1)]))),
              "constant_concepts": int((sd < 1e-8).sum()), "row_common_component_removed_spectrum": spectrum(scores-scores.mean(axis=1, keepdims=True)),
              "concepts": len(labels), "patients": len(scores), "survival_columns_loaded": []}
    write_json(args.report / "concept_geometry.json", result)
    print("CONCEPT_GEOMETRY", json.dumps(result), flush=True)
    return not stop


def evaluate(args, config):
    from sklearn.decomposition import PCA
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    if json.loads((args.report / "concept_geometry.json").read_text())["status"] != "PASS_NONCOLLAPSED":
        raise RuntimeError("geometry gate did not pass")
    bundle = np.load(args.cache / "patient_representations.npz", allow_pickle=False)
    clinical = pd.read_csv(args.cache / "clinical_targets.csv", dtype=str).set_index("patient_id").loc[bundle["patient_ids"]]
    seed_rows, fit_rows, paired_rows = [], [], []
    ev = config["evaluation"]

    def metrics(y, prob):
        pred = prob.argmax(axis=1)
        auc = roc_auc_score(y, prob[:, 1]) if prob.shape[1] == 2 else roc_auc_score(y, prob, multi_class="ovr", average="macro")
        return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "macro_ovr_auroc": float(auc), "accuracy": float(accuracy_score(y, pred))}

    def bootstrap(y, apred, bpred):
        # Preserve pairing and use the same patient resample for all seeds.
        groups = [np.flatnonzero(y == c) for c in np.unique(y)]
        rng = np.random.default_rng(ev["bootstrap_seed"])
        differences = []
        for _ in range(ev["bootstrap_replicates"]):
            chosen = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            ca = np.stack([np.mean(apred[:, chosen[y[chosen] == c]] == c, axis=1) for c in np.unique(y)]).mean(axis=0)
            cb = np.stack([np.mean(bpred[:, chosen[y[chosen] == c]] == c, axis=1) for c in np.unique(y)]).mean(axis=0)
            differences.append(float((ca - cb).mean()))
        return {"paired_delta_mean": float(np.mean(differences)),
                "paired_delta_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
                "paired_delta_ci_familywise": np.quantile(differences, [0.00625, 0.99375]).tolist(),
                "scope": "Stratified patient bootstrap of fixed OOF predictions; includes no model-refit uncertainty"}

    for endpoint in ev["targets"]:
        good = clinical[endpoint].notna().to_numpy()
        classes = sorted(clinical.loc[good, endpoint].unique())
        y = np.array([classes.index(t) for t in clinical.loc[good, endpoint]], dtype=int)
        if len(classes) < 2 or min(Counter(y).values()) < ev["target_class_min_count"]:
            raise RuntimeError(f"insufficient classes for target {endpoint}")
        matrices = {"TITAN_embedding": bundle["native"][good], "CRC_concept_cosine": bundle["concept_cosine"][good],
                    "TITAN_embedding_PCA32": bundle["native"][good], "prior_only": np.zeros((sum(good), 1))}
        all_predictions = {name: [] for name in matrices}
        for seed in ev["seeds"]:
            outer = list(StratifiedKFold(ev["outer_folds"], shuffle=True, random_state=seed).split(matrices["TITAN_embedding"], y))
            fold_ids = np.full(len(y), -1)
            predictions = {name: np.full((len(y), len(classes)), np.nan) for name in matrices}
            for fold, (train, test) in enumerate(outer):
                if np.intersect1d(train, test).size:
                    raise RuntimeError("patient fold overlap")
                fold_ids[test] = fold
                inner = list(StratifiedKFold(ev["inner_folds"], shuffle=True, random_state=seed+fold+1).split(np.zeros((len(train), 1)), y[train]))
                for name, x in matrices.items():
                    if name == "prior_only":
                        fitted = DummyClassifier(strategy="prior").fit(x[train], y[train])
                        chosen_c = None
                    else:
                        steps = [("scale", StandardScaler())]
                        if name == "TITAN_embedding_PCA32":
                            steps.append(("pca", PCA(n_components=32, svd_solver="full")))
                        steps.append(("head", LogisticRegression(class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=seed)))
                        search = GridSearchCV(Pipeline(steps), {"head__C": ev["C_candidates"]}, cv=inner,
                                              scoring="balanced_accuracy", n_jobs=1, error_score="raise", refit=True)
                        with threadpool_limits(limits=4):
                            search.fit(x[train], y[train])
                        fitted = search.best_estimator_
                        chosen_c = float(search.best_params_["head__C"])
                    predictions[name][test] = fitted.predict_proba(x[test])
                    fit_rows.append({"endpoint": endpoint, "seed": seed, "fold": fold, "model": name,
                                     "n_train": len(train), "n_test": len(test), "input_dim": x.shape[1], "chosen_C": chosen_c})
                print(f"CLINICAL_FOLD {endpoint} seed={seed} fold={fold+1}/5", flush=True)
            destination = args.cache / "oof"
            destination.mkdir(exist_ok=True)
            for name, prob in predictions.items():
                if not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1), 1):
                    raise RuntimeError("incomplete OOF probabilities")
                np.savez_compressed(destination / f"{endpoint}_{seed}_{name}.npz", patient_ids=bundle["patient_ids"][good],
                                    target=y, class_names=np.array(classes), probabilities=prob, outer_fold=fold_ids)
                seed_rows.append({"endpoint": endpoint, "seed": seed, "model": name, "n": len(y), **metrics(y, prob)})
                all_predictions[name].append(prob.argmax(axis=1))
        a = np.stack(all_predictions["CRC_concept_cosine"])
        for comparator in ("TITAN_embedding", "TITAN_embedding_PCA32", "prior_only"):
            b = np.stack(all_predictions[comparator])
            seed_delta = [balanced_accuracy_score(y, ap) - balanced_accuracy_score(y, bp) for ap, bp in zip(a, b)]
            paired_rows.append({"endpoint": endpoint, "comparison": "CRC_concept_cosine-minus-"+comparator,
                                "seed_deltas": [float(d) for d in seed_delta], "all_seed_deltas_positive": all(d > 0 for d in seed_delta), **bootstrap(y, a, b)})
    pd.DataFrame(seed_rows).to_csv(args.report / "seed_oof_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(args.report / "fit_manifest.csv", index=False)
    write_json(args.report / "paired_comparisons.json", paired_rows)
    stable = [r["endpoint"] for r in paired_rows if r["comparison"].endswith("prior_only") and r["all_seed_deltas_positive"] and r["paired_delta_ci_familywise"][0] > 0]
    write_json(args.report / "clinical_information_gate.json", {"status": "PASS_CLINICAL_INFORMATION" if stable else "NO_STABLE_CLINICAL_INFORMATION",
               "passing_targets": stable, "vocabulary_updated": False, "survival_columns_loaded": [],
               "no_further_model_tuning": True, "OOF_predictions_persisted_locally": True,
               "repeated_seeds_are_not_independent_cohorts": True})
    print("CLINICAL_INFORMATION_GATE", stable, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "infer", "geometry", "evaluate", "all"], default="all")
    parser.add_argument("--features", type=Path, default=Path(r"E:\diffusion_conch_v15_features"))
    parser.add_argument("--model", type=Path, default=Path(r"E:\diffusion_phase6a_titan\model"))
    parser.add_argument("--cache", type=Path, default=Path(r"E:\diffusion_phase6a_titan\cache_v2"))
    parser.add_argument("--report", type=Path, default=ROOT / "reports/phase6a_titan_semantic_preflight")
    parser.add_argument("--clinical", type=Path, default=ROOT / "data/data_8/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt")
    parser.add_argument("--streamed-inputs", action="store_true", help="Wait for atomically uploaded H5, remove only consumed H5 under features root")
    parser.add_argument("--allocation-mode", choices=["official", "chunked"], default="official")
    args = parser.parse_args()
    os.environ["HF_MODULES_CACHE"] = str(args.cache / "hf_modules")
    os.environ["HF_HOME"] = str(args.cache / "hf_home")
    config, vocab, matched, ids = load_protocol(args)
    if args.stage in ["prepare", "all"]:
        clinical_preflight(args, config, matched, ids)
    if args.stage in ["infer", "all"]:
        infer(args, config, vocab, matched, ids)
    passed = True
    if args.stage in ["geometry", "all"]:
        passed = geometry(args, config, vocab)
    if args.stage in ["evaluate", "all"] and passed:
        evaluate(args, config)
    if args.stage == "all":
        status = "COMPLETE" if passed else "STOPPED_AT_COLLAPSE_GATE"
        write_json(args.report / "DONE.json", {"status": status, "utc": datetime.now(timezone.utc).isoformat(),
                   "inference_success": True, "geometry_passed": passed, "clinical_evaluation_performed": passed,
                   "protocol_lock": json.loads((args.cache / "protocol_lock.json").read_text())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
