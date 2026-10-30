"""Strict first-round outcome-free B0--B7 P1 runner.

This file does not launch on import. No survival endpoint is loaded, selected,
or scored. Real fits are restricted to one already locked outer-train cohort
and one outcome-blind inner partition. Profiles perform forward/backward but
zero optimizer updates. P2 survival training is intentionally not implemented.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.semantic_b0b7.models import ImageInputs, ModelConfig, OmicsInputs, SemanticRepresentation, masked_softmax
from src.semantic_b0b7.pretext import PairMetadata, auxiliary_loss, matched_derangement, negative_support


ARMS = ("B3", "B4", "B6a", "B4_MAP_SHUFFLE")
SEEDS = (31, 47, 83)
CHECKPOINTS = (50, 100, 200, 400)
SOURCE_PATHS = ("src/semantic_b0b7/__init__.py", "src/semantic_b0b7/models.py",
                "src/semantic_b0b7/pretext.py", "src/semantic_b0b7/data.py",
                "configs/semantic_b0b7/p0_p1_v1.json", "scripts/semantic_b0b7_p1.py")
HARD_GATES = ("data_endpoint_fold_alignment", "official_text_alignment",
              "prototype_and_relation_contract", "real_forward_backward", "outcome_blind_scope")


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    return {name: sha_file(ROOT / name) for name in SOURCE_PATHS}


def configure_runtime():
    """One deterministic floating-point policy for both zero-step and fits."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_version": str(torch.__version__)}


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if replace else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def scalar(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def tensor(x, dtype=None):
    result = x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
    return result.to(dtype=dtype) if dtype else result


def load_gate_and_inputs(args):
    input_path = Path(args.inputs)
    actual_sha = sha_file(input_path)
    if actual_sha != args.expected_input_sha256:
        raise ValueError("INPUT_SHA_MISMATCH")
    gate = json.loads(Path(args.gate).read_text(encoding="utf-8-sig"))
    if gate.get("schema") != "semantic_b0b7_p0_gate_v1" or gate.get("status") != "PASS":
        raise ValueError("P0_GATE_SCHEMA_OR_STATUS")
    if gate.get("input_sha256") != actual_sha:
        raise ValueError("P0_GATE_INPUT_BINDING")
    current = source_hashes()
    bindings = gate.get("frozen_sources", {})
    # The final P0 gate binds the runner, data, configuration, and all model
    # sources before any profile or fit. Also verify every additional frozen
    # source provided by the gate, not just this runner's minimal dependency set.
    for name in SOURCE_PATHS:
        if bindings.get(name) != current[name]:
            raise ValueError("P0_GATE_SOURCE_BINDING:" + name)
    for name, expected in bindings.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha_file(path) != expected:
            raise ValueError("P0_ADDITIONAL_SOURCE_BINDING:" + name)
    current = dict(bindings)
    hard = gate.get("hard_gates", {})
    if any(hard.get(name, {}).get("status") != "PASS" for name in HARD_GATES):
        raise ValueError("MISSING_OR_FAILED_P0_HARD_GATE")
    probe = gate.get("probe_evidence", {})
    if (probe.get("optimizer_steps") != 0 or probe.get("input_sha256") != actual_sha
            or not probe.get("actual_forward_backward") or not probe.get("gradients_finite")):
        raise ValueError("P0_REAL_PROBE_EVIDENCE_REQUIRED")
    # This is a hash-verified, user-owned execution cache. It can contain typed
    # NumPy metadata, requiring trusted local pickle loading; never remote code.
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    allowed = {"schema", "image", "omics", "target", "target_mask", "coverage",
               "train_indices", "valid_indices", "train", "valid", "metadata",
               "conch_prototypes", "titan_prototypes", "relation_prior",
               "shuffled_relation_prior", "provenance", "program_names"}
    allowed |= {"global_rows", "clinical_raw"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("UNEXPECTED_INPUT_KEYS_OR_ENDPOINT_DATA")
    image = data["image"]
    if set(image) - {x.name for x in fields(ImageInputs)}:
        raise ValueError("UNEXPECTED_IMAGE_KEYS")
    if set(data["omics"]) != {"values", "measured"}:
        raise ValueError("UNEXPECTED_OMICS_KEYS")
    for key in image:
        if image[key] is not None:
            image[key] = tensor(image[key])
    for key in data["omics"]:
        data["omics"][key] = tensor(data["omics"][key])
    n = len(image["patch_values"])
    if n != 409:
        raise ValueError("P1_REQUIRES_EXACT_OUTER_FOLD1_TRAIN409")
    if (image["patch_values"].shape != (n, 512, 768)
            or image["patch_aligned"].shape != (n, 512, 768)
            or image["patch_mask"].shape != (n, 512)
            or image["patch_mask"].dtype != torch.bool
            or data["omics"]["values"].shape != (n, 50, 3)
            or data["omics"]["measured"].shape != (n, 50, 3)
            or data["omics"]["measured"].dtype != torch.bool):
        raise ValueError("FIXED_CONCH512_OMICS50_MEASURED_INPUT_CONTRACT")
    data["target"] = tensor(data["target"], torch.float32)
    data["target_mask"] = tensor(data["target_mask"])
    if data["target"].shape != (n, 50) or data["target_mask"].shape != (n, 50) or data["target_mask"].dtype != torch.bool:
        raise ValueError("RNA50_TARGET_MASK_CONTRACT")
    if not torch.isfinite(data["target"][data["target_mask"]]).all():
        raise ValueError("NONFINITE_MEASURED_TARGET")
    for key in ("conch_prototypes", "titan_prototypes", "relation_prior", "shuffled_relation_prior"):
        data[key] = tensor(data[key], torch.float32)
    if data["conch_prototypes"].shape != (8, 768) or data["titan_prototypes"].shape != (8, 768):
        raise ValueError("FIXED_EIGHT_PROTOTYPE_SHAPE")
    prior, shuffled = data["relation_prior"], data["shuffled_relation_prior"]
    if prior.shape != (8, 50) or shuffled.shape != prior.shape or torch.equal(prior, shuffled):
        raise ValueError("MAPPING_SHUFFLE_MUST_CHANGE_REAL_SUPPORT")
    if (not torch.isfinite(prior).all() or not torch.isfinite(shuffled).all()
            or (prior < 0).any() or (shuffled < 0).any()):
        raise ValueError("INVALID_FIXED_RELATION_PRIOR")
    if (not torch.equal((prior != 0).sum(0), (shuffled != 0).sum(0))
            or not torch.equal((prior != 0).sum(1), (shuffled != 0).sum(1))):
        raise ValueError("MAPPING_SHUFFLE_MUST_PRESERVE_BOTH_DEGREES")
    train = data.get("train_indices", data.get("train"))
    valid = data.get("valid_indices", data.get("valid"))
    data["train_indices"] = tensor(train, torch.long)
    data["valid_indices"] = tensor(valid, torch.long)
    joined = torch.cat([data["train_indices"], data["valid_indices"]])
    if (not len(data["train_indices"]) or not len(data["valid_indices"])
            or not torch.equal(torch.sort(joined).values, torch.arange(n))):
        raise ValueError("TRAIN_VALID_MUST_BE_DISJOINT_EXHAUSTIVE_409")
    required_meta = {"patient", "stage", "cohort", "missing_pattern"}
    if set(data["metadata"]) != required_meta:
        raise ValueError("METADATA_MUST_NOT_CONTAIN_OUTCOMES")
    for key in required_meta:
        item = data["metadata"][key]
        arr = item.detach().cpu().numpy() if torch.is_tensor(item) else np.asarray(item)
        if arr.shape != (n,) or arr.dtype.kind not in "iuUS":
            raise ValueError("TYPED_METADATA_REQUIRED:" + key)
        data["metadata"][key] = arr
    if len(np.unique(data["metadata"]["patient"])) != n:
        raise ValueError("PATIENT_ROWS_MUST_BE_UNIQUE")
    global_rows = tensor(data["global_rows"], torch.long)
    if (global_rows.shape != (n,) or global_rows.min() < 0 or global_rows.max() >= 512
            or not np.array_equal(global_rows.numpy(), data["metadata"]["patient"])):
        raise ValueError("CANONICAL_GLOBAL_ROWS_MUST_MATCH_PATIENT_METADATA")
    data["global_rows"] = global_rows
    clinical_raw = data["clinical_raw"]
    clinical_fields = {"AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE"}
    if not isinstance(clinical_raw, dict) or set(clinical_raw) != clinical_fields:
        raise ValueError("SIX_ORIGINAL_CLINICAL_FIELDS_REQUIRED_NO_OUTCOMES")
    for name, values in clinical_raw.items():
        raw = np.asarray(values)
        if raw.shape != (n,) or raw.dtype.kind not in "US":
            raise ValueError("CLINICAL_RAW_MUST_BE_TYPED_STRING_COLUMNS:" + name)
    if not isinstance(data.get("provenance"), dict):
        raise ValueError("MISSING_INPUT_PROVENANCE")
    return data, {"input_sha256": actual_sha, "gate_sha256": sha_file(args.gate),
                  "frozen_sources": current, "patients": n,
                  "train_patients": len(data["train_indices"]),
                  "valid_patients": len(data["valid_indices"])}


def build_model(data, arm, seed, device):
    image = data["image"]
    config = ModelConfig(clinical_dim=image["clinical"].shape[-1], width=64,
                         heads=4, rank=2, dropout=.1, temperature=.2)
    name = "B4" if arm == "B4_MAP_SHUFFLE" else arm
    relation = data["shuffled_relation_prior"] if arm == "B4_MAP_SHUFFLE" else data["relation_prior"]
    return SemanticRepresentation(name, data["conch_prototypes"], data["titan_prototypes"],
                                   relation, config=config, seed=seed).to(device)


def batch(data, partition, device):
    index = data[partition + "_indices"]
    image = ImageInputs(**{key: value[index].to(device) if value is not None else None
                           for key, value in data["image"].items()})
    genes = OmicsInputs(**{key: value[index].to(device) for key, value in data["omics"].items()})
    target = data["target"][index].to(device)
    measured = data["target_mask"][index].to(device)
    np_index = index.numpy()
    meta = PairMetadata(**{key: value[np_index] for key, value in data["metadata"].items()})
    hard, hard_audit = negative_support(meta, mode="matched")
    random, random_audit = negative_support(meta, mode="random")
    return image, genes, target, measured, meta, torch.as_tensor(hard, device=device), torch.as_tensor(random, device=device), hard_audit, random_audit


def common_negative_support(metadata, seed):
    """Identical effective negative candidates for all four same-seed arms.

    The B6 wrong-positive assignment is fixed before any optimizer update. Its
    candidate pairs are removed from EVERY arm's negative mask, not just B6.
    Two-patient strata may then have zero remaining negatives for ALL arms;
    this loss coverage is reported and never replaced by false negatives.
    """
    mask, before = negative_support(metadata, mode="matched")
    wrong, pairing = matched_derangement(metadata, seed=seed)
    shared = mask.copy()
    shared[np.arange(len(shared)), wrong] = False
    after = {"anchors_with_negative": int(shared.any(1).sum()),
             "candidate_negative_pairs": int(shared.sum()),
             "anchors": len(shared)}
    audit = {"before": before, "after": after, "pairing_derangement": pairing,
             "wrong_assignment_sha256": hashlib.sha256(wrong.astype("<i8").tobytes()).hexdigest(),
             "same_exclusion_for_all_four_arms": True,
             "excluded_wrong_positive_candidates": int(mask.sum() - shared.sum())}
    return shared, wrong, audit


def train_common_support(train_batch, seed):
    values = list(train_batch)
    mask, wrong, audit = common_negative_support(values[4], seed)
    values[5] = torch.as_tensor(mask, device=values[5].device)
    values[7] = audit
    return tuple(values), wrong, audit


def actual_pair_support(scores_valid, negatives, positive=None):
    """Effective support AFTER excluding assigned positives in each direction."""
    n = len(scores_valid)
    rows = torch.arange(n, device=scores_valid.device)
    pos = rows if positive is None else positive
    candidate = negatives & scores_valid
    candidate = candidate.clone()
    candidate[rows, pos] = False
    inverse = torch.argsort(pos)
    row_count, col_count = candidate.sum(1), candidate.sum(0)
    valid_i = scores_valid[rows, pos] & (row_count > 0)
    valid_g = scores_valid[inverse, rows] & (col_count > 0)
    def counts(count, used):
        return {"anchors_used": int(used.sum().item()),
                "negative_count_mean_on_used": scalar(count[used].double().mean()) if used.any() else None,
                "negative_count_min_on_used": int(count[used].min().item()) if used.any() else None,
                "negative_count_max_on_used": int(count[used].max().item()) if used.any() else None,
                "denominator_candidates_mean_on_used": scalar((count[used] + 1).double().mean()) if used.any() else None}
    return {"image_anchors_used": int(valid_i.sum().item()),
            "gene_anchors_used": int(valid_g.sum().item()),
            "candidate_negative_pairs": int(candidate.sum().item()),
            "image_to_gene": counts(row_count, valid_i),
            "gene_to_image": counts(col_count, valid_g),
            "assigned_positive_excluded_from_negative_count": True}


def checked_actual_pair_support(objective, scores_valid, negatives, positive=None):
    result = actual_pair_support(scores_valid, negatives, positive)
    if (result["image_anchors_used"] != objective["pair"]["image_anchors_used"]
            or result["gene_anchors_used"] != objective["pair"]["gene_anchors_used"]):
        raise ValueError("ACTUAL_PAIR_LOSS_SUPPORT_AUDIT_MISMATCH")
    return result


def device_sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@contextmanager
def preserve_replay_rng(device):
    """Model rebuilding/replay must not perturb the next training update.

    The frozen model constructor calls torch.manual_seed inside a CPU-only
    fork. torch.manual_seed also resets all CUDA generators, so an outer CUDA
    fork must protect every visible CUDA device, not just the CPU generator.
    A CPU execution protects the CPU generator without initializing CUDA.
    """
    device = torch.device(device)
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        yield


def profile(args):
    runtime = configure_runtime()
    data, binding = load_gate_and_inputs(args)
    torch.set_num_threads(args.threads)
    torch.manual_seed(31)
    device = torch.device(args.device)
    train, _, _ = train_common_support(batch(data, "train", device), 31)
    image, genes, target, measured, _, negatives, *_ = train
    model = build_model(data, "B4", 31, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    device_sync(device)
    start = time.perf_counter()
    output = model(image, genes)
    objective = auxiliary_loss(output, target, measured, negatives, lambda_pair=.1)
    pair_support = checked_actual_pair_support(objective, output["pair"]["valid"], negatives)
    objective["loss"].backward()
    device_sync(device)
    elapsed = time.perf_counter() - start
    gradients = {name: {"l2": scalar(p.grad.norm()) if p.grad is not None else None,
                         "finite": bool(torch.isfinite(p.grad).all()) if p.grad is not None else False}
                 for name, p in model.named_parameters()}
    if any(v["l2"] is None or v["l2"] == 0 or not v["finite"] for v in gradients.values()):
        raise ValueError("REAL_FULL_BATCH_PROFILE_HAS_INACTIVE_OR_INVALID_PARAMETER")
    receipt = {"schema": "semantic_b0b7_p1_actual_profile_v1", "status": "PASS", **binding,
               "device_type": device.type, "dtype": "float32", "optimizer_steps": 0,
               "actual_forward_backward": True, "all_parameter_gradients_finite_nonzero": True,
               "full_train_batch_patients": len(target), "seconds_forward_backward": elapsed,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "loss": scalar(objective["loss"]), "gradient_norms": gradients,
               "parameter_counts": model.parameter_counts(), "hard_negative_support": train[7],
               "runtime": runtime,
               "actual_pair_support": pair_support,
               "neither_checkpoint_nor_predictions_saved": True, "scientific_performance_evidence": False}
    write_json(args.output, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "optimizer_steps", "seconds_forward_backward", "peak_cuda_allocated_bytes")}))


def validate_profile(args, binding):
    profile_receipt = json.loads(Path(args.profile).read_text(encoding="utf-8-sig"))
    if (profile_receipt.get("schema") != "semantic_b0b7_p1_actual_profile_v1"
            or profile_receipt.get("status") != "PASS"
            or profile_receipt.get("optimizer_steps") != 0
            or profile_receipt.get("frozen_sources") != binding["frozen_sources"]
            or profile_receipt.get("input_sha256") != binding["input_sha256"]
            or profile_receipt.get("gate_sha256") != binding["gate_sha256"]
            or profile_receipt.get("runtime") != configure_runtime()
            or not profile_receipt.get("all_parameter_gradients_finite_nonzero")):
        raise ValueError("PROFILE_NOT_AN_ACTUAL_ZERO_STEP_HASH_BOUND_SUCCESS")
    return sha_file(args.profile)


def validate_common_extension(extension, private_root, binding):
    """Bind the shared extension to twelve immutable completed400 receipts."""
    if (extension.get("schema") != "semantic_b0b7_p1_common_extension_v1"
            or not extension.get("extension_required") or extension.get("complete_cells") != 12
            or extension.get("input_sha256") != binding["input_sha256"]
            or extension.get("frozen_sources") != binding["frozen_sources"]):
        raise ValueError("UNQUALIFIED_COMMON_EXTENSION")
    proofs = extension.get("completed400_receipts", [])
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(proofs) != 12 or {(p["arm"], p["seed"]) for p in proofs} != expected:
        raise ValueError("COMMON_EXTENSION_REQUIRES_ALL_TWELVE_COMPLETION_BINDINGS")
    for proof in proofs:
        directory = Path(private_root) / "cells" / (proof["arm"] + "_seed" + str(proof["seed"]))
        original = directory / "cell_epoch400.private.json"
        if not original.exists():
            original = directory / "cell.private.json"
        if sha_file(original) != proof["sha256"]:
            raise ValueError("COMMON_EXTENSION_COMPLETED400_RECEIPT_SHA_MISMATCH")
        receipt = json.loads(original.read_text(encoding="utf-8"))
        if (receipt.get("status") != "COMPLETE" or receipt.get("optimizer_steps") != 400
                or receipt.get("input_sha256") != binding["input_sha256"]
                or receipt.get("frozen_sources") != binding["frozen_sources"]
                or {s["epoch"] for s in receipt["states"]} != set(CHECKPOINTS)
                or any(not s["actual_replay"] or s["replay_max_error"] != 0 for s in receipt["states"])):
            raise ValueError("COMMON_EXTENSION_REQUIRES_ACTUAL_COMPLETE400_REPLAYS")


def safe_correlation(a, b, *, ranks=False):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    if ranks:
        from scipy.stats import rankdata
        a, b = rankdata(a), rankdata(b)
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else None


def program_metrics(prediction, target, measured):
    pred, truth, mask = map(np.asarray, (prediction, target, measured))
    if pred.shape != truth.shape or mask.shape != truth.shape or not np.isfinite(pred).all():
        raise ValueError("INVALID_PROGRAM_METRIC_ARRAYS")
    per_program = []
    for k in range(truth.shape[1]):
        use = mask[:, k]
        per_program.append({"program_index": k, "patients": int(use.sum()),
                            "pearson": safe_correlation(pred[use, k], truth[use, k]),
                            "spearman": safe_correlation(pred[use, k], truth[use, k], ranks=True)})
    def mean_finite(items):
        valid = [x for x in items if x is not None]
        return float(np.mean(valid)) if valid else None
    error = np.where(mask, pred - np.where(mask, truth, 0.), 0.)
    absolute = np.abs(error)
    huber = np.where(absolute <= 1., .5 * error**2, absolute - .5)
    counts = mask.sum(1)
    use = counts > 0
    per_patient = (huber * mask).sum(1) / np.maximum(counts, 1)
    profiles = [safe_correlation(pred[i, mask[i]], truth[i, mask[i]]) for i in range(len(pred))]
    return {"patients": len(pred), "patients_with_measurement": int(use.sum()),
            "measured_target_cells": int(mask.sum()),
            "coverage_fraction": float(mask.mean()),
            "macro_program_pearson": mean_finite(x["pearson"] for x in per_program),
            "macro_program_spearman": mean_finite(x["spearman"] for x in per_program),
            "valid_pearson_programs": sum(x["pearson"] is not None for x in per_program),
            "patient_equal_masked_huber": float(np.mean(per_patient[use])) if use.any() else None,
            "patient_equal_profile_pearson_secondary": mean_finite(profiles),
            "pearson_definition": "each program across unique patients, then equal supported-program mean",
            "huber_definition": "within-patient mean over measured programs, then equal patients",
            "profile_pearson_secondary_definition": "within-patient across-program profile correlation, distinct from program Pearson",
            "per_program": per_program}


def retrieval_metrics(scores, valid, negative):
    scores, valid, negative = map(np.asarray, (scores, valid, negative))
    n = len(scores)
    def direction(s, v, neg):
        positive = np.diag(s)
        supported = neg & v & ~np.eye(n, dtype=bool)
        eligible = np.diag(v) & supported.any(1)
        gaps, wins, reciprocal = [], [], []
        for i in np.flatnonzero(eligible):
            negatives = s[i, supported[i]]
            gaps.append(float(positive[i] - negatives.mean()))
            greater = int((negatives > positive[i]).sum())
            tied = int((negatives == positive[i]).sum())
            wins.append(0. if greater else 1. / (1 + tied))
            reciprocal.append(1. / (1 + greater + .5 * tied))
        return {"anchors": n, "anchors_covered": int(eligible.sum()),
                "coverage": float(eligible.mean()),
                "mean_negative_count": float(supported.sum(1)[eligible].mean()) if eligible.any() else None,
                "paired_gap_mean": float(np.mean(gaps)) if gaps else None,
                "top1_tie_averaged": float(np.mean(wins)) if wins else None,
                "mean_reciprocal_midrank": float(np.mean(reciprocal)) if reciprocal else None}
    return {"image_to_gene": direction(scores, valid, negative),
            "gene_to_image": direction(scores.T, valid.T, negative.T),
            "positive_identity_at_evaluation": True,
            "validation_gene_bank_used_only_for_retrieval_scoring": True}


def evaluate(model, valid_batch):
    image, genes, target, measured, meta, hard, random, *_ = valid_batch
    model.eval()
    with torch.no_grad():
        output = model(image, genes)
        objective = auxiliary_loss(output, target, measured, hard, lambda_pair=.1)
    pred = output["program_prediction"].detach().cpu()
    score = output["pair"]["scores"].detach().cpu()
    valid = output["pair"]["valid"].detach().cpu()
    truth, mask = target.cpu().numpy(), measured.cpu().numpy()
    report = {"auxiliary_loss": scalar(objective["loss"]),
              "program_loss": scalar(objective["program"]["loss"]),
              "pair_hard_loss": scalar(objective["pair"]["loss"]),
              "actual_pair_support": checked_actual_pair_support(objective, output["pair"]["valid"], hard),
              "program": program_metrics(pred.numpy(), truth, mask),
              "retrieval_hard": retrieval_metrics(score.numpy(), valid.numpy(), hard.cpu().numpy()),
              "retrieval_random": retrieval_metrics(score.numpy(), valid.numpy(), random.cpu().numpy()),
              "strata": {}}
    for variable in ("stage", "missing_pattern", "cohort"):
        groups = {}
        for group in np.unique(getattr(meta, variable)):
            use = getattr(meta, variable) == group
            ix = np.ix_(use, use)
            groups[str(group)] = {
                "program": program_metrics(pred.numpy()[use], truth[use], mask[use]),
                "retrieval_hard": retrieval_metrics(score.numpy()[ix], valid.numpy()[ix], hard.cpu().numpy()[ix]),
                "retrieval_random": retrieval_metrics(score.numpy()[ix], valid.numpy()[ix], random.cpu().numpy()[ix])}
        report["strata"][variable] = groups
    return report, {"prediction": pred, "pair_scores": score, "pair_valid": valid}


def cell(args):
    if args.arm not in ARMS or args.seed not in SEEDS or args.max_epochs not in (400, 800):
        raise ValueError("CELL_OUTSIDE_PRELOCKED_MATRIX")
    if args.max_epochs == 800 and not args.resume:
        raise ValueError("800_IS_ONLY_A_COMPLETE_GROUP400_TO800_EXTENSION")
    if args.resume and args.max_epochs != 800:
        raise ValueError("RESUME_IS_ONLY_THE_PRELOCKED400_TO800_EXTENSION")
    runtime = configure_runtime()
    data, binding = load_gate_and_inputs(args)
    profile_sha = validate_profile(args, binding)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    directory = Path(args.private_root) / "cells" / (args.arm + "_seed" + str(args.seed))
    if directory.exists() and not args.resume:
        raise FileExistsError("CELL_EXISTS_NO_SILENT_OVERWRITE")
    directory.mkdir(parents=True, exist_ok=True)
    train, wrong_assignment, common_audit = train_common_support(batch(data, "train", device), args.seed)
    valid = batch(data, "valid", device)
    model = build_model(data, args.arm, args.seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    positive = None
    pairing_audit = {"identity_positives": True}
    if args.arm == "B6a":
        pairing_audit = dict(common_audit["pairing_derangement"])
        positive = torch.as_tensor(wrong_assignment, dtype=torch.long, device=device)
    history, states, start_epoch, invocations = [], [], 1, []
    if args.resume:
        previous = json.loads((directory / "cell.private.json").read_text(encoding="utf-8"))
        if (args.max_epochs != 800 or previous["optimizer_steps"] != 400
                or previous["input_sha256"] != binding["input_sha256"]
                or previous["frozen_sources"] != binding["frozen_sources"]):
            raise ValueError("ONLY_LOCKED400_TO800_COMMON_EXTENSION_SUPPORTED")
        if not args.extension_gate:
            raise ValueError("GROUP_EXTENSION_GATE_REQUIRED")
        extension = json.loads(Path(args.extension_gate).read_text(encoding="utf-8"))
        validate_common_extension(extension, args.private_root, binding)
        state_path = directory / "state_epoch400.private.pt"
        if sha_file(state_path) != previous["states"][-1]["sha256"]:
            raise ValueError("RESUME_CHECKPOINT_SHA_MISMATCH")
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device=device)
        history, states, start_epoch = previous["history"], previous["states"], 401
        invocations = previous.get("invocations", [])
        # Preserve the original completed400 receipt byte-for-byte before
        # updating only the convenience pointer to the completed800 receipt.
        old_receipt = directory / "cell.private.json"
        archive = directory / "cell_epoch400.private.json"
        old_bytes = old_receipt.read_bytes()
        if archive.exists():
            if archive.read_bytes() != old_bytes:
                raise ValueError("HISTORICAL400_RECEIPT_ALREADY_DIFFERS")
        else:
            pending = directory / "cell_epoch400.private.pending"
            with pending.open("xb") as stream:
                stream.write(old_bytes)
            pending.rename(archive)
    checkpoint_set = set(CHECKPOINTS) | ({800} if args.max_epochs == 800 else set())
    started = time.perf_counter()
    image, genes, target, measured, _, negatives, *_ = train
    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(image, genes)
        objective = auxiliary_loss(output, target, measured, negatives,
                                   positive=positive, lambda_pair=.1)
        pair_support = checked_actual_pair_support(objective, output["pair"]["valid"], negatives, positive)
        if not torch.isfinite(objective["loss"]):
            raise FloatingPointError("NONFINITE_AUXILIARY_LOSS")
        objective["loss"].backward()
        gradient_norms = {name: scalar(p.grad.norm()) if p.grad is not None else None
                          for name, p in model.named_parameters()}
        if any(v is None or not math.isfinite(v) for v in gradient_norms.values()):
            raise FloatingPointError("MISSING_OR_NONFINITE_GRADIENT")
        optimizer.step()
        row = {"epoch": epoch, "optimizer_steps": epoch,
               "train_auxiliary_loss": scalar(objective["loss"]),
               "train_program_loss": scalar(objective["program"]["loss"]),
               "train_pair_loss": scalar(objective["pair"]["loss"]),
               "actual_pair_support": pair_support,
               "gradient_norms": gradient_norms,
               "weight_l2": {name: scalar(p.detach().norm()) for name, p in model.named_parameters()}}
        if epoch in checkpoint_set:
            metrics, replay_expected = evaluate(model, valid)
            row["validation"] = metrics
            snapshot = {"schema": "semantic_b0b7_p1_state_v1", "arm": args.arm, "seed": args.seed,
                        "epoch": epoch, "input_sha256": binding["input_sha256"],
                        "frozen_sources": binding["frozen_sources"], "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                        "expected_validation": replay_expected, "validation": metrics}
            path = directory / ("state_epoch" + str(epoch).zfill(3) + ".private.pt")
            if path.exists():
                raise FileExistsError("CHECKPOINT_ALREADY_EXISTS")
            torch.save(snapshot, path)
            # Actual saved-state replay is observational: restore the exact
            # pre-replay CPU/CUDA RNGs before the next optimizer epoch. This
            # also makes saved400 -> resumed800 follow the continuous path.
            with preserve_replay_rng(device):
                saved = torch.load(path, map_location=device, weights_only=False)
                replay_model = build_model(data, args.arm, args.seed, device)
                replay_model.load_state_dict(saved["model"])
                replay_metrics, replay_arrays = evaluate(replay_model, valid)
                maximum = 0.
                for key, expected in saved["expected_validation"].items():
                    actual = replay_arrays[key]
                    if not torch.equal(actual, expected.cpu()):
                        raise ValueError("ACTUAL_STATE_REPLAY_ARRAY_MISMATCH:" + key)
                    if actual.dtype != torch.bool:
                        maximum = max(maximum, scalar((actual - expected.cpu()).abs().max()))
                if json.dumps(replay_metrics, sort_keys=True, allow_nan=False) != json.dumps(metrics, sort_keys=True, allow_nan=False):
                    raise ValueError("ACTUAL_STATE_REPLAY_METRIC_MISMATCH")
                del replay_model, saved
            states.append({"epoch": epoch, "file": path.name, "sha256": sha_file(path),
                           "actual_replay": True, "replay_max_error": maximum})
        history.append(row)
        device_sync(device)
        row["elapsed_seconds_including_checkpoint_replay_if_any"] = time.perf_counter() - epoch_started
        if epoch % 10 == 0:
            print(json.dumps({"arm": args.arm, "seed": args.seed, "epoch": epoch,
                              "train_auxiliary_loss": row["train_auxiliary_loss"]}), flush=True)
    invocations.append({"first_epoch": start_epoch, "last_epoch": args.max_epochs,
                        "optimizer_updates_this_invocation": args.max_epochs - start_epoch + 1,
                        "elapsed_seconds": time.perf_counter() - started})
    receipt = {"schema": "semantic_b0b7_p1_cell_v1", "status": "COMPLETE", **binding,
               "profile_sha256": profile_sha, "arm": args.arm, "seed": args.seed,
               "optimizer_steps": args.max_epochs, "optimizer": "Adam", "lr": 3e-4,
               "lambda_pair": .1, "rank": 2, "width": 64, "dropout": .1, "temperature": .2,
               "full_train_batch": True, "patient_epoch_equals_one_optimizer_update": True,
               "dss_or_other_outcomes_loaded": False, "phase": "P1_ONLY",
               "representation_is_unresidualized_phase_a": True,
               "pairing_audit": pairing_audit, "negative_support": train[7],
               "random_negative_support_evaluation_only": train[8],
               "history": history, "states": states, "elapsed_seconds_this_invocation": time.perf_counter() - started,
               "invocations": invocations,
               "parameter_counts": model.parameter_counts()}
    receipt["runtime"] = runtime
    write_json(directory / "cell.private.json", receipt, replace=bool(args.resume))
    print(json.dumps({"status": "COMPLETE", "arm": args.arm, "seed": args.seed,
                      "updates": args.max_epochs, "actual_replay_states": len(states)}))


def frozen_pathology_features(data):
    image = data["image"]
    raw, projected, valid = image["patch_values"], image["patch_aligned"], image["patch_mask"]
    prototype = torch.nn.functional.normalize(data["conch_prototypes"], dim=-1)
    slide = image.get("patch_slide")
    if slide is None:
        slide = torch.zeros_like(valid, dtype=torch.long)
    chunks = []
    for i in range(0, len(raw), 16):
        mask, s = valid[i:i+16], slide[i:i+16]
        values = torch.where(mask[..., None], raw[i:i+16], 0.)
        aligned = torch.nn.functional.normalize(torch.where(mask[..., None], projected[i:i+16], 0.), dim=-1)
        logits = (aligned @ prototype.T).transpose(1, 2) / .2
        aggregate = torch.zeros(len(mask), 8, 768)
        counts = torch.zeros(len(mask), 1, 1)
        for sid in range(int(s[mask].max()) + 1 if mask.any() else 0):
            keep = mask & (s == sid)
            weights = masked_softmax(logits, keep[:, None, :])
            aggregate += weights @ values
            counts += keep.any(-1)[:, None, None]
        chunks.append((aggregate / counts.clamp_min(1)).flatten(1))
    return torch.cat([torch.cat(chunks), image["titan_native"], image["titan_aligned"], image["titan_scores"]], 1).numpy().astype(np.float64)


def fit_masked_ridge(x_train, x_valid, y_train, measured, *, alpha=1.):
    """Measured-only per-program dual ridge, with actual linear-system counts."""
    center, scale = x_train.mean(0), x_train.std(0)
    scale[scale < 1e-8] = 1.
    x, z = (x_train - center) / scale, (x_valid - center) / scale
    prediction = np.empty((len(z), y_train.shape[1]), dtype=float)
    fit_states = [None] * y_train.shape[1]
    groups = {}
    for k in range(y_train.shape[1]):
        groups.setdefault(measured[:, k].tobytes(), []).append(k)
    for columns in groups.values():
        mask = measured[:, columns[0]]
        if mask.sum() < 2:
            raise ValueError("INSUFFICIENT_MEASURED_PROGRAM_FOR_RIDGE")
        a, y = x[mask], y_train[np.ix_(mask, columns)]
        local_mean, target_mean = a.mean(0), y.mean(0)
        a = a - local_mean
        dual = np.linalg.solve(a @ a.T + alpha * np.eye(len(a)), y - target_mean)
        coefficient = a.T @ dual
        prediction[:, columns] = (z - local_mean) @ coefficient + target_mean
        for j, k in enumerate(columns):
            fit_states[k] = {"coefficient": coefficient[:, j], "feature_local_center": local_mean,
                             "target_mean": float(target_mean[j]), "measured_train_patients": int(mask.sum())}
    return prediction, {"standardize_center": center, "standardize_scale": scale,
                        "program_fits": fit_states, "linear_system_scalar_rhs": y_train.shape[1],
                        "linear_solver_calls_factoring_shared_masks": len(groups), "alpha": alpha}


def baselines(args):
    data, binding = load_gate_and_inputs(args)
    train, valid = data["train_indices"].numpy(), data["valid_indices"].numpy()
    y = data["target"].numpy()
    mask = data["target_mask"].numpy()
    clinical = data["image"]["clinical"].numpy().astype(np.float64)
    pathology = frozen_pathology_features(data)
    metrics, states = {}, {}
    for name, features in (("CLINICAL", clinical), ("PATHOLOGY", pathology),
                            ("PATHOLOGY_CLINICAL", np.concatenate([pathology, clinical], 1))):
        prediction, state = fit_masked_ridge(features[train], features[valid], y[train], mask[train])
        metrics[name] = program_metrics(prediction, y[valid], mask[valid])
        states[name] = {"fit": state, "validation_prediction": prediction}
    median = np.array([np.median(y[train, k][mask[train, k]]) for k in range(50)])
    metrics["TRAIN_PATIENT_MEDIAN"] = program_metrics(np.broadcast_to(median, (len(valid), 50)), y[valid], mask[valid])
    directory = Path(args.private_root) / "baselines"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "ridge_states.private.pt"
    if state_path.exists():
        raise FileExistsError("BASELINES_EXIST_NO_SILENT_OVERWRITE")
    torch.save({"states": states, "median": median, **binding}, state_path)
    receipt = {"schema": "semantic_b0b7_p1_baselines_v1", "status": "COMPLETE", **binding,
               "decoder_models_fitted": 3, "scalar_program_linear_system_solves": 150,
               "linear_solver_calls_factoring_shared_masks": sum(x["fit"]["linear_solver_calls_factoring_shared_masks"] for x in states.values()),
               "median_optimizer_fits": 0, "alpha": 1., "metrics": metrics,
               "pathology_features": "frozen CONCH eight text-pool raw slots plus TITAN native/aligned/S32; equal slide mean",
               "dss_or_other_outcomes_loaded": False,
               "private_artifact_sha256": sha_file(state_path)}
    write_json(directory / "baselines.private.json", receipt)
    print(json.dumps({"status": "COMPLETE", "decoder_models": 3, "scalar_linear_system_solves": 150}))


def report(args):
    data, binding = load_gate_and_inputs(args)
    root = Path(args.private_root)
    cells, extension_votes, completed400_receipts = [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            path = root / "cells" / (arm + "_seed" + str(seed)) / "cell.private.json"
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if (receipt.get("status") != "COMPLETE" or receipt.get("arm") != arm or receipt.get("seed") != seed
                    or receipt.get("input_sha256") != binding["input_sha256"]
                    or receipt.get("frozen_sources") != binding["frozen_sources"]
                    or receipt.get("optimizer_steps") not in (400, 800)):
                raise ValueError("CELL_COMPLETION_BINDING_MISMATCH")
            expected_epochs = set(CHECKPOINTS) | ({800} if receipt["optimizer_steps"] == 800 else set())
            if {x["epoch"] for x in receipt["states"]} != expected_epochs:
                raise ValueError("MISSING_ACTUAL_CHECKPOINT_REPLAYS")
            for state in receipt["states"]:
                if (not state["actual_replay"] or state["replay_max_error"] != 0
                        or sha_file(path.parent / state["file"]) != state["sha256"]):
                    raise ValueError("STATE_ARTIFACT_OR_REPLAY_MISMATCH")
            checkpoint_rows = {x["epoch"]: x["validation"] for x in receipt["history"] if "validation" in x}
            if set(checkpoint_rows) != expected_epochs:
                raise ValueError("VALIDATION_CHECKPOINT_HISTORY_MUST_MATCH_SAVED_STATES")
            if any(not math.isfinite(row["auxiliary_loss"]) for row in checkpoint_rows.values()):
                raise ValueError("NONFINITE_CHECKPOINT_SELECTION_LOSS")
            selected_epoch = min(checkpoint_rows,
                                 key=lambda epoch: (checkpoint_rows[epoch]["auxiliary_loss"], epoch))
            vote = checkpoint_rows[400]["auxiliary_loss"] < checkpoint_rows[200]["auxiliary_loss"]
            extension_votes.append(vote)
            original = path.parent / "cell_epoch400.private.json" if receipt["optimizer_steps"] == 800 else path
            completed400_receipts.append({"arm": arm, "seed": seed, "sha256": sha_file(original)})
            cells.append({"arm": arm, "seed": seed, "optimizer_steps": receipt["optimizer_steps"],
                          "actual_replayed_states": len(receipt["states"]),
                          "checkpoint_validation": checkpoint_rows,
                          "selection_rule": "validation_Laux_then_lower_epoch_tie",
                          "selected_epoch": selected_epoch,
                          "selected_validation": checkpoint_rows[selected_epoch],
                          "train_learning_curve": [{k: row[k] for k in
                              ("epoch", "train_auxiliary_loss", "train_program_loss", "train_pair_loss", "elapsed_seconds_including_checkpoint_replay_if_any")}
                              for row in receipt["history"]],
                          "invocations": receipt["invocations"],
                          "pairing_audit": receipt["pairing_audit"],
                          "negative_support": receipt["negative_support"],
                          "actual_train_pair_support_by_epoch": [{"epoch": row["epoch"], **row["actual_pair_support"]}
                                                                    for row in receipt["history"]],
                          "parameter_counts": receipt["parameter_counts"],
                          "private_receipt_sha256": sha_file(path)})
    extension = {"schema": "semantic_b0b7_p1_common_extension_v1", **binding,
                 "complete_cells": len(cells), "improved200to400_cells": int(sum(extension_votes)),
                 "completed400_receipts": completed400_receipts,
                 "extension_required": sum(extension_votes) >= 6,
                 "rule": "if at least 6/12 cells have strictly lower held-out auxiliary loss at400 than200, extend ALL12 once to800",
                 "not_a_DSS_selection_rule": True}
    already_extended = sum(x["optimizer_steps"] == 800 for x in cells)
    if already_extended not in (0, 12):
        raise ValueError("COMMON_EXTENSION_MUST_FINISH_ALL_TWELVE")
    baseline_path = root / "baselines" / "baselines.private.json"
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (base.get("status") != "COMPLETE" or base.get("input_sha256") != binding["input_sha256"]
            or base.get("frozen_sources") != binding["frozen_sources"]
            or sha_file(baseline_path.parent / "ridge_states.private.pt") != base["private_artifact_sha256"]):
        raise ValueError("BASELINE_ARTIFACT_BINDING")
    result = {"schema": "semantic_b0b7_p1_public_aggregate_v1", "status": "P1_EVALUATED",
              **binding, "neural_fits": 12, "decoder_fits": 3,
              "scalar_program_linear_system_solves": base["scalar_program_linear_system_solves"],
              "optimizer_steps": sum(x["optimizer_steps"] for x in cells),
              "actual_checkpoint_replays": sum(x["actual_replayed_states"] for x in cells),
              "baseline_metrics": base["metrics"], "cells": cells,
              "common_extension": extension, "all_twelve_extended": already_extended == 12,
              "budget_limited": extension["extension_required"] and already_extended != 12,
              "dss_models_fitted": 0, "dss_scores_selected": 0,
              "no_outer_test_inputs_loaded": True,
              "mechanism_claim": "P1 auxiliary correspondence only; no DSS prognosis, morphology ground truth, or independent external validation",
              "validation_is_inner_development_and_has_been_used_for_auxiliary_checkpoint_selection": True,
              "P2_started": False}
    write_json(args.output, result)
    if args.extension_output:
        write_json(args.extension_output, extension)
    print(json.dumps({"status": result["status"], "fits": 12, "actual_replays": result["actual_checkpoint_replays"],
                      "extension_required": extension["extension_required"], "budget_limited": result["budget_limited"]}))


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    for name in ("profile", "cell", "baselines", "report"):
        p = sub.add_parser(name)
        p.add_argument("--inputs", required=True)
        p.add_argument("--expected-input-sha256", required=True)
        p.add_argument("--gate", required=True)
        p.add_argument("--threads", type=int, default=2)
        if name in ("profile", "cell"):
            p.add_argument("--device", default="cuda")
        if name in ("profile", "report"):
            p.add_argument("--output", required=True)
        if name in ("cell", "baselines", "report"):
            p.add_argument("--private-root", required=True)
        if name == "cell":
            p.add_argument("--profile", required=True)
            p.add_argument("--arm", choices=ARMS, required=True)
            p.add_argument("--seed", type=int, choices=SEEDS, required=True)
            p.add_argument("--max-epochs", type=int, choices=(400, 800), default=400)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--extension-gate")
        if name == "report":
            p.add_argument("--extension-output")
    return cli


def main():
    args = parser().parse_args()
    {"profile": profile, "cell": cell, "baselines": baselines, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
