"""General outcome-free P1-R/Phase-A execution, with pre-optimizer locks.

The old P1 runner/model remain unchanged. No survival scorer, endpoint loader,
or teacher is imported. Final outer refits use a fixed inner-selected epoch and
do not evaluate or load validation targets during the fit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import semantic_b0b7_p1 as legacy
from src.semantic_b0b7.models import ModelConfig, ImageInputs, OmicsInputs
from src.semantic_b0b7.pretext import auxiliary_loss, masked_program_loss
from src.semantic_b0b7.p1r_models import RepairRepresentation, RepairSpec, ScoreMomentSurrogate

REGISTRY = "configs/semantic_b0b7/p1r_variants_v1.json"
REQUIRED_SOURCES = ("src/semantic_b0b7/models.py", "src/semantic_b0b7/pretext.py",
                    "src/semantic_b0b7/p1r_models.py", "src/semantic_b0b7/p1r_data.py",
                    "scripts/semantic_b0b7_p1.py", "scripts/semantic_b0b7_p1r.py", REGISTRY)
OUTCOME_KEYS = {"t", "e", "time", "event", "DSS", "OS", "PFI", "pfi_t", "pfi_e", "offset", "risk"}


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def tensor_hash(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(array.shape).encode() + str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def registry():
    return json.loads((ROOT / REGISTRY).read_text(encoding="utf-8"))


def variant_definition(variant, lock=None):
    if isinstance(variant, dict):
        value = dict(variant)
    else:
        book = registry()
        definitions = book["new_fit_variants"] + book["reference_variants"]
        if lock:
            definitions += lock.get("replication_variants", [])
        found = [v for v in definitions if v["name"] == variant]
        if len(found) != 1:
            raise ValueError("UNKNOWN_OR_DUPLICATE_PRELOCKED_VARIANT")
        value = dict(found[0])
    name = value.pop("name")
    return name, RepairSpec(**value)


def validate_scope(data, train_indices, valid_indices, *, evaluate_valid):
    if OUTCOME_KEYS.intersection(data) or OUTCOME_KEYS.intersection(data.get("metadata", {})):
        raise ValueError("OUTCOME_BEARING_PHASE_A_PAYLOAD_FORBIDDEN")
    if OUTCOME_KEYS.intersection(data.get("clinical_raw", {})) or data.get("provenance", {}).get("outcomes_loaded", False):
        raise ValueError("NESTED_OUTCOME_ACCESS_FORBIDDEN")
    n = len(data["image"]["patch_values"])
    train = torch.as_tensor(train_indices, dtype=torch.long).cpu()
    valid = torch.as_tensor(valid_indices if valid_indices is not None else [], dtype=torch.long).cpu()
    if train.ndim != 1 or len(train) < 2 or len(torch.unique(train)) != len(train) or train.min() < 0 or train.max() >= n:
        raise ValueError("INVALID_CURRENT_TRAIN_ROWS")
    if valid.ndim != 1 or (len(valid) and (len(torch.unique(valid)) != len(valid) or valid.min() < 0 or valid.max() >= n)):
        raise ValueError("INVALID_CURRENT_VALID_ROWS")
    if np.intersect1d(train.numpy(), valid.numpy()).size:
        raise ValueError("TRAIN_VALID_PATIENT_OVERLAP")
    if evaluate_valid and not len(valid):
        raise ValueError("INNER_SELECTION_REQUIRES_VALIDATION_ROWS")
    if not evaluate_valid and len(valid):
        # Explicit empty evaluation scope, not a held-out bank silently ignored.
        raise ValueError("FIXED_EPOCH_FIT_MUST_NOT_BE_GIVEN_EVALUATION_ROWS")
    if data["target"].shape != (n, 50) or data["target_mask"].shape != (n, 50):
        raise ValueError("MEASURED_RNA50_TARGET_CONTRACT")
    if data["target_mask"].dtype != torch.bool:
        raise ValueError("TARGET_MEASUREMENT_MASK_MUST_BE_BOOLEAN")
    if set(data["metadata"]) != {"patient", "stage", "cohort", "missing_pattern"}:
        raise ValueError("UNEXPECTED_PHASE_A_METADATA")
    patients = np.asarray(data["metadata"]["patient"])
    if len(np.unique(patients)) != n:
        raise ValueError("DUPLICATE_PATIENT_ROWS")
    return train, valid


def lock_binding(lock, data, input_sha256):
    if isinstance(lock, (str, Path)):
        path = Path(lock)
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        lock_sha = legacy.sha_file(path)
    else:
        value = dict(lock)
        lock_sha = canonical_hash(value)
    if value.get("schema") not in ("semantic_b0b7_p1r_lock_v1", "semantic_b0b7_p2_phase_a_lock_v1"):
        raise ValueError("PHASE_A_MASTER_LOCK_SCHEMA")
    if value.get("status") not in ("LOCKED_BEFORE_OPTIMIZER", "LOCKED_BEFORE_P1R_OPTIMIZER"):
        raise ValueError("MASTER_LOCK_MUST_PRECEDE_OPTIMIZER")
    source = value.get("frozen_sources", {})
    if not set(REQUIRED_SOURCES).issubset(source):
        raise ValueError("INCOMPLETE_PHASE_A_FROZEN_SOURCE_SET")
    for name, expected in source.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or legacy.sha_file(path) != expected:
            raise ValueError("FROZEN_SOURCE_CHANGED:" + name)
    if not isinstance(input_sha256, str) or len(input_sha256) != 64:
        raise ValueError("INPUT_SHA_BINDING_REQUIRED")
    allowed = value.get("allowed_input_sha256", [])
    if input_sha256 not in allowed:
        provenance = data.get("provenance", {})
        parents = value.get("expected_frozen_parent_input_sha256", [])
        if isinstance(parents, str):
            parents = [parents]
        if (not value.get("allow_derived_partition_payloads", False)
                or provenance.get("parent_frozen_input_sha256") not in parents
                or provenance.get("raw_container_sha256") != value.get("expected_raw_container_sha256")
                or provenance.get("outcomes_loaded") is not False):
            raise ValueError("UNQUALIFIED_PARTITION_INPUT_SOURCE")
    return value, {"master_lock_sha256": lock_sha, "frozen_sources": source,
                   "input_sha256": input_sha256}


def data_batch(data, indices, device):
    view = {**data, "train_indices": indices}
    return legacy.batch(view, "train", device)


def training_huber_tolerance(target, measured):
    y, mask = target.detach().cpu().numpy(), measured.detach().cpu().numpy()
    centers = []
    for k in range(y.shape[1]):
        if not mask[:, k].any():
            raise ValueError("NO_MEASURED_TRAIN_TARGET_FOR_PROGRAM")
        centers.append(np.median(y[:, k][mask[:, k]]))
    prediction = torch.as_tensor(np.broadcast_to(np.asarray(centers), y.shape).copy(), device=target.device, dtype=target.dtype)
    baseline = float(masked_program_loss(prediction, target, measured)["loss"].detach().cpu())
    return baseline * .01, baseline


def make_model(data, spec, seed, train_indices, device, *, saved=None):
    config = ModelConfig(clinical_dim=data["image"]["clinical"].shape[-1])
    if saved:
        config = ModelConfig(**saved["model_config"])
    surrogate = None
    if spec.score_route == "train_moment_surrogate":
        if saved:
            state = saved["model"]
            surrogate = ScoreMomentSurrogate(*(state["score_surrogate." + k].cpu() for k in
                ("projection", "native_mean", "whitening", "coloring", "score_mean")),
                saved["surrogate_metadata"]["fit_patients"], saved["surrogate_metadata"]["moment_error"])
        else:
            surrogate = ScoreMomentSurrogate.fit(data["image"]["titan_native"][train_indices],
                                                 data["image"]["titan_scores"][train_indices])
    relation = data["shuffled_relation_prior"] if spec.shuffled_map else data["relation_prior"]
    if spec.shuffled_map:
        original = data["relation_prior"]
        if (torch.equal(relation, original)
                or not torch.equal((relation > 0).sum(0), (original > 0).sum(0))
                or not torch.equal((relation > 0).sum(1), (original > 0).sum(1))):
            raise ValueError("MATCHED_DEGREE_SHUFFLE_NOT_QUALIFIED")
    # Frozen legacy constructors seed CUDA inside CPU-only forks. An outer
    # fork makes both initial construction and replay observational to RNGs.
    with legacy.preserve_replay_rng(device):
        constructor = RepairRepresentation
        if spec.source_arm.startswith("B2"):
            from src.semantic_b0b7.p2_models import CrossAttentionRepresentation
            constructor = CrossAttentionRepresentation
        model = constructor(data["conch_prototypes"], data["titan_prototypes"], relation,
            spec=spec, config=config, seed=seed, score_surrogate=surrogate).to(device)
    if saved:
        model.load_state_dict(saved["model"], strict=True)
    return model


def evaluate_model(model, valid, lambda_pair):
    metrics, arrays = legacy.evaluate(model, valid)
    # legacy.evaluate performs one actual forward; all unweighted components
    # are preserved while the current lambda-specific aggregate is explicit.
    metrics["auxiliary_loss"] = float(torch.tensor(metrics["program_loss"], dtype=torch.float32)
        + lambda_pair * torch.tensor(metrics["pair_hard_loss"], dtype=torch.float32))
    metrics["lambda_pair"] = lambda_pair
    prediction = arrays["prediction"].numpy()
    target, measured = valid[2].detach().cpu().numpy(), valid[3].detach().cpu().numpy()
    ratios = []
    for row in metrics["program"]["per_program"]:
        k = row["program_index"]
        keep = measured[:, k]
        sd_y = float(np.std(target[keep, k])) if keep.any() else None
        sd_p = float(np.std(prediction[keep, k])) if keep.any() else None
        ratio = sd_p / sd_y if sd_y is not None and sd_y > 0 else None
        row.update(target_sd=sd_y, prediction_sd=sd_p, prediction_to_target_sd_ratio=ratio)
        if ratio is not None:
            ratios.append(ratio)
    metrics["median_prediction_to_target_sd_ratio"] = float(np.median(ratios)) if ratios else None
    return metrics, arrays


def hard_gap(metrics):
    values = [metrics["retrieval_hard"][d]["paired_gap_mean"] for d in ("image_to_gene", "gene_to_image")]
    values = [x for x in values if x is not None and math.isfinite(x)]
    return sum(values) / len(values) if values else -math.inf


def select_checkpoint(history, tolerance):
    rows = [x for x in history if "validation" in x]
    if not rows:
        raise ValueError("NO_LEGAL_AUXILIARY_VALIDATION_CHECKPOINT")
    best = min(x["validation"]["program"]["patient_equal_masked_huber"] for x in rows)
    eligible = [x for x in rows if x["validation"]["program"]["patient_equal_masked_huber"] <= best + tolerance]
    return min(eligible, key=lambda x: (-hard_gap(x["validation"]), x["epoch"]))


def relation_trace(model, outputs, negatives, positive):
    mask = outputs["gene_mask"]
    prior = model.relation_prior
    known = (prior > 0)[None] & mask[:, None, :]
    unknown = (prior == 0)[None] & mask[:, None, :]
    trace = {"known_edges_per_patient_mean": float(known.sum((1, 2)).float().mean().cpu()),
             "unknown_edges_per_patient_mean": float(unknown.sum((1, 2)).float().mean().cpu()),
             "valid_programs_without_known_support_mean": float((mask & ~known.any(1)).sum(1).float().mean().cpu()),
             "unknown_logit_value": float(model.unknown_logit.detach().cpu()),
             "relation_mode": model.spec.relation_mode}
    if model.spec.relation_mode == "legacy":
        mass_known = (prior[None] * mask[:, None, :]).sum((1, 2))
        mass_unknown = unknown.sum((1, 2)) * torch.sigmoid(model.unknown_logit)
        trace["unknown_mass_fraction_mean"] = float((mass_unknown / (mass_known + mass_unknown).clamp_min(1e-12)).mean().detach().cpu())
    else:
        diagnostics = outputs["pair"]["relation_diagnostics"]
        trace["effective_known_fraction_mean"] = float(diagnostics["known_effective_fraction"].mean().cpu())
        n = len(outputs["pair"]["scores"])
        rows = torch.arange(n, device=mask.device)
        assigned = rows if positive is None else positive
        for key in ("known_weighted_contribution", "unknown_weighted_contribution"):
            value = outputs["pair"][key].detach()
            selected = value[rows, assigned]
            valid_negative = negatives & outputs["pair"]["valid"]
            negative = value[valid_negative]
            trace[key] = {"assigned_positive_mean": float(selected.mean().cpu()),
                          "assigned_positive_std": float(selected.std(unbiased=False).cpu()),
                          "negative_mean": float(negative.mean().cpu()) if len(negative) else None,
                          "negative_std": float(negative.std(unbiased=False).cpu()) if len(negative) else None}
    return trace


def run_cell(data, variant, seed, output, lock, *, input_sha256, train_indices=None,
             valid_indices=None, fixed_epoch=None, evaluate_valid=True, device="cuda", threads=2):
    """Train exactly one locked Phase-A unit; returns a private actual receipt."""
    config, binding = lock_binding(lock, data, input_sha256)
    name, spec = variant_definition(variant, config)
    if spec.source_arm.startswith("B2") and "src/semantic_b0b7/p2_models.py" not in config["frozen_sources"]:
        raise ValueError("P2_ATTENTION_FACTORY_SOURCE_MUST_BE_FROZEN")
    allowed_variants = config.get("allowed_variant_names")
    if allowed_variants is not None and name not in allowed_variants:
        raise ValueError("VARIANT_OUTSIDE_CURRENT_MASTER_LOCK")
    if seed not in (31, 47, 83):
        raise ValueError("UNLOCKED_INITIALIZATION_SEED")
    if train_indices is None:
        train_indices = data["train_indices"]
    if valid_indices is None and evaluate_valid:
        valid_indices = data["valid_indices"]
    train_rows, valid_rows = validate_scope(data, train_indices, valid_indices, evaluate_valid=evaluate_valid)
    checkpoints = tuple(config.get("checkpoints", (50, 100, 200, 400)))
    if checkpoints != (50, 100, 200, 400):
        raise ValueError("THIS_VERSION_LOCKS_COMMON50_100_200_400_CHECKPOINTS")
    if not evaluate_valid and fixed_epoch not in checkpoints:
        raise ValueError("OUTER_FINAL_REQUIRES_FIXED_INNER_SELECTED_EPOCH")
    if evaluate_valid and fixed_epoch is not None:
        raise ValueError("DO_NOT_MIX_FIXED_EPOCH_AND_INNER_CHECKPOINT_SELECTION")
    maximum = 400 if evaluate_valid else int(fixed_epoch)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    runtime = legacy.configure_runtime()
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device)
    train, wrong, support = legacy.train_common_support(data_batch(data, train_rows, device), seed)
    validation = data_batch(data, valid_rows, device) if evaluate_valid else None
    tolerance, median_train_huber = training_huber_tolerance(train[2], train[3])
    model = make_model(data, spec, seed, train_rows, device)
    scope_binding = {**binding, "variant": name, "spec": asdict(spec), "seed": seed,
        "train_rows_sha256": tensor_hash(np.asarray(data["metadata"]["patient"])[train_rows.numpy()]),
        "valid_rows_sha256": tensor_hash(np.asarray(data["metadata"]["patient"])[valid_rows.numpy()]) if evaluate_valid else None,
        "train_patients": len(train_rows), "valid_patients": len(valid_rows) if evaluate_valid else 0,
        "fixed_epoch": fixed_epoch, "evaluate_valid": evaluate_valid,
        "train_median_program_huber": median_train_huber, "program_huber_tolerance": tolerance,
        "selection_rule": "min_program_Huber_then_within0.01_trainmedianHuber_higher_hardgap_then_earlier_epoch",
        "runtime": runtime, "DSS_OS_PFI_accessed": False}
    legacy.write_json(destination / "PRE_FIT_INPUT_BINDING.private.json", scope_binding)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    positive = torch.as_tensor(wrong, dtype=torch.long, device=device) if spec.wrong_aux or spec.source_arm == "B6a" else None
    history, states = [], []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, maximum + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(train[0], train[1])
        objective = auxiliary_loss(outputs, train[2], train[3], train[5],
                                   positive=positive, lambda_pair=spec.lambda_pair)
        if not torch.isfinite(objective["loss"]):
            raise FloatingPointError("P1R_NONFINITE_AUXILIARY_OBJECTIVE")
        objective["loss"].backward()
        gradients, inactive = {}, []
        for parameter_name, parameter in model.named_parameters():
            gradient = parameter.grad
            value = float(gradient.norm().detach().cpu()) if gradient is not None else None
            if value is not None and not math.isfinite(value):
                raise FloatingPointError("P1R_NONFINITE_GRADIENT")
            gradients[parameter_name] = value
            if gradient is None or value == 0:
                inactive.append(parameter_name)
        for mandatory in ("query_projection.weight", "query_residual", "raw_value.weight"):
            if mandatory in inactive:
                raise ValueError("P1R_DEAD_IMAGE_READING_PATH:" + mandatory)
        if spec.lambda_pair > 0 and inactive:
            raise ValueError("P1R_UNEXPECTED_INACTIVE_PARAMETERS:" + ",".join(inactive))
        optimizer.step()
        row = {"epoch": epoch, "optimizer_step": epoch, "train_program_loss": float(objective["program"]["loss"].detach().cpu()),
            "train_pair_loss": float(objective["pair"]["loss"].detach().cpu()),
            "train_auxiliary_loss": float(objective["loss"].detach().cpu()),
            "gradient_norms": gradients, "inactive_parameters": inactive,
            "active_parameter_elements": sum(p.numel() for parameter_name, p in model.named_parameters()
                                               if parameter_name not in inactive),
            "weight_norms": {parameter_name: float(p.detach().norm().cpu()) for parameter_name, p in model.named_parameters()},
            "relation_trace": relation_trace(model, outputs, train[5], positive),
            "actual_pair_support": legacy.checked_actual_pair_support(objective, outputs["pair"]["valid"], train[5], positive)}
        if epoch in checkpoints or epoch == maximum:
            if evaluate_valid:
                metrics, arrays = evaluate_model(model, validation, spec.lambda_pair)
                row["validation"] = metrics
            else:
                # No held-out inputs or targets. Saved-state replay uses the
                # current fit-training bank solely as a numerical unit check.
                model.eval()
                with torch.no_grad():
                    out = model(train[0], train[1])
                arrays = {"prediction": out["program_prediction"].cpu(),
                          "pair_scores": out["pair"]["scores"].cpu(), "pair_valid": out["pair"]["valid"].cpu()}
            surrogate_meta = ({"fit_patients": model.score_surrogate.fit_patients,
                               "moment_error": model.score_surrogate.moment_error}
                              if hasattr(model, "score_surrogate") else None)
            snapshot = {"schema": "semantic_b0b7_p1r_state_v1", **scope_binding, "epoch": epoch,
                "model": model.state_dict(), "model_config": asdict(model.config),
                "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "surrogate_metadata": surrogate_meta, "expected_replay": arrays,
                "validation": row.get("validation")}
            path = destination / ("state_epoch" + str(epoch).zfill(3) + ".private.pt")
            torch.save(snapshot, path)
            with legacy.preserve_replay_rng(device):
                saved = torch.load(path, map_location="cpu", weights_only=False)
                replay = make_model(data, spec, seed, train_rows, device, saved=saved)
                replay.eval()
                with torch.no_grad():
                    output_replayed = replay(*(validation[:2] if evaluate_valid else train[:2]))
                actual = {"prediction": output_replayed["program_prediction"].cpu(),
                          "pair_scores": output_replayed["pair"]["scores"].cpu(),
                          "pair_valid": output_replayed["pair"]["valid"].cpu()}
                for key in actual:
                    if not torch.equal(actual[key], arrays[key]):
                        raise ValueError("P1R_ACTUAL_SAVED_STATE_REPLAY_MISMATCH:" + key)
                del replay, saved
            states.append({"epoch": epoch, "file": path.name, "sha256": legacy.sha_file(path),
                           "actual_replay": True, "replay_max_error": 0.})
        history.append(row)
        if epoch % 25 == 0:
            print(json.dumps({"variant": name, "seed": seed, "epoch": epoch,
                              "train_program_huber": row["train_program_loss"]}), flush=True)
    selected = select_checkpoint(history, tolerance) if evaluate_valid else {"epoch": maximum}
    receipt = {"schema": "semantic_b0b7_p1r_cell_v1", "status": "COMPLETE", **scope_binding,
        "identity": "NEW_FIT", "optimizer_steps": maximum, "history": history, "states": states,
        "selected_epoch": selected["epoch"], "selected_validation": selected.get("validation"),
        "selected_checkpoint_sha256": next(s["sha256"] for s in states if s["epoch"] == selected["epoch"]),
        "source_state_file": next(s["file"] for s in states if s["epoch"] == selected["epoch"]),
        "negative_support": support, "parameter_counts": model.parameter_counts(),
        "lambda0_aux_only_parameters_not_trained": spec.lambda_pair == 0.,
        "P2_main_pretraining_eligible": spec.lambda_pair > 0.,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    legacy.write_json(destination / "cell.private.json", receipt)
    return receipt


def load_checkpoint_model(path, *, device="cuda", freeze=True, expected_input_sha256=None):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") != "semantic_b0b7_p1r_state_v1":
        raise ValueError("NOT_A_P1R_CHECKPOINT")
    if expected_input_sha256 is not None and saved["input_sha256"] != expected_input_sha256:
        raise ValueError("CHECKPOINT_INPUT_SCOPE_MISMATCH")
    state = saved["model"]
    stub = {"image": {"clinical": torch.empty(1, saved["model_config"]["clinical_dim"])},
            "conch_prototypes": state["conch_prototypes"], "titan_prototypes": state["titan_prototypes"],
            "relation_prior": state["relation_prior"], "shuffled_relation_prior": state["relation_prior"]}
    spec = RepairSpec(**saved["spec"])
    # Shuffling already occurred and is cryptographically bound in saved state.
    construction = RepairSpec(**{**asdict(spec), "shuffled_map": False})
    model = make_model(stub, construction, saved["seed"], torch.empty(0, dtype=torch.long),
                       torch.device(device), saved=saved)
    model.spec = spec
    model.eval()
    if freeze:
        model.requires_grad_(False)
    return model, saved


def extract_frozen_features(model, data, indices, *, batch_size=32):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("FEATURE_EXTRACTION_REQUIRES_FROZEN_EVAL_ENCODER")
    indices = torch.as_tensor(indices, dtype=torch.long).cpu()
    device = next(model.parameters()).device
    collected = {}
    for start in range(0, len(indices), batch_size):
        take = indices[start:start + batch_size]
        # Feature extraction never reads targets or constructs validation
        # negative banks, and it also supports a one-patient final chunk.
        image = ImageInputs(**{key: value[take].to(device) if value is not None else None
                               for key, value in data["image"].items()})
        genes = OmicsInputs(**{key: value[take].to(device) for key, value in data["omics"].items()})
        with torch.no_grad():
            image_out, gene_out = model.image_only(image), model.encode_omics(genes)
            outputs = {**image_out, **gene_out,
                "additive_image": image_out["image_embedding"],
                "additive_gene": gene_out["gene_embedding"],
                "additive_clinical": image_out["clinical_embedding"],
                "pre_residual_visual_tokens": image_out["visual_tokens"],
                "pre_residual_gene_tokens": gene_out["gene_tokens"],
                "interaction_features": model.mechanism_from_tokens(image_out["visual_tokens"],
                    gene_out["gene_tokens"], gene_out["gene_mask"], additive=model.spec.source_arm == "B1")}
        for name in ("additive_image", "additive_gene", "additive_clinical", "pre_residual_visual_tokens", "pre_residual_gene_tokens", "gene_mask", "interaction_features"):
            collected.setdefault(name, []).append(outputs[name].detach().cpu().numpy())
        collected.setdefault("omics_measured_mask", []).append(genes.measured.cpu().numpy())
    return {name: np.concatenate(values) for name, values in collected.items()}


pretrain_partition = run_cell


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inputs", required=True)
    cli.add_argument("--expected-input-sha256", required=True)
    cli.add_argument("--lock", required=True)
    cli.add_argument("--variant", required=True)
    cli.add_argument("--seed", type=int, required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--threads", type=int, default=2)
    cli.add_argument("--fixed-epoch", type=int)
    cli.add_argument("--no-evaluate-valid", action="store_true")
    args = cli.parse_args()
    if legacy.sha_file(args.inputs) != args.expected_input_sha256:
        raise ValueError("PRIVATE_INPUT_FILE_SHA_MISMATCH")
    data = torch.load(args.inputs, map_location="cpu", weights_only=False)
    result = run_cell(data, args.variant, args.seed, args.output, args.lock,
        input_sha256=args.expected_input_sha256, device=args.device, threads=args.threads,
        fixed_epoch=args.fixed_epoch, evaluate_valid=not args.no_evaluate_valid,
        valid_indices=[] if args.no_evaluate_valid else None)
    print(json.dumps({"status": result["status"], "optimizer_steps": result["optimizer_steps"],
                      "selected_epoch": result["selected_epoch"], "actual_replays": len(result["states"])}))


if __name__ == "__main__":
    main()
