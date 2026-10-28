"""R6 target private contracts and fold-local wrapper; no R5 dependency.

Source/full-risk assets are verified and reused, never fitted here. Target
locks are created only after three complete source fits plus independent
replays exist. Missing assets fail closed; no placeholder source lock is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_run import load_inputs, write, torch_write
from scripts.bgb_en_r6_pretrain import validate_lock as validate_source_lock, current_code_hashes
from src.bgb_en.data import read_patch_cache, sha256
from src.bgb_en.protocol import fixed_inner_splits, array_hash
from src.bgb_en.replay import apply_saved_preprocessing, replay_full_risks
from src.bgb_en.training import PatchBank
from src.bgb_en.omics import SparsePathwayEncoder
from src.bgb_en.r6_source import canonical_hash
from src.bgb_en.r6_source_model import SourceProgramModel
from src.bgb_en.r6_omics import GeneValueBank, GeneEvidenceTransform, SparseProgramEncoder, load_source_program_mapping
from src.bgb_en.r6_pathology import (FrozenSourceProvenance, NativePatchGeometry,
    frozen_source_patch_encode, source_model_state_sha256)
from src.bgb_en.r6_survival import R6SurvivalModel, MeanGeneQueryContext, TARGET_ARMS
from src.phase10_trimodal.reference import pairs

PARTITIONS, SEEDS = (0, 1, 2), (31, 47, 83)
FULL_SOURCE_CONTROLS = ("MATCHED", "WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH")
TARGET_SOURCE_FILES = ("src/bgb_en/r6_survival.py", "src/bgb_en/r6_training.py", "src/bgb_en/r6_pathology.py",
    "src/bgb_en/r6_omics.py", "src/bgb_en/r6_source.py", "src/bgb_en/r6_source_model.py",
    "scripts/bgb_en_r6_common.py", "scripts/bgb_en_r6_run.py", "scripts/bgb_en_r6_pretrain.py",
    "scripts/bgb_en_r6_qualify_target.py", "scripts/bgb_en_r6_replay.py",
    "scripts/bgb_en_r6_schedule_target.py", "scripts/bgb_en_r6_report_target.py",
    "src/bgb_en/replay.py", "src/phase10_trimodal/math.py", "src/phase10_trimodal/reference.py")
FIXED_DETERMINISTIC_RUNTIME = dict(CUBLAS_WORKSPACE_CONFIG=":4096:8", deterministic_algorithms=True,
    warn_only=False, cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
    cudnn_benchmark=False, cudnn_deterministic=True)


def require(value, code):
    if not bool(value):
        raise ValueError(code)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def configure_deterministic_runtime():
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (None, ":4096:8"), "R6_INCOMPATIBLE_CUBLAS_WORKSPACE_CONFIG")
    require(not torch.cuda.is_initialized() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
            "R6_CUBLAS_ENVIRONMENT_MUST_PRECEDE_CUDA_INITIALIZATION")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return dict(FIXED_DETERMINISTIC_RUNTIME)


def validate_deterministic_runtime():
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
            and torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled()
            and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
            and not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic,
            "R6_LOCKED_DETERMINISTIC_RUNTIME_REQUIRED")
    return dict(FIXED_DETERMINISTIC_RUNTIME)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    for name in ("assets", "private", "primary-private", "source-private", "config", "source-config"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--launch-gate", type=Path)
    p.add_argument("--launch-gate-sha256")
    return p


def validate_target_config(config):
    require(config.get("schema") == "bgb_en_r6_inner_v1", "R6_TARGET_CONFIG_SCHEMA_REQUIRED")
    training = config["training"]
    required = dict(epochs=80, evaluation_epochs=[20, 40, 80], primary_checkpoint_epoch=80,
        optimizer="Adam", learning_rate=.001, dropout=0., batch_size=8,
        parameter_L2=.0001, total_delta_L2=.1, interaction_delta_L2=.1,
        gradient_clip_norm=5., gradient_cache_replay_max_error=2e-5,
        saved_risk_replay_max_error=2e-6)
    require(all(training.get(k) == v for k, v in required.items()), "R6_FIXED_TARGET_TRAINING_CONTRACT_CHANGED")
    prototype = config["prototype"]
    require(prototype.get("outer_fold") == 1 and prototype.get("inner_partitions") == list(PARTITIONS)
            and prototype.get("neural_seeds") == list(SEEDS) and prototype.get("outer_test_access") is False,
            "R6_TARGET_SPLIT_OR_SEED_CHANGED")
    readout = config["readout"]
    require(readout.get("main_scale") == .025 and readout.get("cross_scale") == .05
            and readout.get("weight_initialization_std") == .002
            and readout.get("output_tanh") is False, "R6_FIXED_LINEAR_RISK_RECIPE_CHANGED")
    require(tuple(config["trainable_arms"]) == TARGET_ARMS and config.get("formal_target_neural_fits") == 90,
            "R6_TARGET_FINITE_MATRIX_CHANGED")
    require(config.get("primary_endpoint") == "DSS" and config.get("secondary_endpoint") == "PFI_frozen_DSS_risk_only",
            "R6_TARGET_ENDPOINT_CHANGED")
    return config


def finite_patterns(mask):
    mask = np.asarray(mask, bool)
    packed = np.packbits(mask, axis=1, bitorder="little")
    labels = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in packed])
    return labels, hashlib.sha256(str(mask.shape).encode() + packed.tobytes()).hexdigest()


def hungarian_query_mismatch(indices, *, missing_pattern, cancer, stage, platform, seed):
    """Fixed RNA-mask/cancer/stage/platform lexicographic priority, no endpoints."""
    fields = [np.asarray(x).astype(str) for x in (missing_pattern, cancer, stage, platform)]
    source = np.asarray(indices)
    require(source.ndim == 1 and source.dtype.kind in "iu" and len(source) >= 2
            and len(np.unique(source)) == len(source) and len({len(v) for v in fields}) == 1,
            "R6_INVALID_QUERY_PARTITION")
    idx = source.astype(np.int64, copy=False)
    require(((idx >= 0) & (idx < len(fields[0]))).all(), "R6_QUERY_PARTITION_OUT_OF_RANGE")
    n, base = len(idx), len(idx) + 1
    mismatch = [v[idx, None] != v[None, idx] for v in fields]
    integer = sum(m.astype(np.int64) * base ** (3 - at) for at, m in enumerate(mismatch))
    cost = integer.astype(float) + np.random.default_rng(seed).random((n, n)) * (.125 / n)
    np.fill_diagonal(cost, np.inf)
    row, col = linear_sum_assignment(cost)
    result = np.arange(len(fields[0]))
    result[idx[row]] = idx[col]
    require(np.all(result[idx] != idx) and set(result[idx]) == set(idx), "R6_QUERY_PAIRING_NOT_PARTITION_DERANGEMENT")
    names = ("RNA_finite_pattern", "cancer", "clinical_stage", "platform")
    return result, dict(n=n, fixed_points=0, partition_bijection=True, priority=list(names),
        matched_counts={name: int((~m[row, col]).sum()) for name, m in zip(names, mismatch)},
        total_integer_cost=int(integer[row, col].sum()), seed=int(seed),
        random_tie_break_total_bound=.125, mapping_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
        outcomes_accessed=False)


def qualified_prepared(args, ctx, partition):
    require(partition in PARTITIONS, "R6_UNKNOWN_INNER_PARTITION")
    stem = f"outer1_inner{partition}"
    path = args.primary_private / "prepared" / (stem + ".json")
    payload = path.with_suffix(".pt")
    metadata = read(path)
    require(metadata["status"] == "COMPLETE" and metadata["lineage"] == ctx["lineage"],
            "R6_UPSTREAM_PREPARED_LINEAGE_CHANGED")
    require(sha256(payload) == metadata["sha256"], "R6_UPSTREAM_PREPARED_PAYLOAD_CHANGED")
    if ctx.get("lock") is not None:
        require(ctx["lock"]["prepared_receipt_sha256"][stem] == sha256(path), "R6_LOCKED_PREPARED_RECEIPT_CHANGED")
    prepared = torch.load(payload, map_location="cpu", weights_only=False)
    raw = ctx["raw"]
    train, valid = fixed_inner_splits(raw["folds"], raw["e"], outer=1)[partition]
    require(np.array_equal(prepared["train"], train) and np.array_equal(prepared["valid"], valid)
            and np.all(raw["folds"][np.r_[train, valid]] != 1), "R6_INNER_SPLIT_OR_OUTER_SEAL_CHANGED")
    require(prepared["fit_hash"] == array_hash(train) and prepared["valid_hash"] == array_hash(valid),
            "R6_PREPROCESSING_PATIENT_SCOPE_CHANGED")
    held_all = []
    require(len(prepared["teachers"]) == 3, "R6_EXACT_THREE_REUSED_TEACHERS_REQUIRED")
    for group, teacher in enumerate(prepared["teachers"]):
        fit, held = np.asarray(teacher["train"]), np.asarray(teacher["held"])
        require(not np.intersect1d(fit, held).size and set(fit) | set(held) == set(train)
                and not np.intersect1d(np.r_[fit, held], valid).size
                and teacher["preprocessing"]["fit_hash"] == array_hash(fit), "R6_TEACHER_PATIENT_OR_PREPROCESS_SCOPE_CHANGED")
        require(np.all(prepared["teacher_groups"][held] == group), "R6_SAME_TEACHER_GROUP_CHANGED")
        held_all.extend(held.tolist())
    require(sorted(held_all) == sorted(train.tolist())
            and prepared["pair_teacher_rule"] == "both_pair_members_in_one_excluded_teacher_block",
            "R6_CROSSFIT_COVERAGE_OR_BETA_PAIR_CONTRACT_CHANGED")
    require(prepared["features"]["state"]["fit_hash"] == array_hash(train), "R6_MAIN_FEATURE_FIT_NOT_TRAIN_ONLY")
    molecular = prepared["features"]["state"]["molecular"]
    require(molecular["fit_hash"] == array_hash(train), "R6_ORIGINAL_GENE_SCALE_NOT_TRAIN_ONLY")
    expected_mask = np.isfinite(raw["rna"]) & (np.asarray(molecular["measured"])[None] > 0)
    require(np.array_equal(prepared["features"]["rna_mask"], expected_mask)
            and prepared["features"]["rna"].shape == raw["rna"].shape
            and np.isfinite(prepared["features"]["rna"][expected_mask]).all(),
            "R6_ORIGINAL_GENE_MEASUREMENT_MASK_CHANGED")
    # Recompute the scalar gradient at the fixed saved beta, without optimizing.
    left, right = [], []
    for group in np.unique(prepared["teacher_groups"][train]):
        group_indices = train[prepared["teacher_groups"][train] == group]
        i, j = pairs(raw["t"][group_indices], raw["e"][group_indices])
        left.extend(group_indices[i]); right.extend(group_indices[j])
    i, j = np.asarray(left), np.asarray(right)
    require(len(i) > 0, "R6_SAVED_BETA_HAS_NO_SAME_TEACHER_PAIRS")
    difference = prepared["RNA"][i] - prepared["RNA"][j]
    margin = prepared["CP"][i] - prepared["CP"][j] + prepared["beta"] * difference
    gradient = np.mean(-expit(-margin) * difference) + prepared["beta"]
    require(np.isfinite(gradient) and abs(gradient) <= 1e-6
            and prepared["scalar_audit"]["gradient_max"] <= 1e-6, "R6_FIXED_SAME_TEACHER_BETA_REPLAY_FAILED")
    require(np.array_equal(prepared["base"], prepared["CP"] + prepared["beta"] * prepared["RNA"]),
            "R6_CP_SCALAR_OFFSET_ELEMENTWISE_CHANGED")
    return prepared, dict(prepared_receipt_sha256=sha256(path), prepared_artifact_sha256=sha256(payload),
        train_hash=array_hash(train), valid_hash=array_hash(valid), train_n=len(train), valid_n=len(valid),
        teachers_reused=3, teacher_refits=0, scalar_refits=0, scalar_gradient_replayed=float(abs(gradient)),
        upstream_supervised_fits_reused_not_new=14, scalar_beta=float(prepared["beta"]),
        risk_distribution=metadata["risk_distribution"], teacher_heldout_labels_excluded_from_teacher_fit=True,
        validation_labels_excluded_from_beta_and_teachers=True)


def _model_from_source_payload(state, mapping, *, device):
    require(state.get("schema") == "bgb_en_r6_source_checkpoint_v1"
            and state["cell"]["arm"] == "S7" and state["epochs"] == state["optimizer_updates"] == 80,
            "R6_FULL_SOURCE_CHECKPOINT_BUDGET_CHANGED")
    reader = SparseProgramEncoder(mapping, gene_encoder=GeneEvidenceTransform())
    model = SourceProgramModel("S7", tissue_prior=state["statistics"]["tissue_prior"],
                               gene_reader=reader, init_seed=state["cell"]["seed"])
    model.load_transfer_state(state["transfer"], freeze=True)
    return model.to(device)


def source_foundations(args, mapping):
    config = read(args.source_config)
    validate_source_lock(config)
    source_config_sha = sha256(args.source_config)
    models, audits = {}, {}
    for control in FULL_SOURCE_CONTROLS:
        directory = args.source_private / "source_cells" / f"S7_{control}_h-1_seed31"
        completion, replay, checkpoint = directory / "COMPLETE.json", directory / "REPLAY.json", directory / "state_epoch080.private.pt"
        done, verified = read(completion), read(replay)
        expected = dict(arm="S7", control=control, heldout_donor=-1, seed=31)
        require(done["status"] == "COMPLETE_R6_SOURCE_FIT"
                and verified["status"] == "PASS_COMPLETE_R6_SOURCE_CHECKPOINT_REPLAY"
                and all(done["cell"].get(k) == v and verified["cell"].get(k) == v for k, v in expected.items())
                and done["epochs"] == done["optimizer_updates"] == 80
                and done["actual_new_optimizer_fits"] == 1
                and verified["optimizer_updates"] == verified["new_optimizer_fits"] == 0,
                "R6_COMPLETE_AND_REPLAYED_FULL_SOURCE_REQUIRED")
        require(done["config_sha256"] == verified["config_sha256"] == source_config_sha
                and done["source_code_sha256"] == verified["source_code_sha256"] == current_code_hashes()
                and verified["completion_sha256"] == sha256(completion)
                and done["artifact_sha256"][checkpoint.name] == sha256(checkpoint),
                "R6_FULL_SOURCE_CONFIG_CODE_OR_PAYLOAD_CHANGED")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(state["config_sha256"] == source_config_sha and state["source_code_sha256"] == current_code_hashes()
                and all(state["cell"].get(k) == v for k, v in expected.items()), "R6_SOURCE_STATE_IDENTITY_CHANGED")
        model = _model_from_source_payload(state, mapping, device=args.device)
        audit = dict(checkpoint_sha256=sha256(checkpoint), completion_sha256=sha256(completion),
            replay_sha256=sha256(replay), complete_source_state_sha256=source_model_state_sha256(model),
            source_config_sha256=source_config_sha, source_control=control, external_full_source_seed=31)
        models[control] = model
        audits[control] = audit
    # A complete S7 random structure with no ST supervision or label-derived
    # tissue frequency. The uniform constant also keeps this control's state
    # independent of every trained source model's observed class balance.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        reader = SparseProgramEncoder(mapping, gene_encoder=GeneEvidenceTransform())
        random_model = SourceProgramModel("S7", tissue_prior=torch.full((3,), 1 / 3, dtype=torch.float32),
            gene_reader=reader, init_seed=31).requires_grad_(False).eval().to(args.device)
    random_hash = source_model_state_sha256(random_model)
    models["RANDOM_NO_ST"] = random_model
    audits["RANDOM_NO_ST"] = dict(complete_source_state_sha256=random_hash, external_full_source_seed=31,
        source_control="RANDOM_NO_ST", full_structure_freshly_initialized=True, ST_weights_reused=False,
        only_external_training_tissue_prior_reused=False, source_tissue_labels_reused=False,
        tissue_prior_source="fixed_uniform_three_tissue_categories",
        tissue_prior=random_model.train_tissue_prior.detach().cpu().tolist(),
        source_new_optimizer_fits=0, source_training_claimed=False)
    return models, audits


def verify_target_lock(lock, ctx, args):
    require(lock.get("schema") == "bgb_en_r6_target_protocol_lock_v1"
            and lock.get("status") == "LOCKED_BEFORE_TARGET_FITS", "R6_IMMUTABLE_TARGET_LOCK_REQUIRED")
    contents = {k: v for k, v in lock.items() if k not in ("contract_sha256", "created_UTC")}
    require(canonical_hash(contents) == lock["contract_sha256"], "R6_TARGET_LOCK_CONTENT_CHANGED")
    require(lock.get("fixed_deterministic_runtime") == validate_deterministic_runtime(),
            "R6_DETERMINISTIC_RUNTIME_CONTRACT_CHANGED")
    require(all(sha256(ROOT / name) == digest for name, digest in lock["sources"].items()), "R6_TARGET_LOCKED_SOURCE_CHANGED")
    require(lock["config_sha256"] == sha256(args.config) and lock["source_config_sha256"] == sha256(args.source_config)
            and lock["source_full_states"] == ctx["source_audits"], "R6_TARGET_CONFIG_OR_FULL_SOURCE_CHANGED")
    for control in FULL_SOURCE_CONTROLS:
        origin = lock["source_full_states"][control]
        directory = args.source_private / "source_cells" / f"S7_{control}_h-1_seed31"
        for name, field in (("state_epoch080.private.pt", "checkpoint_sha256"),
                            ("COMPLETE.json", "completion_sha256"), ("REPLAY.json", "replay_sha256")):
            require(sha256(directory / name) == origin[field], "R6_LOCKED_FULL_SOURCE_FILE_CHANGED")
    fingerprints = {k: ctx["raw"]["audit"][k] for k in ("cohort_sha256", "endpoints_sha256")}
    require(all(lock["data_fingerprints"][k] == v for k, v in fingerprints.items()), "R6_TARGET_DATA_LINEAGE_CHANGED")
    require(lock["RNA_finite_mask_sha256"] == ctx["mask_sha256"]
            and lock["UNI2_regions_READY_sha256"] == sha256(Path(ctx["assets"]["patch_uni2"]) / "READY.json"),
            "R6_TARGET_RNA_MASK_OR_NATIVE_PATCH_RECEIPT_CHANGED")


def context(args, *, locked=True):
    require(not args.private.resolve().is_relative_to(ROOT), "R6_PATIENT_ARTIFACTS_MUST_BE_OUTSIDE_PROJECT_TREE")
    original = argparse.Namespace(assets=args.assets, config=ROOT / "configs/bgb_en/prototype_v1.json")
    assets, legacy_config, raw, member, lineage = load_inputs(original)
    config = validate_target_config(read(args.config))
    require(len(raw["t"]) == 512 and int(raw["e"].sum()) == 65, "R6_CANONICAL_DSS_COHORT_CHANGED")
    mapping = load_source_program_mapping(ROOT / "configs/sami_surv_stage_b_programs_v1.json")
    sources, audits = source_foundations(args, mapping)
    patterns, mask_sha = finite_patterns(np.isfinite(raw["rna"]))
    ctx = dict(assets=assets, legacy_config=legacy_config, raw=raw, member=member, lineage=lineage,
               config=config, mapping=mapping, sources=sources, source_audits=audits,
               patterns=patterns, mask_sha256=mask_sha, lock=None)
    if locked:
        ctx["lock"] = read(args.private / "targetLOCK.json")
        verify_target_lock(ctx["lock"], ctx, args)
    return ctx


def create_lock(args, ctx):
    require(ctx.get("lock") is None, "R6_CREATE_LOCK_REQUIRES_UNLOCKED_CONTEXT")
    require(args.launch_gate is not None and args.launch_gate_sha256 == sha256(args.launch_gate)
            and read(args.launch_gate).get("r6_formal_training_allowed") is True,
            "R6_TARGET_CREATION_REQUIRES_EXPLICIT_PASSED_R5_LAUNCH_GATE")
    origins = {f"outer1_inner{part}": qualified_prepared(args, ctx, part)[1]["prepared_receipt_sha256"] for part in PARTITIONS}
    names = sorted(set(ctx["lineage"]["source"]) | set(TARGET_SOURCE_FILES) |
                   {str(args.config.resolve().relative_to(ROOT)).replace("\\", "/"),
                    "configs/sami_surv_stage_b_programs_v1.json"})
    sources = {name: sha256(ROOT / name) for name in names}
    lock = dict(schema="bgb_en_r6_target_protocol_lock_v1", status="LOCKED_BEFORE_TARGET_FITS",
        fixed_deterministic_runtime=validate_deterministic_runtime(),
        sources=sources, config_sha256=sha256(args.config), source_config_sha256=sha256(args.source_config),
        prepared_receipt_sha256=origins, source_full_states=ctx["source_audits"],
        data_fingerprints={k: ctx["raw"]["audit"][k] for k in ("cohort_sha256", "endpoints_sha256")},
        RNA_finite_mask_sha256=ctx["mask_sha256"],
        UNI2_regions_READY_sha256=sha256(Path(ctx["assets"]["patch_uni2"]) / "READY.json"),
        launch_gate_sha256=args.launch_gate_sha256, planned_neural_fits=90, teacher_refits=0,
        outer_test_access=False, source_family_selected_by_DSS=False)
    lock["contract_sha256"] = canonical_hash(lock)
    path = args.private / "targetLOCK.json"
    if path.exists():
        require(read(path) == lock, "R6_EXISTING_TARGET_LOCK_DIFFERS_REFUSE_OVERWRITE")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(lock, handle, indent=2, allow_nan=False)
    return lock


def pairing(ctx, prepared, partition):
    raw, options = ctx["raw"], ctx["config"]["pairing"]
    mapping, audits = np.arange(len(raw["rna"])), {}
    for name, indices in (("train", prepared["train"]), ("validation", prepared["valid"])):
        seed = options["base_seed"] + partition * options["partition_seed_stride"] + options["valid_seed_offset"] * (name == "validation")
        current, audit = hungarian_query_mismatch(indices, missing_pattern=ctx["patterns"], cancer=raw["cancer"],
            stage=raw["clinical_raw"]["AJCC_PATHOLOGIC_TUMOR_STAGE"], platform=raw["platform"], seed=seed)
        mapping[indices] = current[indices]
        audits[name] = audit
    return mapping, audits


class EncodedPatchBank:
    """CPU evidence cache, transferred only for the requested microbatch."""
    def __init__(self, encoded, neighbor_indices, device):
        self.encoded, self.neighbor_indices, self.device = encoded, neighbor_indices, device

    def select(self, indices):
        ids = indices.detach().cpu()
        patch = {key: value[ids].to(self.device) for key, value in self.encoded.items() if torch.is_tensor(value)}
        patch["diagnostics"] = self.encoded["diagnostics"]
        return patch, self.neighbor_indices[ids].to(self.device)


def patch_bank(args, ctx, source_key, *, create=False):
    """Creation is a qualification action; formal runners require READY cache."""
    model, origin = ctx["sources"][source_key], ctx["source_audits"][source_key]
    source_hash = origin["complete_source_state_sha256"]
    ready_hash = sha256(Path(ctx["assets"]["patch_uni2"]) / "READY.json")
    encoder_source_hash = sha256(ROOT / "src/bgb_en/r6_pathology.py")
    key = canonical_hash(dict(source_state_sha256=source_hash, UNI2_READY_sha256=ready_hash,
                              patch_encoder_source_sha256=encoder_source_hash,
                              feature_unit="float32_per_row_L2_norm_division_reject_norm_le_1e-12"))
    directory = args.private / "patch_cache" / key
    receipt, payload = directory / "READY.json", directory / "evidence.private.pt"
    if not receipt.exists():
        require(create, "R6_FROZEN_PATCH_CACHE_MUST_BE_QUALIFIED_BEFORE_TARGET_FIT")
        directory.mkdir(parents=True, exist_ok=True)
        claim = directory / "BUILD_CLAIM.json"
        with claim.open("x", encoding="utf-8") as handle:
            json.dump(dict(cache_key=key, new_optimizer_fits=0), handle)
        raw = PatchBank(read_patch_cache(ctx["assets"]["patch_uni2"]), "cpu")
        require(raw.input_dim == 1536 and raw.step == 512., "R6_NATIVE_UNI2_1536_STEP512_REQUIRED")
        if source_key == "RANDOM_NO_ST":
            # Explicit random-reference artifact/receipt, never a source fit.
            random_payload, random_receipt = directory / "random_source.private.pt", directory / "RANDOM_REFERENCE.json"
            torch_write(random_payload, model.transfer_state())
            write(random_receipt, dict(status="COMPLETE_RANDOM_S7_NO_ST_NO_OPTIMIZER",
                artifact_sha256=sha256(random_payload), source_state_sha256=source_hash,
                source_origin=origin, new_optimizer_fits=0, training_claimed=False))
            checkpoint_hash, fit_hash = sha256(random_payload), sha256(random_receipt)
        else:
            checkpoint_hash, fit_hash = origin["checkpoint_sha256"], origin["replay_sha256"]
        provenance = FrozenSourceProvenance(checkpoint_hash, fit_hash, source_hash)
        pieces = {}
        for start in range(0, len(raw.x), 8):
            result = frozen_source_patch_encode(model, raw.x[start:start+8].to(args.device),
                raw.mask[start:start+8].to(args.device), source_provenance=provenance)
            for name, value in result.items():
                if name in ("states", "patch_mask", "program_score_prior"):
                    pieces.setdefault(name, []).append(value.detach().cpu())
        encoded = {name: torch.cat(values) for name, values in pieces.items()}
        encoded["diagnostics"] = dict(source_state_sha256=source_hash, frozen_source_only=True)
        geometry = NativePatchGeometry().build(raw.xy, raw.mask)
        torch_write(payload, dict(encoded=encoded, neighbor_indices=geometry.neighbor_indices))
        write(receipt, dict(status="COMPLETE_R6_FROZEN_PATCH_CACHE", cache_key=key,
            artifact_sha256=sha256(payload), source_state_sha256=source_hash, UNI2_READY_sha256=ready_hash,
            patch_encoder_source_sha256=encoder_source_hash,
            no_target_parameters_or_outcomes_in_cache=True, new_optimizer_fits=0))
    metadata = read(receipt)
    require(metadata["status"] == "COMPLETE_R6_FROZEN_PATCH_CACHE" and metadata["cache_key"] == key
            and metadata["source_state_sha256"] == source_hash and metadata["UNI2_READY_sha256"] == ready_hash
            and metadata["patch_encoder_source_sha256"] == encoder_source_hash
            and metadata["artifact_sha256"] == sha256(payload), "R6_FROZEN_PATCH_CACHE_SHA_CHANGED")
    saved = torch.load(payload, map_location="cpu", weights_only=False)
    return EncodedPatchBank(saved["encoded"], saved["neighbor_indices"], args.device), sha256(receipt)


class R6PatientModel(nn.Module):
    def __init__(self, arm, prepared, membership, raw_symbols, source, encoded_bank, device, *, seed=31, source_kind):
        super().__init__()
        self.device, self.arm, self.patch_bank = device, arm, encoded_bank
        self.fit_indices, self.valid_indices = np.asarray(prepared["train"]), np.asarray(prepared["valid"])
        feature = prepared["features"]
        self.inputs = {key: torch.as_tensor(feature[key], device=device) for key in ("rna", "rna_mask")}
        self.offset = torch.as_tensor(prepared["base"], device=device, dtype=torch.float32)
        self.rna_component = torch.as_tensor(prepared["beta"] * prepared["RNA"], device=device, dtype=torch.float32)
        self.query_map = np.arange(len(self.offset))
        self.intervention = "wrong_query" if arm == "B9" else "matched"
        self.mean_context = None
        self.gene_bank = GeneValueBank(raw_symbols, source.gene_reader.mapping.gene_symbols).to(device)
        self.core = R6SurvivalModel(arm, source, source_kind=source_kind, init_seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hallmark = SparsePathwayEncoder(membership, mode="pool", width=32, patterns=4)
        require(self.hallmark.rna.pathways == 50, "R6_HALLMARK_AUXILIARY_MUST_HAVE_50_SEPARATE_PROGRAMS")
        self.hallmark.pool_query.requires_grad_(False)
        self.hallmark.risk_head.requires_grad_(False)
        self.to(device)

    def genes(self, indices):
        return self.gene_bank(self.inputs["rna"][indices], self.inputs["rna_mask"][indices])

    def refresh_training_query_mean(self, batch_size=16):
        total, counts = None, None
        with torch.no_grad():
            for start in range(0, len(self.fit_indices), batch_size):
                idx = torch.as_tensor(self.fit_indices[start:start+batch_size], device=self.device)
                out = self.core.foundation.encode_gene_programs(self.genes(idx))
                mask = out["program_mask"]
                numerator = torch.where(mask[..., None], out["tokens"], torch.zeros_like(out["tokens"])).sum(0)
                count = mask.sum(0)
                total = numerator if total is None else total + numerator
                counts = count if counts is None else counts + count
        self.mean_context = MeanGeneQueryContext((total / counts.clamp_min(1)[:, None]).detach(),
            (counts > 0).detach(), len(self.fit_indices))
        return self.mean_context

    def forward(self, indices, control=None):
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        if control == "matched_query":
            intervention = "matched"
        else:
            intervention = self.intervention if control is None else control
        conditioning = (self.intervention if control in
                        ("off", "interaction_off", "value_off", "attention_permutation") else intervention)
        patch, neighborhood = self.patch_bank.select(indices)
        other = torch.as_tensor(self.query_map[indices.detach().cpu().numpy()], device=self.device)
        gene_indices = other if intervention == "whole_omics" else indices
        genes = self.genes(gene_indices)
        aux = self.hallmark(**{key: value[gene_indices] for key, value in self.inputs.items()})
        kwargs = dict(neighborhood=neighborhood, hallmark_tokens=aux["tokens"], hallmark_mask=aux["token_mask"])
        offset = self.offset[indices]
        if conditioning == "wrong_query":
            context = self.core.foundation.encode_gene_programs(self.genes(other))
            kwargs.update(query_context=context["tokens"], query_mask=context["program_mask"])
        elif conditioning == "mean_query":
            require(self.mean_context is not None, "R6_MISSING_TRAIN_ONLY_MEAN_QUERY")
            kwargs["mean_context"] = self.mean_context
        if intervention == "interaction_off":
            intervention = "off"
        elif intervention == "attention_permutation":
            intervention = conditioning
            kwargs["assignment_shuffle_seed"] = 270699
        elif intervention == "whole_omics":
            offset = offset - self.rna_component[indices] + self.rna_component[other]
            intervention = "matched"
        output = self.core(patch, genes, offset, intervention=intervention, **kwargs)
        if control == "whole_omics":
            output["diagnostics"].update(own_gene_values_replaced=True, RNA_main_effect_replaced=True,
                                          offset_untouched=False, whole_omics_includes_frozen_RNA_scalar=True)
        output["hallmark_auxiliary"] = aux
        return output

    def regularization(self):
        return 1e-4 * sum(p.square().sum() for p in self.parameters() if p.requires_grad)

    def parameter_budget(self):
        return dict(registered=sum(p.numel() for p in self.parameters()),
            requires_grad=sum(p.numel() for p in self.parameters() if p.requires_grad),
            core=self.core.parameter_budget(), Hallmark_only_unimodal_auxiliary=True,
            frozen_source_in_optimizer=False, same_effective_capacity_or_FLOPs_claimed=False)


def construct(args, ctx, prepared, arm, seed):
    source_key = "RANDOM_NO_ST" if arm in ("B2", "B13") else "WRONG_DONOR" if arm == "B8" else "MATCHED"
    source_kind = "RANDOM_NO_ST" if source_key == "RANDOM_NO_ST" else "WRONG_DONOR_ST" if source_key == "WRONG_DONOR" else "MATCHED_ST"
    bank, bank_receipt = patch_bank(args, ctx, source_key)
    model = R6PatientModel(arm, prepared, ctx["member"], ctx["raw"]["rna_symbols"], ctx["sources"][source_key],
        bank, args.device, seed=seed, source_kind=source_kind)
    model.query_map, pairing_audit = pairing(ctx, prepared, args.partition)
    return model, dict(source_key=source_key, source_origin=ctx["source_audits"][source_key],
                       patch_cache_receipt_sha256=bank_receipt, pairing=pairing_audit)


def public_snapshot_guard():
    directory = ROOT / "reports/bgb_en"
    return {str(p.relative_to(directory)): sha256(p) for p in directory.rglob("*") if p.is_file()}


def validate_qualification(args, ctx):
    path = args.private / "QUALIFICATION.json"
    value = read(path)
    require(value.get("status") == "PASS_R6_TARGET_REAL_DATA_AND_SOURCE_TRANSFER"
            and value.get("contract_sha256") == ctx["lock"]["contract_sha256"],
            "R6_TARGET_REAL_QUALIFICATION_REQUIRED_BEFORE_FIT")
    return sha256(path)
