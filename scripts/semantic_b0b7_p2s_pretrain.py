"""Locked, outcome-free P2 training-budget diagnostic; original sources unchanged.

Only outer1's three existing inner partitions, MAIN/ATTN_MATCHED, seeds
31/47/83 and pair weights .1/.03 are legal. Weight .1 resumes the actual Adam
and RNG state at epoch 400; .03 starts at zero. Both report fixed 400/800
checkpoints without selecting an epoch or opening a DSS endpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

STARTUP_THREAD_ENVIRONMENT = {key: os.environ.get(key) for key in
    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import semantic_b0b7_p1r as phase
from scripts import semantic_b0b7_p2_matrix as matrix
from src.semantic_b0b7.pretext import auxiliary_loss

SCRIPT = "scripts/semantic_b0b7_p2s_pretrain.py"
TEST = "tests/test_semantic_b0b7_p2s_pretrain.py"
P2_SHA = "b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c"
TRAINING = {"learning_rate": .0003, "optimizer": "Adam", "threads": 2,
    "checkpoint_epochs": [400, 800], "maximum_epoch": 800,
    "eval_only_at_fixed_checkpoints": True, "use_DSS": False,
    "loss": "original_masked_program_Huber_plus_lambda_pair_InfoNCE"}


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    return phase.legacy.sha_file(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def cell_key(split, family, seed, pair_weight):
    return split, family, int(seed), float(pair_weight)


def validate_diagnostic_lock(lock, master, *, lock_sha, root=ROOT):
    require(lock.get("schema") == "semantic_b0b7_p2s_lock_v1" and
        lock.get("status") == "LOCKED_BEFORE_P2S_OPTIMIZER", "P2S_PREOPTIMIZER_LOCK_REQUIRED")
    require(lock.get("parent_master_sha256") == P2_SHA and
        lock.get("parent_git_commit") == "dab64dff52b1170f9a11ceb0833d9d49424fa63a", "P2S_PARENT_MASTER_CHANGED")
    require(lock.get("training") == TRAINING, "P2S_TRAINING_PROTOCOL_CHANGED")
    sources = lock.get("frozen_sources", {})
    require(set(master["frozen_sources"]).issubset(sources) and {SCRIPT, TEST}.issubset(sources),
            "P2S_SOURCE_CLOSURE_INCOMPLETE")
    require(all(sources[k] == v for k, v in master["frozen_sources"].items()),
            "P2S_MUST_NOT_REBIND_FORMAL_SCIENCE")
    for name, digest in sources.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file() and file_sha(path) == digest,
                "P2S_FROZEN_SOURCE_CHANGED:" + name)
    expected = set(itertools.product(("outer1_inner0", "outer1_inner1", "outer1_inner2"),
                                    ("MAIN", "ATTN_MATCHED"), (31, 47, 83), (.1, .03)))
    cells = lock.get("cells", [])
    actual = [cell_key(x["split"], x["family"], x["seed"], x["lambda_pair"]) for x in cells]
    require(len(actual) == 36 and set(actual) == expected, "P2S_FINITE36_MATRIX_CHANGED")
    for cell in cells:
        for key in ("input_sha256", "parent_receipt_sha256", "parent_binding_sha256",
                    "parent_state400_sha256"):
            require(isinstance(cell.get(key), str) and len(cell[key]) == 64,
                    "P2S_PARENT_CELL_HASH_MISSING:" + key)
    require(isinstance(lock_sha, str) and len(lock_sha) == 64, "P2S_LOCK_BYTE_HASH_REQUIRED")
    return {cell_key(x["split"], x["family"], x["seed"], x["lambda_pair"]): x for x in cells}


def validate_parent_snapshot(saved, receipt, *, spec, input_sha256, seed, train_hash, valid_hash):
    require(saved.get("schema") == "semantic_b0b7_p1r_state_v1" and saved.get("epoch") == 400,
            "P2S_REQUIRES_REAL_FORMAL400_STATE")
    require(saved.get("input_sha256") == input_sha256 and saved.get("seed") == seed and
        saved.get("spec") == asdict(spec) and saved.get("train_rows_sha256") == train_hash and
        saved.get("valid_rows_sha256") == valid_hash, "P2S_PARENT_STATE_SCOPE_CHANGED")
    require(saved.get("evaluate_valid") is True and saved.get("fixed_epoch") is None and
        saved.get("DSS_OS_PFI_accessed") is False and receipt.get("DSS_OS_PFI_accessed") is False,
        "P2S_PARENT_MUST_BE_OUTCOME_FREE_INNER")
    require(receipt.get("optimizer_steps") == 400 and len(receipt.get("history", [])) == 400,
            "P2S_PARENT_HISTORY_NOT_COMPLETE400")
    require([r.get("epoch") for r in receipt["history"]] == list(range(1, 401)),
            "P2S_PARENT_HISTORY_DISCONTINUOUS")
    require(set(saved.get("expected_replay", {})) == {"prediction", "pair_scores", "pair_valid"},
            "P2S_PARENT_REPLAY_TENSORS_MISSING")
    require(torch.is_tensor(saved.get("cpu_rng")) and saved["cpu_rng"].dtype == torch.uint8 and
        saved["cpu_rng"].ndim == 1, "P2S_PARENT_CPU_RNG_MISSING")
    require(torch.is_tensor(saved.get("cuda_rng")) and saved["cuda_rng"].dtype == torch.uint8 and
        saved["cuda_rng"].ndim == 1, "P2S_PARENT_CUDA_RNG_MISSING")
    validate_adam_snapshot(saved["optimizer"], epoch=400)


def validate_adam_snapshot(state, *, epoch):
    groups, values = state.get("param_groups", []), state.get("state", {})
    require(len(groups) == 1 and groups[0].get("lr") == .0003 and
        groups[0].get("betas") == (.9, .999) and groups[0].get("weight_decay") == 0 and
        groups[0].get("eps") == 1e-8 and not groups[0].get("amsgrad", False),
        "P2S_ADAM_HYPERPARAMETERS_CHANGED")
    require(len(values) == len(groups[0]["params"]) and len(values) > 0,
            "P2S_ADAM_STATE_INCOMPLETE")
    for value in values.values():
        require(set(value) == {"step", "exp_avg", "exp_avg_sq"} and
            float(value["step"].item()) == epoch and
            torch.isfinite(value["exp_avg"]).all().item() and
            torch.isfinite(value["exp_avg_sq"]).all().item(), "P2S_ADAM_STATE_NOT_EXACT_EPOCH")


def restore_adam_rng(model, saved, device):
    """Restore after all construction/data setup; no reseeding after this call."""
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"].cpu())
    if torch.device(device).type == "cuda":
        require(saved.get("cuda_rng") is not None, "P2S_CUDA_CONTINUATION_RNG_REQUIRED")
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
    return optimizer


def replay_values(model, batch):
    model.eval()
    with torch.no_grad():
        outputs = model(batch[0], batch[1])
    return {"prediction": outputs["program_prediction"].cpu(),
        "pair_scores": outputs["pair"]["scores"].cpu(), "pair_valid": outputs["pair"]["valid"].cpu()}


def assert_replay(saved, actual):
    require(set(actual) == set(saved["expected_replay"]), "P2S_REPLAY_KEY_SET_CHANGED")
    for key, value in actual.items():
        require(torch.equal(value, saved["expected_replay"][key]), "P2S_ACTUAL_REPLAY_MISMATCH:" + key)


def checked_replay(data, spec, seed, train_rows, device, path, validation):
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    with phase.legacy.preserve_replay_rng(device):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        replay = phase.make_model(data, spec, seed, train_rows, device, saved=saved)
        assert_replay(saved, replay_values(replay, validation))
        del replay, saved
    require(torch.equal(torch.get_rng_state(), before_cpu), "P2S_REPLAY_CHANGED_CPU_RNG")
    if device.type == "cuda":
        require(torch.equal(torch.cuda.get_rng_state(device), before_cuda), "P2S_REPLAY_CHANGED_CUDA_RNG")
    return {"actual_replay": True, "replay_max_error": 0., "replay_preserved_training_rng": True}


def runtime_guard(threads, device):
    require(STARTUP_THREAD_ENVIRONMENT == {key: "2" for key in STARTUP_THREAD_ENVIRONMENT},
            "P2S_REQUIRES_FORMAL_THREAD_ENV_BEFORE_NUMPY_IMPORT")
    require(threads == 2 and str(device).startswith("cuda"), "P2S_FORMAL_GPU_THREADS_CHANGED")
    runtime = phase.legacy.configure_runtime()
    require(runtime["cublas_workspace_config"] == ":4096:8", "P2S_CUBLAS_POLICY_CHANGED")
    torch.set_num_threads(threads)
    from threadpoolctl import threadpool_info
    pools = [{k: p.get(k) for k in ("user_api", "internal_api", "prefix", "version", "num_threads")}
             for p in threadpool_info()]
    require(all(p["num_threads"] == 2 for p in pools if p["user_api"] == "blas"),
            "P2S_BLAS_POOL_NOT_FORMAL2")
    return {**runtime, "startup_thread_environment": STARTUP_THREAD_ENVIRONMENT,
            "torch_cpu_threads": torch.get_num_threads(), "threadpools": pools}


def run_diagnostic_cell(data, master, lock, *, lock_sha, cell, formal_private, output, device="cuda", threads=2):
    """One qualified fixed-budget cell. Never selects by DSS or auxiliary score."""
    validate_diagnostic_lock(lock, master, lock_sha=lock_sha)
    split, family, seed, pair_weight = cell_key(cell["split"], cell["family"], cell["seed"], cell["lambda_pair"])
    require(cell in lock["cells"], "P2S_CELL_NOT_IN_PRELOCKED_MATRIX")
    require(matrix.training_input_binding(data, master["assets"]["frozen_images_sha256"]) == cell["input_sha256"],
            "P2S_INPUT_SCOPE_CHANGED")
    train_rows, valid_rows = phase.validate_scope(data, data["train_indices"], data["valid_indices"], evaluate_valid=True)
    train_hash = phase.tensor_hash(np.asarray(data["metadata"]["patient"])[train_rows.numpy()])
    valid_hash = phase.tensor_hash(np.asarray(data["metadata"]["patient"])[valid_rows.numpy()])
    parent = Path(formal_private) / "phase_a" / split / (family + "_seed" + str(seed))
    receipt_path, binding_path = parent / "cell.private.json", parent / "MATRIX_BINDING.private.json"
    require(file_sha(receipt_path) == cell["parent_receipt_sha256"] and
        file_sha(binding_path) == cell["parent_binding_sha256"], "P2S_PARENT_RECEIPT_BYTES_CHANGED")
    parent_receipt, parent_binding = read(receipt_path), read(binding_path)
    require(parent_binding["p2_lock_sha256"] == P2_SHA and
        parent_binding["cell_receipt_sha256"] == cell["parent_receipt_sha256"] and
        parent_binding["training_only_input_sha256"] == cell["input_sha256"] and
        parent_binding["split"] == split and parent_binding["family"] == family and parent_binding["seed"] == seed,
        "P2S_PARENT_BINDING_CHANGED")
    state400 = [s for s in parent_receipt["states"] if s["epoch"] == 400]
    require(len(state400) == 1 and state400[0]["actual_replay"] is True and
        state400[0]["sha256"] == cell["parent_state400_sha256"], "P2S_PARENT400_NOT_VERIFIED")
    source_path = parent / state400[0]["file"]
    require(source_path.parent.resolve() == parent.resolve() and file_sha(source_path) == cell["parent_state400_sha256"],
            "P2S_PARENT400_BYTES_CHANGED")
    original_spec = phase.variant_definition(master["phase_a_families"][family])[1]
    require(original_spec.lambda_pair == .1 and not original_spec.wrong_aux and not original_spec.shuffled_map,
            "P2S_UNEXPECTED_PARENT_VARIANT")
    spec = replace(original_spec, lambda_pair=pair_weight)
    name = "P2S_" + family + ("_PAIR01" if pair_weight == .1 else "_PAIR003")
    require(not Path(output).exists(), "P2S_OUTPUT_MUST_BE_NEW_NO_PARTIAL_OPTIMIZER_RESUME")
    runtime = runtime_guard(threads, device)
    require(all(runtime.get(k) == value for k, value in parent_receipt["runtime"].items()),
            "P2S_FORMAL_NUMERICAL_RUNTIME_CHANGED")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(device)
    train, wrong, support = phase.legacy.train_common_support(phase.data_batch(data, train_rows, device), seed)
    require(support == parent_receipt["negative_support"], "P2S_COMMON_NEGATIVE_SUPPORT_CHANGED")
    validation = phase.data_batch(data, valid_rows, device)
    tolerance, train_median = phase.training_huber_tolerance(train[2], train[3])
    require(tolerance == parent_receipt["program_huber_tolerance"] and
        train_median == parent_receipt["train_median_program_huber"], "P2S_TRAINING_TARGET_SCALE_CHANGED")
    parent_saved = None
    if pair_weight == .1:
        parent_saved = torch.load(source_path, map_location="cpu", weights_only=False)
        validate_parent_snapshot(parent_saved, parent_receipt, spec=original_spec,
            input_sha256=cell["input_sha256"], seed=seed, train_hash=train_hash, valid_hash=valid_hash)
        model = phase.make_model(data, spec, seed, train_rows, device, saved=parent_saved)
        optimizer = restore_adam_rng(model, parent_saved, device)
        first_epoch = 401
    else:
        model = phase.make_model(data, spec, seed, train_rows, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
        first_epoch = 1
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    binding = {"master_lock_sha256": lock_sha, "base_p2_master_lock_sha256": P2_SHA,
        "frozen_sources": lock["frozen_sources"], "input_sha256": cell["input_sha256"],
        "variant": name, "spec": asdict(spec), "seed": seed, "split": split, "family": family,
        "train_rows_sha256": train_hash, "valid_rows_sha256": valid_hash,
        "train_patients": len(train_rows), "valid_patients": len(valid_rows), "fixed_epoch": None,
        "evaluate_valid": True, "program_huber_tolerance": tolerance,
        "train_median_program_huber": train_median, "selection_rule": "NO_SELECTION_FIXED400_800_DIAGNOSTIC",
        "runtime": runtime, "DSS_OS_PFI_accessed": False,
        "negative_mask_sha256": phase.tensor_hash(train[5]),
        "wrong_assignment_sha256": support["wrong_assignment_sha256"], "parent_cell": cell}
    phase.legacy.write_json(destination / "PRE_FIT_INPUT_BINDING.private.json", binding)
    states, history = [], []
    if pair_weight == .1:
        replay = checked_replay(data, spec, seed, train_rows, device, source_path, validation)
        metrics400 = parent_saved["validation"]
        states.append({"epoch": 400, "file": str(source_path.resolve()), "sha256": file_sha(source_path),
            "identity": "VERIFIED_FORMAL400_REFERENCE_NO_NEW_UPDATES", "validation": metrics400, **replay})
        del parent_saved
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(first_epoch, 801):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(train[0], train[1])
        objective = auxiliary_loss(outputs, train[2], train[3], train[5], positive=None, lambda_pair=pair_weight)
        require(bool(torch.isfinite(objective["loss"])), "P2S_NONFINITE_AUXILIARY_OBJECTIVE")
        objective["loss"].backward()
        gradients, inactive = {}, []
        for parameter_name, parameter in model.named_parameters():
            value = float(parameter.grad.norm().detach().cpu()) if parameter.grad is not None else None
            require(value is None or math.isfinite(value), "P2S_NONFINITE_GRADIENT:" + parameter_name)
            gradients[parameter_name] = value
            if value is None or value == 0:
                inactive.append(parameter_name)
        require(not inactive, "P2S_UNEXPECTED_INACTIVE_PARAMETERS:" + ",".join(inactive))
        optimizer.step()
        row = {"epoch": epoch, "optimizer_step": epoch,
            "train_program_loss": float(objective["program"]["loss"].detach().cpu()),
            "train_pair_loss": float(objective["pair"]["loss"].detach().cpu()),
            "train_auxiliary_loss": float(objective["loss"].detach().cpu()),
            "gradient_norms": gradients, "inactive_parameters": inactive,
            "active_parameter_elements": sum(p.numel() for p in model.parameters()),
            "weight_norms": {k: float(p.detach().norm().cpu()) for k, p in model.named_parameters()},
            "relation_trace": phase.relation_trace(model, outputs, train[5], None),
            "actual_pair_support": phase.legacy.checked_actual_pair_support(objective, outputs["pair"]["valid"], train[5], None)}
        if epoch in (400, 800):
            metrics, arrays = phase.evaluate_model(model, validation, pair_weight)
            row["validation"] = metrics
            validate_adam_snapshot(optimizer.state_dict(), epoch=epoch)
            snapshot = {"schema": "semantic_b0b7_p1r_state_v1", **binding, "epoch": epoch,
                "model": model.state_dict(), "model_config": asdict(model.config),
                "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device), "surrogate_metadata": None,
                "expected_replay": arrays, "validation": metrics}
            path = destination / ("state_epoch" + str(epoch) + ".private.pt")
            torch.save(snapshot, path)
            replay = checked_replay(data, spec, seed, train_rows, device, path, validation)
            states.append({"epoch": epoch, "file": path.name, "sha256": file_sha(path),
                "identity": "NEW_P2S_SAVED_STATE", "validation": metrics, **replay})
        history.append(row)
        if epoch % 25 == 0:
            print(json.dumps({"family": family, "seed": seed, "split": split, "lambda_pair": pair_weight,
                "epoch": epoch, "train_program_huber": row["train_program_loss"]}), flush=True)
    require([s["epoch"] for s in states] == [400, 800] and
        all(s["actual_replay"] for s in states), "P2S_FIXED_CHECKPOINTS_INCOMPLETE")
    receipt = {"schema": "semantic_b0b7_p2s_cell_v1", "status": "COMPLETE", **binding,
        "identity": "CONTINUED_FORMAL400" if pair_weight == .1 else "NEW_FIT0_TO800",
        "optimizer_steps": 801 - first_epoch, "optimizer_lifetime_steps": 800,
        "resume_epoch": first_epoch - 1, "history": history, "states": states,
        "checkpoint_paths": {str(s["epoch"]): str((destination / s["file"]).resolve()) for s in states},
        "selected_epoch": None, "no_DSS_or_auxiliary_epoch_selection": True,
        "negative_support": support, "parameter_counts": model.parameter_counts(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    phase.legacy.write_json(destination / "cell.private.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "expected-lock-sha256", "p2-lock", "private", "output-newdir", "assets",
                 "legacy-root", "frozen-images", "split", "family"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--expected-p2-lock-sha256", default=P2_SHA)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lambda-pair", type=float, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    require(file_sha(a.lock) == a.expected_lock_sha256 and a.expected_p2_lock_sha256 == P2_SHA,
            "P2S_LOCK_BYTES_CHANGED")
    lock = read(a.lock)
    old_args = argparse.Namespace(**{**vars(a), "lock": a.p2_lock,
        "expected_lock_sha256": a.expected_p2_lock_sha256})
    master = matrix.verify_lock(old_args)
    cells = validate_diagnostic_lock(lock, master, lock_sha=a.expected_lock_sha256)
    key = cell_key(a.split, a.family, a.seed, a.lambda_pair)
    require(key in cells, "P2S_CELL_NOT_PRELOCKED")
    runtime_guard(a.threads, a.device)
    data, _, _, input_sha = matrix.build_inputs(old_args, master)
    require(input_sha == cells[key]["input_sha256"], "P2S_BUILT_INPUT_NOT_EXACT_PARENT")
    result = run_diagnostic_cell(data, master, lock, lock_sha=a.expected_lock_sha256, cell=cells[key],
        formal_private=a.private, output=a.output_newdir, device=a.device, threads=a.threads)
    print(json.dumps({"status": result["status"], "optimizer_steps": result["optimizer_steps"],
        "actual_replays": len(result["states"]), "epoch_selection": False, "DSS_read": False}))


if __name__ == "__main__":
    main()
