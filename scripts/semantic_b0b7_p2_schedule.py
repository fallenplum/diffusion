"""Finite P2 scheduling with a global inner-selection-before-outer barrier.

Only JSON receipts and byte hashes are inspected here. The matrix worker owns
all model/data access. Complete hash-qualified units may be reused; incomplete
optimizers are never resumed. Signals/failures stop only owned process groups.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (31, 47, 83)
CHECKPOINTS = (50, 100, 200, 400)
GIB = 1024 ** 3
OWNED = []


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".writing")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def checked_path(root, relative, expected=None):
    root = Path(root).resolve()
    name = Path(relative)
    require(not name.is_absolute() and ".." not in name.parts, "UNSAFE_PRIVATE_RELATIVE_PATH")
    path = root / name
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), "MISSING_OR_ESCAPING_ARTIFACT")
    if expected is not None:
        require(sha(path) == expected, "PRIVATE_ARTIFACT_HASH_CHANGED:" + str(relative))
    return path


def verify_lock(args):
    require(sha(args.lock) == args.expected_lock_sha256, "P2_MASTER_LOCK_CHANGED")
    lock = read(args.lock)
    require(lock.get("status") == "LOCKED_BEFORE_OUTER_EVALUATION", "P2_MASTER_LOCK_NOT_FROZEN")
    required = {"scripts/semantic_b0b7_p2_schedule.py", "scripts/semantic_b0b7_p2_matrix.py",
                "scripts/semantic_b0b7_p1r.py", "src/semantic_b0b7/p2_models.py",
                "src/semantic_b0b7/p2_survival.py"}
    require(required <= set(lock["frozen_sources"]), "SCHEDULER_MATRIX_AND_MODELS_MUST_BE_FROZEN")
    for name, expected in lock["frozen_sources"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and sha(path) == expected,
                "P2_FROZEN_SOURCE_CHANGED:" + name)
    families, readouts = lock["phase_a_families"], lock["family_readouts"]
    require(bool(families) and set(families) == set(readouts), "FAMILY_SOURCE_AND_READOUT_SET_MISMATCH")
    arms = []
    for family, values in readouts.items():
        require(re.fullmatch(r"[A-Za-z0-9_]+", family) and isinstance(values, list) and bool(values), "INVALID_FAMILY_NAME_OR_READOUTS")
        require(len(values) == len(set(values)) and all(re.fullmatch(r"[A-Za-z0-9_]+", x) for x in values), "DUPLICATE_OR_UNSAFE_READOUT")
        arms.extend(values)
    require(len(arms) == len(set(arms)), "READOUT_CANNOT_BE_WRITTEN_BY_MULTIPLE_SOURCE_FAMILIES")
    require(lock["B0_R4_EXACT"]["report_with_family"] in families, "B0_REPORT_FAMILY_IS_NOT_SCHEDULED")
    require(sha(args.frozen_images) == lock["assets"]["frozen_images_sha256"], "FROZEN_FM_INPUT_CHANGED")
    for key in ("assets_json_sha256", "manifest_sha256", "private_assets_sha256"):
        if key in lock["assets"]:
            require(sha(args.assets) == lock["assets"][key], "PRIVATE_ASSET_MANIFEST_CHANGED")
    return lock


def verify_sources_only(args, lock):
    require(sha(args.lock) == args.expected_lock_sha256, "P2_MASTER_LOCK_CHANGED_DURING_RUN")
    for relative, expected in lock["frozen_sources"].items():
        require(sha(ROOT / relative) == expected, "P2_SOURCE_CHANGED_DURING_RUN:" + relative)


def plan(args):
    value = read(Path(args.private) / "SPLITS.private.json")
    require(value["p2_lock_sha256"] == args.expected_lock_sha256, "SPLIT_PLAN_MASTER_LOCK_CHANGED")
    partitions = value["partitions"]
    expected = {f"outer{o}_inner{i}" for o in range(1, 6) for i in range(3)} | {f"outer{o}_final" for o in range(1, 6)}
    require(set(partitions) == expected, "EXACT_FIFTEEN_INNER_AND_FIVE_OUTER_PARTITIONS_REQUIRED")
    for name, entry in partitions.items():
        fit, evaluation = entry["fit"], entry["evaluation"]
        require(len(fit) == len(set(fit)) and len(evaluation) == len(set(evaluation))
                and not set(fit) & set(evaluation) and all(isinstance(x, int) and 0 <= x < 512 for x in fit + evaluation),
                "PATIENT_SCOPE_OVERLAP_OR_INVALID_ROW")
        if name.endswith("final"):
            require(entry["role"] == "outer_final" and len(fit) + len(evaluation) == 512,
                    "FINAL_PARTITION_CONTRACT")
        else:
            require(entry["role"] == "inner" and len(fit) + len(evaluation) in (409, 410),
                    "INNER_PARTITION_CONTRACT")
    return partitions


def matrix_command(args, command, *, split=None, family=None, seed=None):
    result = [sys.executable, str(ROOT / "scripts/semantic_b0b7_p2_matrix.py"), command]
    for name in ("lock", "expected_lock_sha256", "private", "assets", "legacy_root", "frozen_images"):
        result += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    result += ["--threads", str(args.threads)]
    for name, value in (("split", split), ("family", family), ("seed", seed)):
        if value is not None:
            result += ["--" + name, str(value)]
    return result


def resources(path):
    v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1.exists():
        memory = {"usage_bytes": int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text()),
                  "limit_bytes": int(v1.read_text())}
    elif Path("/sys/fs/cgroup/memory.max").exists():
        maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        memory = {"usage_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text()),
                  "limit_bytes": None if maximum == "max" else int(maximum)}
    else:
        raise ValueError("REAL_CGROUP_MEMORY_TELEMETRY_REQUIRED_NO_HOST_RAM_ESTIMATE")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    values = []
    for line in gpu.stdout.strip().splitlines():
        total, used, utilization = (int(x.strip()) for x in line.split(","))
        values.append({"total_MiB": total, "used_MiB": used, "utilization_percent": utilization})
    require(bool(values), "GPU_TELEMETRY_MISSING")
    return {"cgroup_memory": memory, "filesystem_free_bytes": shutil.disk_usage(path).free,
            "GPUs": values, "epoch_seconds": time.time()}


def admission(resource, *, gpu):
    memory = resource["cgroup_memory"]
    if resource["filesystem_free_bytes"] < 2 * GIB or memory["usage_bytes"] > 90 * GIB:
        return False
    if memory["limit_bytes"] is not None and memory["limit_bytes"] - memory["usage_bytes"] < 8 * GIB:
        return False
    if gpu and any(row["total_MiB"] - row["used_MiB"] < 2048 for row in resource["GPUs"]):
        return False
    return True


def gpu_reservation(lock, split, family, resource):
    """Reserve peak memory before a just-started worker allocates anything."""
    role = "outer_final" if split.endswith("final") else "inner"
    budgets = lock.get("resource_admission", {}).get("gpu_peak_budget_bytes", {}).get(role, {})
    if family in budgets:
        value = budgets[family]
        require(isinstance(value, (int, float)) and value > 0, "INVALID_PROFILED_GPU_PEAK_RESERVATION")
        return int(value), "master_lock_profile_budget"
    source = lock["phase_a_families"][family].get("source_arm", "")
    if source in ("B2", "B2_NATIVE", "B2_NATIVE_QKVO"):
        # No unmeasured native pair graph is packed beside other jobs. This
        # conservative fallback is stricter than a maximum of two natives.
        total = min(g["total_MiB"] for g in resource["GPUs"]) * 1024 ** 2
        baseline = max(g["used_MiB"] for g in resource["GPUs"]) * 1024 ** 2
        return max(1, total - 2 * GIB - baseline), "unprofiled_native_exclusive_gpu_not_a_peak_measurement"
    # Explicit safety fallbacks for historical width64 /272-patient residents;
    # they are never reported as P2 measurements. Real E0 per-family budgets
    # in the master lock take priority, especially for409-patient refits.
    return int((7. if role == "outer_final" else 4.5) * GIB), "conservative_execution_fallback_not_profile_evidence"


def reservation_admission(resource, active, reservation, baseline_gpu_bytes=0):
    if not admission(resource, gpu=True):
        return False
    total = min(g["total_MiB"] for g in resource["GPUs"]) * 1024 ** 2
    reserved = sum(entry.get("reserved_gpu_bytes", 0) for entry in active.values())
    # GPU usage outside the expected reserved residents also consumes budget.
    actual = max(g["used_MiB"] for g in resource["GPUs"]) * 1024 ** 2
    external_or_over_budget = max(baseline_gpu_bytes, actual - reserved, 0)
    return reserved + reservation + external_or_over_budget <= total - 2 * GIB


def job_gpu_admission(lock, job, resource, active, baseline_gpu_bytes=0):
    reservation, qualification = gpu_reservation(lock, job["split"], job["family"], resource)
    if qualification.startswith("unprofiled_native"):
        if any(entry["kind"] == "cell" for entry in active.values()):
            return False
        if max(g["used_MiB"] for g in resource["GPUs"]) > 1024:
            return False
    return reservation_admission(resource, active, reservation, baseline_gpu_bytes)


def host_reservation_admission(resource, active, additional, baseline):
    maximum = 90 * GIB
    if resource["cgroup_memory"]["limit_bytes"] is not None:
        maximum = min(maximum, resource["cgroup_memory"]["limit_bytes"] - 8 * GIB)
    reserved = sum(entry.get("reserved_host_bytes", 0) for entry in active.values())
    predicted = max(resource["cgroup_memory"]["usage_bytes"], baseline + reserved) + additional
    return predicted <= maximum


def stop_owned():
    live = [p for p in OWNED if p.poll() is None]
    for p in live:
        try:
            if os.name == "nt":
                p.terminate()
            else:
                os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for p in live:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                p.kill()
            else:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


atexit.register(stop_owned)


def chosen_epoch(receipt):
    rows = [x for x in receipt["history"] if "validation" in x]
    best = min(x["validation"]["program"]["patient_equal_masked_huber"] for x in rows)
    use = [x for x in rows if x["validation"]["program"]["patient_equal_masked_huber"] <= best + receipt["program_huber_tolerance"]]
    def gap(row):
        values = [row["validation"]["retrieval_hard"][d]["paired_gap_mean"] for d in ("image_to_gene", "gene_to_image")]
        values = [v for v in values if v is not None and math.isfinite(v)]
        return sum(values) / len(values) if values else -math.inf
    return min(use, key=lambda x: (-gap(x), x["epoch"]))["epoch"]


def checked_anchor(args, split):
    relative = Path("anchors") / (split + ".private.json")
    path = checked_path(args.private, relative)
    receipt = read(path)
    require(receipt["p2_lock_sha256"] == args.expected_lock_sha256 and receipt["split"] == split,
            "ANCHOR_SCOPE_OR_LOCK_CHANGED")
    require(receipt.get("old_scores_reused") is False, "OLD_ANCHOR_SCORES_CANNOT_REPLACE_REAL_REFITS")
    artifact = checked_path(args.private, relative.with_suffix(".pkl"), receipt["artifact_sha256"])
    require(receipt["Cox_optimizer_fits"] > 0 and finite_tree(receipt), "ANCHOR_COUNT_OR_FINITE_AUDIT_INVALID")
    return {"receipt_sha256": sha(path), "artifact_sha256": sha(artifact),
            "Cox_optimizer_fits": receipt["Cox_optimizer_fits"], "elapsed_seconds": receipt.get("elapsed_seconds")}


def checked_phase(args, lock, split, family, seed):
    directory = Path(args.private) / "phase_a" / split / (family + "_seed" + str(seed))
    receipt_path = checked_path(directory, "cell.private.json")
    receipt = read(receipt_path)
    binding = read(checked_path(directory, "MATRIX_BINDING.private.json"))
    role = "outer_final" if split.endswith("final") else "inner"
    require(receipt.get("status") == "COMPLETE" and receipt.get("identity") == "NEW_FIT"
            and receipt["seed"] == seed and receipt["frozen_sources"] == lock["phase_a_lock"]["frozen_sources"]
            and receipt["master_lock_sha256"] == canonical_hash(lock["phase_a_lock"])
            and receipt["DSS_OS_PFI_accessed"] is False and receipt["P2_main_pretraining_eligible"] is True,
            "PHASE_A_COMPLETION_SOURCE_OR_SCOPE_INVALID")
    require(binding["p2_lock_sha256"] == args.expected_lock_sha256 and binding["split"] == split
            and binding["family"] == family and binding["seed"] == seed
            and binding["cell_receipt_sha256"] == sha(receipt_path)
            and binding["training_only_input_sha256"] == receipt["input_sha256"], "MATRIX_SOURCE_BINDING_CHANGED")
    prep = binding["shared_preprocessing"]
    checked_path(Path(args.private) / "preprocessing", prep["file"], prep["sha256"])
    steps = receipt["optimizer_steps"]
    require(steps in CHECKPOINTS and len(receipt["history"]) == steps
            and [x["epoch"] for x in receipt["history"]] == list(range(1, steps + 1)), "PHASE_A_OPTIMIZER_BUDGET_CHANGED")
    expected_epochs = {e for e in CHECKPOINTS if e <= steps}
    require(len(receipt["states"]) == len(expected_epochs) and {x["epoch"] for x in receipt["states"]} == expected_epochs,
            "PHASE_A_SAVED_REPLAY_SET_CHANGED")
    state_hashes = []
    for state in receipt["states"]:
        checked_path(directory, state["file"], state["sha256"])
        require(state["actual_replay"] is True and state["replay_max_error"] == 0., "PHASE_A_REPLAY_FAILED")
        state_hashes.append({"epoch": state["epoch"], "sha256": state["sha256"]})
    for row in receipt["history"]:
        require(finite_tree(row) and all(v is not None and v > 0 for v in row["gradient_norms"].values()),
                "PHASE_A_MISSING_ZERO_OR_NONFINITE_GRADIENT")
    if role == "inner":
        require(steps == 400 and receipt["evaluate_valid"] is True and receipt["valid_patients"] > 0
                and receipt["selected_epoch"] == chosen_epoch(receipt), "INNER_SELECTION_OR_FULL_BUDGET_CHANGED")
    else:
        require(receipt["evaluate_valid"] is False and receipt["valid_patients"] == 0
                and receipt["fixed_epoch"] == steps == receipt["selected_epoch"]
                and not any("validation" in row for row in receipt["history"]), "OUTER_PRETRAIN_SELECTION_LEAK_OR_BUDGET_CHANGED")
    selected = next(s for s in receipt["states"] if s["epoch"] == receipt["selected_epoch"])
    require(selected["file"] == receipt["source_state_file"] and selected["sha256"] == receipt["selected_checkpoint_sha256"],
            "SELECTED_STATE_HASH_CHANGED")
    return {"receipt_sha256": sha(receipt_path), "source_checkpoint_sha256": selected["sha256"],
            "selected_epoch": receipt["selected_epoch"], "optimizer_steps": steps,
            "actual_checkpoint_replays": len(state_hashes), "state_hashes": state_hashes,
            "input_training_sha256": receipt["input_sha256"]}


def checked_risk(args, lock, split, family, seed, phase):
    directory = Path(args.private) / "phase_a" / split / (family + "_seed" + str(seed))
    complete = read(checked_path(directory, "RISK_BUNDLE_COMPLETE.private.json"))
    require(complete["status"] == "COMPLETE" and complete["p2_lock_sha256"] == args.expected_lock_sha256
            and complete["frozen_source_checkpoint_sha256"] == phase["source_checkpoint_sha256"], "RISK_BUNDLE_BINDING_CHANGED")
    final = split.endswith("final")
    if final:
        outer = int(split[5])
        chosen = read(Path(args.private) / "ALL_OUTER_SELECTIONS_FROZEN.private.json")["entries"][str(outer)]["risk"]
        expected = {(arm, chosen[arm][str(seed)]["ridge_lambda"]) for arm in lock["family_readouts"][family]}
    else:
        expected = {(arm, lam) for arm in lock["family_readouts"][family] for lam in (.1, 1., 10.)}
    require(len(complete["records"]) == len(expected)
            and {(r["arm"], r["ridge_lambda"]) for r in complete["records"]} == expected, "READOUT_BUNDLE_INCOMPLETE_OR_DUPLICATE")
    records = []
    for arm, lam in sorted(expected):
        path = checked_path(Path(args.private) / "risk" / split / (arm + "_seed" + str(seed)),
                            "lambda" + str(float(lam)) + ".private.json")
        row = read(path)
        binding = row["source_and_scope_bindings"]
        require(row["schema"] == "semantic_b0b7_p2_risk_cell_v1" and row["status"] == "FITTED_AND_EVALUATED"
                and row["role"] == ("outer_final" if final else "inner") and row["DSS_coefficients"] == 26
                and row["risk_optimizer_fits"] == 1 and row["ridge_lambda"] == lam
                and binding["lock_sha256"] == args.expected_lock_sha256 and binding["split"] == split
                and binding["arm"] == arm and binding["seed"] == seed and binding["phase_a_source_family"] == family
                and binding["phase_a_selected_state_sha256"] == phase["source_checkpoint_sha256"]
                and binding["input_training_sha256"] == phase["input_training_sha256"], "RISK_SCOPE_OR_SOURCE_CHANGED")
        checked_path(path.parent, path.with_suffix(".pkl").name, row["artifact_sha256"])
        optimizer = row["risk_optimizer"]
        replay = row["saved_linear_head_replay"]
        require(finite_tree(row) and optimizer["parameters"] == optimizer["active_design_columns"] == 26
                and optimizer["gradient_max"] <= optimizer["gradient_tolerance"]
                and optimizer["oracle_loss_abs_error"] <= 1e-10
                and replay["actual_linear_head_replay"] is True and replay["neural_forward"] is False
                and replay["max_abs_risk_error"] <= 1e-12 and replay["fold_metric_components_equal"] is True,
                "RISK_CONVERGENCE_FINITE_OR_REPLAY_AUDIT_FAILED")
        records.append({"arm": arm, "ridge_lambda": lam, "receipt_sha256": sha(path),
                        "artifact_sha256": row["artifact_sha256"], "risk_optimizer_fits": 1})
    return records


def checked_cell(args, lock, split, family, seed):
    phase = checked_phase(args, lock, split, family, seed)
    risks = checked_risk(args, lock, split, family, seed, phase)
    return {"split": split, "family": family, "seed": seed, "phase_a": phase, "risk_fits": risks}


def jobs(partitions, families):
    return [{"split": split, "family": family, "seed": seed,
             "key": split + "_" + family + "_seed" + str(seed)}
            for split in partitions for family in sorted(families) for seed in SEEDS]


def inner_choice(args, outer, family, seed):
    selected, bindings = [], []
    for inner in range(3):
        path = Path(args.private) / "phase_a" / f"outer{outer}_inner{inner}" / (family + "_seed" + str(seed)) / "cell.private.json"
        row = read(path)
        selected.append(int(row["selected_epoch"]))
        bindings.append({"inner": inner, "receipt_sha256": sha(path),
                         "selected_state_sha256": row["selected_checkpoint_sha256"]})
    return {"epoch": int(sorted(selected)[1]), "inner_selected_epochs": selected,
            "selection_rule": "median_of_three_outcome_free_inner_selected_checkpoints", "bindings": bindings}


def risk_choice(args, outer, arm, seed):
    summaries, bindings = [], []
    for lam in (.1, 1., 10.):
        metrics = []
        for inner in range(3):
            path = Path(args.private) / "risk" / f"outer{outer}_inner{inner}" / (arm + "_seed" + str(seed)) / ("lambda" + str(lam) + ".private.json")
            row = read(path)
            metrics.append(row["evaluation_metrics"])
            bindings.append({"inner": inner, "lambda": lam, "receipt_sha256": sha(path), "artifact_sha256": row["artifact_sha256"]})
        hn, hd = sum(r["Harrell_num"] for r in metrics), sum(r["Harrell_den"] for r in metrics)
        require(hd > 0, "INNER_HARRELL_UNSUPPORTED")
        supported = all(r["Uno60_valid"] and r["Uno_den"] is not None and r["Uno_den"] > 0 for r in metrics)
        un = sum(r["Uno_num"] for r in metrics) if supported else None
        ud = sum(r["Uno_den"] for r in metrics) if supported else None
        summaries.append({"ridge_lambda": lam, "Harrell": hn / hd, "Harrell_num": hn, "Harrell_den": hd,
            "Uno60": un / ud if supported else None, "Uno_num": un, "Uno_den": ud, "Uno60_complete_support": supported})
    selected = max(summaries, key=lambda r: (r["Harrell"], r["Uno60"] if r["Uno60"] is not None else -math.inf, r["ridge_lambda"]))
    return {"ridge_lambda": selected["ridge_lambda"], "selection_rule": "fold_micro_Harrell_then_Uno60_then_stronger_ridge",
            "candidate_summaries": summaries, "uses_outer_test": False, "confidence_intervals_computed": False,
            "bindings": bindings}


def verify_frozen_selection(args, lock):
    path = Path(args.private) / "ALL_OUTER_SELECTIONS_FROZEN.private.json"
    selected = read(path)
    require(selected["p2_lock_sha256"] == args.expected_lock_sha256
            and selected["status"] == "ALL_FIVE_OUTERS_SELECTED_BEFORE_FIRST_OUTER_SCORE"
            and selected["outer_results_read"] is False and set(selected["entries"]) == {str(x) for x in range(1, 6)},
            "GLOBAL_OUTER_SELECTION_BARRIER_INVALID")
    for outer in range(1, 6):
        row = selected["entries"][str(outer)]
        require(set(row["pretraining"]) == set(lock["phase_a_families"]), "FROZEN_PRETRAIN_FAMILY_SET")
        require(set(row["risk"]) == {arm for values in lock["family_readouts"].values() for arm in values},
                "FROZEN_RISK_READOUT_SET")
        for family, arms in lock["family_readouts"].items():
            require(set(row["pretraining"][family]) == {str(s) for s in SEEDS}, "FROZEN_PRETRAIN_SEED_SET")
            for seed in SEEDS:
                require(row["pretraining"][family][str(seed)] == inner_choice(args, outer, family, seed),
                        "FROZEN_PRETRAIN_CHOICE_CHANGED")
                for arm in arms:
                    require(set(row["risk"][arm]) == {str(s) for s in SEEDS}, "FROZEN_RISK_SEED_SET")
                    require(row["risk"][arm][str(seed)] == risk_choice(args, outer, arm, seed),
                            "FROZEN_RISK_CHOICE_CHANGED")
    binding_path = Path(args.private) / "ALL_OUTER_SELECTION_BARRIER_BINDING.private.json"
    binding = {"status": "VERIFIED_ALL_FIVE_OUTER_SELECTIONS_BEFORE_FINAL_STAGE",
               "p2_lock_sha256": args.expected_lock_sha256, "selection_file_sha256": sha(path)}
    if binding_path.exists():
        require(read(binding_path) == binding, "GLOBAL_SELECTION_FILE_CHANGED_AFTER_BARRIER")
    else:
        root = Path(args.private)
        require(not (root / "B0").exists() and not any((root / "risk" / f"outer{o}_final").exists() for o in range(1, 6)),
                "CANNOT_CREATE_NEW_GLOBAL_BARRIER_AFTER_OUTER_RESULTS_EXIST")
        write(binding_path, binding)
    return binding


def environment(args):
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT), OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    return env


def synchronous_command(args, command, run_id):
    folder = Path(args.private) / "scheduler_logs"
    folder.mkdir(exist_ok=True)
    path = folder / (run_id + "_" + command + ".log")
    with path.open("xb") as stream:
        process = subprocess.Popen(matrix_command(args, command), cwd=ROOT, env=environment(args),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        OWNED.append(process)
        require(process.wait() == 0, "MATRIX_CONTROL_COMMAND_FAILED:" + command)


def run_stage(args, lock, names, stage, run_id):
    pending_anchors = list(names)
    pending_cells = jobs(names, lock["phase_a_families"])
    active, completed_anchors, completed_cells = {}, {}, []
    reused, new = [], []
    root = Path(args.private)
    for split in list(pending_anchors):
        anchor = root / "anchors" / (split + ".private.json")
        artifact = anchor.with_suffix(".pkl")
        if anchor.exists():
            completed_anchors[split] = checked_anchor(args, split)
            pending_anchors.remove(split)
        elif artifact.exists():
            raise ValueError("PARTIAL_ANCHOR_MAY_NOT_BE_SILENTLY_REFIT:" + split)
    for job in list(pending_cells):
        directory = root / "phase_a" / job["split"] / (job["family"] + "_seed" + str(job["seed"]))
        if directory.exists():
            receipt = checked_cell(args, lock, job["split"], job["family"], job["seed"])
            require(job["split"] in completed_anchors, "COMPLETED_CELL_MISSING_ANCHOR")
            completed_cells.append(receipt)
            reused.append(job["key"])
            pending_cells.remove(job)
    logs = root / "scheduler_logs"
    logs.mkdir(exist_ok=True)
    host_baseline, gpu_baseline = 0, 0
    if pending_anchors or pending_cells:
        initial_resource = resources(root)
        host_baseline = initial_resource["cgroup_memory"]["usage_bytes"]
        gpu_baseline = max(g["used_MiB"] for g in initial_resource["GPUs"]) * 1024 ** 2
    try:
        while pending_anchors or pending_cells or active:
            for key, entry in list(active.items()):
                process = entry["process"]
                if process.poll() is None:
                    continue
                entry["log"].close()
                require(process.returncode == 0, "OWNED_MATRIX_CHILD_FAILED:" + key)
                if entry["kind"] == "anchor":
                    split = entry["job"]["split"]
                    completed_anchors[split] = checked_anchor(args, split)
                else:
                    job = entry["job"]
                    completed_cells.append(checked_cell(args, lock, job["split"], job["family"], job["seed"]))
                    new.append(job["key"])
                    print(json.dumps({"stage": stage, "complete_gpu_cells": len(completed_cells),
                                      "total_gpu_cells": len(names) * len(lock["phase_a_families"]) * len(SEEDS),
                                      "completed_key": key}), flush=True)
                del active[key]
            telemetry = resources(root)
            if not active:
                host_baseline = telemetry["cgroup_memory"]["usage_bytes"]
            if not any(entry["kind"] == "cell" for entry in active.values()):
                gpu_baseline = max(g["used_MiB"] for g in telemetry["GPUs"]) * 1024 ** 2
            for kind in ("anchor", "cell"):
                limit = args.cpu_parallel if kind == "anchor" else args.max_parallel
                while (sum(e["kind"] == kind for e in active.values()) < limit
                        and admission(telemetry, gpu=kind == "cell")
                        and host_reservation_admission(telemetry, active, (2 if kind == "anchor" else 4) * GIB, host_baseline)):
                    if kind == "anchor":
                        if not pending_anchors:
                            break
                        job = {"split": pending_anchors.pop(0)}
                        key = "anchor_" + job["split"]
                    else:
                        eligible = [job for job in pending_cells if job["split"] in completed_anchors
                            and job_gpu_admission(lock, job, telemetry, active, gpu_baseline)]
                        if not eligible:
                            break
                        job = eligible[0]
                        pending_cells.remove(job)
                        key = job["key"]
                    verify_sources_only(args, lock)
                    command = matrix_command(args, "anchor" if kind == "anchor" else "cell",
                        split=job["split"], family=job.get("family"), seed=job.get("seed"))
                    handle = (logs / (run_id + "_" + key + ".log")).open("xb")
                    process = subprocess.Popen(command, cwd=ROOT, env=environment(args), stdout=handle,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    OWNED.append(process)
                    active[key] = {"kind": kind, "job": job, "process": process, "log": handle}
                    active[key]["reserved_host_bytes"] = (2 if kind == "anchor" else 4) * GIB
                    if kind == "cell":
                        reservation, qualification = gpu_reservation(lock, job["split"], job["family"], telemetry)
                        active[key].update(reserved_gpu_bytes=reservation, reservation_qualification=qualification)
                    telemetry = resources(root)
            waiting = bool((pending_anchors or pending_cells) and not active)
            value = {"schema": "semantic_b0b7_p2_scheduler_state_v1", "status": "WAITING_FOR_ADMISSION" if waiting else "RUNNING" if pending_anchors or pending_cells or active else "COMPLETE",
                "stage": stage, "p2_lock_sha256": args.expected_lock_sha256, "scheduler_source_sha256": sha(__file__),
                "complete_anchors": completed_anchors, "complete_cells": completed_cells,
                "reused_complete_cells_not_new_fits": reused, "new_complete_cells_this_invocation": new,
                "pending_anchors": pending_anchors, "pending_cells": pending_cells,
                "active": [{"key": key, "pid": e["process"].pid, "kind": e["kind"], "job": e["job"],
                            "reserved_gpu_bytes": e.get("reserved_gpu_bytes", 0),
                            "reserved_host_bytes": e.get("reserved_host_bytes", 0),
                            "reservation_qualification": e.get("reservation_qualification")} for key, e in active.items()],
                "resource": telemetry, "max_gpu_cells": args.max_parallel, "max_cpu_anchor_processes": args.cpu_parallel,
                "gpu_baseline_reserved_bytes": gpu_baseline, "host_baseline_bytes": host_baseline,
                "no_partial_optimizer_resume": True}
            write(root / (stage.upper() + "_LIVE_STATE.private.json"), value, replace=True)
            if pending_anchors or pending_cells or active:
                time.sleep(args.poll_seconds)
    except BaseException:
        stop_owned()
        failure = {"status": "FAILED_STOPPED_OWNED_CHILDREN_NO_AUTORETRY", "stage": stage,
            "p2_lock_sha256": args.expected_lock_sha256, "completed_cells": completed_cells,
            "pending_anchors": pending_anchors, "pending_cells": pending_cells,
            "partial_outputs_preserved": True, "run_id": run_id}
        write(root / (stage.upper() + "_FAILURE_" + run_id + ".private.json"), failure)
        raise
    finally:
        for entry in active.values():
            entry["log"].close()
    result = {"schema": "semantic_b0b7_p2_scheduler_stage_v1", "status": "COMPLETE_ENGINEERING_GATE_ONLY",
        "stage": stage, "p2_lock_sha256": args.expected_lock_sha256, "completed_cells": completed_cells,
        "anchor_receipts": completed_anchors, "new_neural_fits_this_invocation": len(new),
        "reused_complete_cells_this_invocation": len(reused), "new_cell_keys": new, "reused_cell_keys": reused,
        "total_qualified_phase_a_units": len(completed_cells),
        "total_qualified_phase_a_updates": sum(c["phase_a"]["optimizer_steps"] for c in completed_cells),
        "total_qualified_saved_state_replays": sum(c["phase_a"]["actual_checkpoint_replays"] for c in completed_cells),
        "total_qualified_risk_optimizer_fits": sum(len(c["risk_fits"]) for c in completed_cells),
        "performance_or_CI_threshold_used_as_gate": False}
    write(root / (stage.upper() + "_COMPLETE_" + run_id + ".private.json"), result)
    return result


def all_inner_qualified(args, lock):
    result = []
    for outer in range(1, 6):
        for inner in range(3):
            split = f"outer{outer}_inner{inner}"
            checked_anchor(args, split)
            for family in sorted(lock["phase_a_families"]):
                for seed in SEEDS:
                    result.append(checked_cell(args, lock, split, family, seed))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "expected-lock-sha256", "private", "assets", "legacy-root", "frozen-images"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stage", choices=("e1", "inner_remaining", "final", "all"), default="all")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--cpu-parallel", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=5.)
    args = parser.parse_args()
    require(1 <= args.max_parallel <= 6 and 1 <= args.cpu_parallel <= 2 and 1 <= args.poll_seconds <= 30,
            "UNPROFILED_CONCURRENCY_OR_POLL_INTERVAL")
    root = Path(args.private).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = verify_lock(args)
    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
    manager = root / "ACTIVE_SCHEDULER_OWNER.private.json"
    owner = {"pid": os.getpid(), "run_id": run_id, "p2_lock_sha256": args.expected_lock_sha256}
    write(manager, owner)
    def on_signal(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    started = time.monotonic()
    completed_stages = []
    try:
        if not (root / "SPLITS.private.json").exists():
            synchronous_command(args, "split-plan", run_id)
        plan(args)
        e1_names = [f"outer1_inner{i}" for i in range(3)]
        if args.stage in ("e1", "all"):
            completed_stages.append(run_stage(args, lock, e1_names, "e1", run_id))
        if args.stage in ("inner_remaining", "all"):
            # E1 engineering evidence is rechecked before any other outer's
            # inner optimization; no score winner or confidence gate is used.
            for split in e1_names:
                checked_anchor(args, split)
                for family in lock["phase_a_families"]:
                    for seed in SEEDS:
                        checked_cell(args, lock, split, family, seed)
            rest = [f"outer{o}_inner{i}" for o in range(2, 6) for i in range(3)]
            completed_stages.append(run_stage(args, lock, rest, "inner_remaining", run_id))
        if args.stage in ("final", "all"):
            inner = all_inner_qualified(args, lock)
            selected = root / "ALL_OUTER_SELECTIONS_FROZEN.private.json"
            if not selected.exists():
                require(not (root / "B0").exists() and not any((root / "risk" / f"outer{o}_final").exists() for o in range(1, 6)),
                        "OUTER_EVALUATION_EXISTS_BEFORE_GLOBAL_SELECTION_BARRIER")
                synchronous_command(args, "freeze-outer-selection", run_id)
            barrier = verify_frozen_selection(args, lock)
            write(root / ("PRE_FINAL_BARRIER_" + run_id + ".private.json"), {
                **barrier, "qualified_inner_source_units": len(inner), "outer_test_scores_used": False})
            completed_stages.append(run_stage(args, lock, [f"outer{o}_final" for o in range(1, 6)], "final", run_id))
        result = {"schema": "semantic_b0b7_p2_scheduler_invocation_v1", "status": "COMPLETE_REQUESTED_FINITE_STAGES",
            "requested_stage": args.stage, "p2_lock_sha256": args.expected_lock_sha256,
            "scheduler_source_sha256": sha(__file__), "stages": completed_stages,
            "elapsed_wall_seconds": time.monotonic() - started,
            "new_neural_fits_this_invocation": sum(x["new_neural_fits_this_invocation"] for x in completed_stages),
            "source_families": sorted(lock["phase_a_families"]), "confidence_intervals_computed": False,
            "no_partial_optimizer_resume": True}
        write(root / ("SCHEDULER_COMPLETE_" + run_id + ".private.json"), result)
        print(json.dumps({"status": result["status"], "new_neural_fits": result["new_neural_fits_this_invocation"]}), flush=True)
    finally:
        stop_owned()
        if manager.is_file() and read(manager) == owner:
            manager.unlink() # only this invocation's own small ownership marker


if __name__ == "__main__":
    main()
