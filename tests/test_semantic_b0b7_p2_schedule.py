"""Synthetic receipt/resource tests, no models, endpoints, or subprocess fits."""
from argparse import Namespace
from types import SimpleNamespace
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("semantic_p2_scheduler_tests",
    Path(__file__).resolve().parents[1] / "scripts/semantic_b0b7_p2_schedule.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return S.sha(path)


def fake_cell(tmp_path, split="outer1_inner0", family="FAMILY", seed=31):
    """Files are explicitly synthetic bytes, not fake scientific reports."""
    args = Namespace(private=str(tmp_path), expected_lock_sha256="a" * 64)
    phase_lock = {"frozen_sources": {"synthetic": "b" * 64}}
    lock = {"phase_a_families": {family: {"source_arm": "B4"}},
            "family_readouts": {family: ["B4"]}, "phase_a_lock": phase_lock}
    directory = tmp_path / "phase_a" / split / (family + "_seed" + str(seed))
    states = []
    for epoch in S.CHECKPOINTS:
        filename = "state" + str(epoch) + ".private.pt"
        digest = put(directory / filename, ("SYNTHETIC_NOT_A_MODEL_" + str(epoch)).encode())
        states.append({"epoch": epoch, "file": filename, "sha256": digest, "actual_replay": True, "replay_max_error": 0.})
    def valid():
        return {"program": {"patient_equal_masked_huber": .1},
                "retrieval_hard": {d: {"paired_gap_mean": .2} for d in ("image_to_gene", "gene_to_image")}}
    history = [{"epoch": e, "gradient_norms": {"synthetic_layer": 1.},
                **({"validation": valid()} if e in S.CHECKPOINTS else {})} for e in range(1, 401)]
    receipt = {"status": "COMPLETE", "identity": "NEW_FIT", "seed": seed,
        "frozen_sources": phase_lock["frozen_sources"], "master_lock_sha256": S.canonical_hash(phase_lock),
        "DSS_OS_PFI_accessed": False, "P2_main_pretraining_eligible": True, "optimizer_steps": 400,
        "history": history, "states": states, "selected_epoch": 50, "source_state_file": states[0]["file"],
        "selected_checkpoint_sha256": states[0]["sha256"], "input_sha256": "c" * 64,
        "program_huber_tolerance": .0001, "evaluate_valid": True, "valid_patients": 137}
    digest = put(directory / "cell.private.json", receipt)
    prep = put(tmp_path / "preprocessing" / "SYNTHETIC.private.pkl", b"SYNTHETIC_PREPROCESSING_BYTES_ONLY")
    put(directory / "MATRIX_BINDING.private.json", {"p2_lock_sha256": args.expected_lock_sha256,
        "split": split, "family": family, "seed": seed, "cell_receipt_sha256": digest,
        "training_only_input_sha256": receipt["input_sha256"],
        "shared_preprocessing": {"file": "SYNTHETIC.private.pkl", "sha256": prep}})
    records = []
    for lam in (.1, 1., 10.):
        risk_dir = tmp_path / "risk" / split / ("B4_seed" + str(seed))
        name = "lambda" + str(lam) + ".private.pkl"
        artifact = put(risk_dir / name, b"SYNTHETIC_NOT_A_PICKLE_OR_RISK_ARRAY")
        risk = {"schema": "semantic_b0b7_p2_risk_cell_v1", "status": "FITTED_AND_EVALUATED",
            "role": "inner", "DSS_coefficients": 26, "risk_optimizer_fits": 1, "ridge_lambda": lam,
            "source_and_scope_bindings": {"lock_sha256": args.expected_lock_sha256, "split": split,
                "arm": "B4", "seed": seed, "phase_a_source_family": family,
                "phase_a_selected_state_sha256": receipt["selected_checkpoint_sha256"],
                "input_training_sha256": receipt["input_sha256"]},
            "artifact_sha256": artifact,
            "risk_optimizer": {"parameters": 26, "active_design_columns": 26,
                "gradient_max": 1e-7, "gradient_tolerance": 2e-5, "oracle_loss_abs_error": 0.},
            "saved_linear_head_replay": {"actual_linear_head_replay": True, "neural_forward": False,
                "max_abs_risk_error": 0., "fold_metric_components_equal": True},
            "evaluation_metrics": {"Harrell_num": 9., "Harrell_den": 10., "Uno60_valid": True,
                "Uno_num": 8., "Uno_den": 10.}}
        put(risk_dir / ("lambda" + str(lam) + ".private.json"), risk)
        records.append({"arm": "B4", "ridge_lambda": lam})
    put(directory / "RISK_BUNDLE_COMPLETE.private.json", {"status": "COMPLETE", "records": records,
        "p2_lock_sha256": args.expected_lock_sha256,
        "frozen_source_checkpoint_sha256": receipt["selected_checkpoint_sha256"]})
    return args, lock, directory


def resource(used=100, host=20 * S.GIB, limit=96 * S.GIB, disk=20 * S.GIB):
    return {"cgroup_memory": {"usage_bytes": host, "limit_bytes": limit},
            "filesystem_free_bytes": disk, "GPUs": [{"total_MiB": 32768, "used_MiB": used}]}


def test_finite_jobs_come_from_declared_families_not_hardcoded_eight_arms():
    jobs = S.jobs(["outer1_inner0", "outer1_inner1", "outer1_inner2"], {"SOURCE_A": {}, "SOURCE_B": {}})
    assert len(jobs) == 18 and len({j["key"] for j in jobs}) == 18


def test_profile_reservation_prevents_startup_oom_when_workers_have_not_allocated_yet():
    r = resource(used=100)
    active = {"a": {"reserved_gpu_bytes": 12 * S.GIB}, "b": {"reserved_gpu_bytes": 12 * S.GIB}}
    assert not S.reservation_admission(r, active, 12 * S.GIB)
    assert S.reservation_admission(r, {"a": active["a"]}, 12 * S.GIB)


def test_per_family_role_budgets_and_native_exclusive_fallback_are_explicit():
    lock = {"phase_a_families": {"NATIVE": {"source_arm": "B2_NATIVE"}, "NORMAL": {"source_arm": "B4"}},
            "resource_admission": {"gpu_peak_budget_bytes": {"inner": {"NORMAL": 4 * S.GIB}, "outer_final": {"NORMAL": 6 * S.GIB}}}}
    assert S.gpu_reservation(lock, "outer1_inner0", "NORMAL", resource())[0] == 4 * S.GIB
    assert S.gpu_reservation(lock, "outer1_final", "NORMAL", resource())[0] == 6 * S.GIB
    job = {"split": "outer1_inner0", "family": "NATIVE"}
    assert S.job_gpu_admission(lock, job, resource(), {})
    assert not S.job_gpu_admission(lock, job, resource(), {"a": {"kind": "cell", "reserved_gpu_bytes": 1}})


@pytest.mark.parametrize("r", [resource(disk=S.GIB), resource(host=91*S.GIB), resource(host=90*S.GIB, limit=96*S.GIB), resource(used=32000)])
def test_actual_cgroup_disk_gpu_margins_are_required(r):
    assert not S.admission(r, gpu=True)


def test_host_peak_reservations_cover_startup_before_rss_rises():
    r = resource(host=70 * S.GIB)
    active = {str(i): {"reserved_host_bytes": 4 * S.GIB} for i in range(4)}
    assert not S.host_reservation_admission(r, active, 4 * S.GIB, 70 * S.GIB)


def test_hash_qualified_complete_cell_can_be_reused_without_deserializing_arrays(tmp_path):
    args, lock, _ = fake_cell(tmp_path)
    result = S.checked_cell(args, lock, "outer1_inner0", "FAMILY", 31)
    assert result["phase_a"]["optimizer_steps"] == 400
    assert result["phase_a"]["actual_checkpoint_replays"] == 4 and len(result["risk_fits"]) == 3


def test_partial_risk_bundle_cannot_be_silently_resumed_as_complete(tmp_path):
    args, lock, directory = fake_cell(tmp_path)
    (directory / "RISK_BUNDLE_COMPLETE.private.json").unlink()
    with pytest.raises(ValueError, match="MISSING_OR_ESCAPING"):
        S.checked_cell(args, lock, "outer1_inner0", "FAMILY", 31)


def test_mutated_checkpoint_bytes_prevent_complete_cell_reuse(tmp_path):
    args, lock, directory = fake_cell(tmp_path)
    (directory / "state50.private.pt").write_bytes(b"TAMPERED_SYNTHETIC_BYTES")
    with pytest.raises(ValueError, match="HASH_CHANGED"):
        S.checked_cell(args, lock, "outer1_inner0", "FAMILY", 31)


def test_risk_source_or_convergence_mismatch_rejected(tmp_path):
    args, lock, _ = fake_cell(tmp_path)
    path = tmp_path / "risk/outer1_inner0/B4_seed31/lambda0.1.private.json"
    row = S.read(path)
    row["risk_optimizer"]["gradient_max"] = .1
    put(path, row)
    with pytest.raises(ValueError, match="CONVERGENCE_FINITE"):
        S.checked_cell(args, lock, "outer1_inner0", "FAMILY", 31)


def test_same_num_cells_cannot_replace_missing_expected_readout(tmp_path):
    args, lock, directory = fake_cell(tmp_path)
    path = directory / "RISK_BUNDLE_COMPLETE.private.json"
    row = S.read(path)
    row["records"][2] = row["records"][1]
    put(path, row)
    with pytest.raises(ValueError, match="INCOMPLETE_OR_DUPLICATE"):
        S.checked_cell(args, lock, "outer1_inner0", "FAMILY", 31)


def test_global_barrier_rechecks_all_five_outers_before_creating_marker(tmp_path):
    args = lock = None
    for outer in range(1, 6):
        for inner in range(3):
            for seed in S.SEEDS:
                args, lock, _ = fake_cell(tmp_path, f"outer{outer}_inner{inner}", seed=seed)
    entries = {}
    for outer in range(1, 6):
        entries[str(outer)] = {"pretraining": {"FAMILY": {str(seed): S.inner_choice(args, outer, "FAMILY", seed) for seed in S.SEEDS}},
                              "risk": {"B4": {str(seed): S.risk_choice(args, outer, "B4", seed) for seed in S.SEEDS}}}
    path = tmp_path / "ALL_OUTER_SELECTIONS_FROZEN.private.json"
    selected = {"p2_lock_sha256": args.expected_lock_sha256,
                "status": "ALL_FIVE_OUTERS_SELECTED_BEFORE_FIRST_OUTER_SCORE", "outer_results_read": False, "entries": entries}
    put(path, selected)
    assert S.verify_frozen_selection(args, lock)["selection_file_sha256"] == S.sha(path)
    selected["entries"]["5"]["risk"]["B4"]["83"]["ridge_lambda"] = .1
    put(path, selected)
    with pytest.raises(ValueError, match="FROZEN_RISK_CHOICE_CHANGED"):
        S.verify_frozen_selection(args, lock)


def test_owned_child_termination_never_searches_or_kills_unowned_processes(monkeypatch):
    class Child:
        def __init__(self, pid, done=False): self.pid, self.done = pid, done
        def poll(self): return 0 if self.done else None
        def wait(self, timeout): self.done=True
    live, done = Child(1001), Child(1002, True)
    monkeypatch.setattr(S, "OWNED", [live, done])
    killed = []
    # Replace only the scheduler's module reference; mutating global os.name
    # would make pytest/pathlib attempt PosixPath on a Windows host.
    monkeypatch.setattr(S, "os", SimpleNamespace(name="posix", killpg=lambda pid, sig: killed.append(pid)))
    S.stop_owned()
    assert killed == [1001]


def test_finite_stage_launches_anchor_before_three_seed_cells_and_reuses_complete_without_gpu(tmp_path, monkeypatch):
    args = Namespace(private=str(tmp_path), expected_lock_sha256="a"*64,
        lock="SYNTHETIC_LOCK", assets="SYNTHETIC_ASSETS", legacy_root="SYNTHETIC_LEGACY",
        frozen_images="SYNTHETIC_FM", threads=2, max_parallel=6, cpu_parallel=2, poll_seconds=1.)
    lock = {"phase_a_families": {"FAMILY": {"source_arm": "B4"}},
            "family_readouts": {"FAMILY": ["B4"]},
            "phase_a_lock": {"frozen_sources": {"synthetic": "b"*64}}}
    commands = []
    class SyntheticProcess:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.pid = 2000 + len(commands)
            self.returncode = 0
            kind = command[2]
            split = command[command.index("--split")+1]
            if kind == "anchor":
                p = tmp_path / "anchors" / (split + ".private.pkl")
                digest = put(p, b"SYNTHETIC_ANCHOR_NOT_A_REAL_PICKLE")
                put(p.with_suffix(".json"), {"p2_lock_sha256": args.expected_lock_sha256,
                    "split": split, "artifact_sha256": digest, "Cox_optimizer_fits": 52,
                    "old_scores_reused": False})
            else:
                assert kind == "cell" and (tmp_path / "anchors" / (split+".private.json")).is_file()
                family = command[command.index("--family")+1]
                seed = int(command[command.index("--seed")+1])
                fake_cell(tmp_path, split, family, seed)
        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode
    monkeypatch.setattr(S.subprocess, "Popen", SyntheticProcess)
    monkeypatch.setattr(S, "verify_sources_only", lambda *args: None)
    monkeypatch.setattr(S, "resources", lambda *args: resource())
    monkeypatch.setattr(S.time, "sleep", lambda *args: None)
    monkeypatch.setattr(S, "OWNED", [])
    first = S.run_stage(args, lock, ["outer1_inner0"], "SYNTHETIC_ONLY", "run1")
    assert [c[2] for c in commands] == ["anchor", "cell", "cell", "cell"]
    assert first["new_neural_fits_this_invocation"] == 3 and first["total_qualified_risk_optimizer_fits"] == 9
    def forbidden_gpu(*args): raise AssertionError("REUSE_MUST_NOT_REQUIRE_GPU_OR_START_OPTIMIZER")
    monkeypatch.setattr(S, "resources", forbidden_gpu)
    commands.clear()
    second = S.run_stage(args, lock, ["outer1_inner0"], "SYNTHETIC_ONLY", "run2")
    assert commands == [] and second["new_neural_fits_this_invocation"] == 0
    assert second["reused_complete_cells_this_invocation"] == 3
