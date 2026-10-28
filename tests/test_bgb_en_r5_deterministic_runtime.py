"""Numerical-runtime routing/immutable-lock tests; no fit or patient data."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import bgb_en_r5_deterministic_runtime as r
from src.bgb_en import r5_protocol as p


class FakeTorch:
    def __init__(self):
        self.__version__ = "synthetic-test-only"
        self.version = SimpleNamespace(cuda=None)
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False))
        self.deterministic, self.warn, self.threads = False, True, 9

    def use_deterministic_algorithms(self, value, *, warn_only):
        self.deterministic, self.warn = value, warn_only

    def are_deterministic_algorithms_enabled(self):
        return self.deterministic

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self.warn

    def set_num_threads(self, value):
        self.threads = value

    def get_num_threads(self):
        return self.threads


def runtime(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("BGBEN_R5_RUNTIME_V2_PRETORCH", "1")
    torch = FakeTorch()
    return torch, r.configure_torch_runtime(torch)


def base_lock(config):
    parent = config["parent_failed_lock"]
    return p.make_protocol_lock(r.ROOT, parent["main_config"], parent["mechanism_config"],
        r.expected_source_names(config), data_fingerprints=parent["data_fingerprints"],
        prepared_receipt_sha256=parent["prepared_receipt_sha256"])


def test_bootstrap_sets_CUBLAS_before_torch_and_rejects_late_import():
    env = {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    assert r.bootstrap_environment(environ=env, imported_modules={})["CUBLAS_environment_set_before_torch_import"]
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    for imported in ({"torch": object()}, {"torch.cuda": object()}):
        with pytest.raises(ValueError, match="PRECEDE_TORCH"):
            r.bootstrap_environment(environ={}, imported_modules=imported)


def test_actual_flag_guard_rejects_drift_and_warn_only(monkeypatch):
    torch, proof = runtime(monkeypatch)
    assert proof["deterministic_algorithms"] and not proof["deterministic_warn_only"]
    assert not proof["cuda_matmul_allow_tf32"] and not proof["cudnn_allow_tf32"]
    torch.warn = True
    with pytest.raises(ValueError, match="NOT_ACTIVE"):
        r.assert_runtime_active(torch, proof)
    torch.warn = False; torch.__version__ = "changed"
    with pytest.raises(ValueError, match="CHANGED_AFTER_BOOTSTRAP"):
        r.assert_runtime_active(torch, proof)


def test_frozen_parent_bytes_and_reused_prepared_hashes_are_exact():
    config = r.load_runtime_config()
    parent = config["parent_failed_lock"]
    assert len(parent["sources"]) == 25 and parent["contract_sha256"] == r.PARENT
    assert len(parent["prepared_receipt_sha256"]) == 3
    assert config["planned_main_neural_fits"] == 63 and config["planned_mechanism_neural_fits"] == 45
    assert not config["old_neural_fits_or_selection_reused"] and config["new_teacher_fits"] == 0


def test_v2_contract_changes_only_runtime_and_legacy_entry_cannot_verify_it(monkeypatch):
    _, proof = runtime(monkeypatch)
    config = r.load_runtime_config(); base = base_lock(config)
    locked = r.make_runtime_lock(base, config, proof)
    assert locked["schema"] == r.V2_SCHEMA and locked["status"] == r.V2_STATUS
    assert locked["contract_sha256"] != r.PARENT
    assert locked["data_fingerprints"] == config["parent_failed_lock"]["data_fingerprints"]
    assert locked["prepared_receipt_sha256"] == config["parent_failed_lock"]["prepared_receipt_sha256"]
    before = deepcopy(locked)
    with pytest.raises(ValueError, match="not immutable"):
        p.verify_protocol_lock(locked, r.ROOT)
    r.verify_runtime_lock(locked, r.ROOT, config, proof, p.verify_protocol_lock)
    assert locked == before  # The v1-compatible shadow never replaces actual state.
    with pytest.raises(ValueError, match="NO_V1_REUSE"):
        r.verify_runtime_lock(config["parent_failed_lock"], r.ROOT, config, proof, p.verify_protocol_lock)


def test_tampered_runtime_data_or_omitted_wrapper_source_fails_closed(monkeypatch):
    _, proof = runtime(monkeypatch)
    config = r.load_runtime_config(); base = base_lock(config)
    missing = deepcopy(base); del missing["sources"][r.WRAPPER]
    with pytest.raises(ValueError, match="CLOSURE_NOT_EXACT"):
        r.make_runtime_lock(missing, config, proof)
    locked = r.make_runtime_lock(base, config, proof)
    changed = deepcopy(locked); changed["numerical_runtime"]["actual_runtime"]["cuda_matmul_allow_tf32"] = True
    with pytest.raises(ValueError, match="CANONICAL_SHA_CHANGED"):
        r.verify_runtime_lock(changed, r.ROOT, config, proof, p.verify_protocol_lock)
    changed["contract_sha256"] = r.canonical(r._contract_content(changed))
    with pytest.raises(ValueError, match="BINDING_CHANGED"):
        r.verify_runtime_lock(changed, r.ROOT, config, proof, p.verify_protocol_lock)
    changed = deepcopy(locked); changed["prepared_receipt_sha256"]["outer1_inner0"] = "0" * 64
    changed["contract_sha256"] = r.canonical(r._contract_content(changed))
    with pytest.raises(ValueError, match="SCIENTIFIC_SCOPE_CHANGED"):
        r.verify_runtime_lock(changed, r.ROOT, config, proof, p.verify_protocol_lock)


def test_new_lock_is_preserved_and_old_qualification_cannot_overwrite_it(tmp_path, monkeypatch):
    _, proof = runtime(monkeypatch)
    config = r.load_runtime_config(); locked = r.make_runtime_lock(base_lock(config), config, proof)
    path = tmp_path / "LOCK.json"
    actual = p.write_protocol_lock(path, locked)
    assert actual["status"] == r.V2_STATUS
    with pytest.raises(ValueError, match="refuse overwrite"):
        p.write_protocol_lock(path, config["parent_failed_lock"])
    assert json.loads(path.read_text()) == actual


@pytest.mark.parametrize("entry", ["qualify", "schedule", "run", "replay", "report", "publication"])
def test_every_child_entry_is_routed_and_report_never_requires_an_old_failure(entry):
    script = r.ENTRIES[entry]
    if entry == "report":
        script = "scripts/bgb_en_r5_report.py"
    original = [sys.executable, "-B", str(r.ROOT / script), "--stage", "main"]
    routed = r.route_child_command(original)
    assert routed[2] == str(r.ROOT / r.WRAPPER)
    assert routed[3:6] == ["--entry", entry, "--"]
    assert routed[-2:] == ["--stage", "main"]
    assert "failure" not in " ".join(routed).lower()
    assert r.route_child_command(routed) is routed
    with pytest.raises(ValueError, match="NO_LEGACY_FAILURE"):
        r.route_child_command([sys.executable, str(r.ROOT / "scripts/bgb_en_r5_resume_reporting.py")])


def test_context_lock_and_scheduler_subprocess_adapters_restore_cleanly(monkeypatch):
    from scripts import bgb_en_r5_common as common
    config = r.load_runtime_config(); torch, proof = runtime(monkeypatch)
    called = []
    def run(command, *args, **kwargs):
        called.append((command, kwargs)); return SimpleNamespace(returncode=0)
    def context(*args, **kwargs):
        return {"synthetic": True}
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(common, "context", context)
    old_verify, old_sources = p.verify_protocol_lock, common.NEW_SOURCES
    with r.installed_runtime(config, proof, torch):
        assert common.context()["synthetic"]
        assert r.WRAPPER in common.NEW_SOURCES and r.CONFIG in common.NEW_SOURCES
        base = common.make_protocol_lock(r.ROOT, common.MAIN_CONFIG, common.MECH_CONFIG,
            r.expected_source_names(config), data_fingerprints=config["parent_failed_lock"]["data_fingerprints"],
            prepared_receipt_sha256=config["parent_failed_lock"]["prepared_receipt_sha256"])
        assert base["status"] == r.V2_STATUS
        p.verify_protocol_lock(base, r.ROOT)
        subprocess.run([sys.executable, str(r.ROOT / "scripts/bgb_en_r5_run.py")], env={"TEST": "1"})
    assert p.verify_protocol_lock is old_verify and common.NEW_SOURCES == old_sources and common.context is context
    assert subprocess.run is run
    assert called[0][0][1] == str(r.ROOT / r.WRAPPER)
    assert called[0][1]["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert called[0][1]["env"]["BGBEN_R5_RUNTIME_V2_REQUIRED"] == r.sha256(r.ROOT / r.CONFIG)


def test_private_and_public_namespaces_preserve_old_run(tmp_path):
    config = r.load_runtime_config()
    private = tmp_path / config["private_directory_basename"]
    r.guard_entry_namespace("schedule", ["--private", str(private)], config)
    with pytest.raises(ValueError, match="OLD_RUN_PRESERVED"):
        r.guard_entry_namespace("run", ["--private", str(tmp_path / "old_run")], config)
    with pytest.raises(ValueError, match="CURRENT_WORKTREE"):
        r.guard_entry_namespace("report", ["--private", str(private), "--output", str(tmp_path / "old_reports")], config)
    with pytest.raises(ValueError, match="CURRENT_WORKTREE"):
        r.guard_entry_namespace("publication", ["--root", str(tmp_path)], config)


def test_runtime_certificate_is_immutable_anonymous_and_not_training_evidence(tmp_path, monkeypatch):
    config = r.load_runtime_config(); _, proof = runtime(monkeypatch)
    monkeypatch.setattr(r, "runtime_binding", lambda *a, **kw: {"schema": "binding", "actual_runtime": proof})
    first = r.write_runtime_receipt(config, proof, root=tmp_path)
    assert not first["complete_108_training_or_GPU_qualification_inferred"]
    assert not first["private_paths_or_patient_fields_present"]
    assert r.write_runtime_receipt(config, proof, root=tmp_path) == first
    path = tmp_path / config["public_runtime_receipt"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="CERTIFICATE_CHANGED"):
        r.write_runtime_receipt(config, proof, root=tmp_path)


def test_fresh_process_applies_real_torch_flags_without_data_or_fit():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-B", str(r.ROOT / r.WRAPPER), "--runtime-check"],
                            cwd=r.ROOT, env=env, capture_output=True, text=True, check=True)
    value = json.loads(result.stdout.strip())
    assert value["status"] == "PASS_RUNTIME_V2_SETTINGS_NO_DATA_OR_FITS"
    assert value["actual_runtime"]["CUBLAS_environment_set_before_torch_import"]
    assert value["actual_runtime"]["deterministic_algorithms"]
    assert not value["actual_runtime"]["cuda_matmul_allow_tf32"]
    assert value["frozen_parent_source_count"] == 25
