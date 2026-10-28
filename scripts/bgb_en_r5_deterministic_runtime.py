"""New numerical R5 contract; frozen scientific entry files remain unchanged.

Every child is routed back through this stdlib-only entry before torch import.
The special v2 lock status deliberately fails the unmodified v1 verifier, so
direct execution of an old entry cannot silently run a v2 fit with v1 flags.
This module performs no IO or runtime mutation merely by being imported.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = "scripts/bgb_en_r5_deterministic_runtime.py"
CONFIG = "configs/bgb_en/r5_deterministic_runtime_v2.json"
PARENT = "9d543b6b73861a163a40b8134a6e24f62094df91321c2202b11b6c0511cf123e"
V2_SCHEMA = "bgb_en_r5_protocol_lock_deterministic_v2"
V2_STATUS = "LOCKED_NUMERICAL_RUNTIME_V2_BEFORE_FORMAL_FITS"
SETTINGS = dict(CUBLAS_WORKSPACE_CONFIG=":4096:8", deterministic_algorithms=True,
    deterministic_warn_only=False, cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
    cudnn_benchmark=False, cudnn_deterministic=True, torch_threads=2)
ENTRIES = dict(qualify="scripts/bgb_en_r5_qualify.py", schedule="scripts/bgb_en_r5_schedule.py",
    run="scripts/bgb_en_r5_run.py", replay="scripts/bgb_en_r5_replay.py",
    report="scripts/bgb_en_r5_report_recovery.py", publication="scripts/bgb_en_r5_publication.py")
SUPPORT = ("scripts/bgb_en_r5_report_recovery.py", "scripts/bgb_en_r5_publication.py")
PUBLIC_ADDITIONS = (WRAPPER, CONFIG, "tests/test_bgb_en_r5_deterministic_runtime.py",
    "reports/bgb_en_r5/V1_REPLAY_FAILURE_SUMMARY.json", "reports/bgb_en_r5/NUMERIC_RUNTIME_DIAGNOSIS.json",
    "reports/bgb_en_r5/NUMERICAL_RUNTIME_V2.json")


def require(value, code):
    if not bool(value):
        raise ValueError(code)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _contract_content(lock):
    return {key: value for key, value in lock.items() if key not in ("created_UTC", "contract_sha256")}


def _source_hashes(root, names):
    root = Path(root).resolve()
    result = {}
    for name in sorted(names):
        relative = Path(name)
        target = root / relative
        require(not relative.is_absolute() and ".." not in relative.parts
                and target.resolve().is_relative_to(root) and target.is_file()
                and not target.is_symlink(), "RUNTIME_SOURCE_PATH_UNSAFE_OR_MISSING")
        result[relative.as_posix()] = sha256(target)
    return result


def load_runtime_config(*, root=ROOT):
    config = json.loads((Path(root) / CONFIG).read_text(encoding="utf-8-sig"))
    require(config.get("schema") == "bgb_en_r5_deterministic_runtime_v2"
            and config.get("status") == "LOCKED_NUMERICAL_RUNTIME_ONLY_BEFORE_ALL_NEW_R5_FITS"
            and config.get("parent_failed_contract_sha256") == PARENT
            and config.get("runtime_settings") == SETTINGS and config.get("entrypoints") == ENTRIES,
            "WRONG_OR_CHANGED_NUMERICAL_RUNTIME_CONFIG")
    require(config.get("planned_main_neural_fits") == 63 and config.get("planned_mechanism_neural_fits") == 45
            and config.get("old_neural_fits_or_selection_reused") is False
            and config.get("full_matrix_retrained_after_runtime_change") is True
            and config.get("upstream_prepared_bundles_reused") == 3 and config.get("new_teacher_fits") == 0
            and config.get("numerical_risk_tolerance") == 2e-6 and config.get("changed_pair_credit_limit") == 0
            and config.get("unchanged_scientific_source_required") is True,
            "RUNTIME_CONFIG_CHANGED_SCIENTIFIC_SCOPE_OR_REPLAY_GATE")
    parent = config["parent_failed_lock"]
    require(parent.get("contract_sha256") == PARENT and canonical(_contract_content(parent)) == PARENT
            and parent.get("schema") == "bgb_en_r5_protocol_lock_v1"
            and parent.get("status") == "LOCKED_BEFORE_FORMAL_FITS", "PARENT_FAILED_CONTRACT_NOT_EXACT")
    require(_source_hashes(root, parent["sources"]) == parent["sources"],
            "FROZEN_R5_SCIENTIFIC_SOURCE_CHANGED")
    require(set(config["runtime_support_sources"]) == set(SUPPORT)
            and _source_hashes(root, SUPPORT) == config["runtime_support_sources"],
            "RUNTIME_REPORTING_SUPPORT_SOURCE_CHANGED")
    require(config.get("private_directory_basename") == "bgb_en_r5_deterministic_v2_private"
            and config.get("public_runtime_receipt") == "reports/bgb_en_r5/NUMERICAL_RUNTIME_V2.json"
            and config.get("publication_additions") == list(PUBLIC_ADDITIONS),
            "RUNTIME_ARTIFACT_NAMESPACE_CHANGED")
    return config


def bootstrap_environment(*, environ=None, imported_modules=None):
    """Must happen in a fresh child before importing torch or its submodules."""
    environ = os.environ if environ is None else environ
    imported_modules = sys.modules if imported_modules is None else imported_modules
    require(not any(name == "torch" or name.startswith("torch.") for name in imported_modules),
            "CUBLAS_ENVIRONMENT_MUST_PRECEDE_TORCH_IMPORT")
    environ["CUBLAS_WORKSPACE_CONFIG"] = SETTINGS["CUBLAS_WORKSPACE_CONFIG"]
    environ["BGBEN_R5_RUNTIME_V2_PRETORCH"] = "1"
    return dict(CUBLAS_environment_set_before_torch_import=True)


def runtime_observation(torch_module):
    return dict(CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        CUBLAS_environment_set_before_torch_import=os.environ.get("BGBEN_R5_RUNTIME_V2_PRETORCH") == "1",
        deterministic_algorithms=bool(torch_module.are_deterministic_algorithms_enabled()),
        deterministic_warn_only=bool(torch_module.is_deterministic_algorithms_warn_only_enabled()),
        cuda_matmul_allow_tf32=bool(torch_module.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch_module.backends.cudnn.allow_tf32),
        cudnn_benchmark=bool(torch_module.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch_module.backends.cudnn.deterministic),
        torch_threads=int(torch_module.get_num_threads()), torch_version=str(torch_module.__version__),
        torch_cuda_build=None if torch_module.version.cuda is None else str(torch_module.version.cuda))


def assert_runtime_active(torch_module, proof=None):
    actual = runtime_observation(torch_module)
    require(actual["CUBLAS_environment_set_before_torch_import"]
            and all(actual[key] == value for key, value in SETTINGS.items()),
            "R5_DETERMINISTIC_RUNTIME_NOT_ACTIVE")
    if proof is not None:
        require(actual == proof, "R5_NUMERICAL_RUNTIME_CHANGED_AFTER_BOOTSTRAP")
    return actual


def configure_torch_runtime(torch_module):
    require(os.environ.get("BGBEN_R5_RUNTIME_V2_PRETORCH") == "1"
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "PRETORCH_ENVIRONMENT_RECEIPT_MISSING")
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.set_num_threads(2)
    return assert_runtime_active(torch_module)


def expected_source_names(config):
    return set(config["parent_failed_lock"]["sources"]) | set(SUPPORT) | {WRAPPER, CONFIG}


def runtime_binding(config, proof, *, root=ROOT):
    return dict(schema="bgb_en_r5_numerical_runtime_binding_v2", configuration=CONFIG,
                configuration_sha256=sha256(Path(root) / CONFIG), wrapper=WRAPPER,
                wrapper_sha256=sha256(Path(root) / WRAPPER), actual_runtime=dict(proof),
                frozen_scientific_source_identical_to_parent=True,
                report_entry_uses_schema_adapter_from_start=True)


def make_runtime_lock(base, config, proof, *, root=ROOT):
    parent = config["parent_failed_lock"]
    unchanged = set(parent) - {"sources", "contract_sha256", "created_UTC"}
    require(all(base.get(key) == parent[key] for key in unchanged),
            "RUNTIME_REPAIR_CHANGED_SCIENTIFIC_OR_DATA_CONTRACT")
    require(base.get("sources") == _source_hashes(root, expected_source_names(config)),
            "RUNTIME_LOCK_SOURCE_CLOSURE_NOT_EXACT")
    result = dict(base, schema=V2_SCHEMA, status=V2_STATUS,
                  numerical_runtime=runtime_binding(config, proof, root=root),
                  parent_failed_contract_sha256=PARENT)
    result.pop("created_UTC", None)
    result["contract_sha256"] = canonical(_contract_content(result))
    require(result["contract_sha256"] != PARENT, "NEW_RUNTIME_REQUIRES_NEW_CONTRACT")
    return result


def verify_runtime_lock(lock, root, config, proof, base_verifier):
    """Verify actual v2 first; the v1-compatible copy is never stored or used."""
    require(lock.get("schema") == V2_SCHEMA and lock.get("status") == V2_STATUS
            and lock.get("parent_failed_contract_sha256") == PARENT, "V2_RUNTIME_LOCK_REQUIRED_NO_V1_REUSE")
    require(canonical(_contract_content(lock)) == lock.get("contract_sha256"),
            "V2_RUNTIME_CONTRACT_CANONICAL_SHA_CHANGED")
    require(lock.get("numerical_runtime") == runtime_binding(config, proof, root=root),
            "V2_RUNTIME_BINDING_CHANGED")
    require(lock.get("sources") == _source_hashes(root, expected_source_names(config)),
            "V2_RUNTIME_OR_FROZEN_SOURCE_BYTES_CHANGED")
    parent = config["parent_failed_lock"]
    for key in ("data_fingerprints", "prepared_receipt_sha256", "planned_main_neural_fits",
                "planned_mechanism_neural_fits", "maximum_finalists", "outer_test_used_for_selection",
                "bootstrap", "confidence_intervals", "p_values", "upstream_commit", "main_config", "mechanism_config"):
        require(lock.get(key) == parent[key], "V2_DATA_PREPARATION_OR_SCIENTIFIC_SCOPE_CHANGED")
    compatible = dict(lock, status="LOCKED_BEFORE_FORMAL_FITS")
    compatible["contract_sha256"] = canonical(_contract_content(compatible))
    base_verifier(compatible, root)


def route_child_command(command, *, root=ROOT):
    """No shell-string rewriting and no second wrapping of an existing child."""
    if not isinstance(command, (list, tuple)) or len(command) < 2:
        return command
    script_at = 1
    while script_at < len(command) and str(command[script_at]) in ("-u", "-B"):
        script_at += 1
    if script_at >= len(command):
        return command
    script = Path(command[script_at]).resolve()
    root = Path(root).resolve()
    if script == (root / WRAPPER).resolve():
        return command
    routes = {(root / relative).resolve(): entry for entry, relative in ENTRIES.items()}
    routes[(root / "scripts/bgb_en_r5_report.py").resolve()] = "report"
    require(script != (root / "scripts/bgb_en_r5_resume_reporting.py").resolve(),
            "V2_REPORTING_HAS_NO_LEGACY_FAILURE_RESUME_ENTRY")
    if script not in routes:
        return command
    return [*command[:script_at], str(root / WRAPPER), "--entry", routes[script], "--", *command[script_at + 1:]]


def guard_entry_namespace(entry, arguments, config, *, root=ROOT):
    root = Path(root).resolve()
    if entry != "publication":
        require("--private" in arguments and arguments.index("--private") + 1 < len(arguments),
                "EXPLICIT_V2_PRIVATE_DIRECTORY_REQUIRED")
        private = Path(arguments[arguments.index("--private") + 1]).resolve()
        require(private.name == config["private_directory_basename"] and not private.is_relative_to(root)
                and not any((p / ".git").exists() for p in (private, *private.parents)),
                "V2_PRIVATE_NAMESPACE_REQUIRED_OLD_RUN_PRESERVED")
    for flag in ("--root", "--output"):
        if flag not in arguments:
            continue
        require(arguments.index(flag) + 1 < len(arguments), "RUNTIME_ENTRY_OPTION_VALUE_MISSING")
        target = Path(arguments[arguments.index(flag) + 1]).resolve()
        require(target == root if flag == "--root" else target.is_relative_to(root / "reports/bgb_en_r5"),
                "V2_PUBLIC_OUTPUT_MUST_STAY_IN_CURRENT_WORKTREE")


def write_runtime_receipt(config, proof, *, root=ROOT):
    value = {**runtime_binding(config, proof, root=root),
        "schema": "bgb_en_r5_numerical_runtime_certificate_v2", "status": "PASS_ACTUAL_RUNTIME_SETTINGS_ONLY",
        "parent_failed_contract_sha256": PARENT, "complete_108_training_or_GPU_qualification_inferred": False,
        "private_paths_or_patient_fields_present": False}
    destination = Path(root) / config["public_runtime_receipt"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
    except FileExistsError:
        # This immutable certificate is produced before the scheduler starts
        # its children, so normal parallel children only verify it.
        require(json.loads(destination.read_text(encoding="utf-8")) == value,
                "EXISTING_V2_RUNTIME_CERTIFICATE_CHANGED")
    return value


@contextmanager
def installed_runtime(config, proof, torch_module, *, root=ROOT):
    """In-memory adapters only; no frozen source file is edited."""
    root = Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    protocol = importlib.import_module("src.bgb_en.r5_protocol")
    common = importlib.import_module("scripts.bgb_en_r5_common")
    base_verify, base_make = protocol.verify_protocol_lock, protocol.make_protocol_lock
    base_context, run = common.context, subprocess.run
    previous = []

    def patch(module, name, value):
        previous.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def verify(lock, target_root):
        assert_runtime_active(torch_module, proof)
        verify_runtime_lock(lock, target_root, config, proof, base_verify)

    def make(*args, **kwargs):
        assert_runtime_active(torch_module, proof)
        return make_runtime_lock(base_make(*args, **kwargs), config, proof, root=root)

    def context(*args, **kwargs):
        assert_runtime_active(torch_module, proof)
        return base_context(*args, **kwargs)

    def routed(command, *args, **kwargs):
        rewritten = route_child_command(command, root=root)
        if rewritten is not command:
            env = dict(os.environ if kwargs.get("env") is None else kwargs["env"])
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            env["BGBEN_R5_RUNTIME_V2_REQUIRED"] = sha256(root / CONFIG)
            kwargs["env"] = env
        return run(rewritten, *args, **kwargs)

    try:
        patch(protocol, "verify_protocol_lock", verify)
        patch(protocol, "make_protocol_lock", make)
        patch(common, "verify_protocol_lock", verify)
        patch(common, "make_protocol_lock", make)
        patch(common, "context", context)
        patch(common, "NEW_SOURCES", tuple(dict.fromkeys((*common.NEW_SOURCES, WRAPPER, CONFIG, *SUPPORT))))
        patch(subprocess, "run", routed)
        # Already imported entries in a test/review process must receive the
        # same adapters as fresh entry imports in a production child.
        for name, module in list(sys.modules.items()):
            if not name.startswith("scripts.bgb_en_r5_") or module is None:
                continue
            if module is common:
                continue
            for field, original, replacement in (("verify_protocol_lock", base_verify, verify),
                    ("make_protocol_lock", base_make, make), ("context", base_context, context)):
                if getattr(module, field, None) is original:
                    patch(module, field, replacement)
        yield
    finally:
        # Freshly imported entry aliases are restored as well, keeping this
        # adapter usable by synthetic tests without polluting v1 tests.
        for name, module in list(sys.modules.items()):
            if name.startswith("scripts.bgb_en_r5_") and module is not None:
                for field, replacement, original in (("verify_protocol_lock", verify, base_verify),
                        ("make_protocol_lock", make, base_make), ("context", context, base_context)):
                    if getattr(module, field, None) is replacement:
                        setattr(module, field, original)
        for module, name, original in reversed(previous):
            setattr(module, name, original)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--entry", choices=tuple(ENTRIES))
    group.add_argument("--runtime-check", action="store_true", help="Inspect settings only; no data, receipts or fitting")
    args, trailing = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if trailing and trailing[0] == "--":
        trailing = trailing[1:]
    config = load_runtime_config()
    bootstrap_environment()
    torch_module = importlib.import_module("torch")
    proof = configure_torch_runtime(torch_module)
    if args.runtime_check:
        require(not trailing, "RUNTIME_CHECK_HAS_NO_TARGET_ARGUMENTS")
        print(json.dumps(dict(status="PASS_RUNTIME_V2_SETTINGS_NO_DATA_OR_FITS", actual_runtime=proof,
                              frozen_parent_source_count=len(config["parent_failed_lock"]["sources"])), sort_keys=True))
        return
    guard_entry_namespace(args.entry, trailing, config)
    write_runtime_receipt(config, proof)
    old_argv = sys.argv
    with installed_runtime(config, proof, torch_module):
        target = ENTRIES[args.entry]
        module = importlib.import_module(target[:-3].replace("/", "."))
        if args.entry == "publication":
            module.ALLOWLIST = frozenset(module.ALLOWLIST) | frozenset(config["publication_additions"])
        sys.argv = [str(ROOT / target), *trailing]
        print(json.dumps(dict(status="R5_RUNTIME_V2_ENTRY_ACTIVE", entry=args.entry,
                              runtime_configuration_sha256=sha256(ROOT / CONFIG)), sort_keys=True), flush=True)
        try:
            module.main()
        finally:
            sys.argv = old_argv
            assert_runtime_active(torch_module, proof)
            load_runtime_config()


if __name__ == "__main__":
    main()
