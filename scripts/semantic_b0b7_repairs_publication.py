"""Exact, append-only P1R/P2 text publication; no training, identifiers or Git.

The original scanner is imported byte-for-byte. Local staging is a preflight;
only final-scan consumes the already-established server-private denylist. This
helper never builds or exports any identifier registry, reads model arrays, or
performs Git operations. Inputs must be explicit file/bytes/SHA allowlists.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_publication as original

SCANNER_SHA = "4c70929d57459a9e97774e6319bd455deedc29fbcedf077131c918c53db60e98"
DENYLIST_SHA = "d73b70939a38277f8f9f693e9958afcc9435893cd05f5c1d6a6d08a100ffe867"
P1R_BASE = "98f42e298975c39e9e3bde70f2250b2334caca8b"
OLD_MANIFEST = "reports/semantic_b0b7/PUBLICATION_MANIFEST.json"
OLD_MANIFEST_SHA = "421f826cce97cef39d398fe774e63fec496238b9f0f6be23485b4b5ff07a5b2d"
OLD_AUDIT = "reports/semantic_b0b7/PUBLICATION_SOURCE_AUDIT.json"
PLAN_ZIP_MEMBER = "_publication/PLAN.json"
HELPER = "scripts/semantic_b0b7_repairs_publication.py"
HELPER_TEST = "tests/test_semantic_b0b7_repairs_publication.py"
PROMPTS = {"P1R": "P1-R_语义机制修复与无DSS资格实验_Codex_Prompt.md",
           "P2": "P2_B0-B7_统一协议三模态DSS生存验证_Codex_Prompt.md"}
MAX_BYTES = 2 * 1024**2
MAX_FILES = 998  # manifest also contains one generated scan receipt, <1000 list entries
# These are receipts from completed real server scans, not new trust supplied
# by an arbitrary input plan. Their full bytes, schema and provenance are fixed.
PRIOR_REAL_SCANS = {
    "reports/semantic_b0b7_p1r/R0_T2_PUBLIC_REPORT_SCAN_RECEIPT.json": {
        "sha256": "1d59cf45d2a0fb703249afe44be4b1199a553ef6167e70e1fe28dd04734e1546",
        "schema": "semantic_b0b7_p1r_r0_t2_public_scan_v1",
        "status": "PASS_ORIGINAL_RULES_AND_EXISTING_IDENTIFIER_REGISTRY", "count": 26,
        "provenance": {"source_R0_index_sha256": "1455452fc0afa529157c4a3b42bf99cd606f9635e58c130d82bbe987ae5d0d71",
            "source_T2_sha256": "e63c8d37734e7456b46caf6278d7ebe99b8ca946b1eaa7f5f849afea455c7826"}},
    "reports/semantic_b0b7_p1r/MATRIX_PUBLICATION_SCAN_RECEIPT.json": {
        "sha256": "c658b4cb9de88908872820a9cbd77002561c40218e85391a6d7010274f033116",
        "schema": "semantic_b0b7_p1r_matrix_publication_scan_v1",
        "status": "PASS_UNCHANGED_ORIGINAL_SCANNER_AND_IDENTIFIER_REGISTRY", "count": 56,
        "provenance": {"master_lock_sha256": "cbbfa73f974773fdd86ba4730533f36e774c72586d8364255472c9307141ac5a",
            "matrix_report_build_sha256": "502a0d281d4cc35fcdc05282f8423b7f671fff88108edc0a7739d9efa4076031"}},
    "reports/semantic_b0b7_p1r/EXACT_CHANCE_PUBLIC_SCAN_RECEIPT.json": {
        "sha256": "87f1db9d7eaef9547bc56bd11842fc802ecae526d41c94a99d76b15e39f69220",
        "schema": "semantic_b0b7_p1r_exact_chance_addendum_scan_v1",
        "status": "PASS_ORIGINAL_RULES_AND_PRIVATE_IDENTIFIER_REGISTRY", "count": 1,
        "provenance": {"source_private_full_audit_sha256": "0d15840ec560aa8eda501fc09b4fe4a4189ddc81df3e2efb46c3ec4b8b7893f0"}}
}
require = original.require
sha = original.sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def encode(value):
    original.validate_public_json(value)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(len(raw) <= MAX_BYTES, "REPAIRS_OVERSIZED_METADATA")
    return raw


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def publication_paths(phase):
    require(phase in ("P1R", "P2"), "REPAIRS_UNKNOWN_PHASE")
    prefix = "reports/semantic_b0b7_" + phase.lower() + "/"
    return prefix + "PUBLICATION_SCAN_RECEIPT.json", prefix + "PUBLICATION_MANIFEST.json"


def safe_relative(name):
    require(isinstance(name, str) and bool(name) and "\\" not in name and ":" not in name and "\x00" not in name,
            "REPAIRS_UNSAFE_RELATIVE_PATH")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "REPAIRS_UNSAFE_RELATIVE_PATH")
    return name


def safe_name(name, phase):
    safe_relative(name)
    scan_receipt, manifest = publication_paths(phase)
    require(name not in (scan_receipt, manifest), "REPAIRS_GENERATED_FILE_CANNOT_BE_CANDIDATE")
    suffix = phase.lower()
    patterns = (rf"src/semantic_b0b7/{suffix}_[A-Za-z0-9_]+\.py",
                rf"scripts/semantic_b0b7_{suffix}[A-Za-z0-9_]*\.py",
                rf"tests/test_semantic_b0b7_{suffix}[A-Za-z0-9_]*\.py",
                rf"configs/semantic_b0b7/{suffix}_[A-Za-z0-9_]+\.json",
                rf"reports/semantic_b0b7_{suffix}/[A-Za-z0-9_]+\.(?:json|csv|md)")
    extra = {PROMPTS[phase]}
    if phase == "P1R":
        extra.update((HELPER, HELPER_TEST, "scripts/semantic_b0b7_partition_probe.py"))
    require(name in extra or any(re.fullmatch(p, name) for p in patterns), "REPAIRS_PATH_OUTSIDE_NEW_PHASE_ALLOWLIST")
    return name


def checked(root, name, expected=None, size=None):
    safe_relative(name)
    root = Path(root).resolve()
    path = root.joinpath(*PurePosixPath(name).parts)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            "REPAIRS_MISSING_OR_ESCAPING_FILE")
    require(not any(p.is_symlink() for p in path.parents if p.is_relative_to(root)), "REPAIRS_SYMLINK_PARENT")
    if expected is not None:
        require(sha(path) == expected, "REPAIRS_PRESERVED_OR_CANDIDATE_SHA_CHANGED")
    if size is not None:
        require(path.stat().st_size == size, "REPAIRS_CANDIDATE_SIZE_CHANGED")
    return path


def validate_rows(rows):
    require(isinstance(rows, list) and 0 < len(rows) <= MAX_FILES and len(rows) == len({r["file"] for r in rows}),
            "REPAIRS_EXACT_UNIQUE_NONEMPTY_ROWS_REQUIRED")
    for row in rows:
        require(set(row) == {"file", "sha256", "bytes"} and re.fullmatch(r"[0-9a-f]{64}", row["sha256"] or "")
                and type(row["bytes"]) is int and 0 < row["bytes"] <= MAX_BYTES,
                "REPAIRS_INVALID_FILE_METADATA_ROW")
        safe_relative(row["file"])
    return sorted(rows, key=lambda row: row["file"])


@contextmanager
def exact_scanner_allowlist(rows):
    require(sha(original.__file__) == SCANNER_SHA, "REPAIRS_ORIGINAL_SCANNER_SOURCE_CHANGED")
    saved = original.ALLOWLIST
    original.ALLOWLIST = frozenset(row["file"] for row in rows)
    try:
        yield
    finally:
        original.ALLOWLIST = saved


def scan_exact(root, rows, pattern):
    rows = sorted(rows, key=lambda row: row["file"])
    with exact_scanner_allowlist(rows):
        actual = [original.scan_file(Path(root).resolve(), row["file"], pattern) for row in rows]
    require(actual == rows, "REPAIRS_SCANNED_EXACT_BYTES_CHANGED")
    return actual


def prior_scan_provenance(root, receipt, indexed):
    """Validate the public source/build bindings without deserializing arrays."""
    schema = receipt["schema"]
    if schema == "semantic_b0b7_p1r_r0_t2_public_scan_v1":
        require(receipt["new_fits_by_reporter"] == receipt["new_model_forwards_by_reporter"] == 0 and
                receipt["patient_level_arrays_or_weights_exported"] is False and receipt["private_identifiers_exported"] is False,
                "REPAIRS_PRIOR_R0_SCOPE_INVALID")
        build_name = "reports/semantic_b0b7_p1r/R0_T2_REPORT_BUILD_AUDIT.json"
        reporter_name = "scripts/semantic_b0b7_p1r_diagnostic_report.py"
        build_schema = "semantic_b0b7_p1r_r0_t2_report_build_v1"
        expected_outputs = 21
    elif schema == "semantic_b0b7_p1r_matrix_publication_scan_v1":
        require(receipt["new_fits"] == receipt["model_forwards"] == 0 and receipt["all_per_file_bytes_hashes_inspected"]
                and receipt["private_identifiers_exported"] is False and receipt["patient_arrays_predictions_or_weights_exported"] is False
                and receipt["opaque_encoded_reports"] is False and receipt["CI_computed"] is False and
                receipt["unchanged_limits"] == {"CSV_max_rows_including_header": 20000, "JSON_max_list_length": 1000,
                    "large_pure_numeric_list_threshold": 128, "text_file_max_bytes": MAX_BYTES},
                "REPAIRS_PRIOR_MATRIX_SCOPE_OR_RULES_INVALID")
        build_name = "reports/semantic_b0b7_p1r/MATRIX_REPORT_BUILD_AUDIT.json"
        reporter_name = "scripts/semantic_b0b7_p1r_report.py"
        build_schema = "semantic_b0b7_p1r_matrix_report_build_v1"
        expected_outputs = 49
    else:
        require(schema == "semantic_b0b7_p1r_exact_chance_addendum_scan_v1" and
                receipt["new_fits"] == receipt["new_model_forwards"] == 0, "REPAIRS_PRIOR_CHANCE_SCOPE_INVALID")
        row = indexed["reports/semantic_b0b7_p1r/P1R_EXACT_RETRIEVAL_CHANCE.json"]
        audit = read(checked(root, row["file"], row["sha256"], row["bytes"]))
        require(audit["schema"] == "semantic_b0b7_p1r_public_exact_retrieval_chance_v1" and
                audit["status"] == "COMPLETE_EXACT_CHANCE_WITH_ACTUAL_SAVED_MASK_VERIFICATION" and
                audit["private_full_audit_sha256"] == receipt["source_private_full_audit_sha256"] and
                audit["DSS_OS_PFI_or_time_events_accessed"] is False and
                audit["patient_identity_or_raw_pair_masks_exported"] is False and
                audit["new_fits"] == audit["neural_model_forwards"] == 0 and not audit["checkpoint_selection_changed"],
                "REPAIRS_PRIOR_CHANCE_PROVENANCE_INVALID")
        checked(root, "scripts/semantic_b0b7_p1r_chance.py", audit["source_sha256"])
        checked(root, "configs/semantic_b0b7/p1r_lock.json", audit["master_lock_sha256"])
        return
    build_row = indexed[build_name]
    build = read(checked(root, build_name, build_row["sha256"], build_row["bytes"]))
    require(build["schema"] == build_schema and build["optimizer_updates"] == build["model_forwards"] == 0,
            "REPAIRS_PRIOR_BUILD_SCHEMA_OR_SCOPE_INVALID")
    require(build["reporter_source_sha256"] == indexed[reporter_name]["sha256"], "REPAIRS_PRIOR_REPORTER_SOURCE_INVALID")
    checked(root, reporter_name, build["reporter_source_sha256"])
    if schema == "semantic_b0b7_p1r_r0_t2_public_scan_v1":
        require(build["source_R0_index_sha256"] == receipt["source_R0_index_sha256"] and
                build["source_T2_sha256"] == receipt["source_T2_sha256"] and
                build["historical_state_fields_changed"] is False and build["raw_patient_arrays_or_weights_loaded"] is False,
                "REPAIRS_PRIOR_R0_BUILD_PROVENANCE_INVALID")
        checked(root, "configs/semantic_b0b7/p1r_t2_lock.json", build["T2_lock_sha256"])
    else:
        require(build_row["sha256"] == receipt["matrix_report_build_sha256"] and
                build["source_master_lock_sha256"] == receipt["master_lock_sha256"] and
                build["checkpoint_or_hyperparameter_selection_changed"] is False and
                build["patient_level_arrays_or_model_weights_loaded_by_reporter"] is False,
                "REPAIRS_PRIOR_MATRIX_BUILD_PROVENANCE_INVALID")
        checked(root, "configs/semantic_b0b7/p1r_lock.json", build["source_master_lock_sha256"])
    require(len(build["files"]) == expected_outputs and len({x["file"] for x in build["files"]}) == expected_outputs,
            "REPAIRS_PRIOR_BUILD_OUTPUT_SET_INVALID")
    for item in build["files"]:
        require(Path(item["file"]).name == item["file"], "REPAIRS_PRIOR_BUILD_PATH_INVALID")
        full = "reports/semantic_b0b7_p1r/" + item["file"]
        require(indexed[full] == {**item, "file": full}, "REPAIRS_PRIOR_BUILD_OUTPUT_HASH_INVALID")


def prior_scan_evidence(root):
    evidence, valid, invalid = {}, [], []
    for name, specification in PRIOR_REAL_SCANS.items():
        if not (Path(root) / name).is_file():
            continue
        try:
            path = checked(root, name, specification["sha256"])
            receipt = read(path)
            require(receipt["schema"] == specification["schema"] and receipt["status"] == specification["status"] and
                    receipt["original_scanner_sha256"] == SCANNER_SHA and receipt["established_denylist_sha256"] == DENYLIST_SHA and
                    receipt["Git_writes"] == 0, "REPAIRS_PRIOR_SCAN_SCHEMA_SCANNER_OR_DENYLIST_INVALID")
            require(all(receipt.get(k) == v for k, v in specification["provenance"].items()), "REPAIRS_PRIOR_SCAN_PROVENANCE_INVALID")
            rows = validate_rows(receipt["files"])
            require(len(rows) == specification["count"], "REPAIRS_PRIOR_SCAN_FILE_COUNT_INVALID")
            for row in rows:
                safe_name(row["file"], "P1R")
            indexed = {row["file"]: row for row in rows}
            prior_scan_provenance(root, receipt, indexed)
            for row in rows:
                evidence.setdefault(row["file"], []).append({"row": row, "receipt_file": name, "receipt_sha256": specification["sha256"]})
            valid.append({"file": name, "sha256": specification["sha256"], "original_scanned_file_count": len(rows),
                          "schema_and_source_provenance_validated": True})
        except (ValueError, OSError, KeyError, TypeError):
            invalid.append({"file": name, "status": "NOT_REUSED_INVALID_OR_INCOMPLETE_PRIOR_SCAN_EVIDENCE"})
    return evidence, valid, invalid


def scan_with_exact_prior_evidence(root, rows, pattern):
    """Same privacy coverage; prior real exact-byte scans avoid duplicate regex."""
    require(sha(original.__file__) == SCANNER_SHA, "REPAIRS_ORIGINAL_SCANNER_SOURCE_CHANGED")
    rows = sorted(rows, key=lambda row: row["file"])
    evidence, valid, invalid = prior_scan_evidence(root)
    pending, coverage = [], []
    for row in rows:
        checked(root, row["file"], row["sha256"], row["bytes"])
        matches = [x for x in evidence.get(row["file"], ()) if x["row"] == row]
        if matches:
            chosen = matches[0]
            coverage.append({**row, "coverage": "REUSED_EXACT_PRIOR_REAL_SERVER_SCAN",
                             "prior_scan_receipt_sha256": chosen["receipt_sha256"]})
        else:
            pending.append(row)
            coverage.append({**row, "coverage": "SCANNED_NOW_WITH_ESTABLISHED_PRIVATE_IDENTIFIER_PATTERN"})
    scan_exact(root, pending, pattern)
    # All bytes, including reused files, must still match after scanning new ones.
    for row in rows:
        checked(root, row["file"], row["sha256"], row["bytes"])
    return rows, {"scanned_now_count": len(pending), "reused_exact_prior_scan_count": len(rows)-len(pending),
        "all_candidate_current_file_hashes_verified": True, "coverage": coverage,
        "qualified_prior_scan_receipts": valid, "unqualified_prior_receipts_not_reused": invalid,
        "new_regex_scan_of_all_files_claimed": False}


def snapshot_original(preserved_root):
    manifest = read(checked(preserved_root, OLD_MANIFEST, OLD_MANIFEST_SHA))
    rows = validate_rows(manifest["files"])
    require(len(rows) == 78 and manifest["file_count_including_audit_excluding_manifest"] == 78
            and manifest["manifest_does_not_hash_itself"] is True, "REPAIRS_EXPECTED_79_OLD_PUBLIC_FILES")
    old = {row["file"]: row["sha256"] for row in rows}
    for row in rows:
        checked(preserved_root, row["file"], row["sha256"], row["bytes"])
    require(OLD_MANIFEST not in old and OLD_AUDIT in old, "REPAIRS_OLD_MANIFEST_CYCLE_OR_MISSING_AUDIT")
    old[OLD_MANIFEST] = OLD_MANIFEST_SHA
    audit = read(checked(preserved_root, OLD_AUDIT, old[OLD_AUDIT]))
    history = audit["historical_public_source_sha256"]
    require(len(history) == audit["historical_files_unchanged"] == 121 and not (set(old) & set(history)),
            "REPAIRS_EXPECTED_121_DISTINCT_HISTORY_FILES")
    for name, digest in history.items():
        require(re.fullmatch(r"[0-9a-f]{64}", digest or ""), "REPAIRS_HISTORY_SHA_INVALID")
        checked(preserved_root, name, digest)
    require(audit["original_privacy_scanner_sha256"] == SCANNER_SHA and audit["established_denylist_sha256"] == DENYLIST_SHA,
            "REPAIRS_OLD_PRIVACY_SOURCE_CHANGED")
    return {"original_79": old, "history_121": history}


def snapshot_prior_p1r(preserved_root, prior_manifest_path, expected_prior_sha):
    require(expected_prior_sha is not None and sha(prior_manifest_path) == expected_prior_sha, "REPAIRS_P1R_MANIFEST_SHA_REQUIRED")
    prior = read(prior_manifest_path)
    receipt_name, manifest_name = publication_paths("P1R")
    checked(preserved_root, manifest_name, expected_prior_sha)
    require(prior["schema"] == "semantic_repairs_publication_manifest_v1" and prior["phase"] == "P1R"
            and prior["base_git_commit"] == P1R_BASE and prior["manifest_does_not_hash_itself"] is True,
            "REPAIRS_PRIOR_P1R_MANIFEST_SCOPE_CHANGED")
    rows = validate_rows(prior["files"])
    require(manifest_name not in {r["file"] for r in rows} and receipt_name in {r["file"] for r in rows},
            "REPAIRS_PRIOR_P1R_MANIFEST_CYCLE")
    for row in rows:
        checked(preserved_root, row["file"], row["sha256"], row["bytes"])
    snapshot = {row["file"]: row["sha256"] for row in rows}
    snapshot[manifest_name] = expected_prior_sha
    return snapshot


def preservation(preserved_root, *, phase, prior_manifest_path=None, expected_prior_sha=None):
    snapshots = snapshot_original(preserved_root)
    if phase == "P2":
        require(prior_manifest_path is not None, "REPAIRS_P2_REQUIRES_PRIOR_P1R_MANIFEST")
        snapshots["prior_p1r"] = snapshot_prior_p1r(preserved_root, prior_manifest_path, expected_prior_sha)
    combined = {}
    for snapshot in snapshots.values():
        for name, digest in snapshot.items():
            require(name not in combined, "REPAIRS_OVERLAPPING_PRESERVATION_SNAPSHOTS")
            combined[name] = digest
    summary = {name: {"file_count": len(values), "files_snapshot_sha256": canonical(values)} for name, values in snapshots.items()}
    return combined, summary


def validate_base(phase, base_commit, prior_p1r_commit=None):
    require(isinstance(base_commit, str) and re.fullmatch(r"[0-9a-f]{40}", base_commit), "REPAIRS_FULL_BASE_SHA_REQUIRED")
    if phase == "P1R":
        require(base_commit == P1R_BASE and prior_p1r_commit is None, "REPAIRS_P1R_EXACT_BASE_CHANGED")
    else:
        require(phase == "P2" and isinstance(prior_p1r_commit, str) and re.fullmatch(r"[0-9a-f]{40}", prior_p1r_commit)
                and prior_p1r_commit == base_commit and base_commit != P1R_BASE,
                "REPAIRS_P2_REQUIRES_EXPLICIT_FULL_P1R_COMMIT")


def build_plan(root, preserved_root, candidates, *, phase, base_commit, prior_p1r_commit=None,
               prior_manifest_path=None, expected_prior_sha=None):
    validate_base(phase, base_commit, prior_p1r_commit)
    rows = validate_rows(candidates)
    old, summaries = preservation(preserved_root, phase=phase, prior_manifest_path=prior_manifest_path,
                                  expected_prior_sha=expected_prior_sha)
    require(not ({row["file"] for row in rows} & set(old)), "REPAIRS_OLD_PUBLISHED_FILE_CANNOT_BE_NEW_CANDIDATE")
    for row in rows:
        checked(root, safe_name(row["file"], phase), row["sha256"], row["bytes"])
    scan_exact(root, rows, re.compile(r"(?!x)x"))
    return {"schema": "semantic_repairs_publication_plan_v1", "phase": phase,
        "status": "LOCAL_PREFLIGHT_ONLY_SERVER_PRIVATE_IDENTIFIER_SCAN_REQUIRED",
        "base_git_commit": base_commit, "explicit_prior_p1r_commit": prior_p1r_commit,
        "prior_p1r_manifest_sha256": expected_prior_sha, "preserved_snapshots": summaries,
        "files": rows, "original_scanner_sha256": SCANNER_SHA, "established_denylist_sha256": DENYLIST_SHA,
        "new_model_fits": 0, "model_arrays_loaded": False, "Git_operations": 0}


def verify_plan(root, preserved_root, plan, *, prior_manifest_path=None):
    require(plan["schema"] == "semantic_repairs_publication_plan_v1" and
            plan["status"] == "LOCAL_PREFLIGHT_ONLY_SERVER_PRIVATE_IDENTIFIER_SCAN_REQUIRED" and
            plan["original_scanner_sha256"] == SCANNER_SHA and plan["established_denylist_sha256"] == DENYLIST_SHA,
            "REPAIRS_PLAN_SCHEMA_OR_PRIVACY_BINDING_CHANGED")
    expected = build_plan(root, preserved_root, plan["files"], phase=plan["phase"], base_commit=plan["base_git_commit"],
        prior_p1r_commit=plan["explicit_prior_p1r_commit"], prior_manifest_path=prior_manifest_path,
        expected_prior_sha=plan["prior_p1r_manifest_sha256"])
    require(plan == expected, "REPAIRS_PLAN_METADATA_CHANGED")
    return expected


def write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)


def zip_entry(name, raw):
    info = zipfile.ZipInfo(name, date_time=(2026, 10, 31, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info, raw


def stage(root, preserved_root, plan_path, expected_plan_sha, zip_path, *, prior_manifest_path=None):
    require(sha(plan_path) == expected_plan_sha, "REPAIRS_PLAN_FILE_SHA_CHANGED")
    plan = verify_plan(root, preserved_root, read(plan_path), prior_manifest_path=prior_manifest_path)
    require(not Path(zip_path).exists(), "REPAIRS_EXISTING_STAGE_ZIP_PRESERVED")
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in plan["files"]:
            raw = checked(root, row["file"], row["sha256"], row["bytes"]).read_bytes()
            archive.writestr(*zip_entry(row["file"], raw))
        archive.writestr(*zip_entry(PLAN_ZIP_MEMBER, Path(plan_path).read_bytes()))
    receipt = {"schema": "semantic_repairs_metadata_stage_v1", "status": "LOCAL_PREFLIGHT_NOT_SERVER_FINAL_SCAN",
        "phase": plan["phase"], "base_git_commit": plan["base_git_commit"],
        "plan_sha256": expected_plan_sha, "archive_sha256": sha(zip_path),
        "candidate_files": len(plan["files"]), "candidate_bytes": sum(r["bytes"] for r in plan["files"]),
        "zip_members": len(plan["files"]) + 1, "each_payload_has_explicit_file_bytes_sha256": True,
        "established_private_identifier_scan_performed": False, "Git_operations": 0}
    # Detect source changes during zip construction without touching Git.
    verify_plan(root, preserved_root, plan, prior_manifest_path=prior_manifest_path)
    return receipt


def unpack_stage(archive_path, expected_archive_sha, plan_path, expected_plan_sha, destination):
    require(sha(archive_path) == expected_archive_sha and sha(plan_path) == expected_plan_sha, "REPAIRS_STAGE_OR_PLAN_SHA_CHANGED")
    plan = read(plan_path)
    rows = validate_rows(plan["files"])
    expected = {row["file"]: row for row in rows}
    expected[PLAN_ZIP_MEMBER] = {"sha256": expected_plan_sha, "bytes": Path(plan_path).stat().st_size}
    require(not Path(destination).exists(), "REPAIRS_EXTRACTION_MUST_USE_NEW_DIRECTORY")
    content = {}
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        require(not archive.comment, "REPAIRS_UNSCANNED_ZIP_COMMENT_BLOCKED")
        require(len(entries) == len(expected) and len(entries) == len({i.filename for i in entries}) and
                {i.filename for i in entries} == set(expected), "REPAIRS_ZIP_EXTRA_MISSING_OR_DUPLICATE_MEMBER")
        for info in entries:
            safe_relative(info.filename)
            mode = info.external_attr >> 16
            require(not info.is_dir() and not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in (0, stat.S_IFREG)
                    and not (info.flag_bits & 1) and not info.comment and not info.extra,
                    "REPAIRS_ZIP_SYMLINK_BINARY_OR_ENCRYPTED_MEMBER")
            row = expected[info.filename]
            require(info.file_size == row["bytes"] and 0 < info.file_size <= MAX_BYTES, "REPAIRS_ZIP_MEMBER_SIZE_CHANGED")
            raw = archive.read(info)
            require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "REPAIRS_ZIP_PAYLOAD_HASH_CHANGED")
            raw.decode("utf-8-sig")
            content[info.filename] = raw
    # All paths and bytes are validated before creating any destination file.
    destination = Path(destination)
    destination.mkdir(parents=True)
    for name, raw in content.items():
        if name != PLAN_ZIP_MEMBER:
            write_new(destination.joinpath(*PurePosixPath(name).parts), raw)
    return plan


def final_scan(archive_path, expected_archive_sha, plan_path, expected_plan_sha, preserved_root, denylist_path,
               output_newdir, *, prior_manifest_path=None):
    require(sha(original.__file__) == SCANNER_SHA, "REPAIRS_ORIGINAL_SCANNER_SOURCE_CHANGED")
    require(sha(denylist_path) == DENYLIST_SHA, "REPAIRS_ESTABLISHED_PRIVATE_DENYLIST_CHANGED")
    # Exactly the original validator and identifier-pattern semantics, no new registry.
    pattern = original.identifier_pattern(original.validate_denylist(read(denylist_path)))
    out = Path(output_newdir)
    require(not out.exists(), "REPAIRS_EXISTING_FINAL_SCAN_OUTPUT_PRESERVED")
    payload = out / "public"
    plan = unpack_stage(archive_path, expected_archive_sha, plan_path, expected_plan_sha, payload)
    verify_plan(payload, preserved_root, plan, prior_manifest_path=prior_manifest_path)
    rows, coverage = scan_with_exact_prior_evidence(payload, plan["files"], pattern)
    receipt_name, manifest_name = publication_paths(plan["phase"])
    receipt = {"schema": "semantic_repairs_server_scan_receipt_v1", "status": "PASS_EXACT_ESTABLISHED_PRIVATE_IDENTIFIER_SCAN",
        "phase": plan["phase"], "base_git_commit": plan["base_git_commit"],
        "explicit_prior_p1r_commit": plan["explicit_prior_p1r_commit"],
        "plan_sha256": expected_plan_sha, "stage_archive_sha256": expected_archive_sha,
        "candidate_files": len(rows), "candidate_bytes": sum(r["bytes"] for r in rows),
        "candidate_rows_sha256": canonical(rows), "preserved_snapshots": plan["preserved_snapshots"],
        "scan_execution": coverage,
        "original_scanner_sha256": SCANNER_SHA, "established_private_denylist_sha256": DENYLIST_SHA,
        "all_text_fields_covered_by_same_original_rules_and_actual_private_identifier_pattern": True,
        "denylist_built_exported_or_modified": False, "patient_mapping_created": False,
        "new_model_fits": 0, "model_or_patient_array_deserializations": 0, "Git_operations": 0,
        "scientific_completion_or_positive_result_inferred_from_privacy_scan": False,
        "manifest_hash_is_not_embedded_here": True}
    receipt_raw = encode(receipt)
    receipt_row = {"file": receipt_name, "bytes": len(receipt_raw), "sha256": hashlib.sha256(receipt_raw).hexdigest()}
    manifest = {"schema": "semantic_repairs_publication_manifest_v1", "phase": plan["phase"],
        "status": receipt["status"], "base_git_commit": plan["base_git_commit"],
        "explicit_prior_p1r_commit": plan["explicit_prior_p1r_commit"],
        "source_scan_receipt_sha256": receipt_row["sha256"], "plan_sha256": expected_plan_sha,
        "preserved_snapshots": plan["preserved_snapshots"],
        "files": sorted(rows + [receipt_row], key=lambda row: row["file"]),
        "file_count_including_scan_receipt_excluding_manifest": len(rows) + 1,
        "manifest_does_not_hash_itself": True, "Git_operations": 0}
    manifest_raw = encode(manifest)
    manifest_row = {"file": manifest_name, "bytes": len(manifest_raw), "sha256": hashlib.sha256(manifest_raw).hexdigest()}
    for row, raw in ((receipt_row, receipt_raw), (manifest_row, manifest_raw)):
        write_new(payload / row["file"], raw)
    generated = scan_exact(payload, sorted([receipt_row, manifest_row], key=lambda row: row["file"]), pattern)
    # Recheck preserved files after the actual scan. Never regenerate old reports.
    old, after = preservation(preserved_root, phase=plan["phase"], prior_manifest_path=prior_manifest_path,
                              expected_prior_sha=plan["prior_p1r_manifest_sha256"])
    require(after == plan["preserved_snapshots"], "REPAIRS_PRESERVED_SNAPSHOT_CHANGED_DURING_SCAN")
    all_rows = sorted(rows + generated, key=lambda row: row["file"])
    for row in all_rows:
        checked(payload, row["file"], row["sha256"], row["bytes"])
    final_archive = out / "PUBLICATION.zip"
    with zipfile.ZipFile(final_archive, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in all_rows:
            archive.writestr(*zip_entry(row["file"], checked(payload, row["file"], row["sha256"], row["bytes"]).read_bytes()))
    binding = {"schema": "semantic_repairs_final_binding_v1", "status": "PASS_METADATA_ONLY_NO_GIT_WRITE",
        "phase": plan["phase"], "base_git_commit": plan["base_git_commit"],
        "publication_manifest_sha256": manifest_row["sha256"], "scan_receipt_sha256": receipt_row["sha256"],
        "plan_sha256": expected_plan_sha, "publication_archive_sha256": sha(final_archive),
        "published_files_including_manifest_and_receipt": len(all_rows),
        "published_bytes": sum(row["bytes"] for row in all_rows), "binding_is_separate_not_hashed_by_manifest": True,
        "candidate_scanned_now_count": coverage["scanned_now_count"],
        "candidate_reused_exact_prior_scan_count": coverage["reused_exact_prior_scan_count"],
        "generated_receipt_and_manifest_scanned_now_count": 2,
        "private_identifier_list_exported": False, "Git_operations": 0}
    write_new(out / "FINAL_BINDING.json", encode(binding))
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "stage", "final-scan"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--preserved-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("P1R", "P2"))
    parser.add_argument("--base-commit")
    parser.add_argument("--prior-p1r-commit")
    parser.add_argument("--prior-p1r-manifest", type=Path)
    parser.add_argument("--expected-prior-p1r-manifest-sha256")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--denylist", type=Path)
    parser.add_argument("--output-newdir", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        require(args.phase is not None and args.base_commit is not None and args.candidates is not None,
                "REPAIRS_PLAN_ARGUMENTS_REQUIRED")
        candidates = read(args.candidates)
        require(set(candidates) == {"files"}, "REPAIRS_CANDIDATE_WRAPPER_MUST_BE_EXACT_FILES")
        plan = build_plan(args.root, args.preserved_root, candidates["files"], phase=args.phase,
            base_commit=args.base_commit, prior_p1r_commit=args.prior_p1r_commit,
            prior_manifest_path=args.prior_p1r_manifest, expected_prior_sha=args.expected_prior_p1r_manifest_sha256)
        write_new(args.plan, encode(plan))
        result = {"status": plan["status"], "phase": args.phase, "plan_sha256": sha(args.plan), "candidate_files": len(plan["files"])}
    elif args.command == "stage":
        require(args.expected_plan_sha256 is not None and args.archive is not None, "REPAIRS_STAGE_ARGUMENTS_REQUIRED")
        result = stage(args.root, args.preserved_root, args.plan, args.expected_plan_sha256, args.archive,
                       prior_manifest_path=args.prior_p1r_manifest)
    else:
        require(all(x is not None for x in (args.archive, args.expected_archive_sha256, args.expected_plan_sha256,
                args.denylist, args.output_newdir)), "REPAIRS_FINAL_SCAN_ARGUMENTS_REQUIRED")
        result = final_scan(args.archive, args.expected_archive_sha256, args.plan, args.expected_plan_sha256,
            args.preserved_root, args.denylist, args.output_newdir, prior_manifest_path=args.prior_p1r_manifest)
    # Hashes/counts only: no private paths, registries, payload text or Git mutations.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
