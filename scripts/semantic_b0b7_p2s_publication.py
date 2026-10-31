"""Append-only P2S text publication, with the unchanged original privacy scanner.

No Git writes, external messages, model loading or scientific fitting occur.
The fixed published parent is checked through read-only Git objects, never a
125-MB checkout/export. ZIP compression is transport only: every UTF-8 member,
including controls and generated metadata, is scanned under original rules.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_publication as original

SCANNER_SHA = "4c70929d57459a9e97774e6319bd455deedc29fbcedf077131c918c53db60e98"
DENYLIST_SHA = "d73b70939a38277f8f9f693e9958afcc9435893cd05f5c1d6a6d08a100ffe867"
BASE_COMMIT = "dab64dff52b1170f9a11ceb0833d9d49424fa63a"
BASE_TREE = "4156773ec3a84fee74b32e797914c0325ef2c1ec"
BRANCH = "research/semantic-b0b7-p2-sufficiency"
BASELINE_SHA = "dc2c84d17e51d8a4411f2aa70b0f3a5feb3de44f5eaf3b0aec8ab8e8d24e4f63"
STUDY_SHA = "3babf37b356e7d27f3cb5da0985b94d0b8d983edadc333eb8333b627539b5c4b"
EXECUTION_SHA = "bc2df57fc7593acc92d652978fe288295c3131e3bfbedeca2cbeed82b92017a0"
HELPER = "scripts/semantic_b0b7_p2s_publication.py"
TEST = "tests/test_semantic_b0b7_p2s_publication.py"
VERIFIER = "scripts/verify_semantic_b0b7_p2s.py"
VERIFIER_TEST = "tests/test_verify_semantic_b0b7_p2s.py"
SCAN_RECEIPT = "reports/semantic_b0b7_p2s/PUBLICATION_SCAN_RECEIPT.json"
MANIFEST = "reports/semantic_b0b7_p2s/PUBLICATION_MANIFEST.json"
PLAN_MEMBER = "_publication/PLAN.json"
BASELINE_MEMBER = "_publication/PARENT472_BASELINE.json"
MAX_BYTES = 2 * 1024 ** 2
MAX_FILES = 998
PROOF_KEYS = frozenset(("study_lock", "execution_lock", "readout_lock", "report_index", "independent", "completion"))
require, sha = original.require, original.sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def encode(value):
    original.validate_public_json(value)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(len(raw) <= MAX_BYTES, "P2S_OVERSIZED_METADATA")
    return raw


def safe_relative(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name and "\x00" not in name,
            "P2S_UNSAFE_RELATIVE_PATH")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == name, "P2S_UNSAFE_RELATIVE_PATH")
    return name


def safe_name(name):
    safe_relative(name)
    require(name not in (SCAN_RECEIPT, MANIFEST), "P2S_GENERATED_FILE_NOT_A_CANDIDATE")
    patterns = (r"src/semantic_b0b7/p2s_[A-Za-z0-9_]+\.py",
        r"scripts/semantic_b0b7_p2s[A-Za-z0-9_]*\.py", r"tests/test_semantic_b0b7_p2s[A-Za-z0-9_]*\.py",
        r"configs/semantic_b0b7/p2s_[A-Za-z0-9_]+\.json",
        r"reports/semantic_b0b7_p2s/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.(?:json|csv|md)")
    require(name in (VERIFIER, VERIFIER_TEST) or any(re.fullmatch(p, name) for p in patterns),
            "P2S_PATH_OUTSIDE_NEW_NAMESPACE")
    return name


def checked(root, name, expected=None, size=None):
    safe_relative(name)
    root = Path(root).resolve(); p = root.joinpath(*PurePosixPath(name).parts)
    require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root), "P2S_MISSING_OR_ESCAPING_FILE")
    require(not any(q.is_symlink() for q in p.parents if q.is_relative_to(root)), "P2S_SYMLINK_PARENT")
    require(expected is None or sha(p) == expected, "P2S_FILE_SHA_CHANGED")
    require(size is None or p.stat().st_size == size, "P2S_FILE_SIZE_CHANGED")
    return p


def validate_rows(rows):
    require(isinstance(rows, list) and 0 < len(rows) <= MAX_FILES and
        len(rows) == len({r["file"] for r in rows}), "P2S_EXACT_UNIQUE_NONEMPTY_FILES_REQUIRED")
    for r in rows:
        require(set(r) == {"file", "bytes", "sha256"} and type(r["bytes"]) is int and
            0 < r["bytes"] <= MAX_BYTES and re.fullmatch(r"[0-9a-f]{64}", r["sha256"] or ""), "P2S_BAD_CANDIDATE_ROW")
        safe_name(r["file"])
    return sorted(rows, key=lambda r: r["file"])


def baseline(path, expected_sha):
    require(expected_sha == BASELINE_SHA and sha(path) == expected_sha, "P2S_FIXED_PARENT472_BASELINE_CHANGED")
    b = read(path)
    require(b["schema"] == "semantic_b0b7_p2s_published_parent472_baseline_v1" and
        b["commit"] == BASE_COMMIT and b["tree"] == BASE_TREE and b["file_count"] == 472 and
        b["all_local_bytes_and_Git_blob_SHA1_verified"] is True and len(b["files"]) == 472,
        "P2S_FIXED_PARENT472_SCOPE_CHANGED")
    require(len({r["file"] for r in b["files"]}) == 472, "P2S_DUPLICATE_PARENT_FILE")
    for r in b["files"]:
        safe_relative(r["file"])
        require(set(r) == {"file", "bytes", "sha256", "git_blob_sha1"} and type(r["bytes"]) is int and r["bytes"] > 0 and
            re.fullmatch(r"[0-9a-f]{64}", r["sha256"]) and re.fullmatch(r"[0-9a-f]{40}", r["git_blob_sha1"]),
            "P2S_BAD_PARENT_METADATA")
    require(sum(r["bytes"] for r in b["files"]) == b["total_bytes"], "P2S_PARENT_BYTE_TOTAL_CHANGED")
    return b


def verify_local_parent(root, b):
    for r in b["files"]:
        raw = checked(root, r["file"], r["sha256"], r["bytes"]).read_bytes()
        require(hashlib.sha1(("blob " + str(len(raw)) + "\0").encode() + raw).hexdigest() == r["git_blob_sha1"],
                "P2S_PARENT_GIT_BLOB_SHA1_CHANGED")
    return {"protected_files": 472, "protected_bytes": b["total_bytes"],
            "baseline_rows_sha256": canonical(b["files"]), "all_local_byte_hashes_verified": True}


def git_command(repo, *args):
    # Only callers below supply fixed read-only Git verbs; never shell strings.
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(["git", "-C", str(repo), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True, env=env).stdout


def git_control_hashes(repo):
    result = {}
    for name in ("HEAD", "index"):
        raw = git_command(repo, "rev-parse", "--git-path", name).decode().strip()
        p = Path(raw); p = p if p.is_absolute() else Path(repo) / p
        result[name] = sha(p) if p.is_file() else None
    return result


def verify_git_parent(repo, b):
    """Stream immutable blobs; does not checkout, export, stage, or write Git."""
    before = git_control_hashes(repo)
    require(git_command(repo, "cat-file", "-t", BASE_COMMIT).strip() == b"commit" and
        git_command(repo, "rev-parse", BASE_COMMIT + "^{tree}").decode().strip() == BASE_TREE,
        "P2S_FIXED_GIT_PARENT_OR_TREE_MISSING")
    entries = {}
    for raw in git_command(repo, "ls-tree", "-rz", "--full-tree", BASE_COMMIT).split(b"\0"):
        if not raw:
            continue
        header, filename = raw.split(b"\t", 1); mode, kind, oid = header.decode().split()
        entries[filename.decode("utf-8")] = mode, kind, oid
    for r in b["files"]:
        require(entries.get(r["file"]) in (("100644", "blob", r["git_blob_sha1"]),
            ("100755", "blob", r["git_blob_sha1"])), "P2S_PARENT_TREE_FILE_BINDING_CHANGED")
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0"}
    p = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        for r in b["files"]:
            p.stdin.write((r["git_blob_sha1"] + "\n").encode()); p.stdin.flush()
            header = p.stdout.readline().decode().strip().split()
            require(header == [r["git_blob_sha1"], "blob", str(r["bytes"])], "P2S_PARENT_BLOB_HEADER_CHANGED")
            remaining = r["bytes"]; h = hashlib.sha256(); g = hashlib.sha1(("blob " + str(remaining) + "\0").encode())
            while remaining:
                data = p.stdout.read(min(remaining, 1024 ** 2))
                require(bool(data), "P2S_TRUNCATED_GIT_BLOB")
                remaining -= len(data); h.update(data); g.update(data)
            require(p.stdout.read(1) == b"\n" and h.hexdigest() == r["sha256"] and
                    g.hexdigest() == r["git_blob_sha1"], "P2S_PARENT_GIT_OBJECT_CONTENT_CHANGED")
        p.stdin.close()
        require(p.wait() == 0, "P2S_GIT_OBJECT_READER_FAILED")
    finally:
        if p.stdin and not p.stdin.closed:
            p.stdin.close()
        # Close a possibly unread malformed blob before wait: otherwise a
        # failed header check could leave cat-file blocked on a full pipe.
        p.stdout.close(); p.stderr.close()
        if p.poll() is None:
            p.wait()
    after = git_control_hashes(repo)
    require(before == after, "P2S_GIT_HEAD_OR_INDEX_CHANGED")
    return {"base_git_commit": BASE_COMMIT, "base_git_tree": BASE_TREE, "protected_files_verified": 472,
        "protected_blob_bytes_streamed": b["total_bytes"], "all_SHA256_and_git_blob_SHA1_exact": True,
        "HEAD_and_index_byte_hashes_unchanged": before == after, "Git_write_operations": 0,
        "checkout_or_BASE_export_performed": False}


@contextmanager
def scanner_allowlist(rows):
    require(sha(original.__file__) == SCANNER_SHA, "P2S_ORIGINAL_SCANNER_SOURCE_CHANGED")
    saved = original.ALLOWLIST
    original.ALLOWLIST = frozenset(r["file"] for r in rows)
    try:
        yield
    finally:
        original.ALLOWLIST = saved


def scan_exact(root, rows, pattern):
    rows = sorted(rows, key=lambda r: r["file"])
    with scanner_allowlist(rows):
        actual = [original.scan_file(Path(root).resolve(), r["file"], pattern) for r in rows]
    require(actual == rows, "P2S_EXACT_SCANNED_BYTES_CHANGED")
    return actual


def projected_report_index(root, public_index, binding, indexed):
    """Independently reconstruct only the three known 154-column CSV sources.

    The unpublishable source CSVs are never written. All cell strings originate
    in explicitly listed, normally scanned public column parts. The source
    INDEX remains byte-identical provenance, not a list of physical payloads.
    """
    if public_index.get("schema") != "semantic_p2s_lossless_publication_report_index_v1":
        return public_index, None
    parent = PurePosixPath(binding["file"]).parent
    require(public_index["source_report_index_file"] == "SOURCE_REPORT_INDEX.json", "P2S_PROJECTED_SOURCE_INDEX_NAME_CHANGED")
    source_name = str(parent / public_index["source_report_index_file"])
    source_row = indexed.get(source_name)
    require(source_row is not None and source_row["sha256"] == public_index["source_report_index_sha256"],
            "P2S_PROJECTED_SOURCE_INDEX_NOT_EXACT_CANDIDATE")
    source = read(checked(root, source_name, source_row["sha256"], source_row["bytes"]))
    require(source["schema"] == "semantic_p2s_receipt_report_index_v1" and
        all(public_index.get(k) == v for k, v in source.items() if k not in ("schema", "files")),
        "P2S_PROJECTION_CHANGED_ORIGINAL_REPORT_CONTROL_FIELDS")
    original = {r["file"]: r for r in source["files"]}
    physical = {r["file"]: r for r in public_index["files"]}
    require(len(original) == len(source["files"]) and len(physical) == len(public_index["files"]),
            "P2S_DUPLICATE_PROJECTED_OR_SOURCE_FILE")
    expected = {name for name in original if re.fullmatch(r"04_all_2376_readout_cells_[0-9]{2}\.csv", name)}
    projections = public_index["column_projection"]
    require(len(expected) == len(projections) == 3 and len({r["source_file"] for r in projections}) == 3 and
        {r["source_file"] for r in projections} == expected, "P2S_ONLY_THE_THREE_KNOWN_CSV_SOURCES_MAY_BE_VIRTUAL")
    for name, row in original.items():
        if name not in expected:
            require(physical.get(name) == row, "P2S_NONPROJECTED_ORIGINAL_REPORT_FILE_MISSING")
    for field in ("projection_source", "projection_test"):
        row = public_index[field]
        require(set(row) == {"file", "sha256"} and indexed.get(row["file"], {}).get("sha256") == row["sha256"],
                "P2S_PROJECTION_SOURCE_OR_TEST_NOT_BOUND")
        prefix = "scripts/semantic_b0b7_p2s" if field == "projection_source" else "tests/test_semantic_b0b7_p2s"
        require(re.fullmatch(re.escape(prefix) + r"[A-Za-z0-9_]*\.py", row["file"]), "P2S_PROJECTION_SOURCE_ROLE_CHANGED")
    summaries = []
    for projection in projections:
        source_file = projection["source_file"]; source_binding = original[source_file]
        require(projection["source_bytes"] == source_binding["bytes"] and projection["source_sha256"] == source_binding["sha256"] and
            str(parent / source_file) not in indexed and source_file not in physical,
            "P2S_PROJECTED_SOURCE_HASH_CHANGED_OR_VIRTUAL_SOURCE_PUBLISHED")
        headers = projection["source_headers"]; count = projection["source_rows"]
        require(isinstance(headers, list) and all(isinstance(v, str) for v in headers) and len(headers) == len(set(headers)) == 154 and
            projection["ordinal_field"] == "aggregate_row_ordinal" and projection["ordinal_start"] == 0 and
            "aggregate_row_ordinal" not in headers and type(count) is int and 0 < count < 20000,
            "P2S_PROJECTED_ORIGINAL_HEADER_OR_ORDINAL_CHANGED")
        values = [[None] * len(headers) for _ in range(count)]; positions = {name: i for i, name in enumerate(headers)}
        parts = projection["parts"]
        require(parts and len(parts) <= 1000 and len(parts) == len({r["file"] for r in parts}), "P2S_DUPLICATE_OR_MISSING_COLUMN_PART")
        for part in parts:
            name = str(parent / safe_relative(part["file"]))
            row = {"file": name, "bytes": part["bytes"], "sha256": part["sha256"]}
            require(indexed.get(name) == row and physical.get(part["file"]) == {**row, "file": part["file"]},
                    "P2S_COLUMN_PART_NOT_EXACT_PHYSICAL_PAYLOAD")
            columns = part["columns"]; start, stop = part["row_start"], part["row_stop_exclusive"]
            require(isinstance(columns, list) and all(isinstance(c, str) for c in columns) and 0 < len(columns) <= 149 and len(columns) == len(set(columns)) and
                all(c in positions for c in columns) and sorted(columns, key=positions.get) == columns and
                type(start) is type(stop) is int and 0 <= start < stop <= count, "P2S_COLUMN_PART_HEADER_OR_RANGE_CHANGED")
            raw = checked(root, name, row["sha256"], row["bytes"]).read_bytes()
            parsed = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
            require(parsed and parsed[0] == ["aggregate_row_ordinal"] + columns and len(parsed) == stop - start + 1,
                    "P2S_COLUMN_PART_CSV_HEADER_OR_ROWS_CHANGED")
            for ordinal, line in enumerate(parsed[1:], start):
                require(len(line) == len(columns) + 1 and line[0] == str(ordinal), "P2S_COLUMN_PART_ORDINAL_CHANGED")
                for column, value in zip(columns, line[1:]):
                    pos = positions[column]
                    require(values[ordinal][pos] is None, "P2S_DUPLICATE_RECONSTRUCTION_CELL")
                    values[ordinal][pos] = value
        require(all(all(v is not None for v in row) for row in values), "P2S_MISSING_RECONSTRUCTION_CELL")
        output = io.StringIO(newline=""); writer = csv.writer(output, lineterminator="\n")
        writer.writerow(headers); writer.writerows(values); raw = output.getvalue().encode("utf-8")
        require(len(raw) == source_binding["bytes"] and hashlib.sha256(raw).hexdigest() == source_binding["sha256"],
                "P2S_ORIGINAL_CSV_BYTE_RECONSTRUCTION_MISMATCH")
        summaries.append({"file": source_file, "bytes": len(raw), "sha256": source_binding["sha256"],
                          "columns": 154, "aggregate_rows": count, "public_parts": len(parts)})
    return source, {"source_report_index_sha256": source_row["sha256"], "virtual_source_files": 3,
        "all_source_CSV_bytes_independently_reconstructed_exact": True, "sources": summaries,
        "original_csv_files_written_or_published": False,
        "only_declared_projection_source_missing_references_exempted": True}


def proof_closure(root, rows, proofs, b):
    require(set(proofs) == PROOF_KEYS, "P2S_COMPLETE_METADATA_PROOFS_REQUIRED")
    indexed = {r["file"]: r for r in rows}; old = {r["file"]: r for r in b["files"]}
    require(not set(indexed) & set(old), "P2S_OLD472_CANNOT_BE_OVERWRITTEN")
    require({HELPER, TEST, VERIFIER, VERIFIER_TEST} <= set(indexed), "P2S_PUBLICATION_AND_VERIFIER_SOURCES_REQUIRED")
    require(indexed[HELPER]["sha256"] == sha(__file__), "P2S_RUNNING_PUBLICATION_HELPER_NOT_THE_CANDIDATE")
    data = {}
    for label, binding in proofs.items():
        require(binding == indexed.get(binding["file"]) and binding["file"].endswith(".json"), "P2S_PROOF_NOT_EXACT_CANDIDATE")
        data[label] = read(checked(root, **{"name": binding["file"], "expected": binding["sha256"], "size": binding["bytes"]}))
    s, e, d, public_index, audit, completion = (data[k] for k in ("study_lock", "execution_lock", "readout_lock", "report_index", "independent", "completion"))
    index, projection_proof = projected_report_index(root, public_index, proofs["report_index"], indexed)
    require(proofs["study_lock"]["sha256"] == STUDY_SHA and proofs["execution_lock"]["sha256"] == EXECUTION_SHA and
        s["schema"] == "semantic_b0b7_p2s_lock_v1" and s["status"] == "LOCKED_BEFORE_P2S_OPTIMIZER" and
        s["parent_git_commit"] == BASE_COMMIT and e["parent_pretraining_lock_sha256"] == STUDY_SHA,
        "P2S_STUDY_EXECUTION_PROVENANCE_CHANGED")
    require(e["schema"] == "semantic_b0b7_p2s_execution_lock_v1" and e["status"] == "LOCKED_BEFORE_P2S_SCHEDULER_LAUNCH" and
        d["schema"] == "semantic_b0b7_p2s_readout_lock_v1" and d["status"] == "LOCKED_BEFORE_P2S_DSS_FITS" and
        d["parent_study_lock_sha256"] == STUDY_SHA and d["expected_feature_states"] == 72 and
        d["expected_Cox_fits"] == 2376 and d["primary_ridge_lambda"] == 1. and
        d["ridge_candidates"] == [.1, 1., 10.] and d["outer_evaluation"] is False, "P2S_READOUT_PROVENANCE_CHANGED")
    require(index["schema"] == "semantic_p2s_receipt_report_index_v1" and
        index["status"] in ("COMPLETE_FIXED_GRID", "COMPLETE_FIXED_GRID_WITH_EXPLICIT_NA") and
        index["pretraining_lock_sha256"] == STUDY_SHA and index["readout_lock_sha256"] == proofs["readout_lock"]["sha256"] and
        index["outer_results_loaded"] is False and index["new_fits_or_forwards"] == 0 and index["self_hash_cycle"] is False,
        "P2S_REPORT_COMPLETION_PROOF_CHANGED")
    require(audit["schema"] == "semantic_p2s_independent_postsave_Z_coefficient_audit_v1" and
        audit["status"] in ("PASS_WITH_EXPLICIT_RANK_NA", "PASS_POSTSAVE_Z_COEFFICIENT_METRIC_AND_OBJECTIVE_AUDIT") and
        audit["readout_jobs"] == 72 and audit["registered_head_positions"] == 2376 and
        audit["study_lock_sha256"] == STUDY_SHA and audit["readout_lock_sha256"] == proofs["readout_lock"]["sha256"] and
        audit["manifest_sha256"] == index["readout_manifest_sha256"] and
        audit["report_arithmetic"]["status"] == "PASS_264_CONFIGURATION_AGGREGATE_ARITHMETIC" and
        audit["report_arithmetic"]["configuration_summaries_checked"] == 264 and
        audit["model_predict_calls"] == audit["producer_scorer_calls"] == audit["new_optimizer_fits"] == audit["new_neural_forwards"] == 0 and
        audit["outer_predictions_read"] is False and audit["scientific_mechanism_decision_assigned"] is False,
        "P2S_INDEPENDENT_AUDIT_PROOF_CHANGED")
    require(completion["schema"] == "semantic_p2s_reporting_completion_v1" and
        completion["scientific_mechanism_decision"] == "NOT_AUTOMATICALLY_ASSIGNED" and
        completion["epoch_pair_weight_or_ridge_winner_selected"] is False and
        completion["CIs_p_values_or_bootstrap_computed"] is False, "P2S_COMPLETION_IS_NOT_AN_AUTOMATIC_SCIENTIFIC_WINNER")
    costs = completion["costs"]
    require(costs["new_optimizer_updates"] == 21600 and costs["actual_checkpoint_replays"] == 72 and costs["new_snapshots"] == 54 and
        costs["reused_original400_snapshots"] == 18 and costs["actual_Cox_fits"] == audit["fitted_heads_audited"] and
        costs["explicit_NA_head_cells"] == audit["explicit_rank_NA"] and
        audit["fitted_heads_audited"] + audit["explicit_rank_NA"] == 2376, "P2S_ACTUAL_COST_OR_NA_PROOF_CHANGED")
    def bound(name, digest):
        row = indexed.get(name, old.get(name))
        require(row is not None and row["sha256"] == digest, "P2S_REPRODUCTION_SOURCE_CLOSURE_MISSING:" + name)
    for sources in (s["frozen_sources"], e["execution_sources"], d["readout_sources"], index["frozen_scientific_sources"], index["readout_source_closure"]):
        for name, digest in sources.items():
            bound(name, digest)
    bound("scripts/semantic_b0b7_p2s_report.py", index["source_sha256"])
    bound(VERIFIER, audit["verifier_source_sha256"])
    bound("scripts/verify_semantic_b0b7_p2.py", audit["independent_helper_source_sha256"])
    for label, value in (("report_index", public_index), ("independent", audit)):
        parent = PurePosixPath(proofs[label]["file"]).parent
        files = value["files"]
        require(len(files) == len({r["file"] for r in files}), "P2S_DUPLICATE_REPORT_PART")
        for r in files:
            name = str(parent / safe_relative(r["file"]))
            require(indexed.get(name) == {"file": name, "bytes": r["bytes"], "sha256": r["sha256"]},
                    "P2S_REPORT_OR_AUDIT_PART_MISSING:" + name)
    return {"metadata_proof_file_sha256": {k: v["sha256"] for k, v in proofs.items()},
        "study_fixed_updates": 21600, "fixed_checkpoint_replays": 72, "registered_readout_positions": 2376,
        "actual_Cox_fits": audit["fitted_heads_audited"], "explicit_rank_NA": audit["explicit_rank_NA"],
        "source_report_and_independent_part_closure_complete": True,
        "lossless_column_projection": projection_proof,
        "privacy_scan_infers_scientific_mechanism": False}


def validate_plan(root, plan, b):
    require(set(plan) == {"schema", "status", "base_git_commit", "base_git_tree", "branch", "parent472_baseline_sha256",
        "parent472_rows_sha256", "preservation", "files", "proofs", "metadata_proof_closure", "original_scanner_sha256",
        "established_denylist_sha256", "private_identifier_scan_performed", "Git_write_operations", "new_fits_or_forwards",
        "publication_helper_source_sha256"}, "P2S_PLAN_FIELDS_CHANGED")
    require(plan["schema"] == "semantic_b0b7_p2s_publication_plan_v1" and
        plan["status"] == "LOCAL_PREFLIGHT_ONLY_ACTUAL_PRIVATE_SCAN_REQUIRED" and plan["base_git_commit"] == BASE_COMMIT and
        plan["base_git_tree"] == BASE_TREE and plan["branch"] == BRANCH and plan["parent472_baseline_sha256"] == BASELINE_SHA and
        plan["original_scanner_sha256"] == SCANNER_SHA and plan["established_denylist_sha256"] == DENYLIST_SHA,
        "P2S_PLAN_SCOPE_CHANGED")
    require(plan["private_identifier_scan_performed"] is False and plan["Git_write_operations"] == plan["new_fits_or_forwards"] == 0 and
        plan["publication_helper_source_sha256"] == sha(__file__) and
        plan["preservation"] == {"protected_files": 472, "protected_bytes": b["total_bytes"],
            "baseline_rows_sha256": canonical(b["files"]), "all_local_byte_hashes_verified": True}, "P2S_PLAN_PROVENANCE_CHANGED")
    rows = validate_rows(plan["files"])
    for r in rows:
        checked(root, r["file"], r["sha256"], r["bytes"])
    proof = proof_closure(root, rows, plan["proofs"], b)
    require(proof == plan["metadata_proof_closure"] and plan["parent472_rows_sha256"] == canonical(b["files"]),
            "P2S_METADATA_PROOF_CHANGED")
    return rows


def build_plan(root, preserved_root, baseline_path, expected_baseline_sha, candidates, proofs):
    b = baseline(baseline_path, expected_baseline_sha)
    preservation = verify_local_parent(preserved_root, b)
    rows = validate_rows(candidates)
    proof = proof_closure(root, rows, proofs, b)
    scan_exact(root, rows, re.compile(r"(?!x)x"))
    return {"schema": "semantic_b0b7_p2s_publication_plan_v1", "status": "LOCAL_PREFLIGHT_ONLY_ACTUAL_PRIVATE_SCAN_REQUIRED",
        "base_git_commit": BASE_COMMIT, "base_git_tree": BASE_TREE, "branch": BRANCH,
        "parent472_baseline_sha256": BASELINE_SHA, "parent472_rows_sha256": canonical(b["files"]),
        "preservation": preservation, "files": rows, "proofs": proofs, "metadata_proof_closure": proof,
        "original_scanner_sha256": SCANNER_SHA, "established_denylist_sha256": DENYLIST_SHA,
        "private_identifier_scan_performed": False, "Git_write_operations": 0, "new_fits_or_forwards": 0,
        "publication_helper_source_sha256": sha(__file__)}


def write_new(path, raw):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(raw)


def zip_entry(name, raw):
    z = zipfile.ZipInfo(name, (2026, 10, 31, 0, 0, 0)); z.create_system = 3
    z.external_attr = (stat.S_IFREG | 0o644) << 16; z.compress_type = zipfile.ZIP_DEFLATED
    return z, raw


def stage(root, preserved_root, baseline_path, baseline_sha, plan_path, plan_sha, archive_path):
    require(sha(plan_path) == plan_sha, "P2S_PLAN_BYTES_CHANGED")
    b = baseline(baseline_path, baseline_sha); verify_local_parent(preserved_root, b)
    plan = read(plan_path); rows = validate_plan(root, plan, b)
    require(not Path(archive_path).exists(), "P2S_STAGE_ARCHIVE_MUST_BE_NEW")
    with zipfile.ZipFile(archive_path, "x") as archive:
        for r in rows:
            archive.writestr(*zip_entry(r["file"], checked(root, r["file"], r["sha256"], r["bytes"]).read_bytes()))
        archive.writestr(*zip_entry(PLAN_MEMBER, Path(plan_path).read_bytes()))
        archive.writestr(*zip_entry(BASELINE_MEMBER, Path(baseline_path).read_bytes()))
    validate_plan(root, plan, b); verify_local_parent(preserved_root, b)
    return {"schema": "semantic_b0b7_p2s_text_stage_v1", "status": "PREFLIGHT_ONLY_NOT_PRIVATE_SCAN",
        "plan_sha256": plan_sha, "archive_sha256": sha(archive_path), "candidate_files": len(rows),
        "transport_control_files": 2, "Git_write_operations": 0}


def unpack_stage(archive_path, archive_sha, plan_path, plan_sha, baseline_path, baseline_sha, output):
    require(sha(archive_path) == archive_sha and sha(plan_path) == plan_sha, "P2S_ARCHIVE_OR_PLAN_CHANGED")
    b = baseline(baseline_path, baseline_sha); plan = read(plan_path); rows = validate_rows(plan["files"])
    expected = {r["file"]: r for r in rows}
    expected.update({PLAN_MEMBER: {"bytes": Path(plan_path).stat().st_size, "sha256": plan_sha},
        BASELINE_MEMBER: {"bytes": Path(baseline_path).stat().st_size, "sha256": baseline_sha}})
    require(not Path(output).exists(), "P2S_EXTRACTION_DIRECTORY_MUST_BE_NEW")
    content = {}
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        require(not archive.comment and len(entries) == len(expected) == len({e.filename for e in entries}) and
            {e.filename for e in entries} == set(expected), "P2S_ZIP_EXTRA_OR_DUPLICATE_MEMBER")
        for entry in entries:
            safe_relative(entry.filename); mode = entry.external_attr >> 16
            require(not entry.is_dir() and not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in (0, stat.S_IFREG) and
                not (entry.flag_bits & 1) and not entry.comment and not entry.extra, "P2S_ZIP_UNSCANNED_METADATA_OR_SYMLINK")
            r = expected[entry.filename]
            require(0 < entry.file_size <= MAX_BYTES and entry.file_size == r["bytes"], "P2S_ZIP_MEMBER_SIZE_CHANGED")
            raw = archive.read(entry); raw.decode("utf-8-sig")
            require(len(raw) == r["bytes"] and hashlib.sha256(raw).hexdigest() == r["sha256"], "P2S_ZIP_PAYLOAD_CHANGED")
            content[entry.filename] = raw
    Path(output).mkdir(parents=True)
    for name, raw in content.items():
        write_new(Path(output) / name, raw)
    validate_plan(output, plan, b)
    return plan, b


def final_scan(archive_path, archive_sha, plan_path, plan_sha, baseline_path, baseline_sha,
               git_repo, denylist_path, output_newdir):
    require(sha(original.__file__) == SCANNER_SHA and sha(denylist_path) == DENYLIST_SHA,
            "P2S_ORIGINAL_SCANNER_OR_ESTABLISHED_DENYLIST_CHANGED")
    pattern = original.identifier_pattern(original.validate_denylist(read(denylist_path)))
    require(not Path(output_newdir).exists(), "P2S_FINAL_OUTPUT_MUST_BE_NEW")
    payload = Path(output_newdir) / "public"
    plan, b = unpack_stage(archive_path, archive_sha, plan_path, plan_sha, baseline_path, baseline_sha, payload)
    git_before = verify_git_parent(git_repo, b)
    rows = scan_exact(payload, plan["files"], pattern)
    controls = [{"file": name, "bytes": (payload / name).stat().st_size, "sha256": sha(payload / name)}
                for name in (PLAN_MEMBER, BASELINE_MEMBER)]
    scan_exact(payload, controls, pattern)
    receipt = {"schema": "semantic_b0b7_p2s_actual_private_scan_v1", "status": "PASS_ORIGINAL_SCANNER_ACTUAL_IDENTIFIER_RULES",
        "base_git_commit": BASE_COMMIT, "base_git_tree": BASE_TREE, "branch": BRANCH,
        "plan_sha256": plan_sha, "stage_archive_sha256": archive_sha, "parent472_baseline_sha256": BASELINE_SHA,
        "original_scanner_sha256": SCANNER_SHA, "established_private_denylist_sha256": DENYLIST_SHA,
        "candidate_files": len(rows), "candidate_bytes": sum(r["bytes"] for r in rows), "candidate_rows_sha256": canonical(rows),
        "candidate_scanned_now_count": len(rows), "candidate_reused_prior_scan_count": 0,
        "control_metadata_scanned_now_count": 2, "generated_files_require_separate_actual_scan": 2,
        "parent472_Git_object_verification": git_before, "metadata_proof_closure": plan["metadata_proof_closure"],
        "private_identifier_values_exported": False, "denylist_built_exported_or_modified": False,
        "new_fits_or_forwards": 0, "Git_write_operations": 0, "manifest_hash_not_embedded": True,
        "scientific_mechanism_inferred_from_privacy_scan": False}
    rr = encode(receipt); receipt_row = {"file": SCAN_RECEIPT, "bytes": len(rr), "sha256": hashlib.sha256(rr).hexdigest()}
    manifest = {"schema": "semantic_b0b7_p2s_publication_manifest_v1", "status": receipt["status"],
        "base_git_commit": BASE_COMMIT, "base_git_tree": BASE_TREE, "branch": BRANCH,
        "parent472_baseline_sha256": BASELINE_SHA, "plan_sha256": plan_sha, "source_scan_receipt_sha256": receipt_row["sha256"],
        "files": sorted(rows + [receipt_row], key=lambda r: r["file"]), "manifest_does_not_hash_itself": True,
        "preserved_old_files": 472, "Git_write_operations": 0}
    mr = encode(manifest); manifest_row = {"file": MANIFEST, "bytes": len(mr), "sha256": hashlib.sha256(mr).hexdigest()}
    write_new(payload / SCAN_RECEIPT, rr); write_new(payload / MANIFEST, mr)
    generated = scan_exact(payload, [receipt_row, manifest_row], pattern)
    require(verify_git_parent(git_repo, b) == git_before, "P2S_PARENT_GIT_OBJECTS_CHANGED_DURING_SCAN")
    all_rows = sorted(rows + generated, key=lambda r: r["file"])
    final_zip = Path(output_newdir) / "PUBLICATION.zip"
    publication_rows = []
    with zipfile.ZipFile(final_zip, "x") as archive:
        for r in all_rows:
            raw = checked(payload, r["file"], r["sha256"], r["bytes"]).read_bytes()
            archive.writestr(*zip_entry(r["file"], raw))
            publication_rows.append({**r, "git_blob_sha1": hashlib.sha1(("blob " + str(len(raw)) + "\0").encode() + raw).hexdigest()})
    binding = {"schema": "semantic_b0b7_p2s_final_publication_binding_v1", "status": "PASS_METADATA_ONLY_NO_GIT_WRITE",
        "base_git_commit": BASE_COMMIT, "base_git_tree": BASE_TREE, "branch": BRANCH,
        "plan_sha256": plan_sha, "parent472_baseline_sha256": BASELINE_SHA, "publication_manifest_sha256": manifest_row["sha256"],
        "scan_receipt_sha256": receipt_row["sha256"], "publication_archive_sha256": sha(final_zip),
        "candidate_scanned_now_count": len(rows), "candidate_reused_prior_scan_count": 0, "generated_scanned_now_count": 2,
        "control_metadata_scanned_now_count": 2, "published_files": len(all_rows), "files": publication_rows,
        "binding_separate_not_hashed_by_manifest": True, "original_scanner_sha256": SCANNER_SHA,
        "established_private_denylist_sha256": DENYLIST_SHA, "preserved_old_files": 472,
        "Git_write_operations": 0, "private_identifier_values_exported": False}
    write_new(Path(output_newdir) / "FINAL_BINDING.json", encode(binding))
    return binding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "stage", "final-scan", "verify-parent-git"))
    p.add_argument("--root", type=Path, default=ROOT); p.add_argument("--preserved-root", type=Path)
    for name in ("baseline", "expected-baseline-sha256", "candidates", "proofs", "plan", "expected-plan-sha256",
                 "archive", "expected-archive-sha256", "git-repo", "denylist", "output-newdir"):
        p.add_argument("--" + name)
    a = p.parse_args()
    if a.command == "plan":
        require(a.preserved_root and a.candidates and a.proofs and a.plan, "P2S_PLAN_ARGUMENTS_REQUIRED")
        candidates, proofs = read(a.candidates), read(a.proofs)
        require(set(candidates) == {"files"}, "P2S_CANDIDATE_WRAPPER_MUST_BE_EXACT_FILES")
        value = build_plan(a.root, a.preserved_root, a.baseline, a.expected_baseline_sha256, candidates["files"], proofs)
        write_new(a.plan, encode(value)); print(json.dumps({"plan_sha256": sha(a.plan), "candidate_files": len(value["files"])}))
    elif a.command == "stage":
        print(json.dumps(stage(a.root, a.preserved_root, a.baseline, a.expected_baseline_sha256,
            a.plan, a.expected_plan_sha256, a.archive)))
    elif a.command == "verify-parent-git":
        print(json.dumps(verify_git_parent(a.git_repo, baseline(a.baseline, a.expected_baseline_sha256))))
    else:
        print(json.dumps(final_scan(a.archive, a.expected_archive_sha256, a.plan, a.expected_plan_sha256,
            a.baseline, a.expected_baseline_sha256, a.git_repo, a.denylist, a.output_newdir)))


if __name__ == "__main__":
    main()
