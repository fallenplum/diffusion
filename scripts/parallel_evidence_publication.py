"""Administrative, explicit-file privacy preflight; performs zero Git operations.

Uses the established exact-identifier denylist and text scanner. This script is
outside both scientific filename globs and never changes scientific files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_publication as inherited

require = inherited.require
sha256 = inherited.sha256
BASE = {"r8":"reports/bgb_en_r8/", "spgel":"reports/spgel/"}
PUBLIC_LOCK = {"r8":"R8_PROTOCOL_LOCK.json", "spgel":"SOURCE_RUN_LOCK.json"}
ADMIN = {"scripts/parallel_evidence_publication.py", "tests/test_parallel_evidence_publication.py"}
EXTRA = {"r8":{"scripts/verify_r8_aggregate_tables.py"}, "spgel":{"scripts/audit_transfer_geometry.py"}}
HISTORY = ("reports/bgb_en", "reports/bgb_en_r5", "reports/bgb_en_r6", "reports/bgb_en_r7")


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "DUPLICATE_PUBLICATION_JSON_KEY")
            value[key] = item
        return value
    def bad_constant(_):
        raise inherited.PublicationError("NONFINITE_PUBLICATION_JSON")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique, parse_constant=bad_constant)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_relative(value):
    require(isinstance(value, str) and "\\" not in value and ":" not in value, "UNSAFE_PUBLICATION_PATH")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts and str(pure) == value, "UNSAFE_PUBLICATION_PATH")
    return value


def allowed_namespace(relative, line, lock):
    safe_relative(relative)
    if relative in ADMIN | EXTRA[line]:
        return True
    if relative.startswith(BASE[line]):
        return Path(relative).suffix.lower() in (".json", ".md", ".csv")
    patterns = ((r"src/bgb_en_r8/[A-Za-z0-9_]+\.py", r"scripts/bgb_en_r8_[A-Za-z0-9_]+\.py",
                 r"tests/test_bgb_en_r8_[A-Za-z0-9_]+\.py", r"configs/bgb_en_r8/[A-Za-z0-9_]+\.json")
                if line == "r8" else
                (r"src/spgel/[A-Za-z0-9_]+\.py", r"scripts/spgel_[A-Za-z0-9_]+\.py",
                 r"tests/test_spgel_[A-Za-z0-9_]+\.py", r"configs/spgel/[A-Za-z0-9_]+\.json"))
    return any(re.fullmatch(p, relative) for p in patterns) and relative in lock["sources"]


@contextmanager
def scanner_allowlist(names):
    previous = inherited.ALLOWLIST
    inherited.ALLOWLIST = frozenset(names)
    try:
        yield
    finally:
        inherited.ALLOWLIST = previous


def inspect(root, relative, pattern):
    receipt = inherited.scan_file(root, relative, pattern)
    # Broaden drive-path detection without weakening any inherited rule.
    text = (root / relative).read_text(encoding="utf-8-sig")
    require(re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", text) is None, "ABSOLUTE_DRIVE_PATH_BLOCKED:"+relative)
    return receipt


def history_snapshot(root):
    rows = []
    for directory in HISTORY:
        folder = root / directory
        for path in sorted(folder.rglob("*")) if folder.exists() else ():
            require(not path.is_symlink(), "HISTORICAL_REPORT_SYMLINK_BLOCKED")
            if path.is_file():
                rows.append(dict(file=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
    return dict(files=len(rows), bytes=sum(r["bytes"] for r in rows), tree_sha256=canonical(rows))


def validate_lock(root, lock, line):
    schema, status = (("bgb_en_r8_frozen_evidence_protocol_lock_v1", "LOCKED_BEFORE_R8_TARGET_FITS")
                      if line == "r8" else ("spgel_source_run_lock_v1", "LOCKED_BEFORE_SPGEL_FITS"))
    require(lock.get("schema") == schema and lock.get("status") == status
            and lock.get("contract_sha256") == canonical({k:v for k,v in lock.items() if k != "contract_sha256"}),
            "ACTUAL_SCIENTIFIC_LOCK_INVALID")
    require(isinstance(lock.get("sources"), dict) and bool(lock["sources"]), "LOCKED_SOURCE_BINDINGS_REQUIRED")
    for name, digest in lock["sources"].items():
        safe_relative(name)
        path = root / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
                and sha256(path) == digest, "LOCKED_SCIENTIFIC_SOURCE_BYTES_CHANGED:"+name)
    if line == "r8":
        require(lock.get("planned_convex_fits") == 81 and lock.get("planned_neural_fits") == 0
                and lock.get("primary_candidate") == "R8-6", "R8_LOCK_BUDGET_CHANGED")
    else:
        require(lock.get("pilot", {}).get("logical_cells") == 72
                and lock.get("pilot", {}).get("actual_saved_states") == 192
                and lock.get("s4", {}).get("neural_fits") == 2
                and lock.get("s4", {}).get("actual_saved_states") == 6, "SOURCE_LOCK_BUDGET_CHANGED")


def validate_complete(root, private, lock, line, listed):
    require(private is not None, "COMPLETE_STAGE_REQUIRES_ACTUAL_PRIVATE_REPLAY_METADATA")
    private = Path(private)
    if line == "r8":
        names = ("R8_RESULTS.json", "R8_DECISION.json", "R8_INDEPENDENT_ARITHMETIC_REVIEW.json")
        require(all(BASE[line]+n in listed for n in names), "COMPLETE_R8_PUBLIC_RESULTS_REQUIRED")
        result = read(root / BASE[line] / names[0]); replay = read(private / "REPLAY_AUDIT.json")
        require(result.get("status") == "COMPLETE_81_CONVEX_FITS_AND_ACTUAL_SAVED_MODEL_REPLAYS"
                and result.get("contract_sha256") == replay.get("contract_sha256") == lock["contract_sha256"]
                and result.get("replay_sha256") == sha256(private / "REPLAY_AUDIT.json")
                and replay.get("status") == "PASS_COMPLETE_R8_81_SAVED_RISK_REPLAYS"
                and replay.get("actual_saved_models") == replay.get("passed_units") == 81
                and replay.get("failed_units") == replay.get("new_optimizer_fits") == replay.get("source_or_teacher_refits") == 0,
                "COMPLETE_R8_ACTUAL_REPLAY_REQUIRED")
        budget = result.get("budget", {})
        require(budget.get("actual_convex_fits") == budget.get("actual_saved_model_replays") == 81
                and budget.get("neural_fits") == budget.get("neural_updates") == 0
                and budget.get("zero_fit_references") == 18, "COMPLETE_R8_COUNT_BUDGET_CHANGED")
        review = read(root / BASE[line] / names[2])
        require(review.get("status") == "PASS_INDEPENDENT_WITHIN_FOLD_WEIGHTED_ARITHMETIC_AND_PAIR_CREDIT_IDENTITY"
                and review.get("aggregate_rows_checked") == 198
                and review.get("paired_endpoint_comparisons_checked") == 20, "R8_INDEPENDENT_ARITHMETIC_REQUIRED")
        for name, digest in review.get("sources", {}).items():
            safe_relative(name)
            require(BASE[line]+name in listed and sha256(root / BASE[line] / name) == digest,
                    "R8_ARITHMETIC_INPUT_BINDING_CHANGED")
        for part in result.get("diagnostic_index", []):
            safe_relative(part["file"])
            name = BASE[line]+part["file"]
            require(name in listed and sha256(root/name) == part["sha256"]
                    and (root/name).stat().st_size == part["bytes"], "R8_DIAGNOSTIC_SHARD_BINDING_CHANGED")
        require(len(result.get("diagnostic_index", [])) == 9, "R8_NINE_DIAGNOSTIC_SHARDS_REQUIRED")
        return dict(actual_fits=81, actual_saved_state_replays=81, zero_fit_references=18,
                    independent_arithmetic_is_not_extra_model_replay=True)
    for stage, units, states, status in (("pilot",72,192,"PASS_COMPLETE_SPGEL_PILOT_72_FITS_192_REPLAYS"),
                                         ("s4",2,6,"PASS_COMPLETE_SPGEL_S4_2_FITS_6_REPLAYS")):
        replay = read(private/stage/"REPLAY_AUDIT.json")
        require(replay.get("status") == status and replay.get("contract_sha256") == lock["contract_sha256"]
                and replay.get("passed_units") == replay.get("expected_units") == units
                and replay.get("passed_states") == replay.get("expected_states") == states
                and replay.get("failed_units") == replay.get("new_optimizer_fits") == replay.get("new_optimizer_updates") == 0,
                "COMPLETE_SOURCE_74_FITS_198_REPLAYS_REQUIRED")
        records = replay.get("units", [])
        require(len(records) == units and all(r.get("status") == "PASS"
            and r.get("actual_saved_state_loaded") is True and r.get("new_optimizer_fits") == r.get("new_optimizer_updates") == 0
            and all(s.get("actual_state_forward_replayed") is True for s in r.get("snapshots", [])) for r in records)
            and sum(len(r.get("snapshots", [])) for r in records) == states, "COMPLETE_SOURCE_ACTUAL_STATES_REQUIRED")
    exported = read(private/"s4/EXPORT_COMPLETE.private.json")
    require(exported.get("status") == "COMPLETE_TWO_REPLAYED_SOURCE_INTERFACES"
            and exported.get("contract_sha256") == lock["contract_sha256"]
            and len(exported.get("exports", [])) == 2 and exported.get("new_optimizer_fits") == 0,
            "COMPLETE_SOURCE_TWO_FROZEN_INTERFACES_REQUIRED")
    require(BASE[line]+"S3_PILOT_SUMMARY.json" in listed, "COMPLETE_SOURCE_PUBLIC_REPORT_REQUIRED")
    summary = read(root/BASE[line]/"S3_PILOT_SUMMARY.json")
    require(summary.get("contract_sha256") == lock["contract_sha256"]
            and summary.get("status") == "COMPLETE_REGISTERED_SOURCE_PILOT_AND_REPLAY"
            and summary.get("s4_export", {}).get("neural_optimizer_fits") == 2
            and summary.get("s4_export", {}).get("actual_saved_state_replays") == 6,
            "COMPLETE_SOURCE_PUBLIC_S4_REPORT_REQUIRED")
    return dict(actual_fits=74, actual_saved_state_replays=198, exported_encoders=2,
                official_HVG_or_independent_confirmation_claimed=False)


def publish(root, files, line, stage, lock_path, denylist_path, *, private=None, copy_public_lock=False):
    root = Path(root).resolve(); require(line in BASE and stage in ("protocol", "complete"), "UNKNOWN_PUBLICATION_SCOPE")
    old_reports = history_snapshot(root)
    lock = read(lock_path); validate_lock(root, lock, line)
    denylist = inherited.validate_denylist(read(denylist_path)); pattern = inherited.identifier_pattern(denylist)
    allow = read(files); rows = allow.get("new_files", allow.get("files"))
    require(isinstance(rows, list) and bool(rows), "EXPLICIT_SHA_SIZE_FILE_ALLOWLIST_REQUIRED")
    names = [safe_relative(r.get("file")) for r in rows]
    require(len(set(names)) == len(names), "DUPLICATE_PUBLICATION_ALLOWLIST_FILE")
    generated = [BASE[line]+"SOURCE_PUBLICATION_AUDIT.json", BASE[line]+"PUBLICATION_MANIFEST.json"]
    require(not set(names) & set(generated), "GENERATED_MANIFEST_MUST_NOT_HASH_ITSELF")
    require(all(allowed_namespace(n,line,lock) for n in names), "UNKNOWN_OR_CROSS_LINE_FILE_BLOCKED")
    public_lock = BASE[line]+PUBLIC_LOCK[line]
    require(public_lock in names, "PUBLIC_LOCK_MUST_BE_EXPLICITLY_LISTED")
    with scanner_allowlist(names+generated):
        if copy_public_lock:
            raw = Path(lock_path).read_bytes()
            with tempfile.TemporaryDirectory(dir=Path(lock_path).resolve().parent) as temp:
                sandbox = Path(temp); target = sandbox/public_lock; target.parent.mkdir(parents=True);target.write_bytes(raw)
                inspect(sandbox, public_lock, pattern)
            target = root/public_lock
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open("xb") as stream:stream.write(raw)
        require((root/public_lock).read_bytes() == Path(lock_path).read_bytes(), "PUBLIC_LOCK_NOT_ORIGINAL_BYTES")
        receipts = []
        for expected in rows:
            name = expected["file"]; actual = inspect(root,name,pattern)
            require(actual["sha256"] == expected.get("sha256") and actual["bytes"] == expected.get("bytes"),
                    "PUBLICATION_ALLOWLIST_BYTES_CHANGED:"+name)
            receipts.append(actual)
        completion = validate_complete(root,private,lock,line,set(names)) if stage == "complete" else dict(
            actual_fit_completion_claimed=False, source_training_can_be_running=True)
        audit = dict(schema="parallel_evidence_publication_audit_v1",status="PASS_EXPLICIT_ANONYMOUS_PREFLIGHT",
            line=line,stage=stage,scientific_contract_sha256=lock["contract_sha256"],actual_lock_file_sha256=sha256(lock_path),
            explicit_input_allowlist_sha256=sha256(files),scanned_input_files=len(receipts),
            scanned_input_bytes=sum(r["bytes"] for r in receipts),completion=completion,
            denylist_provenance={k:denylist[k] for k in ("cohort_sha256","patient_order_sha256","stage_b_sha256")},
            exact_identifier_group_counts={k+"_count":len(v) for k,v in denylist["groups"].items()},
            identifiers_or_private_paths_published=False,git_operations=0,model_loads=0,new_optimizer_fits=0,
            historical_reports_snapshot=old_reports,old_reports_bytes_unchanged=True,
            administrative_script_sha256=sha256(Path(__file__)))
        require(history_snapshot(root) == old_reports, "HISTORICAL_REPORT_BYTES_CHANGED_DURING_PREFLIGHT")
        for name in generated:require(not (root/name).exists(), "PUBLICATION_OUTPUT_EXISTS_NO_OVERWRITE")
        path = root/generated[0];path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf-8") as stream:json.dump(audit,stream,indent=2,allow_nan=False);stream.write("\n")
        audit_receipt = inspect(root,generated[0],pattern)
        manifest = dict(schema="parallel_evidence_publication_manifest_v1",status="PASS_EXPLICIT_FILES_AND_AUDIT_ONLY",
            line=line,stage=stage,scientific_contract_sha256=lock["contract_sha256"],
            actual_lock_file_sha256=sha256(lock_path),files=sorted(receipts+[audit_receipt],key=lambda r:r["file"]),
            excludes_manifest_self_hash=True,git_operations=0,old_reports_bytes_unchanged=True)
        path=root/generated[1]
        with path.open("x",encoding="utf-8") as stream:json.dump(manifest,stream,indent=2,allow_nan=False);stream.write("\n")
        inspect(root,generated[1],pattern)
        require(history_snapshot(root) == old_reports, "HISTORICAL_REPORT_BYTES_CHANGED_AFTER_MANIFEST")
    return dict(status=audit["status"],line=line,stage=stage,input_files=len(receipts),
        input_bytes=audit["scanned_input_bytes"],audit_sha256=sha256(root/generated[0]),
        manifest_sha256=sha256(root/generated[1]),git_operations=0,model_loads=0)


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ("root","files","lock","denylist"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--line",choices=tuple(BASE),required=True);p.add_argument("--stage",choices=("protocol","complete"),required=True)
    p.add_argument("--private",type=Path);p.add_argument("--copy-public-lock",action="store_true")
    a=p.parse_args();print(json.dumps(publish(a.root,a.files,a.line,a.stage,a.lock,a.denylist,
        private=a.private,copy_public_lock=a.copy_public_lock),sort_keys=True))


if __name__ == "__main__":main()
