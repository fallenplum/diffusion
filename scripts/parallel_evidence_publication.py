"""Administrative, explicit-file privacy preflight; performs zero Git operations.

Uses the established exact-identifier denylist and text scanner. This script is
outside both scientific filename globs and never changes scientific files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
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
EXTRA = {"r8":{"scripts/verify_r8_aggregate_tables.py"}, "spgel":{
    "scripts/audit_transfer_geometry.py", "scripts/check_spgel_frozen_delivery.py",
    "scripts/verify_spgel_aggregate_tables.py", "scripts/shard_spgel_public_report.py"}}
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


def check_error(value, code):
    require(type(value) in (int,float) and 0 <= value <= 2e-6, code)


def load_lossless_source_report(root, listed):
    script="scripts/shard_spgel_public_report.py"
    files={"S3_PILOT_SUMMARY.json","S3_CHECKPOINT_DIAGNOSTICS.json","S3_LOSSLESS_SERIALIZATION_AUDIT.json"}
    require(script in listed and all(BASE["spgel"]+n in listed for n in files),
            "COMPLETE_SOURCE_LOSSLESS_SHARDS_REQUIRED")
    spec=importlib.util.spec_from_file_location("source_publication_lossless_reader",root/script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    restored,raw=module.load_original_public_report(root/BASE["spgel"])
    audit=read(root/BASE["spgel"]/"S3_LOSSLESS_SERIALIZATION_AUDIT.json")
    main=root/BASE["spgel"]/"S3_PILOT_SUMMARY.json";diagnostic=root/BASE["spgel"]/"S3_CHECKPOINT_DIAGNOSTICS.json"
    require(audit.get("schema") == "spgel_lossless_public_report_serialization_v1"
        and audit.get("status") == "PASS_EXACT_ORIGINAL_PRODUCER_BYTES_AND_METRICS_RECONSTRUCTION"
        and audit.get("contract_sha256") == restored["contract_sha256"]
        and audit.get("original_producer_sha256") == audit.get("reconstructed_original_sha256") == hashlib.sha256(raw).hexdigest()
        and audit.get("original_semantic_sha256") == module.semantic(restored)
        and audit.get("original_bytes") == len(raw)
        and audit.get("public_main_sha256") == sha256(main) and audit.get("public_main_bytes") == main.stat().st_size
        and audit.get("diagnostic_shard_sha256") == sha256(diagnostic) and audit.get("diagnostic_shard_bytes") == diagnostic.stat().st_size
        and audit.get("donor_rows_preserved") == len(restored["donor_rows"]) == 72
        and audit.get("diagnostic_records_preserved") == len(restored["fixed_checkpoint_diagnostics"]) == 192
        and audit.get("scalar_or_diagnostic_records_removed") == audit.get("new_optimizer_fits") == audit.get("model_loads") == 0
        and audit.get("script_sha256") == sha256(root/script), "SOURCE_LOSSLESS_SERIALIZATION_AUDIT_INVALID")
    return restored,audit


def validate_source_supplements(root,private,lock,listed,serialization):
    names=("SPGEL_HELDOUT_MAP_AUDIT.json","SPGEL_INDEPENDENT_ARITHMETIC_REVIEW.json","SPGEL_EXECUTION_LEDGER.json")
    require(all(BASE["spgel"]+n in listed for n in names)
            and "scripts/verify_spgel_aggregate_tables.py" in listed, "SOURCE_FINAL_INDEPENDENT_EVIDENCE_REQUIRED")
    maps=read(root/BASE["spgel"]/names[0])
    require(maps.get("schema") == "spgel_anonymous_fixed_heldout_map_audit_v1"
        and maps.get("status") == "PASS_ALL_12_SECTIONS_TWO_FIXED_MODELS_SAVED_LODO_EPOCH40_MAPS"
        and maps.get("source_contract_SHA256") == lock["contract_sha256"]
        and maps.get("fixed_source_models_count") == 2 and maps.get("fixed_epoch") == 40
        and maps.get("source_sections_count") == 12 and maps.get("source_spots_per_model") == 13011
        and maps.get("private_PNG_count") == maps.get("private_array_count") == 24
        and maps.get("saved_prediction_arrays_used") == 12
        and len(maps.get("PNG_SHA256", [])) == 24 and len(maps.get("saved_prediction_array_SHA256", [])) == 12
        and maps.get("all_sections_included") is True
        and maps.get("model_or_epoch_or_section_selected_from_heldout") is False
        and maps.get("source_full_fit_maps_are_not_used_as_heldout") is True
        and maps.get("public_contains_coordinates_images_or_spot_predictions") is False
        and maps.get("TCGA_loaded") is False and maps.get("DSS_or_PFI_read") is False
        and maps.get("new_optimizer_fits") == maps.get("new_optimizer_updates") == maps.get("new_model_forward_calls") == 0,
        "SOURCE_ACTUAL_HELDOUT_MAP_AUDIT_INVALID")
    arithmetic=read(root/BASE["spgel"]/names[1])
    require(arithmetic.get("schema") == "spgel_independent_anonymous_arithmetic_review_v1"
        and arithmetic.get("status") == "PASS_INDEPENDENT_DONOR_PROGRAM_RETRIEVAL_MACROS_AND_PAIRED_DIFFERENCES"
        and arithmetic.get("contract_sha256") == lock["contract_sha256"]
        and arithmetic.get("source_summary_sha256") == sha256(root/BASE["spgel"]/"S3_PILOT_SUMMARY.json")
        and arithmetic.get("reviewer_script_sha256") == sha256(root/"scripts/verify_spgel_aggregate_tables.py")
        and arithmetic.get("actual_model_loads") == arithmetic.get("new_optimizer_fits") == 0
        and arithmetic.get("donor_cells_checked") == 72 and arithmetic.get("arm_seed_summaries_checked") == 12
        and arithmetic.get("paired_comparisons_checked") == 20 and arithmetic.get("scalar_metric_checks") == 1316
        and type(arithmetic.get("maximum_absolute_arithmetic_error")) in (int,float)
        and 0 <= arithmetic["maximum_absolute_arithmetic_error"] <= 2e-10
        and arithmetic.get("this_review_is_not_an_additional_checkpoint_replay") is True,
        "SOURCE_INDEPENDENT_ARITHMETIC_REVIEW_INVALID")
    ledger=read(root/BASE["spgel"]/names[2]);counts=ledger.get("actual_counts", {})
    require(ledger.get("schema") == "spgel_execution_ledger_v1"
        and ledger.get("status") == "COMPLETE_FIXED_74_FITS_198_ACTUAL_REPLAYS_TWO_FROZEN_EXPORTS"
        and ledger.get("formal_lock", {}).get("file_sha256") == sha256(private/"SOURCE_RUN_LOCK.json")
        and ledger["formal_lock"].get("internal_contract_sha256") == lock["contract_sha256"]
        and counts.get("formal_fits") == 74 and counts.get("neural_fits") == 62 and counts.get("convex_fits") == 12
        and counts.get("optimizer_updates") == 54080 and counts.get("actual_replay_states") == 198
        and counts.get("new_fits_during_replay_or_export") == counts.get("failed_formal_cells")
            == counts.get("automatic_or_unplanned_retry_fits") == 0,
        "SOURCE_ACTUAL_EXECUTION_LEDGER_INVALID")
    original=[r for r in ledger.get("report_artifacts", []) if r.get("basename") == "S3_PILOT_SUMMARY.json"]
    require(len(original) == 1 and original[0]["sha256"] == serialization["original_producer_sha256"],
            "SOURCE_ORIGINAL_PRODUCER_AND_LEDGER_SHA_DIFFER")
    return dict(private_heldout_maps=24, independent_metadata_arithmetic_checks=1316,
        original_producer_sha256=serialization["original_producer_sha256"],
        original_semantic_sha256=serialization["original_semantic_sha256"], lossless_diagnostic_records=192)


def validate_source_delivery_acceptance(root, private, lock, listed, exported):
    """Bind actual source-interface calls, not merely the existence of exports."""
    script = "scripts/check_spgel_frozen_delivery.py"
    require(script in listed, "COMPLETE_SOURCE_INDEPENDENT_ACCEPTANCE_SCRIPT_REQUIRED")
    matches = []
    for name in sorted(listed):
        if name.startswith(BASE["spgel"]) and name.endswith(".json"):
            value = read(root/name)
            if value.get("schema") == "spgel_actual_frozen_source_delivery_acceptance_v1":
                matches.append((name,value))
    require(len(matches) == 1, "COMPLETE_SOURCE_ACTUAL_INTERFACE_ACCEPTANCE_REQUIRED")
    name, acceptance = matches[0]
    require(acceptance.get("status") == "PASS_TWO_ACTUAL_S4_ENCODERS_FULL_SOURCE_CONTEXT_AND_BOUNDARIES"
        and acceptance.get("source_contract_SHA256") == lock["contract_sha256"]
        and acceptance.get("acceptance_script_SHA256") == sha256(root/script)
        and acceptance.get("source_export_complete_SHA256") == sha256(private/"s4/EXPORT_COMPLETE.private.json")
        and acceptance.get("source_encoders_count") == 2 and acceptance.get("source_sections_count") == 12
        and acceptance.get("source_spots_count") == 13011
        and acceptance.get("new_optimizer_fits") == acceptance.get("new_optimizer_updates") == 0
        and acceptance.get("TCGA_loaded") is False and acceptance.get("DSS_or_PFI_read") is False
        and acceptance.get("source_or_target_labels_selected_model") is False
        and acceptance.get("source_ST_measurements_used_only_for_source_acceptance") is True
        and acceptance.get("public_contains_spot_predictions_coordinates_or_weights") is False
        and acceptance.get("TCGA_bulk_qualification") == "NOT_RUN_REQUIRES_SEPARATE_TRAIN_ONLY_DOMAIN_CONTRACT",
        "COMPLETE_SOURCE_ACTUAL_INTERFACE_ACCEPTANCE_SCOPE_INVALID")
    receipt = private/"s4/frozen_delivery_acceptance_v1/COMPLETE.private.json"
    require(sha256(root/name) == sha256(receipt), "SOURCE_PUBLIC_PRIVATE_INTERFACE_ACCEPTANCE_BYTES_DIFFER")
    audits = acceptance.get("prior_actual_replay_audits", [])
    require(len(audits) == 2 and {r.get("stage") for r in audits} == {"pilot","s4"},
            "SOURCE_INTERFACE_PRIOR_REPLAY_INVENTORY_CHANGED")
    for row in audits:
        stage=row["stage"];units,states=(72,192) if stage=="pilot" else (2,6)
        require(row.get("SHA256") == sha256(private/stage/"REPLAY_AUDIT.json")
                and row.get("replayed_models") == units and row.get("replayed_states") == states,
                "SOURCE_INTERFACE_PRIOR_REPLAY_BINDING_CHANGED")
    rows=acceptance.get("rows", []);by_arm={r.get("arm"):r for r in rows}
    export_by_arm={r.get("arm"):r for r in exported.get("exports", [])}
    require(len(rows) == len(by_arm) == 2 and set(by_arm) == set(export_by_arm) == {"S_A1","S_A5"},
            "SOURCE_INTERFACE_FIXED_ARMS_CHANGED")
    for arm,row in by_arm.items():
        folder=private/"s4/export"/f"{arm}_seed31"
        require(row.get("source_manifest_SHA256") == export_by_arm[arm].get("manifest_sha256")
                == sha256(folder/"source_encoder_manifest.json")
            and row.get("exported_encoder_SHA256") == export_by_arm[arm].get("encoder_state_sha256")
                == sha256(folder/"encoder_state.pt")
            and row.get("original_checkpoint_SHA256") == sha256(private/"s4/cells"/f"{arm}_full_seed31"/"checkpoint_epoch40.pt"),
            "SOURCE_ACCEPTED_INTERFACE_ARTIFACT_BYTES_CHANGED")
        require(row.get("image_sections") == 12 and row.get("image_spots") == 13011
            and row.get("image_API_gene_forward_calls") == 0 and row.get("all_exported_parameters_frozen") is True
            and row.get("new_optimizer_fits") == row.get("new_optimizer_updates") == 0
            and row.get("actual_TCGA_bulk_qualification") == "NOT_RUN_SOURCE_SPOT_VALUES_NOT_PASSED_AS_BULK",
            "SOURCE_ACTUAL_IMAGE_INTERFACE_SCOPE_CHANGED")
        errors=row.get("maximum_errors", {})
        require(set(errors) == {"image_prediction_z","image_embedding","image_program_map"},
                "SOURCE_ACTUAL_IMAGE_INTERFACE_FORWARD_PATHS_MISSING")
        for value in errors.values():check_error(value,"SOURCE_IMAGE_INTERFACE_FORWARD_ERROR")
        conditional=row.get("source_conditional_maximum_errors", {})
        if arm=="S_A5":
            require(set(conditional) == {"source_gene_token","conditional_image_read","conditional_gene_read",
                "paired_scores","route_original_window"} and row.get("source_conditional_batches",0)>0,
                "SOURCE_A5_ACTUAL_CONDITIONAL_PATHS_MISSING")
            for value in conditional.values():check_error(value,"SOURCE_CONDITIONAL_INTERFACE_FORWARD_ERROR")
        else:
            require(conditional == {} and row.get("source_conditional_batches") == 0,
                    "BASELINE_MUST_NOT_CLAIM_TRAINED_CONDITIONAL_PATHS")
        boundary=row.get("padding_multislide", {})
        require(boundary.get("batch_rows") == boundary.get("complete_sections") == 2
                and type(boundary.get("padding_positions")) is int and boundary["padding_positions"]>0,
                "SOURCE_INTERFACE_PADDING_AND_SLIDE_BOUNDARY_CHECKS_MISSING")
        check_error(boundary.get("maximum_error"),"SOURCE_INTERFACE_BOUNDARY_FORWARD_ERROR")
    return dict(actual_interface_encoders=2, actual_source_image_sections_per_encoder=12,
        actual_source_image_spots_per_encoder=13011, source_interface_acceptance_sha256=sha256(root/name),
        new_TCGA_bulk_qualification_required_for_source_publication=False)


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
    summary,serialization=load_lossless_source_report(root,listed)
    require(summary.get("contract_sha256") == lock["contract_sha256"]
            and summary.get("status") == "COMPLETE_REGISTERED_SOURCE_PILOT_AND_REPLAY"
            and summary.get("s4_export", {}).get("neural_optimizer_fits") == 2
            and summary.get("s4_export", {}).get("actual_saved_state_replays") == 6,
            "COMPLETE_SOURCE_PUBLIC_S4_REPORT_REQUIRED")
    acceptance=validate_source_delivery_acceptance(root,private,lock,listed,exported)
    supplements=validate_source_supplements(root,private,lock,listed,serialization)
    return dict(actual_fits=74, actual_saved_state_replays=198, exported_encoders=2,
                official_HVG_or_independent_confirmation_claimed=False,**acceptance,**supplements)


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
