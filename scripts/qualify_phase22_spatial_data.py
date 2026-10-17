"""Qualify arriving, pinned CRC spatial assets; keep case-level outputs private.

Missing/in-progress downloads produce PENDING_DATA_ARRIVAL and can be rerun.
This command performs no training and accepts no survival outcome input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

from src.phase22_spatial_bridge.data_contract import (
    DataContractError, file_sha256, fixed_gene_vocabulary,
    qualify_patch_pair, read_h5ad_raw, validate_split_lock,
)


def _write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise DataContractError("runtime manifest paths must remain below the transferable data root")
    return resolved.relative_to(root).as_posix()


def _public(marker: dict) -> dict:
    ready = marker["sections_ready"]
    counts_by_split = Counter(row["split"] for row in ready)
    donors_by_split = {label: len({(row["study_id"], row["donor_id_private"]) for row in ready if row["split"] == label})
                       for label in ("train", "validation", "test")}
    result = dict(
        schema_version="phase22_external_st_qualification_public_v1",
        qualification_status=marker["qualification_status"],
        source_repository=marker.get("source_repository"), source_revision=marker.get("source_revision"),
        locked_split_sha=marker.get("locked_split_sha"),
        planned_sections=marker.get("planned_sections", 0), sections_ready=len(ready),
        sections_pending=len(marker["sections_pending"]), sections_failed=len(marker["sections_failed"]),
        ready_sections_by_split=dict(counts_by_split), ready_donors_by_split=donors_by_split,
        ready_sections_by_assay=dict(Counter(row["assay"] for row in ready)),
        fixed_gene_count=marker.get("fixed_gene_count"), gene_contract=marker.get("gene_contract"),
        missing_downloads_are_resumable=True, donor_identity_split_checked=True,
        exact_duplicate_source_files_crossing_splits=marker.get("duplicate_source_cross_split", 0),
        actual_spatial_or_survival_model_fits=0, survival_outcomes_read=False,
        historical_embeddings_reused=False, UNI2_feature_extraction_qualification="SEPARATE_REQUIRED_GATE",
        original_WSI_image_sha="NOT_AVAILABLE_PATCH_DERIVATIVES_VERIFIED_INSTEAD",
        known_foundation_pretraining_overlap="UNKNOWN_NOT_VERIFIED",
        license_status="SOURCE_DECLARATIONS_RECORDED_VERSION_AND_WEIGHT_LICENSE_ARE_SEPARATE",
        local_ST_data_qualification_is_not_TCGA_bulk_assay_qualification=True,
        integer_native_crop_tolerance="abs(actual_um-112)<=0.5*source_mpp+1e-6",
        image_and_expression_capture_footprints_recorded_separately=True,
        failure_categories=dict(Counter(row["error_class"] for row in marker["sections_failed"])),
        data_locations="private_E_disk_runtime_only",
    )
    return result


def qualify(args) -> dict:
    root = args.data_root.resolve()
    lock_path = args.split_lock or root/"metadata/CRC16_ST_DONOR_SPLIT_LOCK_PRIVATE.json"
    plan_path = args.asset_plan or root/"metadata/CRC16_QUALIFICATION_ASSET_PLAN.json"
    target_path = args.target_genes or root/"metadata/TARGET_CONSERVATIVE_MEASURED_GENES_PRIVATE.json"
    private_marker = root/"processed/DATA_QUALIFICATION_PRIVATE.json"
    marker = dict(schema_version="phase22_external_st_qualification_v1", qualification_status="PENDING_DATA_ARRIVAL",
                  planned_sections=0, sections_ready=[], sections_pending=[], sections_failed=[],
                  locked_split_path=_relative(lock_path, root), duplicate_source_cross_split=0)
    required = [lock_path, plan_path, target_path, args.panel_genes]
    missing = [path for path in required if not path.is_file()]
    if missing:
        marker["sections_pending"] = [dict(reason="PENDING_METADATA_ARRIVAL", path_private=str(path)) for path in missing]
        _write_json(private_marker, marker)
        _write_json(args.public_report, _public(marker))
        return marker
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_split_lock(lock)
    source_ledger_path = root/"metadata/hest/HEST_v1_3_0.csv"
    if not source_ledger_path.is_file() or file_sha256(source_ledger_path) != lock.get("metadata_sha256"):
        raise DataContractError("fixed source donor ledger must match the prelocked CSV SHA")
    with source_ledger_path.open(encoding="utf-8", newline="") as stream:
        donor_ledger = {entry["id"]: entry for entry in csv.DictReader(stream)}
    genes, gene_contract = fixed_gene_vocabulary(args.panel_genes, target_path)
    marker.update(planned_sections=len(lock["rows"]), locked_split_sha=file_sha256(lock_path),
                  asset_plan_sha=file_sha256(plan_path), fixed_gene_count=len(genes), gene_contract=gene_contract,
                  source_repository=plan["repo"], source_revision=plan["revision"],
                  qualification_code_sha256=dict(
                      script=file_sha256(Path(__file__)),
                      data_contract=file_sha256(Path(__file__).resolve().parents[1]/"src/phase22_spatial_bridge/data_contract.py")))
    expected = {item["path"]: item for item in plan["files"]}
    if len(expected) != len(plan["files"]):
        raise DataContractError("duplicate asset plan paths")
    raw_root = root/"raw/hest"
    output_root = root/"processed/sections"
    output_root.mkdir(parents=True, exist_ok=True)
    duplicate_map = {}
    for row in lock["rows"]:
        section = row["section_id_private"]
        identity = {key: row[key] for key in ("section_id_private", "study_id", "donor_id_private", "split", "assay")}
        files = dict(expression=f"st/{section}.h5ad", patches=f"patches/{section}.h5", metadata=f"metadata/{section}.json")
        if args.only_section and section not in args.only_section:
            marker["sections_pending"].append(dict(identity, reason="NOT_INSPECTED_IN_THIS_BOUNDED_RUN"))
            continue
        pending = [name for name in files.values() if not (raw_root/name).is_file()]
        if pending:
            marker["sections_pending"].append(dict(identity, reason="PENDING_ASSET_ARRIVAL", missing_files=pending))
            continue
        try:
            wrong_size = [name for name in files.values() if name not in expected or (raw_root/name).stat().st_size != expected[name]["bytes"]]
        except FileNotFoundError:
            # The HF client may atomically replace an already-cached file while
            # a resumable download refreshes it. This is arrival, not bad data.
            marker["sections_pending"].append(dict(identity, reason="PENDING_ATOMIC_DOWNLOAD_REPLACEMENT"))
            continue
        if wrong_size:
            marker["sections_pending"].append(dict(identity, reason="PENDING_COMPLETE_ASSET_BYTES", incomplete_files=wrong_size))
            continue
        try:
            hashes = {name: file_sha256(raw_root/relative) for name, relative in files.items()}
            for name, relative in files.items():
                if expected[relative].get("sha256") and hashes[name] != expected[relative]["sha256"]:
                    raise DataContractError("completed asset SHA256 differs from the pinned source plan")
            for kind in ("expression", "patches"):
                duplicate_key = (kind, hashes[kind])
                if duplicate_key in duplicate_map:
                    old = duplicate_map[duplicate_key]
                    if old["split"] != row["split"]:
                        marker["duplicate_source_cross_split"] += 1
                        raise DataContractError("exact duplicate source content crosses locked splits")
                    raise DataContractError("exact duplicate source content would count the same section twice")
            metadata = json.loads((raw_root/files["metadata"]).read_text(encoding="utf-8"))
            if metadata.get("id") != section or metadata.get("st_technology") != row["assay"]:
                raise DataContractError("original metadata section/assay identity does not match source lock")
            ledger = donor_ledger[section]
            if ledger["patient"] != row.get("patient_source_field_verified"):
                raise DataContractError("prelocked donor identity differs from the fixed source CSV")
            donor_adjudication = None
            if str(metadata.get("patient")) != row.get("patient_source_field_verified"):
                # These stale individual JSON patient fields are contradicted
                # by both their own P5/P2 subseries and the later fixed CSV.
                # The published original study independently lists P1/P2/P5
                # as distinct patients. No arbitrary override is accepted.
                corrected = {"TENX147": ("Patient 5", "Xenium In Situ, Sample P5 CRC"),
                             "TENX148": ("Patient 2", "Xenium In Situ, Sample P2 CRC"),
                             "TENX152": ("Patient 2", "Visium CytAssist v2, Sample P2 CRC")}
                expected_patient, expected_subseries = corrected.get(section, (None, None))
                if (expected_patient != ledger["patient"] or
                        str(metadata.get("subseries", "")) != expected_subseries):
                    raise DataContractError("unresolved source patient identity discrepancy")
                donor_adjudication = dict(
                    status="RESOLVED_FIXED_CSV_AND_ORIGINAL_STUDY_NOT_RAW_METADATA_REWRITE",
                    original_JSON_patient_private=metadata.get("patient"),
                    original_JSON_subseries_private=metadata.get("subseries"),
                    authoritative_CSV_patient_private=ledger["patient"],
                    authoritative_CSV_sha256=file_sha256(source_ledger_path),
                    original_JSON_sha256=hashes["metadata"],
                    independent_source="https://pmc.ncbi.nlm.nih.gov/articles/PMC12165841/",
                    source_location="original study Table 1 and Xenium methods: distinct P1CRC/P2CRC/P5CRC")
            expression = read_h5ad_raw(raw_root/files["expression"])
            arrays, audit = qualify_patch_pair(expression, raw_root/files["patches"], metadata, genes)
            if not arrays["measured_mask"].all() and row["split"] != "test":
                raise DataContractError("ST training/validation must measure the prelocked composition vocabulary")
            npz_path, meta_path = output_root/f"{section}.npz", output_root/f"{section}.json"
            cache_signature = dict(source_sha256=hashes, locked_split_sha=marker["locked_split_sha"],
                                   gene_contract=gene_contract, qualification_code_sha256=marker["qualification_code_sha256"])
            previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            if previous.get("cache_signature") == cache_signature and npz_path.is_file() and previous.get("npz_sha256") == file_sha256(npz_path):
                record = previous
            else:
                temporary = npz_path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(npz_path)
                record = dict(identity, qualification_status="PASS_SECTION_DATA_CONTRACT", cache_signature=cache_signature,
                              npz_path=_relative(npz_path, root), npz_sha256=file_sha256(npz_path),
                              coordinate_unit="um", capture_area_um2=audit["expression_capture_footprint"]["nominal_area_um2"],
                              audit=audit, raw_count_unit=audit["count_unit"],
                              source_files={name: dict(path=_relative(raw_root/relative, root), sha256=hashes[name]) for name, relative in files.items()},
                              source_repository=plan["repo"], source_revision=plan["revision"],
                              original_accession=metadata.get("download_page_link1", row.get("source", "UNKNOWN")),
                              source_declared_license=metadata.get("license", "UNKNOWN"),
                              original_WSI_sha256=None, original_WSI_not_downloaded=True,
                              source_pixel_size_um_estimated=metadata.get("pixel_size_um_estimated"),
                              known_pretraining_overlap="UNKNOWN_NOT_INDEPENDENTLY_VERIFIED")
                record["donor_identity_source_adjudication"] = donor_adjudication
                record["full_dictionary_measurement_projection_eligible"] = bool(arrays["measured_mask"].all())
                record["partial_test_measurement_policy"] = (
                    "A_evaluate_observed_gene_conditional_composition_only; "
                    "B_full_composition_projection_ineligible; no_test_based_dictionary_gene_reselection")
                _write_json(meta_path, record)
            ready = dict(identity, npz_path=_relative(npz_path, root), npz_sha256=record["npz_sha256"],
                         metadata_path=_relative(meta_path, root), metadata_sha256=file_sha256(meta_path),
                         patch_path=_relative(raw_root/files["patches"], root), patch_sha256=hashes["patches"])
            marker["sections_ready"].append(ready)
            for kind in ("expression", "patches"):
                duplicate_map[(kind, hashes[kind])] = row
            # A bounded run still leaves a usable, honest marker if interrupted.
            _write_json(private_marker, marker)
            _write_json(args.public_report, _public(marker))
            print("SECTION_QUALIFIED", len(marker["sections_ready"]), "of", len(lock["rows"]), flush=True)
        except (DataContractError, OSError, ValueError, KeyError) as error:
            marker["sections_failed"].append(dict(identity, error_class=type(error).__name__, reason=str(error)))
    marker["qualification_status"] = ("FAIL_DATA_CONTRACT" if marker["sections_failed"] else
        "PENDING_DATA_ARRIVAL" if marker["sections_pending"] else "PASS_EXTERNAL_ST_DATA_CONTRACT")
    _write_json(private_marker, marker)
    _write_json(args.public_report, _public(marker))
    return marker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("E:/diffusion/dataset/spatial_bridge"))
    parser.add_argument("--split-lock", type=Path)
    parser.add_argument("--asset-plan", type=Path)
    parser.add_argument("--target-genes", type=Path)
    parser.add_argument("--panel-genes", type=Path, default=Path("tmp/phase22/assay_measured_gene_intersection_private.json"))
    parser.add_argument("--public-report", type=Path, default=Path("reports/phase22_spatial_bridge/DATA_QUALIFICATION.json"))
    parser.add_argument("--only-section", nargs="*")
    args = parser.parse_args()
    marker = qualify(args)
    print(json.dumps(dict(status=marker["qualification_status"], ready=len(marker["sections_ready"]),
                          pending=len(marker["sections_pending"]), failed=len(marker["sections_failed"]))))
    if marker["qualification_status"] == "FAIL_DATA_CONTRACT":
        sys.exit(2)


if __name__ == "__main__":
    main()
