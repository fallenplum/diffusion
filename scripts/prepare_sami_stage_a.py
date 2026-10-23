"""Prepare qualified private tissue labels and frozen UNI2-h; zero model fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile
import hashlib

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.sami_surv.annotations import (
    ANNOTATION_ARCHIVE, CLASSES, LABEL_MAP, LABEL_MAPPING_SHA256, LABEL_SCHEMA,
    align_annotations, annotation_member, parse_annotation_csv, qualified_feature_section,
    read_json, require, sha256, source_coverage, validate_development_lock, validate_protocol_source,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(args):
    protocol = read_json(args.protocol)
    label_contract = protocol["label_contract"]
    require(label_contract.get("schema_version") == LABEL_SCHEMA
            and label_contract.get("class_order") == list(CLASSES)
            and label_contract.get("mapping_sha256") == LABEL_MAPPING_SHA256,
            "protocol does not bind exact label mapping and order")
    root, output = args.phase22_root.resolve(), args.output.resolve()
    require(not output.is_relative_to(REPO) and not output.is_relative_to(root / "raw"),
            "private prepared data must remain outside repository and source raw data")
    require(not output.exists() or not any(output.iterdir()), "refusing to overwrite existing prepared artifacts")
    archive = args.annotations
    require(archive.stat().st_size == ANNOTATION_ARCHIVE["bytes"]
            and sha256(archive) == ANNOTATION_ARCHIVE["sha256"], "official annotation archive integrity mismatch")
    lock_path = root / "metadata/CRC16_ST_DONOR_SPLIT_LOCK_PRIVATE.json"
    marker_path = root / "processed/DATA_QUALIFICATION_PRIVATE.json"
    lock, marker = read_json(lock_path), read_json(marker_path)
    lock_sha = sha256(lock_path)
    validate_protocol_source(protocol, lock_sha)
    require(marker.get("source_revision") == protocol["source"]["hest_revision"],
            "actual qualified HEST revision differs from Stage A protocol")
    rows, evidence = validate_development_lock(lock, marker, lock_sha)
    ledger = root / "metadata/hest/HEST_v1_3_0.csv"
    require(sha256(ledger) == lock["metadata_sha256"], "original donor ledger SHA mismatch")
    pieces, records, provenance, used = [], [], [], set()
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        require(len(names) == len(set(names)), "duplicate ZIP member names")
        require(len([n for n in names if n.endswith(".csv")]) == 14, "expected original 14 CSV source archive")
        for row in rows:
            X, coords, barcodes, meta, source = qualified_feature_section(
                root, row, evidence[row["section_id_private"]], lock_sha, REPO)
            member = annotation_member(meta, row, names)
            require(member not in used, "same source annotation mapped to multiple sections")
            used.add(member)
            content = zipped.read(member)
            annotation, column = parse_annotation_csv(content)
            raw, categories, y = align_annotations(barcodes, annotation)
            selected = y >= 0
            section, donor = row["section_id_private"], row["donor_id_private"]
            n = int(selected.sum())
            pieces.append(dict(X=X[selected], y=y[selected], coords_um=coords[selected], barcodes=barcodes[selected],
                               donor=np.full(n, donor), section=np.full(n, section)))
            records.append(dict(section=section, donor=donor, barcodes=barcodes, raw_labels=raw,
                                categories=categories, y=y))
            provenance.append(dict(source, annotation_member=member, annotation_column=column,
                                   annotation_csv_sha256=hashlib.sha256(content).hexdigest(),
                                   spots=len(y), primary_spots=n, excluded_spots=int((~selected).sum()),
                                   exact_barcode_set_and_ordered_join=True))
    coverage = source_coverage(records)
    require(coverage["primary_class_counts"] == {"tumor": 4040, "stroma": 5237, "tumor_stroma_mixed": 3734},
            "primary counts differ from locked source audit")
    require(coverage["primary_class_donors"] == {c: 6 for c in CLASSES}, "all classes must cover all six donors")
    data = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    donor_names = sorted(set(data["donor"]))
    outer = [dict(outer_index=i, train_donors=[d for d in donor_names if d != held], heldout_donor=held)
             for i, held in enumerate(donor_names)]
    require(len(data["y"]) == 13011 and sum(len(r["y"]) for r in records) == 17392,
            "prepared data total mismatch")
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "prepared.npz"
    with data_path.open("wb") as stream:
        np.savez_compressed(stream, **data)
    mapping_path = output / "annotation_mapping.npz"
    mapping = {k: np.concatenate([r[k] for r in records]) for k in ("barcodes", "raw_labels", "categories", "y")}
    for k in ("section", "donor"):
        mapping[k] = np.concatenate([np.full(len(r["y"]), r[k]) for r in records])
    with mapping_path.open("wb") as stream:
        np.savez_compressed(stream, **mapping)
    manifest = dict(schema_version="sami_stage_a_prepared_v1", status="PASS_PREPARED_FROZEN_TISSUE3",
                    classes=list(CLASSES), label_mapping=LABEL_MAP, mapping_sha256=LABEL_MAPPING_SHA256,
                    data_file="prepared.npz", data_sha256=sha256(data_path),
                    annotation_mapping_file="annotation_mapping.npz", annotation_mapping_sha256=sha256(mapping_path),
                    protocol_sha256=sha256(args.protocol), original_split_lock_sha256=lock_sha,
                    original_qualification_sha256=sha256(marker_path), original_donor_ledger_sha256=sha256(ledger),
                    annotation_archive_sha256=ANNOTATION_ARCHIVE["sha256"], sections=provenance,
                    outer_donor_folds=outer, donors=6, n_spots=13011, n_features=1536,
                    source_code_sha256={p: sha256(REPO / p) for p in
                        ["scripts/prepare_sami_stage_a.py", "src/sami_surv/annotations.py"]},
                    expression_array_accessed=False, DSS_read=False, original_ST_test_arrays_read=False,
                    original_train_validation_recombined_as_development=True,
                    this_nested_cv_is_not_new_independent_external_confirmation=True)
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    audit = dict(schema_version="sami_stage_a_annotation_audit_public_v1", status="PASS_PREPARED_FROZEN_TISSUE3",
                 prepared_manifest_sha256=sha256(manifest_path), prepared_data_sha256=manifest["data_sha256"],
                 protocol_sha256=manifest["protocol_sha256"], label_mapping_sha256=LABEL_MAPPING_SHA256,
                 annotation_archive=ANNOTATION_ARCHIVE, source="https://zenodo.org/records/7760264",
                 classes=list(CLASSES), sections=12, donors=6, annotated_spots=17392, primary_spots=13011,
                 excluded_spots=4381, feature_dimension=1536, exact_barcode_join_sections=12,
                 duplicate_barcodes_within_section=0, missing_annotation_barcodes=0, missing_feature_barcodes=0,
                 source_feature_receipt_and_sha_verified_sections=12, elementwise_coordinate_barcode_checks=12,
                 **coverage, original_ST_test_arrays_read=False, expression_array_accessed=False, DSS_read=False,
                 actual_model_fits=0, source_raw_files_modified=False, tissue_labels_are_cell_source_truth=False,
                 source_cells_and_features_private=True)
    write_json(args.public_report, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase22-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-report", type=Path, default=REPO / "reports/sami_surv_stage_a/DATA_AUDIT.json")
    result = prepare(parser.parse_args())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
