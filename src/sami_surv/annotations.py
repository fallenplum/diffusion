"""Exact, outcome-free pathology annotation qualification for SAMI Stage A.

This module never trains or reads expression arrays or TCGA outcomes.  IDs and
spot-level mappings belong in private artifacts.  Original source spelling is
preserved; only the finite mapping below groups medically compatible labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np


class AnnotationContractError(ValueError):
    pass


CLASSES = ("tumor", "stroma", "tumor_stroma_mixed")
ANNOTATION_COLUMNS = ("Pathologist Annotations", "Pathologist Annotation",
                      "Pathologist annotation", "Pathologist_KH")
LABEL_SCHEMA = "sami_stage_a_tissue3_labels_v1"
LABEL_MAP = {
    "tumor": "tumor",
    "stroma_fibroblastic_IC high": "stroma",
    "stroma_fibroblastic_IC med": "stroma",
    "stroma_fibroblastic_IC low": "stroma",
    "stroma_fibroblastic_IC_high": "stroma",
    "stroma_fibroblastic_IC_med": "stroma",
    "stroma desmoplastic_IC low": "stroma",
    "stroma desmoplastic_IC med to high": "stroma",
    "stroma_desmoplastic_IC low": "stroma",
    "stroma_desmoplastic_IC med to high": "stroma",
    "tumor&stroma_IC med to high": "tumor_stroma_mixed",
    "tumor&stroma IC med to high": "tumor_stroma_mixed",
    "tumor&stroma_IC low": "tumor_stroma_mixed",
    "tumor&stroma": "tumor_stroma_mixed",
    "non neo epithelium": "EXCLUDE_NON_NEOPLASTIC_EPITHELIUM",
    "epithelium&submucosa": "EXCLUDE_EPITHELIUM_SUBMUCOSA_MIXTURE",
    "epitehlium&submucosa": "EXCLUDE_EPITHELIUM_SUBMUCOSA_MIXTURE",
    "submucosa": "EXCLUDE_SUBMUCOSA",
    "muscularis_IC med to high": "EXCLUDE_MUSCULARIS",
    "IC aggregate_submucosa": "EXCLUDE_IMMUNE_AGGREGATE",
    "IC aggregregate_submucosa": "EXCLUDE_IMMUNE_AGGREGATE",
    "IC aggregate submucosa": "EXCLUDE_IMMUNE_AGGREGATE",
    "IC aggregate_stroma or muscularis": "EXCLUDE_IMMUNE_AGGREGATE",
    "IC aggreates_stroma or muscularis": "EXCLUDE_IMMUNE_AGGREGATE",
    "IC aggragate_stroma or muscularis": "EXCLUDE_IMMUNE_AGGREGATE",
    "IC aggregate_muscularis or stroma": "EXCLUDE_IMMUNE_AGGREGATE",
    "exclude": "EXCLUDE_SOURCE_PATHOLOGIST",
    "": "EXCLUDE_UNLABELED",
}
ANNOTATION_ARCHIVE = {
    "bytes": 702660,
    "md5": "92882eb0db7c2cacc5077cb5fc9578fd",
    "sha256": "2468ad1208468277eaa7693cbaf92735e76d52207e87d534e0ac8fe77fcacb47",
}
MODEL_SHA = "6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e"
MODEL_REVISION = "d517a8dd47902dd7c308b3c36f63bce47e7b9a43"
# One exact source Sample ID has an official _X annotation filename.  Bind the
# documented alias by SHA rather than publishing a source sample identifier.
FILENAME_X_ALIAS_SHA256 = frozenset({"236ee366f9ff3f1fe0dd273f1b0b9f7cd317ecfdb0ba64ba85d24512d5a1a65c"})


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


LABEL_MAPPING_SHA256 = hashlib.sha256(canonical_json(LABEL_MAP).encode("utf-8")).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise AnnotationContractError(message)


def validate_protocol_source(protocol, actual_split_lock_sha):
    """Self-consistent historical receipts cannot replace the new source lock."""
    require(protocol.get("schema_version") == "sami_surv_stage_a_protocol_v1"
            and protocol.get("status") == "LOCKED_BEFORE_MODEL_FIT", "unexpected Stage A protocol")
    require(protocol.get("cohort", {}).get("split_lock_sha256") == actual_split_lock_sha,
            "actual source split SHA differs from locked Stage A protocol")
    source = protocol.get("source", {})
    expected = dict(annotation_archive_bytes=ANNOTATION_ARCHIVE["bytes"],
                    annotation_archive_md5=ANNOTATION_ARCHIVE["md5"],
                    annotation_archive_sha256=ANNOTATION_ARCHIVE["sha256"],
                    encoder="official_frozen_UNI2-h", encoder_revision=MODEL_REVISION,
                    encoder_weight_sha256=MODEL_SHA, encoder_feature_precision="FP32")
    require(all(source.get(k) == v for k, v in expected.items()),
            "protocol annotation or frozen encoder identity mismatch")


def under(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "source path escapes data root")
    return path


def parse_annotation_csv(content):
    """Accept one exact approved two-column schema; never normalize barcodes."""
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    columns = reader.fieldnames
    require(columns is not None and len(columns) == 2 and columns[0] == "Barcode"
            and columns[1] in ANNOTATION_COLUMNS, "unrecognized exact annotation CSV schema")
    annotation_column = columns[1]
    rows = {}
    for row in reader:
        require(set(row) == set(columns) and None not in row.values(), "malformed CSV row")
        barcode, label = row["Barcode"], row[annotation_column]
        require(bool(barcode) and barcode == barcode.strip(), "empty or whitespace-mutated barcode")
        require(barcode not in rows, "duplicate annotation barcode")
        require(label in LABEL_MAP, "unrecognized source annotation label")
        rows[barcode] = label
    require(bool(rows), "empty annotation CSV")
    return rows, annotation_column


def annotation_member(source_metadata, locked_row, names):
    """Use documented Sample ID, not row position or barcode resemblance.

    The Count_ prefix is an exact source metadata convention.  The original
    source archive has one explicit filename alias bound by its exact SHA.
    Neither operation modifies patient or spot identifiers.
    """
    require(source_metadata.get("id") == locked_row["section_id_private"], "wrong source section")
    require(locked_row["study_id"] == "Zenodo_7760264", "wrong source study")
    require(source_metadata.get("patient") == locked_row["patient_source_field_verified"], "wrong source donor")
    require(locked_row["donor_id_private"] == locked_row["study_id"] + ":" + source_metadata["patient"],
            "source donor does not bind the locked donor")
    require(source_metadata.get("st_technology") == "Visium" and locked_row["assay"] == "Visium",
            "Stage A supports locked Visium development sections only")
    require(locked_row["split"] in {"train", "validation"}, "original ST-test is prohibited")
    sample = source_metadata.get("Sample ID", "")
    require(re.fullmatch(r"(?:Count_)?SN\d+_A\d+_Rep[12]", sample) is not None,
            "unrecognized exact source Sample ID")
    source_name = sample.removeprefix("Count_")
    if hashlib.sha256(source_name.encode("utf-8")).hexdigest() in FILENAME_X_ALIAS_SHA256:
        source_name += "_X"
    member = "Pathology_SpotAnnotations/Pathologist_Annotations_" + source_name + ".csv"
    require(member in names, "source metadata has no exact official annotation member")
    return member


def align_annotations(barcodes, annotations):
    barcodes = np.asarray(barcodes)
    require(barcodes.ndim == 1 and barcodes.dtype.kind in "US", "barcodes must be a text vector")
    barcodes = barcodes.astype(str)
    require(len(set(barcodes)) == len(barcodes), "duplicate feature barcode")
    require(set(barcodes) == set(annotations), "annotation and feature barcode sets differ")
    require(all(label in LABEL_MAP for label in annotations.values()), "unrecognized source annotation label")
    raw = np.asarray([annotations[b] for b in barcodes])
    categories = np.asarray([LABEL_MAP[label] for label in raw])
    labels = np.asarray([CLASSES.index(c) if c in CLASSES else -1 for c in categories], dtype=np.int64)
    return raw, categories, labels


def validate_development_lock(lock, marker, lock_sha):
    require(lock.get("status") == "LOCKED_BEFORE_FEATURE_EXTRACTION_OR_TRAINING", "unlocked Phase22 source")
    require(marker.get("schema_version") == "phase22_external_st_qualification_v1"
            and marker.get("qualification_status") == "PASS_EXTERNAL_ST_DATA_CONTRACT"
            and marker.get("locked_split_sha") == lock_sha, "Phase22 source qualification mismatch")
    rows = lock["rows"]
    require(len(rows) == 16 and len({r["section_id_private"] for r in rows}) == 16,
            "expected finite 16-section original lock")
    donor_roles = {}
    for row in rows:
        donor, role = row["donor_id_private"], row["split"]
        require(role in {"train", "validation", "test"}, "unknown source split")
        require(donor not in donor_roles or donor_roles[donor] == role, "donor crosses original source splits")
        donor_roles[donor] = role
    dev = [r for r in rows if r["split"] in {"train", "validation"}]
    require(len(dev) == 12 and len({r["donor_id_private"] for r in dev}) == 6,
            "expected 12 sections and six development donors")
    require(all(r["study_id"] == "Zenodo_7760264" and r["assay"] == "Visium" for r in dev),
            "unexpected development study or assay")
    evidence = marker.get("sections_ready", [])
    require(len(evidence) == 16 and len({r["section_id_private"] for r in evidence}) == 16,
            "source qualification must cover 16 unique sections")
    evidence = {r["section_id_private"]: r for r in evidence}
    require(set(evidence) == {r["section_id_private"] for r in rows}, "source section coverage mismatch")
    return dev, evidence


def qualified_feature_section(root, row, evidence, lock_sha, repo):
    """Verify full-file hashes but decompress only barcodes/coords, never counts."""
    root, repo = Path(root), Path(repo)
    section = row["section_id_private"]
    for key in ("section_id_private", "donor_id_private", "study_id", "split", "assay"):
        require(evidence.get(key) == row[key], "qualified section identity mismatch")
    count_path = under(root, "processed/sections/" + section + ".npz")
    meta_path = under(root, evidence["metadata_path"])
    require(sha256(count_path) == evidence["npz_sha256"], "qualified source NPZ SHA mismatch")
    require(sha256(meta_path) == evidence["metadata_sha256"], "qualified source metadata SHA mismatch")
    meta = read_json(meta_path)
    require(meta.get("qualification_status") == "PASS_SECTION_DATA_CONTRACT"
            and meta.get("section_id_private") == section and meta.get("coordinate_unit") == "um"
            and meta.get("npz_sha256") == evidence["npz_sha256"]
            and meta.get("cache_signature", {}).get("locked_split_sha") == lock_sha
            and meta.get("source_files", {}).get("patches", {}).get("sha256") == evidence["patch_sha256"],
            "qualified physical lineage mismatch")
    source_record = meta["source_files"]["metadata"]
    source_path = under(root, source_record["path"])
    require(sha256(source_path) == source_record["sha256"], "raw source metadata SHA mismatch")
    source_metadata = read_json(source_path)
    footprint = meta["audit"]["image_footprint"]
    actual = float(footprint["actual_native_crop_view_um"])
    mpp = float(meta["audit"]["coordinates"]["mpp_x"])
    require(np.isfinite(actual) and np.isfinite(mpp) and mpp > 0 and
            abs(actual - 112.) <= .5 * mpp + 1e-6 and footprint["target_pixels"] == 224,
            "physical source-pixel crop qualification mismatch")
    with np.load(count_path, allow_pickle=False) as arr:
        barcodes = arr["barcodes"].astype(str)
        coords = arr["coords_um"].astype(np.float64)
    require(barcodes.ndim == 1 and len(set(barcodes)) == len(barcodes) and
            coords.shape == (len(barcodes), 2) and np.isfinite(coords).all(), "malformed barcode/coordinate cache")
    feature_path = under(root, "features/uni2-h/" + section + ".npz")
    receipt_path = feature_path.with_suffix(".json")
    receipt = read_json(receipt_path)
    expected = dict(schema_version="phase22_uni2_feature_section_v1", status="COMPLETE",
                    section_id_private=section, model="UNI2-h", model_sha256=MODEL_SHA,
                    model_revision=MODEL_REVISION, counts_npz_sha256=evidence["npz_sha256"],
                    split_lock_sha256=lock_sha, feature_dimension=1536, image_view_um_nominal=112.,
                    image_mpp_nominal=.5, image_pixels=224, source_patch_sha256=evidence["patch_sha256"],
                    frozen=True, qualification_metadata_sha256=evidence["metadata_sha256"],
                    actual_native_crop_view_um=actual, exact_112um=footprint["exact_112um"],
                    extractor_sha256=sha256(repo / "scripts/extract_phase22_uni2.py"),
                    normalization="official_RGB_uint8_div255_ImageNet_mean_std; no_resize_for_224square",
                    precision="fp32")
    require(all(receipt.get(k) == v for k, v in expected.items()), "UNI2 extraction receipt mismatch")
    require(receipt.get("precision_parity", {}).get("status") == "FP32_NO_REDUCED_PRECISION_GATE",
            "expected qualified full precision feature extraction")
    feature_sha = sha256(feature_path)
    require(receipt.get("features_sha256") == feature_sha, "UNI2 feature SHA mismatch")
    with np.load(feature_path, allow_pickle=False) as visual:
        X = visual["features"].copy()
        require(np.array_equal(visual["barcodes"].astype(str), barcodes)
                and np.array_equal(visual["coords_um"], coords), "feature/coordinate/barcode elementwise mismatch")
    require(X.dtype == np.float32 and X.shape == (len(barcodes), 1536) and np.isfinite(X).all(),
            "malformed frozen UNI2 features")
    provenance = dict(row, features_sha256=feature_sha, feature_receipt_sha256=sha256(receipt_path),
                      source_counts_npz_sha256=evidence["npz_sha256"],
                      qualification_metadata_sha256=evidence["metadata_sha256"],
                      raw_metadata_sha256=source_record["sha256"], source_sample=source_metadata["Sample ID"])
    return X, coords, barcodes, source_metadata, provenance


def source_coverage(records):
    """Aggregate only; never return section, patient, spot, or barcode values."""
    raw_counts, category_counts = Counter(), Counter()
    raw_donors, class_donors = defaultdict(set), defaultdict(set)
    raw_sections = defaultdict(set)
    for rec in records:
        for raw, category in zip(rec["raw_labels"], rec["categories"]):
            raw_counts[str(raw)] += 1
            category_counts[str(category)] += 1
            raw_donors[str(raw)].add(rec["donor"])
            raw_sections[str(raw)].add(rec["section"])
            class_donors[str(category)].add(rec["donor"])
    return dict(raw_label_coverage={k: dict(spots=v, donors=len(raw_donors[k]), sections=len(raw_sections[k]))
                                    for k, v in sorted(raw_counts.items())},
                primary_class_counts={c: category_counts[c] for c in CLASSES},
                primary_class_donors={c: len(class_donors[c]) for c in CLASSES},
                exclusion_counts={k: v for k, v in sorted(category_counts.items()) if k not in CLASSES})
