"""Failure-path and exact-alignment tests; synthetic arrays only."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.sami_surv.annotations import (
    ANNOTATION_ARCHIVE, ANNOTATION_COLUMNS, AnnotationContractError, CLASSES, LABEL_MAP, LABEL_MAPPING_SHA256,
    MODEL_REVISION, MODEL_SHA, align_annotations, annotation_member, canonical_json,
    parse_annotation_csv, qualified_feature_section, sha256, source_coverage,
    under, validate_development_lock, validate_protocol_source,
)


def csv_bytes(rows, column="Pathologist Annotation"):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["Barcode", column])
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def source_row():
    return dict(section_id_private="SYNTHETIC", study_id="Zenodo_7760264",
                patient_source_field_verified="Patient test", donor_id_private="Zenodo_7760264:Patient test",
                split="train", assay="Visium")


def source_metadata():
    return {"id": "SYNTHETIC", "patient": "Patient test", "st_technology": "Visium",
            "Sample ID": "Count_SN001_A123_Rep1"}


def original_lock():
    rows = []
    for d in range(6):
        for s in range(2):
            rows.append(dict(section_id_private=f"S{d}_{s}", donor_id_private=f"donor{d}",
                             study_id="Zenodo_7760264", assay="Visium", split="train" if d < 4 else "validation"))
    for s in range(4):
        rows.append(dict(section_id_private=f"T{s}", donor_id_private=f"testdonor{s//2}",
                         study_id="other", assay="Xenium", split="test"))
    lock = dict(status="LOCKED_BEFORE_FEATURE_EXTRACTION_OR_TRAINING", rows=rows)
    marker = dict(schema_version="phase22_external_st_qualification_v1",
                  qualification_status="PASS_EXTERNAL_ST_DATA_CONTRACT", locked_split_sha="lockedsha",
                  sections_ready=copy.deepcopy(rows))
    return lock, marker


class AnnotationTests(unittest.TestCase):
    def test_exact_supported_schemas(self):
        for column in ANNOTATION_COLUMNS:
            ann, found = parse_annotation_csv(csv_bytes([("ACGT-1", "tumor")], column))
            self.assertEqual(found, column)
            self.assertEqual(ann, {"ACGT-1": "tumor"})

    def test_new_capitalization_not_silently_accepted(self):
        with self.assertRaisesRegex(AnnotationContractError, "schema"):
            parse_annotation_csv(csv_bytes([("ACGT-1", "tumor")], "PATHOLOGIST ANNOTATION"))

    def test_duplicate_barcodes_fail_even_same_label(self):
        with self.assertRaisesRegex(AnnotationContractError, "duplicate"):
            parse_annotation_csv(csv_bytes([("ACGT-1", "tumor"), ("ACGT-1", "tumor")]))

    def test_unknown_label_not_other(self):
        with self.assertRaisesRegex(AnnotationContractError, "label"):
            parse_annotation_csv(csv_bytes([("ACGT-1", "lymphocyte")]))

    def test_blank_and_explicit_exclusion_not_negative_class(self):
        ann, _ = parse_annotation_csv(csv_bytes([("B", ""), ("A", "exclude")]))
        raw, categories, y = align_annotations(np.asarray(["A", "B"]), ann)
        self.assertEqual(raw.tolist(), ["exclude", ""])
        self.assertEqual(y.tolist(), [-1, -1])
        self.assertTrue(all(c.startswith("EXCLUDE_") for c in categories))

    def test_order_independent_exact_join(self):
        ann, _ = parse_annotation_csv(csv_bytes([("B-1", "tumor"), ("A-1", "tumor&stroma")]))
        raw, categories, y = align_annotations(np.asarray(["A-1", "B-1"]), ann)
        self.assertEqual(y.tolist(), [2, 0])
        self.assertEqual(raw.tolist(), ["tumor&stroma", "tumor"])

    def test_suffix_or_subset_match_forbidden(self):
        for barcodes in (["ACGT"], ["ACGT-1", "OTHER-1"]):
            with self.assertRaisesRegex(AnnotationContractError, "sets differ"):
                align_annotations(np.asarray(barcodes), {"ACGT-1": "tumor"})

    def test_duplicate_feature_barcodes_fail(self):
        with self.assertRaisesRegex(AnnotationContractError, "duplicate feature"):
            align_annotations(np.asarray(["A", "A"]), {"A": "tumor"})

    def test_no_whitespace_barcode_normalization(self):
        with self.assertRaisesRegex(AnnotationContractError, "whitespace"):
            parse_annotation_csv(csv_bytes([(" A-1", "tumor")]))

    def test_extra_or_missing_csv_values_fail(self):
        for content in (b'Barcode,Pathologist Annotation\nA,tumor,extra\n',
                        b'Barcode,Pathologist Annotation\nA\n'):
            with self.assertRaisesRegex(AnnotationContractError, "malformed"):
                parse_annotation_csv(content)

    def test_mapping_locked_as_finite_exact_dictionary(self):
        self.assertEqual(len(LABEL_MAP), 28)
        self.assertEqual(LABEL_MAPPING_SHA256, "1e213fb24f7006c2672013d3f32dc8f712e76e921a3dfe89f5739eebf2abf940")
        self.assertEqual(hashlib.sha256(canonical_json(LABEL_MAP).encode()).hexdigest(), LABEL_MAPPING_SHA256)
        self.assertEqual(CLASSES, ("tumor", "stroma", "tumor_stroma_mixed"))

    def test_exact_metadata_source_mapping(self):
        member = "Pathology_SpotAnnotations/Pathologist_Annotations_SN001_A123_Rep1.csv"
        self.assertEqual(annotation_member(source_metadata(), source_row(), [member]), member)

    def test_wrong_section_donor_test_or_study_fail(self):
        member = "Pathology_SpotAnnotations/Pathologist_Annotations_SN001_A123_Rep1.csv"
        for which in ("section", "donor", "donor_lock", "split", "study"):
            row, meta = source_row(), source_metadata()
            if which == "section": meta["id"] = "WRONG"
            if which == "donor": meta["patient"] = "WRONG"
            if which == "donor_lock": row["donor_id_private"] = "WRONG"
            if which == "split": row["split"] = "test"
            if which == "study": row["study_id"] = "WRONG"
            with self.assertRaises(AnnotationContractError):
                annotation_member(meta, row, [member])

    def test_ambiguous_partial_source_name_fail(self):
        meta = source_metadata()
        meta["Sample ID"] = "SN001_A123"
        with self.assertRaisesRegex(AnnotationContractError, "Sample ID"):
            annotation_member(meta, source_row(), [])

    def test_explicit_published_filename_alias(self):
        meta = source_metadata()
        meta["Sample ID"] = "SN001_A123_Rep1"
        synthetic_sha = hashlib.sha256(meta["Sample ID"].encode()).hexdigest()
        member = "Pathology_SpotAnnotations/Pathologist_Annotations_SN001_A123_Rep1_X.csv"
        with patch("src.sami_surv.annotations.FILENAME_X_ALIAS_SHA256", frozenset({synthetic_sha})):
            self.assertEqual(annotation_member(meta, source_row(), [member]), member)
        with self.assertRaisesRegex(AnnotationContractError, "no exact official"):
            annotation_member(meta, source_row(), [member])

    def test_original_lock_selects_only_development(self):
        lock, marker = original_lock()
        rows, _ = validate_development_lock(lock, marker, "lockedsha")
        self.assertEqual(len(rows), 12)
        self.assertNotIn("test", {r["split"] for r in rows})

    def test_cross_donor_split_rejected(self):
        lock, marker = original_lock()
        lock["rows"][-1]["donor_id_private"] = lock["rows"][0]["donor_id_private"]
        with self.assertRaisesRegex(AnnotationContractError, "crosses"):
            validate_development_lock(lock, marker, "lockedsha")

    def test_missing_source_qualification_and_wrong_hash_rejected(self):
        lock, marker = original_lock()
        with self.assertRaisesRegex(AnnotationContractError, "mismatch"):
            validate_development_lock(lock, marker, "different")
        marker["sections_ready"].pop()
        with self.assertRaisesRegex(AnnotationContractError, "16 unique"):
            validate_development_lock(lock, marker, "lockedsha")

    def test_aggregate_does_not_return_identity(self):
        rec = dict(section="private-section", donor="private-donor", raw_labels=np.asarray(["tumor", "exclude"]),
                   categories=np.asarray(["tumor", "EXCLUDE_SOURCE_PATHOLOGIST"]))
        summary = source_coverage([rec])
        self.assertNotIn("private-", json.dumps(summary))
        self.assertEqual(summary["primary_class_counts"]["tumor"], 1)

    def test_source_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(AnnotationContractError, "escapes"):
                under(Path(temp), "../raw_patient_data")

    @staticmethod
    def source_protocol():
        return dict(schema_version="sami_surv_stage_a_protocol_v1", status="LOCKED_BEFORE_MODEL_FIT",
                    cohort={"split_lock_sha256": "lockedsha"},
                    source=dict(annotation_archive_bytes=ANNOTATION_ARCHIVE["bytes"],
                                annotation_archive_md5=ANNOTATION_ARCHIVE["md5"],
                                annotation_archive_sha256=ANNOTATION_ARCHIVE["sha256"],
                                encoder="official_frozen_UNI2-h", encoder_revision=MODEL_REVISION,
                                encoder_weight_sha256=MODEL_SHA, encoder_feature_precision="FP32"))

    def test_protocol_accepts_exact_source_lock(self):
        validate_protocol_source(self.source_protocol(), "lockedsha")

    def test_self_consistent_historical_receipt_cannot_replace_new_lock(self):
        lock, marker = original_lock()
        marker["locked_split_sha"] = "substitutedsha"
        # The original validator is intentionally only a historical-consistency
        # check.  The new protocol independently rejects this replacement.
        validate_development_lock(lock, marker, "substitutedsha")
        with self.assertRaisesRegex(AnnotationContractError, "locked Stage A protocol"):
            validate_protocol_source(self.source_protocol(), "substitutedsha")

    def test_protocol_rejects_changed_archive_or_encoder(self):
        for key in ("annotation_archive_bytes", "annotation_archive_md5", "annotation_archive_sha256",
                    "encoder", "encoder_revision", "encoder_weight_sha256", "encoder_feature_precision"):
            protocol = self.source_protocol()
            protocol["source"][key] = "changed"
            with self.assertRaisesRegex(AnnotationContractError, "identity mismatch"):
                validate_protocol_source(protocol, "lockedsha")

    def test_protocol_must_be_pre_fit_locked(self):
        protocol = self.source_protocol()
        protocol["status"] = "DRAFT"
        with self.assertRaisesRegex(AnnotationContractError, "protocol"):
            validate_protocol_source(protocol, "lockedsha")


class FeatureQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        extractor = self.repo / "scripts/extract_phase22_uni2.py"
        extractor.write_text("# synthetic fixture", encoding="utf-8")
        self.row = source_row()
        self.feature = self.root / "features/uni2-h/SYNTHETIC.npz"
        self.feature.parent.mkdir(parents=True)
        count = self.root / "processed/sections/SYNTHETIC.npz"
        count.parent.mkdir(parents=True)
        self.coords = np.asarray([[1., 2.], [3., 4.]])
        # Object counts cannot be accessed with allow_pickle=False. Qualification
        # must succeed without evaluating this member: only source SHA is used.
        np.savez(count, counts=np.asarray([object()], dtype=object), barcodes=np.asarray(["A", "B"]), coords_um=self.coords)
        np.savez(self.feature, features=np.zeros((2, 1536), dtype=np.float32),
                 barcodes=np.asarray(["A", "B"]), coords_um=self.coords)
        raw = self.root / "source.json"
        raw.write_text(json.dumps(source_metadata()), encoding="utf-8")
        meta = count.with_suffix(".json")
        meta.write_text(json.dumps(dict(qualification_status="PASS_SECTION_DATA_CONTRACT", section_id_private="SYNTHETIC",
                       coordinate_unit="um", npz_sha256=sha256(count), cache_signature={"locked_split_sha": "lock"},
                       source_files={"patches": {"sha256": "patch"}, "metadata": {"path": "source.json", "sha256": sha256(raw)}},
                       audit={"image_footprint": {"actual_native_crop_view_um": 112., "target_pixels": 224, "exact_112um": True},
                              "coordinates": {"mpp_x": .5}})), encoding="utf-8")
        self.evidence = dict(self.row, metadata_path="processed/sections/SYNTHETIC.json", metadata_sha256=sha256(meta),
                             npz_sha256=sha256(count), patch_sha256="patch")
        self.receipt = dict(schema_version="phase22_uni2_feature_section_v1", status="COMPLETE", section_id_private="SYNTHETIC",
                            model="UNI2-h", model_sha256=MODEL_SHA, model_revision=MODEL_REVISION,
                            counts_npz_sha256=sha256(count), split_lock_sha256="lock", feature_dimension=1536,
                            image_view_um_nominal=112., image_mpp_nominal=.5, image_pixels=224, source_patch_sha256="patch",
                            frozen=True, qualification_metadata_sha256=sha256(meta), actual_native_crop_view_um=112.,
                            exact_112um=True, extractor_sha256=sha256(extractor),
                            normalization="official_RGB_uint8_div255_ImageNet_mean_std; no_resize_for_224square",
                            precision="fp32", precision_parity={"status": "FP32_NO_REDUCED_PRECISION_GATE"},
                            features_sha256=sha256(self.feature))
        self.save_receipt()

    def tearDown(self):
        self.temp.cleanup()

    def save_receipt(self):
        self.feature.with_suffix(".json").write_text(json.dumps(self.receipt), encoding="utf-8")

    def qualify(self):
        return qualified_feature_section(self.root, self.row, self.evidence, "lock", self.repo)

    def test_qualified_features_without_reading_expression_member(self):
        X, coords, barcodes, _, provenance = self.qualify()
        self.assertEqual(X.shape, (2, 1536))
        self.assertEqual(barcodes.tolist(), ["A", "B"])
        np.testing.assert_array_equal(coords, self.coords)
        self.assertEqual(provenance["features_sha256"], sha256(self.feature))

    def test_feature_bytes_mutation_rejected(self):
        with self.feature.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(AnnotationContractError, "SHA mismatch"):
            self.qualify()

    def test_wrong_elementwise_coordinate_or_barcode_even_with_updated_sha(self):
        for coords, barcodes in ((self.coords[::-1], ["A", "B"]), (self.coords, ["B", "A"])):
            np.savez(self.feature, features=np.zeros((2, 1536), dtype=np.float32),
                     barcodes=np.asarray(barcodes), coords_um=coords)
            self.receipt["features_sha256"] = sha256(self.feature)
            self.save_receipt()
            with self.assertRaisesRegex(AnnotationContractError, "elementwise"):
                self.qualify()

    def test_nonfinite_or_wrong_dtype_features_fail(self):
        for dtype, value in ((np.float32, np.nan), (np.float64, 0)):
            np.savez(self.feature, features=np.full((2, 1536), value, dtype=dtype),
                     barcodes=np.asarray(["A", "B"]), coords_um=self.coords)
            self.receipt["features_sha256"] = sha256(self.feature)
            self.save_receipt()
            with self.assertRaisesRegex(AnnotationContractError, "malformed"):
                self.qualify()


if __name__ == "__main__":
    unittest.main()
