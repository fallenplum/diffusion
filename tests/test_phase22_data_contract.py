"""Source-scale, identity, physical-alignment, and resumability risk checks."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import sparse

from scripts.qualify_phase22_spatial_data import qualify
from src.phase22_spatial_bridge.data_contract import (
    DataContractError, file_sha256, fixed_gene_vocabulary,
    qualify_patch_pair, read_h5ad_raw, validate_split_lock, xenium_capture_footprint,
)


def _source(tmp_path, dense=False, *, section="SYNTHETIC_ONLY",
            raw_patient="Patient synthetic", ledger_patient=None, subseries="Synthetic fixture"):
    ledger_patient = raw_patient if ledger_patient is None else ledger_patient
    root = tmp_path/"private_data"
    for directory in ("raw/hest/st", "raw/hest/patches", "raw/hest/metadata", "metadata/hest"):
        (root/directory).mkdir(parents=True)
    counts = np.array([[0, 3, 8], [5, 0, 1]], dtype=np.int32)
    centers = np.array([[200., 300.], [500., 600.]])
    expression_path = root/f"raw/hest/st/{section}.h5ad"
    with h5py.File(expression_path, "w") as handle:
        if dense:
            handle.create_dataset("X", data=counts)
        else:
            matrix = sparse.csr_matrix(counts)
            group = handle.create_group("X")
            group.attrs["encoding-type"] = "csr_matrix"
            group.attrs["shape"] = counts.shape
            group.create_dataset("data", data=matrix.data)
            group.create_dataset("indices", data=matrix.indices)
            group.create_dataset("indptr", data=matrix.indptr)
        obs, var = handle.create_group("obs"), handle.create_group("var")
        obs.attrs["_index"], var.attrs["_index"] = "_index", "_index"
        obs.create_dataset("_index", data=np.asarray([b"spot-A", b"spot-B"]))
        var.create_dataset("_index", data=np.asarray([b"A", b"B", b"Blank-1"]))
        handle.create_dataset("obsm/spatial", data=centers)
        handle.create_group("uns")
    patch_path = root/f"raw/hest/patches/{section}.h5"
    with h5py.File(patch_path, "w") as handle:
        handle.create_dataset("barcode", data=np.asarray([[b"spot-B"], [b"spot-A"]]))
        handle.create_dataset("coords", data=centers[::-1]-112.)
        image = handle.create_dataset("img", data=np.zeros((2, 224, 224, 3), dtype=np.uint8))
        for key, value in dict(pixel_size=.5, patch_size_src=224, patch_size_target=224, downsample=1.).items():
            image.attrs[key] = value
    metadata = dict(id=section, patient=raw_patient, subseries=subseries,
                    st_technology="Visium", pixel_size_um_estimated=.5,
                    spot_diameter=55., license="SYNTHETIC_FIXTURE_NOT_RESEARCH_DATA", download_page_link1="synthetic")
    metadata_path = root/f"raw/hest/metadata/{section}.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    ledger_path = root/"metadata/hest/HEST_v1_3_0.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "patient", "subseries"])
        writer.writeheader()
        writer.writerow(dict(id=section, patient=ledger_patient, subseries=subseries))
    identity = dict(section_id_private=section, study_id="synthetic-study", donor_id_private="synthetic-donor",
                    split="train", assay="Visium", patient_source_field_verified=ledger_patient)
    (root/"metadata/CRC16_ST_DONOR_SPLIT_LOCK_PRIVATE.json").write_text(
        json.dumps(dict(rows=[identity], metadata_sha256=file_sha256(ledger_path))), encoding="utf-8")
    files = []
    for relative in (f"st/{section}.h5ad", f"patches/{section}.h5", f"metadata/{section}.json"):
        path = root/"raw/hest"/relative
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=file_sha256(path)))
    (root/"metadata/CRC16_QUALIFICATION_ASSET_PLAN.json").write_text(json.dumps(dict(repo="synthetic-fixture", revision="0"*40, files=files)), encoding="utf-8")
    target_path = root/"metadata/TARGET_CONSERVATIVE_MEASURED_GENES_PRIVATE.json"
    target_path.write_text(json.dumps(dict(gene_symbols_private=["A", "B", "Blank-1"])), encoding="utf-8")
    panel_path = root/"metadata/PANEL_SYNTHETIC.json"
    panel_path.write_text(json.dumps(dict(shared_genes_private=["A", "B", "Blank-1"])), encoding="utf-8")
    return root, section, expression_path, patch_path, metadata, panel_path


def test_csr_dense_raw_count_zero_parity(tmp_path):
    one = _source(tmp_path/"csr")
    two = _source(tmp_path/"dense", dense=True)
    csr, dense = read_h5ad_raw(one[2]), read_h5ad_raw(two[2])
    np.testing.assert_array_equal(csr["matrix"].toarray(), dense["matrix"])
    assert csr["audit"]["source_measured_zero_entries"] == 2
    assert csr["audit"]["control_like_source_genes"] == 1
    assert dense["audit"]["raw_integer_nonnegative_verified"]


def test_elementwise_barcode_mapping_preserves_every_patch(tmp_path):
    _, _, expression_path, patch_path, metadata, _ = _source(tmp_path)
    arrays, audit = qualify_patch_pair(read_h5ad_raw(expression_path), patch_path, metadata, np.asarray(["A", "B"]))
    np.testing.assert_array_equal(arrays["barcodes"], ["spot-B", "spot-A"])
    np.testing.assert_array_equal(arrays["counts"], [[5, 0], [0, 3]])
    np.testing.assert_array_equal(arrays["coords_um"], [[250., 300.], [100., 150.]])
    assert arrays["counts"].dtype == np.int64
    assert audit["all_patches_retained"] and audit["aligned_barcode_elementwise"]
    assert audit["image_footprint"]["nominal_view_um"] == 112.
    assert audit["expression_capture_footprint"]["nominal_diameter_um"] == 55.


def test_missing_patch_barcode_fails_closed(tmp_path):
    _, _, expression_path, patch_path, metadata, _ = _source(tmp_path)
    with h5py.File(patch_path, "a") as handle:
        handle["barcode"][0, 0] = b"absent"
    with pytest.raises(DataContractError, match="lacks a source count"):
        qualify_patch_pair(read_h5ad_raw(expression_path), patch_path, metadata, np.asarray(["A", "B"]))


def test_footprint_rounding_cannot_silently_expand_tolerance(tmp_path):
    _, _, expression_path, patch_path, metadata, _ = _source(tmp_path)
    with h5py.File(patch_path, "a") as handle:
        handle["img"].attrs["patch_size_src"] = 225
    with pytest.raises(DataContractError, match="half-native-pixel"):
        qualify_patch_pair(read_h5ad_raw(expression_path), patch_path, metadata, np.asarray(["A", "B"]))


def test_normalized_or_log_history_without_raw_counts_fails(tmp_path):
    _, _, expression_path, _, _, _ = _source(tmp_path, dense=True)
    with h5py.File(expression_path, "a") as handle:
        values = handle["X"][:].astype(float)
        del handle["X"]
        handle.create_dataset("X", data=np.log1p(values))
    with pytest.raises(DataContractError, match="integer raw counts"):
        read_h5ad_raw(expression_path)
    with h5py.File(expression_path, "a") as handle:
        handle.create_group("uns/log1p")
    with pytest.raises(DataContractError, match="log1p preprocessing"):
        read_h5ad_raw(expression_path)


def test_source_donor_cannot_cross_splits_or_repeat_section():
    row = dict(section_id_private="synthetic-A", study_id="synthetic-study", donor_id_private="synthetic-donor", split="train", assay="Visium")
    with pytest.raises(DataContractError, match="crosses data splits"):
        validate_split_lock(dict(rows=[row, dict(row, section_id_private="synthetic-B", split="test")]))
    with pytest.raises(DataContractError, match="duplicate section"):
        validate_split_lock(dict(rows=[row, row]))


def test_fixed_measurement_vocabulary_removes_controls_without_expression_selection(tmp_path):
    root, _, _, _, _, panel_path = _source(tmp_path)
    genes, audit = fixed_gene_vocabulary(panel_path, root/"metadata/TARGET_CONSERVATIVE_MEASURED_GENES_PRIVATE.json")
    np.testing.assert_array_equal(genes, ["A", "B"])
    assert audit["panel_control_names_excluded"] == 1
    assert "no_expression_value_selection" in audit["rule"]


def test_missing_download_pending_then_arrival_is_idempotent_and_transferable(tmp_path):
    root, section, _, patch_path, _, panel_path = _source(tmp_path)
    args = argparse.Namespace(data_root=root, split_lock=None, asset_plan=None, target_genes=None,
                              panel_genes=panel_path, public_report=tmp_path/"PUBLIC_AGGREGATE.json", only_section=None)
    moved = patch_path.with_suffix(".not_arrived")
    patch_path.rename(moved)
    pending = qualify(args)
    assert pending["qualification_status"] == "PENDING_DATA_ARRIVAL" and not pending["sections_ready"]
    moved.rename(patch_path)
    first = qualify(args)
    assert first["qualification_status"] == "PASS_EXTERNAL_ST_DATA_CONTRACT" and len(first["sections_ready"]) == 1
    marker = root/"processed/DATA_QUALIFICATION_PRIVATE.json"
    original = marker.read_bytes()
    repeated = qualify(args)
    assert marker.read_bytes() == original
    ready = repeated["sections_ready"][0]
    assert ready["npz_path"] == f"processed/sections/{section}.npz"
    assert not Path(ready["patch_path"]).is_absolute()
    with np.load(root/ready["npz_path"], allow_pickle=False) as cache:
        assert set(cache.files) == {"counts", "gene_names", "coords_um", "barcodes", "measured_mask"}
        assert cache["measured_mask"].all()
    public = json.loads(args.public_report.read_text())
    assert "sections_ready" in public and public["sections_ready"] == 1
    assert "synthetic-donor" not in args.public_report.read_text()


@pytest.mark.parametrize("section,patient,subseries", [
    ("TENX147", "Patient 5", "Xenium In Situ, Sample P5 CRC"),
    ("TENX148", "Patient 2", "Xenium In Situ, Sample P2 CRC"),
    ("TENX152", "Patient 2", "Visium CytAssist v2, Sample P2 CRC"),
])
def test_known_stale_patient_field_uses_pinned_ledger_without_rewriting_raw_source(
        tmp_path, section, patient, subseries):
    # Synthetic counts/images isolate the identity check; these are not real
    # source measurements and make no claim about the Xenium footprint.
    root, _, _, _, _, panel_path = _source(tmp_path, section=section,
        raw_patient="Patient 1", ledger_patient=patient, subseries=subseries)
    raw_path = root/f"raw/hest/metadata/{section}.json"
    raw_before = raw_path.read_bytes()
    args = argparse.Namespace(data_root=root, split_lock=None, asset_plan=None, target_genes=None,
                              panel_genes=panel_path, public_report=tmp_path/"PUBLIC.json", only_section=None)
    marker = qualify(args)
    assert marker["qualification_status"] == "PASS_EXTERNAL_ST_DATA_CONTRACT"
    ready = marker["sections_ready"][0]
    record = json.loads((root/ready["metadata_path"]).read_text(encoding="utf-8"))
    adjudication = record["donor_identity_source_adjudication"]
    assert adjudication["status"] == "RESOLVED_FIXED_CSV_AND_ORIGINAL_STUDY_NOT_RAW_METADATA_REWRITE"
    assert adjudication["original_JSON_patient_private"] == "Patient 1"
    assert adjudication["original_JSON_subseries_private"] == subseries
    assert adjudication["authoritative_CSV_patient_private"] == patient
    assert adjudication["authoritative_CSV_sha256"] == file_sha256(root/"metadata/hest/HEST_v1_3_0.csv")
    assert adjudication["original_JSON_sha256"] == file_sha256(raw_path)
    assert raw_path.read_bytes() == raw_before
    assert "donor_identity_source_adjudication" not in json.loads(args.public_report.read_text())


@pytest.mark.parametrize("section,subseries", [
    ("SYNTHETIC_UNREVIEWED", "Xenium In Situ, Sample P5 CRC"),
    ("TENX147", "Xenium In Situ, Sample P2 CRC"),
])
def test_unreviewed_or_subseries_mismatched_patient_conflict_is_rejected(tmp_path, section, subseries):
    root, _, _, _, _, panel_path = _source(tmp_path, section=section,
        raw_patient="Patient 1", ledger_patient="Patient 5", subseries=subseries)
    args = argparse.Namespace(data_root=root, split_lock=None, asset_plan=None, target_genes=None,
                              panel_genes=panel_path, public_report=tmp_path/"PUBLIC.json", only_section=None)
    marker = qualify(args)
    assert marker["qualification_status"] == "FAIL_DATA_CONTRACT"
    assert not marker["sections_ready"]
    assert len(marker["sections_failed"]) == 1
    assert "unresolved source patient identity discrepancy" in marker["sections_failed"][0]["reason"]


def test_source_donor_ledger_content_drift_rejected_before_source_qualification(tmp_path):
    root, _, _, _, _, panel_path = _source(tmp_path)
    ledger_path = root/"metadata/hest/HEST_v1_3_0.csv"
    ledger_path.write_bytes(ledger_path.read_bytes()+b"\n")
    args = argparse.Namespace(data_root=root, split_lock=None, asset_plan=None, target_genes=None,
                              panel_genes=panel_path, public_report=tmp_path/"PUBLIC.json", only_section=None)
    with pytest.raises(DataContractError, match="prelocked CSV SHA"):
        qualify(args)


@pytest.mark.parametrize("section,subseries", [
    ("TENX147", "Xenium In Situ, Sample P5 CRC"),
    ("TENX148", "Xenium In Situ, Sample P2 CRC"),
    ("TENX149", "Xenium In Situ, Sample P1 CRC"),
])
def test_reviewed_xenium_marker_conflict_has_explicit_capture_and_preserves_inputs(section, subseries):
    metadata = dict(id=section, subseries=subseries, st_technology="Xenium",
                    spot_diameter=55., inter_spot_dist=100.)
    original = json.dumps(metadata, sort_keys=True)
    coordinates = np.array([[22.5, 77.25], [122.5, 77.25], [22.5, 177.25]])
    original_coordinates = coordinates.copy()
    audit = xenium_capture_footprint(metadata, coordinates)
    assert audit["nominal_side_um"] == 100.
    assert audit["nominal_area_um2"] == 10000.
    assert audit["original_JSON_spot_diameter_um"] == 55.
    assert audit["rendering_size_metadata_conflict"]
    assert audit["grid_lattice_max_deviation_um"] < 1e-10
    assert audit["original_transcript_mass_conservation_verified"] is False
    assert audit["source_code_revision"] == "3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3"
    assert audit["not_equal_to_image_square"]
    assert json.dumps(metadata, sort_keys=True) == original
    np.testing.assert_array_equal(coordinates, original_coordinates)


@pytest.mark.parametrize("coordinates", [
    np.array([[0., 0.], [100., 0.], [0., 150.]]),
    np.array([[0., 0.], [100., 0.], [100., 0.]]),
])
def test_xenium_reviewed_source_does_not_bypass_physical_grid_or_duplicate_checks(coordinates):
    metadata = dict(id="TENX147", subseries="Xenium In Situ, Sample P5 CRC",
                    st_technology="Xenium", spot_diameter=55., inter_spot_dist=100.)
    with pytest.raises(DataContractError, match="unique 100um pooled-grid"):
        xenium_capture_footprint(metadata, coordinates)


@pytest.mark.parametrize("changes", [
    {"id": "SYNTHETIC_UNREVIEWED_XENIUM"},
    {"subseries": "Xenium In Situ, Sample P2 CRC"},
    {"inter_spot_dist": 95.},
])
def test_xenium_55um_capture_override_is_closed_to_unreviewed_sources(changes):
    metadata = dict(id="TENX147", subseries="Xenium In Situ, Sample P5 CRC",
                    st_technology="Xenium", spot_diameter=55., inter_spot_dist=100.)
    metadata.update(changes)
    with pytest.raises(DataContractError, match="unreviewed Xenium capture-size"):
        xenium_capture_footprint(metadata, np.array([[0., 0.], [100., 0.], [0., 100.]]))


def _partial_measurement_source(tmp_path, role):
    root, section, _, _, _, panel_path = _source(tmp_path)
    lock_path = root/"metadata/CRC16_ST_DONOR_SPLIT_LOCK_PRIVATE.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["rows"][0]["split"] = role
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    # The frozen vocabulary includes a known unmeasured name. Never remove it
    # after seeing a test section or interpret its storage fill as a count.
    target_path = root/"metadata/TARGET_CONSERVATIVE_MEASURED_GENES_PRIVATE.json"
    target_path.write_text(json.dumps(dict(gene_symbols_private=["A", "B", "SYNTHETIC_UNMEASURED"])), encoding="utf-8")
    panel_path.write_text(json.dumps(dict(shared_genes_private=["A", "B", "SYNTHETIC_UNMEASURED"])), encoding="utf-8")
    args = argparse.Namespace(data_root=root, split_lock=None, asset_plan=None, target_genes=None,
                              panel_genes=panel_path, public_report=tmp_path/"PUBLIC.json", only_section=None)
    return root, section, args


def test_partial_test_measurement_keeps_frozen_vocabulary_and_mask_but_excludes_full_projection(tmp_path):
    root, section, args = _partial_measurement_source(tmp_path, "test")
    marker = qualify(args)
    assert marker["qualification_status"] == "PASS_EXTERNAL_ST_DATA_CONTRACT"
    assert marker["fixed_gene_count"] == 3
    ready = marker["sections_ready"][0]
    with np.load(root/ready["npz_path"], allow_pickle=False) as values:
        np.testing.assert_array_equal(values["gene_names"], ["A", "B", "SYNTHETIC_UNMEASURED"])
        np.testing.assert_array_equal(values["measured_mask"], [True, True, False])
        assert values["counts"].shape == (2, 3)
        assert (values["counts"][:, 2] == 0).all()
    record = json.loads((root/ready["metadata_path"]).read_text(encoding="utf-8"))
    assert record["full_dictionary_measurement_projection_eligible"] is False
    assert "B_full_composition_projection_ineligible" in record["partial_test_measurement_policy"]
    assert "no_test_based_dictionary_gene_reselection" in record["partial_test_measurement_policy"]
    assert record["audit"]["unmeasured_storage_fill_is_not_expression"]
    assert not list((root/"processed/sections").glob("*.pt"))


@pytest.mark.parametrize("role", ["train", "validation"])
def test_partial_training_or_validation_measurement_fails_without_vocabulary_reselection(tmp_path, role):
    root, _, args = _partial_measurement_source(tmp_path, role)
    panel_before = args.panel_genes.read_bytes()
    marker = qualify(args)
    assert marker["qualification_status"] == "FAIL_DATA_CONTRACT"
    assert marker["fixed_gene_count"] == 3
    assert not marker["sections_ready"]
    assert "training/validation must measure" in marker["sections_failed"][0]["reason"]
    assert args.panel_genes.read_bytes() == panel_before
    assert not list((root/"processed/sections").glob("*.npz"))
