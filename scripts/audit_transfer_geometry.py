"""Outcome-blind, anonymous inventory of source/TCGA physical-support limits.

Reads H5 metadata and coordinate validity, never endpoints, expression or image
features. Physical units are not inferred from objective magnification.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(assets_path, source_cache, fresh_receipt, history_path):
    assets = json.loads(Path(assets_path).read_text(encoding="utf-8-sig"))
    files = sorted(Path(assets["uni2_root"]).rglob("*.h5"))
    if not files:
        raise ValueError("NO_TCGA_UNI2_H5_FILES")
    geometries = collections.Counter()
    patches = invalid = mismatch = mpp_files = 0
    allowed = ("patch_size", "patch_size_level", "patch_size_level0",
               "patch_level", "overlap", "target_magnification",
               "level0_magnification", "mpp", "mpp_x", "mpp_y")
    attrs_hashes = []
    for path in files:
        with h5py.File(path, "r") as h:
            ds = h["coords"]
            coords = np.asarray(ds).reshape(-1, 2)
            n = int(np.prod(h["features"].shape[:-1]))
            if h["features"].shape[-1] != 1536:
                raise ValueError("TCGA_UNI2_FEATURE_DIMENSION_CHANGED")
            mismatch += int(n != len(coords))
            invalid += int((~np.isfinite(coords).all(axis=1)).sum())
            patches += n
            attrs = {k: str(ds.attrs[k]) for k in allowed if k in ds.attrs}
            key = json.dumps(attrs, sort_keys=True)
            geometries[key] += 1
            attrs_hashes.append(key)
            mpp_files += int(any(k in attrs for k in ("mpp", "mpp_x", "mpp_y")))
    if invalid or mismatch:
        raise ValueError("NONFINITE_OR_MISALIGNED_TCGA_COORDINATES")
    manifest = Path(source_cache) / "manifest.private.json"
    source = json.loads(manifest.read_text())["source_audit"]
    fresh = json.loads(Path(fresh_receipt).read_text())
    if fresh["status"] != "PASS_FRESH_RAW_FULL_LIBRARY_AND_EXISTING_CACHE_REPLAY":
        raise ValueError("FRESH_SOURCE_REQUALIFICATION_NOT_PASS")
    return {
        "schema": "source_to_tcga_geometry_manifest_v1",
        "status": "PASS_METADATA_AND_COORDINATE_VALIDITY_PHYSICAL_EQUIVALENCE_UNPROVEN",
        "DSS_PFI_or_gene_values_read": False,
        "source": {
            "donors": source["source_donors"], "sections": source["source_sections"],
            "spots": source["primary_spots"], "image_feature_dimension": 1536,
            "image_feature_unit": source["feature_normalization"]["unit"],
            "image_support_nominal_micrometers": [112, 112],
            "image_support_nominal_pixels": [224, 224],
            "image_support_nominal_mpp": .5,
            "image_support_evidence": "historical_extraction_contract_not_new_pixel_remeasurement",
            "molecular_support": "Visium_nominal_55um_capture_circle_not_112um_image_square",
            "other_platforms": "Xenium_100um_square_pooling_requires_file_specific_qualification_not_55um_plotting_diameter",
            "radius_um": source["radius_um"],
            "maximum_neighbors_including_self": source["maximum_neighbors_including_self"],
            "cross_section_edges": source["cross_section_edges"],
            "new_raw_pixel_or_transcript_level_geometry_replay": False,
            "source_cache_manifest_sha256": digest(manifest),
            "fresh_raw_expression_requalification_sha256": digest(fresh_receipt),
            "historical_geometry_report_sha256": digest(history_path),
        },
        "TCGA_UNI2_inventory": {
            "slides": len(files), "patches": patches, "dimension": 1536,
            "geometry_attribute_groups": [dict(attributes=json.loads(k), slides=v)
                                          for k, v in sorted(geometries.items())],
            "coordinate_feature_length_mismatches": mismatch,
            "nonfinite_coordinate_rows": invalid,
            "H5_files_with_coordinate_MPP_attribute": mpp_files,
            "coordinate_unit": "native_level0_pixels_from_H5_extraction_metadata",
            "micrometer_equivalence_to_source": "UNVERIFIED_NO_PER_SLIDE_MPP_CONTRACT",
            "geometry_attributes_multiset_sha256": hashlib.sha256(
                json.dumps(sorted(attrs_hashes), separators=(",", ":")).encode()).hexdigest(),
            "cross_encoder_row_index_pairing_used": False,
            "patient_specific_identifiers_coordinates_or_file_paths_exported": False,
        },
        "transfer_limits": [
            "Same_UNI2_encoder_dimension_does_not_prove_equal_physical_field_of_view",
            "Do_not_apply_300um_source_neighborhood_to_unscaled_TCGA_pixel_coordinates",
            "Source_spot_and_TCGA_bulk_RNA_are_different_measurement_supports",
            "R8_R6_S7_source_has_no_spatial_GNN_and_uses_equal_nonempty_WSI_pooling",
            "SPGEL_xy_um_requires_caller_verified_micrometers_API_cannot_detect_mislabeled_pixel_coordinates",
            "No_local_patch_gene_correspondence_in_TCGA_is_claimed",
        ],
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    for k in ("assets", "source-cache", "fresh-receipt", "history", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    result = audit(a.assets, a.source_cache, a.fresh_receipt, a.history)
    if a.output.exists():
        raise FileExistsError("GEOMETRY_RECEIPT_EXISTS_NO_OVERWRITE")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output_sha256": digest(a.output),
                      "slides": result["TCGA_UNI2_inventory"]["slides"],
                      "patches": result["TCGA_UNI2_inventory"]["patches"]}))


if __name__ == "__main__":
    main()
