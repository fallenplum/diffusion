"""HEST raw-count / patch qualification without survival outcomes.

Source expression remains raw count data.  Fixed gene vocabulary uses declared
measurement names only, with no variance, expression, or outcome selection.
Only private runtime manifests contain section, donor, gene, or barcode IDs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
from scipy import sparse


class DataContractError(ValueError):
    pass


CONTROL_PATTERN = re.compile(r"^(?:blank|neg(?:ative)?control|unassignedcodeword|deprecatedcodeword|controlprobe|clonotype)", re.IGNORECASE)


def xenium_capture_footprint(source_metadata: dict, centers_um: np.ndarray) -> dict:
    """Qualify publisher-pooled bins, preserving stale rendering-size fields.

    HEST's transcript pooling uses 100um square bins. Its image registration
    helper defaults to a 55um drawing marker; that marker is not the capture
    bin. The reviewed finite source subset is the only allowed 55->100
    adjudication. This does not verify original transcript mass conservation.
    """
    declared = float(source_metadata.get("spot_diameter", np.nan))
    reviewed = {
        "TENX147": "Xenium In Situ, Sample P5 CRC",
        "TENX148": "Xenium In Situ, Sample P2 CRC",
        "TENX149": "Xenium In Situ, Sample P1 CRC",
    }
    conflict = not np.isclose(declared, 100.)
    if conflict and not (np.isclose(declared, 55.) and source_metadata.get("id") in reviewed and
            reviewed.get(source_metadata.get("id")) == source_metadata.get("subseries") and
            np.isclose(float(source_metadata.get("inter_spot_dist", np.nan)), 100.)):
        raise DataContractError("unreviewed Xenium capture-size metadata conflict")
    xy = np.asarray(centers_um, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise DataContractError("Xenium grid requires finite physical coordinates")
    units = (xy - xy.min(0)) / 100.
    deviation_um = float(np.max(np.abs(units - np.rint(units))) * 100.)
    if deviation_um > 1e-5 or len(np.unique(xy, axis=0)) != len(xy):
        raise DataContractError("Xenium positions do not match unique 100um pooled-grid coordinates")
    return dict(
        shape="HEST_aggregated_square_grid_not_single_cell_or_2um_bin", nominal_side_um=100.,
        nominal_area_um2=10000., source_field="official_Hest_transcript_pooling_100um_square_bin_and_physical_grid_witness",
        original_JSON_spot_diameter_um=declared, rendering_size_metadata_conflict=bool(conflict),
        source_code_revision="3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3",
        pooling_source="https://github.com/mahmoodlab/HEST/blob/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/src/hest/readers.py#L1120-L1226",
        rendering_marker_source="https://github.com/mahmoodlab/HEST/blob/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/src/hest/utils.py#L1133-L1157",
        grid_lattice_max_deviation_um=deviation_um, original_transcript_mass_conservation_verified=False,
        not_equal_to_image_square=True,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(values) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in array], dtype=str)


def _index(group: h5py.Group) -> np.ndarray:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode()
    if key not in group or not isinstance(group[key], h5py.Dataset):
        raise DataContractError("H5AD dataframe index dataset is unavailable")
    return _text(group[key][:])


def fixed_gene_vocabulary(panel_json: Path, target_gene_json: Path) -> tuple[np.ndarray, dict]:
    panel = json.loads(Path(panel_json).read_text(encoding="utf-8"))
    target = json.loads(Path(target_gene_json).read_text(encoding="utf-8"))
    panel_names = panel.get("shared_genes_private", panel.get("gene_symbols_private"))
    target_names = target.get("gene_symbols_private")
    if not isinstance(panel_names, list) or not isinstance(target_names, list):
        raise DataContractError("explicit named measurement vocabulary and target gene metadata are required")
    if any(not isinstance(g, str) or not g for g in panel_names+target_names):
        raise DataContractError("gene metadata must contain nonempty exact gene-symbol strings")
    controls = {g for g in panel_names if CONTROL_PATTERN.match(g)}
    names = np.asarray(sorted((set(panel_names)-controls) & set(target_names)), dtype=str)
    if len(names) < 1 or len(names) > 1024:
        raise DataContractError("fixed vocabulary must contain 1..1024 genes")
    return names, dict(
        rule="exact_named_measurement_vocabulary_intersection_target_finite_gene_metadata_no_expression_value_selection",
        panel_json_sha256=file_sha256(panel_json), target_gene_json_sha256=file_sha256(target_gene_json),
        panel_named_genes=len(set(panel_names)), panel_control_names_excluded=len(controls), selected_genes=len(names),
        panel_provenance_notes=panel.get("notes", []), alias_reconciliation=False,
    )


def validate_split_lock(lock: dict) -> None:
    rows = lock.get("rows")
    if not isinstance(rows, list) or not rows:
        raise DataContractError("split lock requires nonempty rows")
    section_ids, donor_split = set(), {}
    for row in rows:
        for key in ("section_id_private", "study_id", "donor_id_private", "split", "assay"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise DataContractError(f"missing split-lock identity field {key}")
        if row["section_id_private"] in section_ids:
            raise DataContractError("duplicate section identity in split lock")
        section_ids.add(row["section_id_private"])
        if row["split"] not in {"train", "validation", "test"}:
            raise DataContractError("unrecognized split label")
        donor = (row["study_id"], row["donor_id_private"])
        if donor in donor_split and donor_split[donor] != row["split"]:
            raise DataContractError("one source donor crosses data splits")
        donor_split[donor] = row["split"]


def _read_matrix(node: h5py.Group | h5py.Dataset):
    if isinstance(node, h5py.Dataset):
        matrix = node[:]
        if matrix.ndim != 2:
            raise DataContractError("dense expression must have [spot,gene] shape")
        values = matrix.reshape(-1)
        encoding = "dense"
    else:
        encoding = node.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        if encoding not in {"csr_matrix", "csc_matrix"}:
            raise DataContractError("only explicit dense/CSR/CSC expression encoding is supported")
        shape = tuple(int(x) for x in node.attrs.get("shape", ()))
        if len(shape) != 2:
            raise DataContractError("sparse expression shape is missing")
        values, indices, indptr = node["data"][:], node["indices"][:], node["indptr"][:]
        constructor = sparse.csr_matrix if encoding == "csr_matrix" else sparse.csc_matrix
        matrix = constructor((values, indices, indptr), shape=shape)
        if not matrix.has_canonical_format:
            raise DataContractError("duplicate/unsorted sparse entries require explicit source repair")
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
        raise DataContractError("source expression is not finite nonnegative integer raw counts")
    if np.any(values > 2**53-1):
        raise DataContractError("raw counts exceed supported exact integer range")
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
        nonzeros = int(np.count_nonzero(matrix.data))
    else:
        nonzeros = int(np.count_nonzero(matrix))
    return matrix, dict(encoding=encoding, source_spots=int(matrix.shape[0]), source_genes=int(matrix.shape[1]),
                        source_nonzero_count_entries=nonzeros,
                        source_measured_zero_entries=int(np.prod(matrix.shape))-nonzeros,
                        raw_integer_nonnegative_verified=True)


def read_h5ad_raw(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        matrix_path, var_group = "X", handle["var"]
        if "layers/counts" in handle:
            matrix_path = "layers/counts"
        elif "raw/X" in handle and "raw/var" in handle:
            matrix_path, var_group = "raw/X", handle["raw/var"]
        elif "uns/log1p" in handle:
            raise DataContractError("X has log1p preprocessing history without an explicit raw/counts matrix")
        matrix, audit = _read_matrix(handle[matrix_path])
        genes, barcodes = _index(var_group), _index(handle["obs"])
        if matrix.shape != (len(barcodes), len(genes)):
            raise DataContractError("expression dimensions do not match barcode/gene metadata")
        if len(set(genes)) != len(genes) or len(set(barcodes)) != len(barcodes):
            raise DataContractError("duplicate genes or observation barcodes require explicit source repair")
        if "obsm/spatial" not in handle:
            raise DataContractError("physical source spatial coordinates are missing")
        coords = np.asarray(handle["obsm/spatial"][:], dtype=np.float64)
        if coords.shape != (len(barcodes), 2) or not np.all(np.isfinite(coords)):
            raise DataContractError("source spatial coordinates must be finite [spot,2]")
        audit.update(matrix_path=matrix_path, preprocessing_history_keys=list(handle.get("uns", {}).keys()),
                     control_like_source_genes=int(sum(bool(CONTROL_PATTERN.match(g)) for g in genes)),
                     original_count_unit="source_integer_UMI_or_transcript_count_not_normalized_expression",
                     absent_gene_is_unmeasured_not_biological_zero=True)
    return dict(matrix=matrix, gene_names=genes, barcodes=barcodes, coords_px=coords, audit=audit)


def qualify_patch_pair(expression: dict, patch_path: Path, source_metadata: dict,
                       fixed_genes: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Align every actual patch to its count row and qualify source-native FOV."""
    with h5py.File(patch_path, "r") as handle:
        barcode_key = "barcode" if "barcode" in handle else "barcodes"
        if barcode_key not in handle or "coords" not in handle or "img" not in handle:
            raise DataContractError("patch H5 requires barcode, coords, and img datasets")
        barcodes = _text(handle[barcode_key][:])
        corners = np.asarray(handle["coords"][:], dtype=np.float64)
        image = handle["img"]
        if len(set(barcodes)) != len(barcodes):
            raise DataContractError("duplicate patch barcodes")
        if image.shape != (len(barcodes), 224, 224, 3) or image.dtype != np.uint8:
            raise DataContractError("patch image contract requires uint8 [N,224,224,3] RGB")
        if corners.shape != (len(barcodes), 2) or not np.all(np.isfinite(corners)):
            raise DataContractError("patch crop coordinates must be finite [N,2]")
        if len(barcodes) < 1:
            raise DataContractError("empty patch image dataset")
        # Decode representative data, not just its shape; no pixel-derived labels.
        sample_indices = sorted(set([0, len(barcodes)//2, len(barcodes)-1]))
        for index in sample_indices:
            if image[index].shape != (224, 224, 3):
                raise DataContractError("patch image read failed")
        attrs = {str(k): (v.item() if np.ndim(v) == 0 and hasattr(v, "item") else v) for k, v in image.attrs.items()}
    try:
        source_mpp = float(attrs["pixel_size"])
        crop_pixels = int(attrs["patch_size_src"])
        target_pixels = int(attrs["patch_size_target"])
        downsample = float(attrs["downsample"])
    except (KeyError, TypeError, ValueError) as error:
        raise DataContractError("patch pixel scale / source crop metadata is incomplete") from error
    if not np.isfinite(source_mpp) or source_mpp <= 0 or crop_pixels < 1 or target_pixels != 224:
        raise DataContractError("invalid patch physical scale metadata")
    nominal_mpp = source_mpp*downsample
    if not np.isclose(nominal_mpp, .5, rtol=0, atol=1e-8):
        raise DataContractError("patch nominal target mpp is not 0.5 um/px")
    metadata_mpp = source_metadata.get("pixel_size_um_estimated")
    if metadata_mpp is None or not np.isfinite(metadata_mpp) or not np.isclose(source_mpp, metadata_mpp, rtol=0, atol=1e-8):
        raise DataContractError("patch source pixel_size not verified against original metadata estimate")
    actual_view = crop_pixels*source_mpp
    rounding_bound = .5*source_mpp+1e-6
    if abs(actual_view-112.) > rounding_bound:
        raise DataContractError("native source crop exceeds the locked half-native-pixel 112um rounding tolerance")
    lookup = {barcode: i for i, barcode in enumerate(expression["barcodes"])}
    if any(barcode not in lookup for barcode in barcodes):
        raise DataContractError("a patch barcode lacks a source count observation")
    order = np.asarray([lookup[b] for b in barcodes], dtype=np.int64)
    if not np.array_equal(expression["barcodes"][order], barcodes):
        raise DataContractError("elementwise barcode alignment failed")
    centers_px = expression["coords_px"][order]
    center_error = np.linalg.norm((corners+crop_pixels/2)-centers_px, axis=1)
    if np.max(center_error) > 1.5:
        raise DataContractError("patch crop centers do not correspond to source expression centers within 1.5 native pixels")
    assay = str(source_metadata.get("st_technology", ""))
    diameter = float(source_metadata.get("spot_diameter", np.nan))
    if assay == "Visium" and np.isclose(diameter, 55.):
        capture = dict(shape="circle", nominal_diameter_um=55., nominal_area_um2=float(np.pi*27.5**2),
                       source_field="metadata.spot_diameter", not_equal_to_image_square=True)
    elif assay == "Xenium":
        capture = xenium_capture_footprint(source_metadata, centers_px*source_mpp)
    else:
        raise DataContractError("assay capture footprint is unqualified (only explicit 55um Visium / 100um Xenium grid)")
    gene_lookup = {g: i for i, g in enumerate(expression["gene_names"])}
    measured = np.asarray([g in gene_lookup for g in fixed_genes], dtype=bool)
    if not measured.any():
        raise DataContractError("no fixed bridge genes are measured in this section")
    columns = np.asarray([gene_lookup[g] for g in fixed_genes[measured]], dtype=np.int64)
    selected = expression["matrix"][order][:, columns]
    if sparse.issparse(selected):
        selected = selected.toarray()
    counts = np.zeros((len(barcodes), len(fixed_genes)), dtype=np.int64)
    counts[:, measured] = np.asarray(selected, dtype=np.int64)
    arrays = dict(counts=counts, gene_names=np.asarray(fixed_genes, dtype=str), coords_um=centers_px*source_mpp,
                  barcodes=barcodes, measured_mask=measured)
    audit = dict(expression["audit"], patch_count=len(barcodes), source_observations_without_patch=len(lookup)-len(barcodes),
                 aligned_barcode_elementwise=True, all_patches_retained=True,
                 coordinates=dict(source="h5ad.obsm.spatial expression-center pixels times patch.img.pixel_size",
                                  unit="um", mpp_x=source_mpp, mpp_y=source_mpp,
                                  source_crop_coord_semantics="patch.coords native-pixel upper-left",
                                  maximum_center_alignment_error_native_pixels=float(np.max(center_error))),
                 image_footprint=dict(nominal_view_um=112., actual_native_crop_view_um=actual_view,
                                      native_crop_pixels=crop_pixels, target_pixels=224,
                                      nominal_mpp=.5, effective_native_crop_mpp=actual_view/224,
                                      locked_rounding_tolerance_um=rounding_bound,
                                      exact_112um=False if actual_view != 112. else True,
                                      preprocessing_contract="source-native rounded crop -> uint8 RGB224; new frozen UNI2 extraction required"),
                 expression_capture_footprint=capture, fixed_gene_count=len(fixed_genes),
                 measured_fixed_genes=int(measured.sum()), unmeasured_fixed_genes=int((~measured).sum()),
                 measured_zero_entries=int(np.sum(counts[:, measured] == 0)),
                 unmeasured_storage_fill=0, unmeasured_storage_fill_is_not_expression=True,
                 count_unit="raw_integer_UMI_or_transcript_count", known_pretraining_overlap="UNKNOWN_NOT_INDEPENDENTLY_VERIFIED")
    return arrays, audit
