"""Explicit, hash-bound official assets; no discovery of old source weights."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd

from .official import DATASET_REVISION, UPSTREAM_BLOBS, log1p_only_selected_expression, require


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "DUPLICATE_JSON_KEY")
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def bound_file(reference):
    require(set(reference) >= {"path", "sha256"}, "EXPLICIT_FILE_PATH_AND_SHA256_REQUIRED")
    path = Path(reference["path"])
    require(path.is_file() and not path.is_symlink() and sha256(path) == reference["sha256"],
            "HEST_ASSET_BYTES_CHANGED")
    return path


def certificate(reference):
    value = read_json(bound_file(reference))
    require(value.get("status") == reference["expected_status"]
            and reference["expected_status"].startswith("PASS"), "HEST_QUALIFICATION_NOT_PASS")
    return value


def cross_bind_certificates(assets, data_certificate, feature_certificate):
    """Every runtime byte reference must be covered by the actual audits.

    File SHA plus a generic PASS label is insufficient: a stale PASS receipt
    must not authorize another expression matrix, feature bank or split.
    """
    require(data_certificate.get("schema") == "hest_official_asset_requalification_20261029_v1"
        and data_certificate.get("dataset_revision") == DATASET_REVISION,
        "OFFICIAL_DATA_CERTIFICATE_VERSION_OR_DATASET_CHANGED")
    require(feature_certificate.get("schema") == "hest_official8_bf16_feature_qualification_v1"
        and feature_certificate.get("dataset_revision") == DATASET_REVISION,
        "OFFICIAL_FEATURE_CERTIFICATE_VERSION_OR_DATASET_CHANGED")
    samples = feature_certificate.get("samples", [])
    samples = list(samples.values()) if isinstance(samples, dict) else samples
    require(isinstance(samples, list) and len(samples) == 8, "EIGHT_ACTUAL_FEATURE_CERTIFICATE_ROWS_REQUIRED")
    feature_rows = {(row["task"], row["official_sample_id"]): row for row in samples}
    require(len(feature_rows) == 8
        and set(feature_rows) == {(name, specimen) for name, task in assets["tasks"].items()
                                 for specimen in task["samples"]}, "FEATURE_CERTIFICATE_SAMPLE_SCOPE_CHANGED")
    producer_identities = set()
    for name, task in assets["tasks"].items():
        qualified = data_certificate["tasks"][name]
        require(task["gene_file"]["sha256"] == qualified["official_gene_file_sha256"],
                "OFFICIAL_HVG_CERTIFICATE_BINDING_CHANGED")
        folds = qualified["folds"]
        require(len(folds) == 2 and {row["fold"] for row in folds} == {0, 1}, "DATA_CERTIFICATE_FOLDS_CHANGED")
        for row in folds:
            for split in ("train", "test"):
                require(task["splits"][f"{split}_{row['fold']}"]["sha256"] == row[split]["sha256"],
                        "OFFICIAL_SPLIT_CERTIFICATE_BINDING_CHANGED")
        expression_rows = {row["expression_sha256"]: row for row in qualified["samples"].values()}
        require(len(expression_rows) == 4
            and {s["expression"]["sha256"] for s in task["samples"].values()} == set(expression_rows),
            "OFFICIAL_EXPRESSION_CERTIFICATE_SCOPE_CHANGED")
        for specimen, sample in task["samples"].items():
            expression = expression_rows[sample["expression"]["sha256"]]
            feature = feature_rows[(name, specimen)]
            require(sample["patches"]["sha256"] == expression["official_patch"]["expected_sha256"]
                and sample.get("donor_code") == expression["donor_code"],
                "EXPRESSION_PATCH_OR_DONOR_CERTIFICATE_BINDING_CHANGED")
            require(feature["source_patch_sha256"] == sample["patches"]["sha256"]
                and feature["feature_sha256"] == sample["features"]["sha256"]
                and Path(feature["feature_file"]).name == Path(sample["features"]["path"]).name
                and feature["forward_precision"] == "bfloat16" and feature["stored_dtype"] == "float32"
                and feature["row_L2_normalized"] is False and feature["feature_dimension"] == 1536
                and 0 < int(feature["spots_count"]) <= int(expression["expression_spots"]),
                "ACTUAL_BF16_FEATURE_CERTIFICATE_BINDING_CHANGED")
            sample["_qualified_feature_rows"] = int(feature["spots_count"])
            producer_identities.add((feature["model_sha256"], feature["model_revision"],
                feature["config_sha256"], feature["extractor_sha256"]))
    require(len(producer_identities) == 1 and all(len(str(x)) == 64 for x in
        (next(iter(producer_identities))[0], next(iter(producer_identities))[2], next(iter(producer_identities))[3])),
        "ALL_EIGHT_FEATURES_REQUIRE_SAME_PINNED_MODEL_CONFIG_AND_EXTRACTOR")


def validate_assets(path):
    value = read_json(path)
    require(set(value) == {"schema", "status", "dataset_revision", "tasks", "qualification",
        "feature_qualification", "upstream_code"} and value["schema"] == "spgel_hest_official_assets_v1"
        and value["status"] == "PASS_PINNED_OFFICIAL_DATA_AND_EIGHT_NEW_BF16_FEATURES"
        and value["dataset_revision"] == DATASET_REVISION, "NEW_OFFICIAL_ASSETS_SCHEMA_REQUIRED")
    require(set(value["tasks"]) == {"COAD", "READ"}, "EXACT_OFFICIAL_COAD_READ_TASKS_REQUIRED")
    data_certificate = certificate(value["qualification"])
    feature_certificate = certificate(value["feature_qualification"])
    cross_bind_certificates(value, data_certificate, feature_certificate)
    require(set(value["upstream_code"]) == set(UPSTREAM_BLOBS), "THREE_PINNED_PRIMARY_FILES_REQUIRED")
    for name, expected in UPSTREAM_BLOBS.items():
        payload = bound_file(value["upstream_code"][name]).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        require(actual == expected, "OFFICIAL_PRIMARY_GIT_BLOB_CHANGED")
    for task in value["tasks"].values():
        require(set(task["splits"]) == {"train_0", "test_0", "train_1", "test_1"}
                and len(task["samples"]) == 4, "ACTUAL_TWO_FOLDS_FOUR_SAMPLES_REQUIRED")
        genes = read_json(bound_file(task["gene_file"]))["genes"]
        require(len(genes) == len(set(genes)) == 50, "EXACT_FIFTY_OFFICIAL_ORDERED_GENES_REQUIRED")
        for fold in (0, 1):
            frames = [pd.read_csv(bound_file(task["splits"][f"{s}_{fold}"])) for s in ("train", "test")]
            for frame in frames:
                require(set(frame.columns) >= {"sample_id", "patches_path", "expr_path"}
                    and frame.sample_id.is_unique and len(frame), "OFFICIAL_SPLIT_COLUMNS_OR_DUPLICATES")
            train, test = [set(frame.sample_id.astype(str)) for frame in frames]
            require(not train.intersection(test) and train.union(test) == set(task["samples"]),
                    "OFFICIAL_SAMPLE_SPLIT_OVERLAP_OR_OMISSION")
        for sample in task["samples"].values():
            require(sample["feature_keys"] == {"embeddings": "embeddings", "barcodes": "barcodes", "coords": "coords"},
                    "OFFICIAL_NEW_FEATURE_KEYS_REQUIRED")
            require(sample["feature_contract"]["forward_precision"] == "bfloat16"
                and sample["feature_contract"]["row_L2_normalized"] is False
                and sample["feature_contract"]["image_only"] is True
                and sample["feature_contract"]["feature_dimension"] == 1536,
                "BF16_RAW_UNI2_FEATURE_CONTRACT_REQUIRED")
            # RGB patches need not be copied to the fitting machine. Their SHA
            # is already bound by the extraction/physical QA certificates.
            require(len(sample["patches"]["sha256"]) == 64, "ORIGINAL_PATCH_SHA_REQUIRED")
    return value


@dataclass
class Partition:
    features: np.ndarray
    targets: np.ndarray
    barcodes: np.ndarray
    specimen_index: np.ndarray
    specimen_ids: list[str]
    donor_codes: list
    genes: list[str]

    @property
    def ordering_sha256(self):
        return canonical_hash([[self.specimen_ids[int(k)], str(barcode)]
            for k, barcode in zip(self.specimen_index, self.barcodes)])


def load_partition(assets, dataset, fold, split):
    require(dataset in ("COAD", "READ") and fold in (0, 1) and split in ("train", "test"),
            "UNREGISTERED_OFFICIAL_PARTITION")
    task = assets["tasks"][dataset]
    genes = list(read_json(bound_file(task["gene_file"]))["genes"])
    frame = pd.read_csv(bound_file(task["splits"][f"{split}_{fold}"]))
    xs, ys, bs, codes, donors = [], [], [], [], []
    ids = frame.sample_id.astype(str).tolist()
    for index, specimen in enumerate(ids):
        spec = task["samples"][specimen]
        csv_row = frame.iloc[index]
        require(Path(spec["expression"]["path"]).name == Path(str(csv_row["expr_path"])).name
            and Path(spec["patches"]["path"]).name == Path(str(csv_row["patches_path"])).name,
            "OFFICIAL_SPLIT_SPECIMEN_FILE_BINDING_MISMATCH")
        with h5py.File(bound_file(spec["features"]), "r") as handle:
            def attr(name):
                value = handle.attrs.get(name)
                return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else value
            require(attr("official_sample_id") == specimen
                and attr("source_patch_sha256") == spec["patches"]["sha256"]
                and attr("dataset_revision") == DATASET_REVISION
                and attr("forward_precision") == "bfloat16"
                and attr("row_L2_normalized") is not None and not bool(attr("row_L2_normalized")),
                "OFFICIAL_FEATURE_SPECIMEN_PATCH_PRECISION_ATTR_BINDING")
            features = handle["embeddings"][:]
            barcodes = handle["barcodes"][:].flatten().astype(str)
            coords = handle["coords"][:]
        require(features.ndim == 2 and features.shape == (len(barcodes), 1536)
            and features.dtype == np.float32 and np.isfinite(features).all()
            and coords.shape == (len(barcodes), 2) and np.isfinite(coords).all()
            and len(set(barcodes)) == len(barcodes) and len(barcodes) > 0,
            "QUALIFIED_FEATURE_SHAPE_DTYPE_COORDS_OR_BARCODE")
        require(len(barcodes) == spec.get("_qualified_feature_rows", len(barcodes)),
                "ACTUAL_FEATURE_ROWS_DIFFER_FROM_QUALIFICATION_CERTIFICATE")
        adata = anndata.read_h5ad(bound_file(spec["expression"]))
        require(adata.obs_names.is_unique and adata.var_names.is_unique
            and set(barcodes).issubset(set(adata.obs_names)) and set(genes).issubset(set(adata.var_names)),
            "EXPRESSION_ROW_OR_GENE_IDENTITY_MISMATCH")
        selected = adata[barcodes.tolist(), genes]
        targets = log1p_only_selected_expression(selected.X)
        require(targets.shape == (len(features), 50) and np.isfinite(targets).all(), "HEST_TARGET_SHAPE_OR_FINITE")
        xs.append(features);ys.append(targets);bs.append(barcodes)
        codes.append(np.full(len(features), index, np.int64));donors.append(spec.get("donor_code"))
    return Partition(np.concatenate(xs), np.concatenate(ys), np.concatenate(bs),
                     np.concatenate(codes), ids, donors, genes)
