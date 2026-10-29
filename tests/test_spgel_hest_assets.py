"""Synthetic row/scale/provenance adversaries, not real HEST qualification."""
import copy
import hashlib
import json
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.spgel_hest import assets as module
from src.spgel_hest.official import DATASET_REVISION


def save_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")
    return dict(path=str(path), sha256=module.sha256(path))


def sync_synthetic_certificates(output, root):
    """Artificial certificate fixture; never used to qualify real E assets."""
    data = dict(schema="hest_official_asset_requalification_20261029_v1", status="PASS_SYNTHETIC_DATA_ONLY",
                dataset_revision=DATASET_REVISION, tasks={})
    features = dict(schema="hest_official8_bf16_feature_qualification_v1", status="PASS_SYNTHETIC_FEATURES_ONLY",
                    dataset_revision=DATASET_REVISION, samples=[])
    for task_name, task in output["tasks"].items():
        q = dict(official_gene_file_sha256=task["gene_file"]["sha256"], folds=[], samples={})
        for fold in (0, 1):
            q["folds"].append(dict(fold=fold, train=dict(sha256=task["splits"][f"train_{fold}"]["sha256"]),
                                   test=dict(sha256=task["splits"][f"test_{fold}"]["sha256"])))
        for index, (name, sample) in enumerate(task["samples"].items()):
            expression = anndata.read_h5ad(sample["expression"]["path"])
            with h5py.File(sample["features"]["path"], "r") as handle:
                spots = len(handle["embeddings"])
            q["samples"][f"anonymous_{index}"] = dict(expression_sha256=sample["expression"]["sha256"],
                official_patch=dict(expected_sha256=sample["patches"]["sha256"]), donor_code=sample["donor_code"],
                expression_spots=expression.n_obs)
            features["samples"].append(dict(task=task_name, official_sample_id=name,
                feature_sha256=sample["features"]["sha256"], feature_file=f"{task_name}/{name}.h5",
                source_patch_sha256=sample["patches"]["sha256"], forward_precision="bfloat16", stored_dtype="float32",
                row_L2_normalized=False, feature_dimension=1536, spots_count=spots,
                model_sha256="b"*64, model_revision="SYNTHETIC_REVISION", config_sha256="c"*64, extractor_sha256="d"*64))
        data["tasks"][task_name] = q
    output["qualification"] = save_json(root / "data_qualification.json", data)
    output["qualification"]["expected_status"] = data["status"]
    output["feature_qualification"] = save_json(root / "feature_qualification.json", features)
    output["feature_qualification"]["expected_status"] = features["status"]


@pytest.fixture
def asset_fixture(tmp_path, monkeypatch):
    certificate = save_json(tmp_path / "qualification.json", dict(status="PASS_SYNTHETIC_ASSET_TEST_ONLY"))
    certificate["expected_status"] = "PASS_SYNTHETIC_ASSET_TEST_ONLY"
    primary, blobs = {}, {}
    for name in ("benchmark.py", "trainer.py", "st_dataset.py"):
        payload = ("# synthetic fixture, never a real official certificate: " + name + "\n").encode()
        path = tmp_path / name;path.write_bytes(payload)
        blobs[name] = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        primary[name] = dict(path=str(path), sha256=module.sha256(path))
    monkeypatch.setattr(module, "UPSTREAM_BLOBS", blobs)
    output = dict(schema="spgel_hest_official_assets_v1",
        status="PASS_PINNED_OFFICIAL_DATA_AND_EIGHT_NEW_BF16_FEATURES", dataset_revision=DATASET_REVISION,
        tasks={}, qualification=certificate, feature_qualification=certificate, upstream_code=primary)
    for dataset in ("COAD", "READ"):
        folder = tmp_path / dataset;folder.mkdir()
        genes = [f"G{k}" for k in range(50)]
        task = dict(gene_file=save_json(folder / "var_50genes.json", dict(genes=genes)), samples={}, splits={})
        for index in range(4):
            name = f"SYN_{dataset}_{index}"
            values = np.arange(6 * 50).reshape(6, 50).astype(np.float64) + index
            adata = anndata.AnnData(sparse.csr_matrix(values))
            adata.var_names = genes[::-1];adata.obs_names = [f"BARCODE{k}" for k in range(6)]
            expr = folder / (name + ".h5ad");adata.write_h5ad(expr)
            feature = folder / (name + ".h5")
            rng = np.random.default_rng(index)
            with h5py.File(feature, "w") as handle:
                handle.attrs.update(official_sample_id=name, source_patch_sha256="a" * 64,
                    dataset_revision=DATASET_REVISION, forward_precision="bfloat16", row_L2_normalized=False)
                handle.create_dataset("embeddings", data=rng.normal(size=(6, 1536)).astype(np.float32))
                handle.create_dataset("barcodes", data=np.array([f"BARCODE{k}".encode() for k in reversed(range(6))])[:, None])
                handle.create_dataset("coords", data=np.c_[np.arange(6), np.zeros(6)])
            task["samples"][name] = dict(expression=dict(path=str(expr), sha256=module.sha256(expr)),
                features=dict(path=str(feature), sha256=module.sha256(feature)),
                patches=dict(path=str(folder / (name + ".h5")), sha256="a" * 64),
                feature_keys=dict(embeddings="embeddings", barcodes="barcodes", coords="coords"),
                feature_contract=dict(forward_precision="bfloat16", row_L2_normalized=False,
                    image_only=True, feature_dimension=1536), donor_code=None if index == 0 else index)
        ids = list(task["samples"])
        for fold in (0, 1):
            for split, chosen in (("train", ids[:2] if fold == 0 else ids[2:]),
                                  ("test", ids[2:] if fold == 0 else ids[:2])):
                path = folder / f"{split}_{fold}.csv"
                pd.DataFrame(dict(sample_id=chosen, expr_path=[x + ".h5ad" for x in chosen],
                                  patches_path=[x + ".h5" for x in chosen])).to_csv(path, index=False)
                task["splits"][f"{split}_{fold}"] = dict(path=str(path), sha256=module.sha256(path))
        output["tasks"][dataset] = task
    sync_synthetic_certificates(output, tmp_path)
    path = tmp_path / "assets.json";save_json(path, output)
    return output, path


def test_exact_fifty_genes_and_barcode_permutation_are_honored(asset_fixture):
    _, path = asset_fixture
    assets = module.validate_assets(path)
    partition = module.load_partition(assets, "READ", 0, "train")
    # Both barcode and gene order are reversed relative to stored AnnData.
    assert partition.targets[0, 0] == np.log1p(299.)
    assert partition.targets[5, 49] == 0.
    assert partition.targets.dtype == np.float64 and partition.features.dtype == np.float32
    assert partition.targets.shape == (12, 50)
    assert partition.donor_codes[0] is None
    assert len(set(partition.barcodes)) < len(partition.barcodes)  # Valid across samples.
    assert len(partition.ordering_sha256) == 64  # Composite sample+barcode identities.


def test_incomplete_feature_status_and_legacy_unit_l2_cannot_pass(asset_fixture):
    assets, path = asset_fixture
    bad = copy.deepcopy(assets);bad["status"] = "INCOMPLETE_FEATURES_NOT_QUALIFIED_DO_NOT_TRAIN"
    save_json(path, bad)
    with pytest.raises(ValueError, match="NEW_OFFICIAL_ASSETS_SCHEMA"):
        module.validate_assets(path)
    bad = copy.deepcopy(assets)
    next(iter(bad["tasks"]["READ"]["samples"].values()))["feature_contract"]["row_L2_normalized"] = True
    save_json(path, bad)
    with pytest.raises(ValueError, match="BF16_RAW_UNI2_FEATURE_CONTRACT"):
        module.validate_assets(path)


def test_changing_primary_bytes_and_refreshing_only_manifest_sha_is_rejected(asset_fixture):
    assets, path = asset_fixture
    primary = assets["upstream_code"]["trainer.py"]
    Path(primary["path"]).write_bytes(b"# substituted upstream\n")
    primary["sha256"] = module.sha256(primary["path"])
    save_json(path, assets)
    with pytest.raises(ValueError, match="PRIMARY_GIT_BLOB_CHANGED"):
        module.validate_assets(path)


def test_cross_specimen_expression_file_is_rejected_even_with_reused_barcodes(asset_fixture):
    assets, _ = asset_fixture
    samples = assets["tasks"]["READ"]["samples"]
    left, right = list(samples)[:2]
    samples[left]["expression"] = copy.deepcopy(samples[right]["expression"])
    with pytest.raises(ValueError, match="SPECIMEN_FILE_BINDING_MISMATCH"):
        module.load_partition(assets, "READ", 0, "train")


@pytest.mark.parametrize("component", ["gene_file", "expression", "features", "donor_code"])
def test_updated_runtime_sha_or_donor_cannot_reuse_stale_pass_certificate(asset_fixture, component):
    assets, path = asset_fixture
    task = assets["tasks"]["READ"]
    if component == "gene_file":
        task["gene_file"] = save_json(Path(task["gene_file"]["path"]), dict(genes=[f"OTHER{k}" for k in range(50)]))
    else:
        sample = next(iter(task["samples"].values()))
        if component == "donor_code":
            sample["donor_code"] = "FORGED_DISTINCT_PATIENT"
        else:
            sample[component]["sha256"] = "e" * 64
    save_json(path, assets)
    with pytest.raises(ValueError, match="CERTIFICATE"):
        module.validate_assets(path)


def test_official_qc_image_subset_selects_expression_by_barcode_without_padding(asset_fixture):
    assets, path = asset_fixture
    sample = next(iter(assets["tasks"]["READ"]["samples"].values()))
    feature = Path(sample["features"]["path"])
    with h5py.File(feature, "r") as handle:
        arrays = {key: handle[key][:][[0, 2, 4]] for key in ["embeddings", "barcodes", "coords"]}
        attributes = dict(handle.attrs)
    with h5py.File(feature, "w") as handle:
        handle.attrs.update(attributes)
        for key, value in arrays.items():
            handle.create_dataset(key, data=value)
    sample["features"]["sha256"] = module.sha256(feature)
    sync_synthetic_certificates(assets, path.parent)
    save_json(path, assets)
    qualified = module.validate_assets(path)
    partition = module.load_partition(qualified, "READ", 0, "train")
    assert len(partition.features) == 9  # 3 QC-selected image rows plus the other specimen's 6.
    assert partition.barcodes[:3].tolist() == ["BARCODE5", "BARCODE3", "BARCODE1"]
    assert np.array_equal(partition.targets[:3, 0], np.log1p([299., 199., 99.]))
