"""Run-lock and scope guards; no real official assets are read here."""
import copy
import json
from pathlib import Path

import pytest
from tests.test_spgel_hest_assets import asset_fixture, sync_synthetic_certificates

from src.spgel_hest import execution


def test_actual_recipe_is_accepted_and_four_cell_scope_is_fixed(tmp_path):
    path = Path(__file__).parents[1] / "configs/spgel_hest/ridge_v1.json"
    data = execution.recipe(path)
    assert data["logical_fits"] == 4 and data["seed"] == 1
    bad = copy.deepcopy(data);bad["logical_fits"] = 8
    target = tmp_path / "bad.json";target.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="EXACT_FOUR_OFFICIAL_RIDGE"):
        execution.recipe(target)
    bad = copy.deepcopy(data);bad["targets"]["library_normalization"] = True
    target.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="TARGET_CONTRACT_CHANGED"):
        execution.recipe(target)


def test_prior_source_pilot_lock_is_never_an_official_hvg_lock(tmp_path):
    path = Path(__file__).parents[1] / "configs/spgel_hest/ridge_v1.json"
    (tmp_path / "HEST_RUN_LOCK.json").write_text(json.dumps(dict(schema="spgel_source_run_lock_v1")))
    with pytest.raises(ValueError, match="OFFICIAL_RUN_LOCK_ASSETS_CODE_OR_RUNTIME_CHANGED"):
        execution.validate_lock(tmp_path / "no_assets.json", path, tmp_path)


def test_real_private_outputs_are_outside_scientific_checkout():
    with pytest.raises(ValueError, match="NEW_OFFICIAL_PRIVATE_NAMESPACE"):
        execution.private_directory(execution.ROOT / "tmp" / "forbidden_official_patient_outputs")


def test_no_old_source_modules_or_neural_arms_in_new_official_recipe():
    source = (Path(__file__).parents[1] / "src/spgel_hest/execution.py").read_text(encoding="utf-8")
    assert "from src.spgel." not in source and "from ..spgel" not in source
    assert "S_A5" not in source and "torch.optim" not in source


def test_preview_donor_is_unconfirmed_and_never_fabricated_unique():
    from types import SimpleNamespace
    train = SimpleNamespace(donor_codes=[None])
    test = SimpleNamespace(donor_codes=["known0", "known1", "known2"])
    assert execution.identity_status("COAD", train, test) == "COAD_IDENTITY_UNCONFIRMED"
    with pytest.raises(ValueError, match="NOT_FABRICATED_UNIQUE_ID"):
        execution.identity_status("COAD", SimpleNamespace(donor_codes=["fake_unique_preview"]), test)
    assert execution.identity_status("READ", SimpleNamespace(donor_codes=["A", "A"]),
        SimpleNamespace(donor_codes=["B", "B"])) == "READ_TWO_DONORS_VERIFIED_NO_CROSSING"


def test_actual_synthetic_four_fits_model_reload_report_and_forged_sha_detection(asset_fixture, monkeypatch, tmp_path):
    import anndata
    import h5py
    import joblib
    import numpy as np
    from scipy import sparse
    from src.spgel_hest.assets import sha256
    from src.spgel_hest.official import DATASET_REVISION
    assets, path = asset_fixture
    # Extend only synthetic tensors to make the exact PCA256 feasible.
    for task_name, task in assets["tasks"].items():
        for index, (name, spec) in enumerate(task["samples"].items()):
            rng = np.random.default_rng(index + 2)
            adata = anndata.AnnData(sparse.csr_matrix(rng.poisson(3, size=(129, 50)).astype(np.float64)))
            adata.obs_names = [f"BARCODE{k}" for k in range(129)]
            adata.var_names = [f"G{k}" for k in reversed(range(50))]
            adata.write_h5ad(spec["expression"]["path"])
            spec["expression"]["sha256"] = sha256(spec["expression"]["path"])
            with h5py.File(spec["features"]["path"], "w") as handle:
                handle.attrs.update(official_sample_id=name, source_patch_sha256="a" * 64,
                    dataset_revision=DATASET_REVISION, forward_precision="bfloat16", row_L2_normalized=False)
                handle.create_dataset("embeddings", data=rng.normal(size=(129, 1536)).astype(np.float32))
                handle.create_dataset("barcodes", data=np.array([f"BARCODE{k}".encode() for k in reversed(range(129))])[:, None])
                handle.create_dataset("coords", data=np.c_[np.arange(129), np.zeros(129)])
            spec["features"]["sha256"] = sha256(spec["features"]["path"])
            if task_name == "READ":
                spec["donor_code"] = "KNOWN_A" if index < 2 else "KNOWN_B"
    sync_synthetic_certificates(assets, path.parent)
    path.write_text(json.dumps(assets), encoding="utf-8")
    private = tmp_path / "synthetic_private";private.mkdir()
    monkeypatch.setattr(execution, "private_directory", lambda p: Path(p))
    config = Path(__file__).parents[1] / "configs/spgel_hest/ridge_v1.json"
    lock = execution.make_lock(path, config, private)
    assert len(lock["cells"]) == 4
    qualified = execution.read_json(private / "HEST_QUALIFICATION.private.json")
    assert qualified["new_PCA_fits"] == qualified["new_Ridge_fits"] == 0
    for dataset in ["COAD", "READ"]:
        for fold in [0, 1]:
            result = execution.fit_cell(path, config, private, dataset, fold)
            assert result["new_Ridge_fits"] == 1 and result["registered_Ridge_coefficients"] == 12800
    replay = execution.replay_all(path, config, private)
    assert replay["passed_models"] == 4 and replay["new_Ridge_fits"] == 0
    assert all(x["actual_saved_model_loaded"] and x["actual_forward_replayed"] for x in replay["units"])
    report = execution.report(path, config, private, tmp_path / "anonymous_report.json")
    assert report["tasks"]["COAD"]["donor_separation_status"] == "COAD_IDENTITY_UNCONFIRMED"
    assert report["tasks"]["READ"]["donor_separation_status"] == "READ_TWO_DONORS_VERIFIED_NO_CROSSING"
    artifact = private / "cells/READ_fold1_seed1/predictions.private.npz"
    original_bytes = artifact.read_bytes()
    artifact.write_bytes(original_bytes + b"SYNTHETIC_AFTER_REPLAY_TAMPER")
    with pytest.raises(ValueError, match="ARTIFACT_CHANGED_AFTER_REPLAY"):
        execution.report(path, config, private, tmp_path / "must_not_be_published.json")
    artifact.write_bytes(original_bytes)
    with pytest.raises(ValueError, match="NO_AUTOMATIC_RETRY"):
        execution.fit_cell(path, config, private, "COAD", 0)
    folder = private / "cells/COAD_fold0_seed1"
    model = joblib.load(folder / "model.private.joblib");model.estimator.coef_[0, 0] += 100.
    joblib.dump(model, folder / "model.private.joblib")
    done = execution.read_json(folder / "COMPLETE.json")
    done["model_sha256"] = sha256(folder / "model.private.joblib")
    (folder / "COMPLETE.json").write_text(json.dumps(done), encoding="utf-8")
    with pytest.raises(ValueError, match="ACTUAL_MODEL_REPLAY_PREDICTION_ERROR"):
        execution.replay_cell(path, config, private, "COAD", 0)
