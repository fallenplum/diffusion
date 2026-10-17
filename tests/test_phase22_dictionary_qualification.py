"""Dictionary qualification failures must have receipts, never qualified arrays."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from scripts import run_phase22_external_spatial as runner


def setup(monkeypatch, tmp_path, *, nmf_converged=True, oracle_converged=True):
    counts = np.array([[8, 1, 1], [5, 3, 2], [2, 7, 1], [1, 1, 8], [3, 2, 5]])
    genes = np.array(["synthetic-g1", "synthetic-g2", "synthetic-g3"])
    sections = [SimpleNamespace(counts=counts[:3], gene_names=genes, measured=np.ones(3, bool), donor_id="synthetic-A"),
                SimpleNamespace(counts=counts[3:], gene_names=genes, measured=np.ones(3, bool), donor_id="synthetic-B")]
    monkeypatch.setattr(runner, "qualification_rows", lambda *args, **kwargs:
                        (sections, "a" * 64, {"synthetic-A": "train", "synthetic-B": "train"}))
    rng = np.random.default_rng(31)
    dictionary = rng.dirichlet(np.ones(3), size=32)
    fit = SimpleNamespace(dictionary=dictionary, gene_names_private=tuple(genes),
                          selected_gene_indices_private=np.arange(3), sampled_spot_indices_private=np.arange(5),
                          metadata={"converged": nmf_converged, "partition_role": "ST-train",
                                    "split_lock_sha256": "a" * 64, "max_iterations": 500})
    monkeypatch.setattr(runner, "fit_molecular_dictionary", lambda *args, **kwargs: fit)
    monkeypatch.setattr(runner, "oracle_coefficients", lambda *args, **kwargs:
                        (np.full((5, 32), 1. / 32), {"converged": oracle_converged, "iterations": 1000}))
    qualification = tmp_path / "SYNTHETIC_QUALIFICATION_ONLY.json"
    qualification.write_text("{}", encoding="utf-8")
    target = tmp_path / "SYNTHETIC_GENES_ONLY.json"
    target.write_text(json.dumps({"gene_symbols_private": genes.tolist()}), encoding="utf-8")
    return SimpleNamespace(data_root=tmp_path, split_lock=tmp_path / "unused_lock.json",
                           qualification=qualification, target_measured_genes=target,
                           output_root=tmp_path / "private_output")


@pytest.mark.parametrize("stage,settings", [
    ("KL_NMF_DICTIONARY", {"nmf_converged": False}),
    ("TRAIN_FIXED_DICTIONARY_ORACLE", {"oracle_converged": False}),
])
def test_nonconvergence_writes_failure_only_and_zero_formal_fit_count(monkeypatch, tmp_path, stage, settings):
    args = setup(monkeypatch, tmp_path, **settings)
    with pytest.raises(ValueError, match="did not converge"):
        runner.prepare_dictionary(args)
    folder = args.output_root / "dictionary"
    receipt = runner.read_json(folder / "DICTIONARY_QUALIFICATION_FAILED_PRIVATE.json")
    assert receipt["failed_stage"] == stage
    assert receipt["formal_spatial_model_fits"] == 0
    assert receipt["qualified_dictionary_or_model_written"] is False
    assert not (folder / "dictionary_private.npz").exists()
    assert not (folder / "DICT_RECEIPT_PRIVATE.json").exists()


def test_qualified_prepare_and_load_bind_new_schema_scale_and_source(monkeypatch, tmp_path):
    args = setup(monkeypatch, tmp_path)
    runner.prepare_dictionary(args)
    d, genes, receipt, digest = runner.load_dictionary(args.output_root)
    assert d.shape == (32, 3) and len(genes) == 3
    assert receipt["schema_version"] == runner.DICTIONARY_SCHEMA
    assert receipt["dictionary_npz_sha256"] == digest
    assert receipt["omega_calibration"]["training_donors"] == 2
    assert receipt["omega_calibration"]["oracle_converged"] is True
    assert receipt["omega_calibration"]["validation_or_test_profiles_read"] is False
    assert receipt["identity"]["measurement_scale_protocol_sha256"] == runner.SCALE_PROTOCOL_SHA
    assert "src/phase22_spatial_bridge/measurement_scale.py" in receipt["identity"]["source_hashes"]
    assert not (args.output_root / "dictionary/DICTIONARY_QUALIFICATION_FAILED_PRIVATE.json").exists()
