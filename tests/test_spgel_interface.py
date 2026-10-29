import copy
import numpy as np
import pytest
import torch

from tests.test_spgel_contracts import source
from src.spgel.contracts import SourceStatistics, canonical_hash
from src.spgel.interface import FrozenSpatialProgramInterface
from src.spgel.models import SpatialProgramEncoder


def interface(arm="S_A5"):
    data = source(); train, held = data.split(0)
    stats = SourceStatistics.fit(data, train, heldout_donors=[0])
    model = SpatialProgramEncoder(arm, data.up, data.down, data.gene_symbols, image_mlp_hidden=527)
    manifest = dict(source_gene_membership_sha256="a" * 64)
    frozen = FrozenSpatialProgramInterface(model, stats, gene_symbols=data.gene_symbols, source_manifest=manifest)
    return frozen, data, held, stats


def test_frozen_image_interface_has_real_shape_masks_and_image_only_inputs():
    frozen, data, held, _ = interface()
    x, xy = data.features[held][None], data.xy_um[held][None]
    slide = data.section[held][None]; mask = np.ones(slide.shape, bool); mask[0, -1] = False
    out = frozen.encode_image_spots(x, xy, slide, mask)
    assert out["program_embeddings"].shape == (1, len(held), 4, 128)
    assert out["program_pred"].shape == (1, len(held), 4)
    assert not out["transcriptome_used"] and not out["tissue_truth_used"]
    assert np.count_nonzero(out["program_embeddings"][0, -1]) == 0
    assert not any(p.requires_grad for p in frozen.model.parameters())
    with pytest.raises(TypeError):
        frozen.encode_image_spots(x, xy, slide, mask, rna=data.genes[held])


def test_bulk_interface_rejects_source_spot_scaler_and_wrong_gene_order():
    frozen, data, held, stats = interface()
    value = stats.genes_z(data, held).values.numpy()
    contract = dict(schema="spgel_bulk_scale_contract_v1", fit_scope="current_R8_train_only",
        uses_source_spot_scaler=False, values_are_current_train_gene_Z=True,
        gene_order_sha256=canonical_hash(data.gene_symbols.tolist()))
    good = frozen.encode_bulk_genes(value, data.gene_mask[held], "a" * 64, contract)
    assert good["program_tokens"].shape == (len(held), 4, 128)
    assert not good["bulk_and_spot_measurements_equivalent_claimed"]
    bad = dict(contract, uses_source_spot_scaler=True)
    with pytest.raises(ValueError, match="SEPARATE_TRAIN_ONLY_SCALE_CONTRACT"):
        frozen.encode_bulk_genes(value, data.gene_mask[held], "a" * 64, bad)
    with pytest.raises(ValueError, match="BULK_GENE_ORDER_CHANGED"):
        frozen.encode_bulk_genes(value, data.gene_mask[held], "a" * 64, dict(contract, gene_order_sha256="b" * 64))


def test_plain_image_baseline_does_not_claim_a_trained_conditional_gene_branch():
    frozen, data, held, _ = interface("S_A1")
    mask = np.ones((1, len(held)), bool)
    out = frozen.encode_image_spots(data.features[held][None], data.xy_um[held][None], data.section[held][None], mask)
    assert out["native_learned_program_slots"] is False
    with pytest.raises(ValueError, match="NO_TRAINED_MOLECULAR_ENCODER"):
        frozen.encode_bulk_genes(data.genes[held], data.gene_mask[held], "a" * 64, {})


def test_zero_value_masked_wsi_rows_never_become_artificial_evidence():
    frozen, _, _, _ = interface()
    token = np.ones((2, 4, 128), np.float32)
    value = np.full((2, 5, 128), np.nan, np.float32)
    mask = np.zeros((2, 5), bool)
    out = frozen.route_bulk_to_wsi(token, value, np.zeros((2, 5, 2)), mask)
    assert not out["mask"].any() and np.count_nonzero(out["program_embeddings"]) == 0
