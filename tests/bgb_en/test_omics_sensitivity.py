import importlib.util
from pathlib import Path

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[2] / 'scripts/bgb_en_omics_sensitivity.py'
SPEC = importlib.util.spec_from_file_location('bgb_en_sensitivity_tests', SOURCE)
aux = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aux)


def fixture():
    rng = np.random.default_rng(714); n, genes = 7, 11
    x = rng.normal(size=(n, genes)).astype(np.float32)
    prepared = {'features': {'rna': x, 'rna_mask': np.ones_like(x, bool),
                            'cna_gain': np.zeros_like(x), 'cna_loss': np.zeros_like(x),
                            'cna_mask': np.ones_like(x, bool), 'C': np.zeros((n, 3), np.float32)},
                'train': np.arange(5), 'valid': np.arange(5, 7), 'beta': .2,
                'RNA': np.linspace(-.5, .5, n), 'CP': np.zeros(n), 'base': np.zeros(n)}
    member = torch.zeros(5, genes, dtype=torch.bool)
    for k in range(5): member[k, k:k+4] = True
    return prepared, member


def test_factor_width_changes_only_readout_dimensions_and_preserves_direct_rna():
    prepared, member = fixture()
    narrow = aux.make_variant(prepared, member, 'O8_F2', 31, 'cpu', backend='ssm')
    wide = aux.make_variant(prepared, member, 'O8_F8', 31, 'cpu', backend='ssm')
    assert narrow.omics.factor[0].weight.shape == (2, 32)
    assert wide.omics.factor[0].weight.shape == (8, 32)
    for name, value in narrow.omics.state_dict().items():
        if not name.startswith(('factor.', 'risk_head.')):
            assert torch.equal(value, wide.omics.state_dict()[name])
    assert torch.equal(narrow.offset, wide.offset)
    assert torch.equal(narrow.offset, torch.tensor(prepared['RNA'], dtype=torch.float32))


def test_random_order_variants_keep_identical_trainable_core_initialization():
    prepared, member = fixture()
    for variant, arm in [('O5_RANDOM_ORDERS', 'O5'), ('O6_RANDOM_ORDERS', 'O6')]:
        torch.manual_seed(31)
        core = aux.MechanismModel('R2', arm, prepared, member, None, 'cpu', gamma=.3, backend='ssm')
        changed = aux.make_variant(prepared, member, variant, 31, 'cpu', backend='ssm')
        for name, parameter in core.named_parameters():
            assert torch.equal(parameter, dict(changed.named_parameters())[name])
        assert torch.equal(changed.omics.sequence.orders, aux.locked_orders(5))
        assert not torch.equal(changed.omics.sequence.orders, core.omics.sequence.orders)
        for row in changed.omics.sequence.orders:
            assert torch.equal(torch.sort(row).values, torch.arange(5))


def test_order_null_is_fixed_across_training_seeds_not_newly_sampled_per_epoch():
    prepared, member = fixture()
    a = aux.make_variant(prepared, member, 'O5_RANDOM_ORDERS', 31, 'cpu', backend='ssm')
    b = aux.make_variant(prepared, member, 'O5_RANDOM_ORDERS', 47, 'cpu', backend='ssm')
    assert torch.equal(a.omics.sequence.orders, b.omics.sequence.orders)
    assert not torch.equal(a.omics.rna.edge_weight, b.omics.rna.edge_weight)


def test_synthetic_qualification_performs_no_optimizer_or_real_cohort_reads(tmp_path):
    result = aux.synthetic_qualification(tmp_path / 'qualification.json')
    assert result['optimizer_steps'] == result['real_cohort_reads'] == 0
    assert len(result['variants']) == 4
    assert all(row['weights_unchanged'] and row['direct_RNA_risk_unchanged'] for row in result['variants'])
    assert result['formal_backend_remains_official_mamba'] is True
