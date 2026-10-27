import copy

import pytest
import torch

from src.bgb_en.omics import (SparsePathwayEncoder, SparseProgramPool, SetAttention,
                             DiagonalSelectiveSSM, MultiOrderSequence,
                             load_gmt_membership, shuffled_membership)


def fixture(mode='pool'):
    torch.manual_seed(172)
    membership = torch.tensor([[1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0]]).bool()
    model = SparsePathwayEncoder(membership, mode=mode).double()
    rna = torch.randn(3, 6, dtype=torch.float64)
    mask = torch.ones_like(rna, dtype=torch.bool)
    mask[1, :3] = False
    mask[2] = False
    return model, rna, mask


@pytest.mark.parametrize('mode', ['pool', 'set', 'ssm', 'set_ssm', 'factors'])
def test_finite_nonzero_molecular_gradient_and_masked_empty_patient(mode):
    model, rna, mask = fixture(mode)
    rna.requires_grad_(True)
    out = model(rna, mask)
    assert out['tokens'].shape == (3, 3, 32)
    assert out['embedding'].shape == (3, 32)
    assert torch.equal(out['tokens'][2], torch.zeros_like(out['tokens'][2]))
    assert out['risk'][2] == 0
    out['risk'].square().sum().backward()
    assert torch.isfinite(rna.grad).all()
    assert rna.grad[mask].abs().sum() > 0
    assert rna.grad[~mask].abs().sum() == 0


def test_nonfinite_missing_values_are_not_observations():
    model, rna, mask = fixture()
    changed = rna.clone(); changed[~mask] = float('nan')
    assert torch.equal(model(rna, mask)['risk'], model(changed, mask)['risk'])
    changed[0, 0] = float('nan')
    with pytest.raises(ValueError, match='observed'):
        model(changed, mask)


@pytest.mark.parametrize('mode', ['pool', 'set', 'ssm', 'set_ssm', 'cna_gate'])
def test_joint_gene_order_alignment(mode):
    model, rna, mask = fixture(mode)
    gain, loss = rna.abs(), rna.square()
    permutation = torch.tensor([5, 1, 3, 0, 4, 2])
    if mode == 'cna_gate':
        before = model(rna, mask, gain, loss, mask)['risk']
        after = model(rna[:, permutation], mask[:, permutation], gain[:, permutation],
                      loss[:, permutation], mask[:, permutation], gene_ids=permutation)['risk']
    else:
        before = model(rna, mask)['risk']
        after = model(rna[:, permutation], mask[:, permutation], gene_ids=permutation)['risk']
    assert torch.equal(before, after)


def test_sparse_projection_has_multiple_gene_patterns_not_only_mean():
    torch.manual_seed(7)
    pool = SparseProgramPool(torch.ones(1, 4).bool(), patterns=4).double()
    x = torch.tensor([[1., -1., 2., -2.], [-1., 1., -2., 2.]], dtype=torch.float64)
    _, _, _, patterns = pool(x, torch.ones_like(x).bool())
    assert patterns[0, 0, 0] == patterns[1, 0, 0] == 0
    assert not torch.equal(patterns[0, 0, 1:], patterns[1, 0, 1:])


def test_pathway_set_attention_permutation_equivariance():
    torch.manual_seed(8)
    block = SetAttention().double()
    x = torch.randn(2, 5, 32, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 0, 1, 0], [0, 0, 0, 0, 0]]).bool()
    order = torch.tensor([4, 2, 0, 1, 3])
    assert torch.allclose(block(x[:, order], mask[:, order]), block(x, mask)[:, order], atol=1e-14)


def test_shuffled_annotation_preserves_size_overlap_and_fixed_reproducibility():
    member = torch.eye(8).bool(); member[:, :3] = True
    shuffled, order = shuffled_membership(member, seed=98)
    again, again_order = shuffled_membership(member, seed=98)
    assert torch.equal(shuffled, again) and torch.equal(order, again_order)
    assert torch.equal(shuffled.sum(1), member.sum(1))
    assert torch.equal(shuffled.int() @ shuffled.int().T, member.int() @ member.int().T)
    assert not torch.equal(shuffled, member)


def test_cna_gain_loss_remain_distinct_and_missing_cna_is_neutral():
    model, rna, mask = fixture('cna_gate')
    gain = torch.ones_like(rna); loss = torch.zeros_like(rna)
    observed = torch.ones_like(mask)
    a = model(rna, mask, gain, loss, observed)
    b = model(rna, mask, loss, gain, observed)
    assert not torch.allclose(a['risk'], b['risk'])
    missing = torch.zeros_like(mask)
    a = model(rna, mask, gain * float('nan'), loss * float('nan'), missing)
    rna_tokens = model.rna(rna, mask)[0]
    assert torch.equal(a['tokens'], rna_tokens)
    with pytest.raises(ValueError, match='explicit measurement mask'):
        model(rna, mask)


def test_cna_centered_channels_are_allowed_but_axes_must_align():
    model, rna, mask = fixture('cna_gate')
    result = model(rna, mask, rna - 2., rna + .7, mask)
    assert torch.isfinite(result['risk']).all()
    with pytest.raises(ValueError, match='align'):
        model(rna, mask, rna[:, :4], rna, mask)


def test_explicit_ssm_invalid_tokens_do_not_modify_recurrent_state():
    torch.manual_seed(6)
    block = DiagonalSelectiveSSM().double()
    x = torch.randn(1, 4, 32, dtype=torch.float64)
    valid = torch.tensor([[1, 0, 1, 0]]).bool()
    full = block(x, valid)
    compact = block(x[:, [0, 2]], torch.ones(1, 2).bool())
    # Linear projection uses different GEMM row shapes for padded/compact
    # batches. CPU BLAS can differ by one double ULP before the same scan.
    assert torch.allclose(full[:, [0, 2]], compact, atol=1e-14, rtol=1e-14)
    assert torch.equal(full[:, [1, 3]], torch.zeros_like(full[:, [1, 3]]))


def test_gmt_exact_symbol_mapping_and_hash_guard(tmp_path):
    p = tmp_path / 'fixed.gmt'
    p.write_text('P1\turl1\tA\tB\tA\nP2\turl2\tB\tC\n', encoding='utf-8')
    matrix, audit = load_gmt_membership(p, ['C', 'B', 'D'])
    assert matrix.tolist() == [[False, True, False], [True, True, False]]
    assert audit['external_gene_counts'] == [2, 2]
    with pytest.raises(ValueError, match='SHA256'):
        load_gmt_membership(p, ['C', 'B', 'D'], expected_sha256='wrong')


def test_factor_widths_and_no_data_selection_api():
    member = torch.ones(2, 7).bool()
    for rank in (2, 4, 8):
        model = SparsePathwayEncoder(member, mode='factors', factors=rank)
        result = model(torch.randn(2, 7), torch.ones(2, 7).bool())
        assert result['factors'].shape == (2, rank)
        assert model.parameter_audit()['direct_rna_risk_is_external']
    with pytest.raises(ValueError, match='2, 4 or 8'):
        SparsePathwayEncoder(member, mode='factors', factors=3)


def test_mamba_equal_length_compaction_matches_single_patient_output_and_gradient():
    class SyntheticScan(torch.nn.Module):
        """Deterministic CPU scan API stand-in, explicitly not official Mamba."""
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(32, 32)
            self.calls = []

        def forward(self, x):
            self.calls.append(tuple(x.shape))
            return torch.cumsum(self.projection(x), dim=1)

    torch.manual_seed(149)
    batched = MultiOrderSequence(6, backend='ssm').double()
    batched.backend = 'mamba'
    batched.scan = SyntheticScan().double()
    legacy = copy.deepcopy(batched)
    values = torch.randn(5, 6, 32, dtype=torch.float64, requires_grad=True)
    old_values = values.detach().clone().requires_grad_(True)
    valid = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]).bool()
    actual = batched(values, valid)
    normalized = legacy.norm(old_values)
    messages = []
    for order in legacy.orders:
        x, m = normalized[:, order], valid[:, order]
        out = torch.zeros_like(x)
        for patient in range(len(x)):
            if m[patient].any():
                out[patient, m[patient]] = legacy.scan(x[patient, m[patient]][None])[0]
        messages.append(out[:, torch.argsort(order)])
    expected = (old_values + torch.stack(messages).mean(0)) * valid[..., None]
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(actual[~valid], torch.zeros_like(actual[~valid]))
    actual.square().mean().backward()
    expected.square().mean().backward()
    assert torch.allclose(values.grad, old_values.grad, atol=1e-12, rtol=1e-12)
    for a, b in zip(batched.parameters(), legacy.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-12, rtol=1e-12)
    assert len(batched.scan.calls) == 2 * len(batched.orders)
    assert len(legacy.scan.calls) == 4 * len(legacy.orders)
    assert all(shape[1] in (3, 6) for shape in batched.scan.calls)
