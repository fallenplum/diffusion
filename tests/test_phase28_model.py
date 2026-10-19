import copy
import pytest
import torch
from torch import nn
from src.phase28_reread.model import BidirectionalReread, EvidenceRead, SharedRouting, masked_softmax


class SyntheticSpatial(nn.Module):
    """Explicit synthetic-only identity; never constructed by the formal runner."""
    def forward(self, value, plan):
        return value


def fixture(arm='C11', dtype=torch.float64):
    torch.manual_seed(2801)
    genes = 17
    membership = torch.zeros(108, genes, dtype=torch.bool)
    for k in range(100):
        membership[k, (k // 2) % genes] = True
        membership[k, ((k // 2) + 3) % genes] = True
        membership[k, ((k // 2) + 7) % genes] = True
    membership[100:] = True
    model = BidirectionalReread(genes, membership, SyntheticSpatial(), arm=arm, path_dim=12).to(dtype)
    x = torch.randn(genes, 3, dtype=dtype, requires_grad=True)
    mask = torch.ones_like(x, dtype=torch.bool); mask[0] = False; mask[1, 1:] = False
    ids = torch.arange(genes)
    slides = [{'features': torch.randn(9, 12, dtype=dtype),
               'plan': {'region_indices': [[0, 1, 2, 3], [4, 5, 6, 7, 8]]}}]
    return model, x, mask, ids, slides


def test_empty_mask_rows_are_zero_and_have_finite_gradients():
    score = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[0] * 5, [1, 0, 1, 0, 0], [1] * 5], dtype=torch.bool)
    result = masked_softmax(score, mask)
    assert torch.equal(result[0], torch.zeros(5, dtype=result.dtype))
    assert torch.allclose(result[1:].sum(-1), torch.ones(2, dtype=result.dtype))
    result.square().sum().backward()
    assert torch.isfinite(score.grad).all() and torch.equal(score.grad[0], torch.zeros(5, dtype=result.dtype))


def test_singleton_original_evidence_cannot_change_by_query_only():
    torch.manual_seed(2)
    reader = EvidenceRead(7).double()
    e = torch.randn(1, 7, dtype=torch.float64)
    support = torch.ones(4, 1, dtype=torch.bool)
    a, _ = reader(torch.randn(4, 128, dtype=torch.float64), reader.prepare(e), support)
    b, _ = reader(torch.randn(4, 128, dtype=torch.float64), reader.prepare(e), support)
    assert torch.equal(a, b)


def test_shared_score_null_equations_and_invalid_sources():
    torch.manual_seed(3)
    router = SharedRouting().double()
    p, g = torch.randn(3, 128, dtype=torch.float64), torch.randn(5, 128, dtype=torch.float64)
    pv, gv = torch.tensor([True, True, False]), torch.tensor([True, False, True, True, False])
    mp, mg, d = router(p, g, pv, gv)
    score = d['score']
    hand_p = torch.softmax(torch.cat((score.masked_fill(~gv[None, None, :], -torch.inf), torch.zeros(4, 3, 1)), -1), -1)
    hand_g = torch.softmax(torch.cat((score.transpose(-1, -2).masked_fill(~pv[None, None, :], -torch.inf), torch.zeros(4, 5, 1)), -1), -1)
    assert torch.allclose(d['path_probability'], hand_p)
    assert torch.allclose(d['gene_probability'], hand_g)
    assert torch.equal(mp[2], torch.zeros(128, dtype=mp.dtype))
    assert torch.equal(mg[1], torch.zeros(128, dtype=mg.dtype))
    zero_p, _, no_genes = router(p, g, pv, torch.zeros(5, dtype=torch.bool))
    assert torch.equal(zero_p, torch.zeros_like(zero_p))
    assert torch.equal(no_genes['path_probability'][..., -1], torch.ones(4, 3, dtype=p.dtype))


@pytest.mark.parametrize('arm', ['C00', 'C10', 'C01', 'C11', 'DIRECT'])
def test_zero_interaction_exactly_restores_own_learned_base(arm):
    model, x, mask, ids, slides = fixture(arm)
    with torch.no_grad():
        model.main_path.coefficient.normal_(std=.1)
        model.main_gene.axis.normal_(std=.1)
    result = model(x, mask, ids, slides, offset=.7, interaction_off=True)
    assert torch.equal(result['risk'], result['base'])
    assert result['interaction'] == 0
    encoded = model.encode(x, mask, ids, slides)
    assert all(torch.equal(a, b) for a, b in zip(result['path_final'], encoded['path0']))
    assert torch.equal(result['gene_final'], encoded['gene0'])


def test_zero_bias_free_query_projections_give_exact_zero_difference():
    model, x, mask, ids, slides = fixture()
    nn.init.zeros_(model.up.weight); nn.init.zeros_(model.ug.weight)
    result = model(x, mask, ids, slides, details=True)
    assert result['interaction'] == 0
    assert all(torch.equal(row['difference'], torch.zeros_like(row['difference'])) for layer in result['trace'] for row in layer['path'])
    assert all(torch.equal(layer['gene_difference'], torch.zeros_like(layer['gene_difference'])) for layer in result['trace'])


def test_joint_gene_identity_value_mask_permutation_preserves_predictions():
    model, x, mask, ids, slides = fixture()
    expected = model(x, mask, ids, slides)
    permutation = torch.randperm(len(ids))
    changed = model(x[permutation], mask[permutation], ids[permutation], slides)
    assert torch.allclose(expected['risk'], changed['risk'], rtol=0, atol=1e-12)
    assert torch.allclose(expected['gene_final'], changed['gene_final'], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match='identities'):
        model(x, mask, torch.zeros_like(ids), slides)


def test_naturally_missing_values_and_identity_do_not_generate_gene_evidence():
    model, x, mask, ids, slides = fixture()
    mask[:] = False
    encoded = model.encode(x, mask, ids, slides)
    assert not encoded['gene_valid'].any()
    assert torch.equal(encoded['gene_evidence'], torch.zeros_like(encoded['gene_evidence']))
    assert torch.equal(encoded['gene0'], torch.zeros_like(encoded['gene0']))
    result = model.read_encoded(encoded, 0.)
    assert result['path_interaction'] == result['gene_interaction'] == 0


def test_constant_value_control_cancels_reread_without_changing_main_anchor():
    model, x, mask, ids, slides = fixture()
    encoded = model.encode(x, mask, ids, slides)
    normal = model.read_encoded(encoded, .7)
    control = model.read_encoded(encoded, .7, constant_path_value=True, constant_gene_value=True)
    assert control['interaction'] == 0 and torch.equal(normal['base'], control['base'])


def test_multislide_aggregation_is_equal_WSI_mean_and_order_invariant():
    model, x, mask, ids, slides = fixture()
    other = {'features': torch.randn(4, 12, dtype=x.dtype), 'plan': {'region_indices': [[0, 1, 2, 3]]}}
    a = model(x, mask, ids, slides + [other])
    b = model(x, mask, ids, [other] + slides)
    assert torch.allclose(a['risk'], b['risk'], atol=1e-12, rtol=0)
    single = model(x, mask, ids, slides)
    duplicate = model(x, mask, ids, slides + slides)
    assert torch.equal(single['risk'], duplicate['risk'])


def test_cross_direction_gradients_are_nonzero_without_auxiliary_forcing():
    model, x, mask, ids, slides = fixture()
    encoded = model.encode(x, mask, ids, slides)
    result = model.read_encoded(encoded, .7, details=True)
    grad_gene = torch.autograd.grad(result['path_interaction'], x, retain_graph=True)[0]
    grad_patch = torch.autograd.grad(result['gene_interaction'], encoded['path_evidence'], retain_graph=True)[0]
    assert torch.isfinite(grad_gene).all() and grad_gene.abs().max() > 0
    assert torch.isfinite(grad_patch).all() and grad_patch.abs().max() > 0
    for layer in result['trace']:
        assert layer['gene_change'].abs().max() <= .1
        assert all(v['change'].abs().max() <= .1 for v in layer['path'])


def test_frozen_own_context_bank_replays_matched_layers_exactly():
    model, x, mask, ids, slides = fixture()
    encoded = model.encode(x, mask, ids, slides)
    expected = model.read_encoded(encoded, .7)
    context = [{'gene': v['gene'], 'gene_valid': encoded['gene_valid'],
                'path': v['path'], 'path_valid': encoded['path_valid']} for v in expected['states']]
    replay = model.read_encoded(encoded, .7, context=context)
    assert torch.equal(expected['risk'], replay['risk'])
    assert torch.equal(expected['gene_final'], replay['gene_final'])


def test_foreign_context_trace_reports_actual_graphs_in_each_direction():
    model, x, mask, ids, slides = fixture()
    encoded = model.encode(x, mask, ids, slides)
    normal = model.read_encoded(encoded, .7)
    context = [{'gene': v['gene']+1, 'gene_valid': encoded['gene_valid'],
                'path': [p-1 for p in v['path']], 'path_valid': encoded['path_valid']}
               for v in normal['states']]
    result = model.read_encoded(encoded, .7, context=context, details=True)
    first = result['trace'][0]
    _, _, expected_path = model.routing(encoded['path0'][0], context[0]['gene'], encoded['path_valid'][0], encoded['gene_valid'])
    _, _, expected_gene = model.routing(context[0]['path'][0], encoded['gene0'], encoded['path_valid'][0], encoded['gene_valid'])
    assert torch.equal(first['actual_path_routing'][0]['score'], expected_path['score'])
    assert torch.equal(first['actual_gene_routing'][0]['score'], expected_gene['score'])
    assert not torch.equal(first['actual_path_routing'][0]['score'], first['own_routing_reference'][0]['score'])
