import numpy as np
import torch
from types import SimpleNamespace
from scripts.bgb_en_f7_determinism_diagnostic import collect_forward, compare_captures, compare_topk, topk_boundary
from src.bgb_en.interaction import masked_softmax
from src.bgb_en.training import MechanismModel


def test_competition_topk_amplifies_a_tiny_boundary_perturbation_without_randomness():
    # Exactly the frozen competition -> fixed top64 -> key normalization. The
    # test demonstrates sensitivity, not that CUDA nondeterminism was observed.
    score = torch.zeros(1, 1, 2, 65, dtype=torch.float64)
    score[..., 0, :63] = .1
    a, b = score.clone(), score.clone()
    a[..., 0, 63], a[..., 0, 64] = 1e-7, -1e-7
    b[..., 0, 63], b[..., 0, 64] = -1e-7, 1e-7
    valid = torch.ones_like(score, dtype=torch.bool)
    def frozen_attention(value):
        weight = masked_softmax(value, valid, dim=-2)
        chosen = torch.topk(weight, 64, dim=-1).indices
        selected = torch.zeros_like(valid).scatter(-1, chosen, True)
        weight = weight*selected
        return weight/weight.sum(-1, keepdim=True)
    wa, wb = frozen_attention(a), frozen_attention(b)
    audit = compare_topk(a, b, wa, wb)
    values = torch.zeros(65); values[63], values[64] = 100., -100.
    assert audit['score_max_abs_error'] == 2e-7
    assert audit['changed_support_query_rows'] > 0
    assert float(((wa-wb)*values).sum(-1).abs().max()) > 1.
    assert 0 < topk_boundary(a, valid)['mean_margin'] < 1e-6


def test_boundary_summary_uses_full_valid_keys_including_unselected_ones():
    score = torch.zeros(1, 1, 2, 66)
    valid = torch.ones_like(score, dtype=torch.bool)
    assert topk_boundary(score, valid)['exact_boundary_ties'] == 2
    valid[..., 64:] = False
    assert topk_boundary(score, valid)['rows_with_more_than_topk_keys'] == 0


def test_actual_f7_cpu_no_grad_grad_replay_is_readonly_and_matches_fixed_shape():
    torch.manual_seed(741)
    rng = np.random.default_rng(741)
    n, genes = 8, 12
    features = dict(rna=rng.normal(size=(n, genes)).astype(np.float32), rna_mask=np.ones((n, genes), bool),
        cna_gain=np.zeros((n, genes), np.float32), cna_loss=np.zeros((n, genes), np.float32),
        cna_mask=np.ones((n, genes), bool), C=np.zeros((n, 2), np.float32))
    prepared = dict(features=features, train=np.arange(4), valid=np.arange(4, 8), base=np.zeros(n),
                    RNA=np.zeros(n), CS=np.zeros(n), beta=1.)
    bank = SimpleNamespace(x=torch.randn(n, 1, 66, 768), mask=torch.ones(n, 1, 66, dtype=torch.bool),
        xy=torch.zeros(n, 1, 66, 2), region_ids=torch.zeros(n, 1, 66, dtype=torch.long), step=1., input_dim=768)
    bank.xy[..., 0] = torch.arange(66)
    membership = torch.arange(genes)[None, :]//4 == torch.arange(3)[:, None]
    model = MechanismModel('R3', 'F7', prepared, membership, bank, 'cpu').train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    no_grad = collect_forward(model, torch.arange(4), False)
    grad = collect_forward(model, torch.arange(4), True)
    comparison = compare_captures(no_grad, grad)
    assert comparison['risk_within_unchanged_gradient_cache_threshold']
    assert comparison['topk']['changed_support_query_rows'] == 0
    assert 'omics.rna:0' in no_grad and 'interaction.path_match:0' in no_grad
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())
