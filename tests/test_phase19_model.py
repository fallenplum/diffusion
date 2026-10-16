"""Mechanism-facing checks of conditioning, identities and strict pairwise risk."""
import itertools

import numpy as np
import torch

from src.phase19_attention.model import AttentionModel, SeparateMolecularTokens


def configuration(dropout=0.):
    return {'hidden': 8, 'rank': 4, 'training': {'dropout': dropout}}


def membership():
    result = np.zeros((8, 16), bool)
    for k in range(8):
        result[k, 2*k:2*k+2] = True
    return result


def model_inputs(batch=2):
    torch.manual_seed(191901)
    dtype = torch.float64
    visual = torch.randn(batch, 5, 1536, dtype=dtype)
    mask = torch.ones(batch, 5, dtype=torch.bool)
    mask[0, -1] = False
    clinical = torch.randn(batch, 3, dtype=dtype)
    r, g, l = [torch.randn(batch, 16, dtype=dtype) for _ in range(3)]
    rm, cm = torch.ones_like(r), torch.ones_like(r)
    x = torch.randn(batch, 7, dtype=dtype)
    return visual, mask, clinical, r, g, l, rm, cm, x


def build(arm, dropout=0.):
    torch.manual_seed(998)
    return AttentionModel(3, membership(), np.arange(7) / 20., np.zeros(3), arm,
                          configuration(dropout)).double()


def test_same_seed_all_seven_arms_have_identical_complete_initial_state():
    models = [build('A' + str(k)) for k in range(7)]
    first = models[0].state_dict()
    for model in models[1:]:
        assert first.keys() == model.state_dict().keys()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(first[name], value, atol=0, rtol=0, msg=name)


def test_24_tokens_keep_rna_and_cna_mask_independent_and_gain_loss_separate():
    torch.manual_seed(19)
    encoder = SeparateMolecularTokens(membership(), 8).double()
    r, g, l = [torch.randn(2, 16, dtype=torch.float64) for _ in range(3)]
    rm, cm = torch.ones_like(r), torch.ones_like(r)
    rm[1] = 0; cm[0] = 0
    token, valid = encoder(r, g, l, rm, cm)
    assert token.shape == (2, 24, 8) and valid.shape == (2, 24)
    assert valid[0, :8].all() and not valid[0, 8:].any()
    assert not valid[1, :8].any() and valid[1, 8:].all()
    assert token[0, 8:].count_nonzero() == 0 and token[1, :8].count_nonzero() == 0
    rg, _ = encoder(r, g + 3., l, rm, cm)
    rl, _ = encoder(r, g, l - 2., rm, cm)
    torch.testing.assert_close(rg[:, :8], token[:, :8], atol=0, rtol=0)
    torch.testing.assert_close(rg[:, 16:], token[:, 16:], atol=0, rtol=0)
    torch.testing.assert_close(rl[:, :16], token[:, :16], atol=0, rtol=0)
    assert not torch.allclose(rg[1, 8:16], token[1, 8:16])
    assert not torch.allclose(rl[1, 16:], token[1, 16:])


def test_fixed_context_keeps_nonzero_global_bilinear_term_and_factor_gradients():
    model, args = build('A2'), model_inputs()
    model.eval()
    with torch.no_grad():
        model.attention.c.weight[:, 1:].normal_(std=.3)
    risk, diag = model(*args, diagnostic=True)
    assert diag['trilinear_scores'].norm() > 0
    wrong = model(*args, context_clinical=args[2].flip(0))
    torch.testing.assert_close(risk, wrong, atol=0, rtol=0)
    risk.square().sum().backward()
    assert model.attention.a.weight.grad.norm() > 0
    assert model.attention.b.weight.grad.norm() > 0
    # Fixed clinical mean is zero, while its augmented intercept stays one.
    assert model.attention.c.weight.grad[:, 0].norm() > 0
    assert model.attention.c.weight.grad[:, 1:].count_nonzero() == 0


def test_dynamic_clinical_starts_equal_but_context_parameters_can_learn():
    model, args = build('A3'), model_inputs()
    model.eval()
    risk = model(*args)
    wrong = model(*args, context_clinical=-args[2])
    torch.testing.assert_close(risk, wrong, atol=0, rtol=0)
    assert model.attention.c.weight[:, 1:].count_nonzero() == 0
    risk.square().sum().backward()
    assert model.attention.c.weight.grad[:, 1:].norm() > 0


def test_wrong_context_changes_interaction_and_preserves_matched_main_effect():
    model, args = build('A3'), model_inputs()
    model.eval()
    with torch.no_grad():
        model.attention.c.weight[:, 1:].normal_(std=.8)
    risk, matched = model(*args, diagnostic=True)
    wrong, altered = model(*args, context_clinical=args[2].flip(0), diagnostic=True)
    torch.testing.assert_close(matched['main_risk'], altered['main_risk'], atol=0, rtol=0)
    torch.testing.assert_close(risk - wrong, matched['interaction_risk'] - altered['interaction_risk'], atol=1e-13, rtol=1e-13)
    assert not torch.allclose(matched['interaction_risk'], altered['interaction_risk'], atol=1e-12, rtol=1e-12)
    wrong_o, changed_o = model(*args, context_omics=tuple(v.flip(0) for v in args[3:8]), diagnostic=True)
    torch.testing.assert_close(matched['main_risk'], changed_o['main_risk'], atol=0, rtol=0)
    assert not torch.allclose(risk, wrong_o, atol=1e-12, rtol=1e-12)


def test_strict_pairwise_risk_has_zero_third_mixed_finite_difference():
    # Hold raw main_x fixed, alter the three interaction inputs independently.
    # An unrestricted post-sum nonlinear head would violate the A6 zero here.
    base = model_inputs(batch=1)
    torch.manual_seed(777)
    p_alt = base[0] + torch.randn_like(base[0]) * .7
    r_alt = base[3] + torch.randn_like(base[3]) * 2.
    c_alt = base[2] + torch.randn_like(base[2]) * 2.
    differences = {}
    for arm in ['A6', 'A3']:
        model = build(arm).eval()
        with torch.no_grad():
            model.attention.c.weight[:, 1:].normal_(std=.8)
            model.head[-1].weight.mul_(10)
        total = torch.zeros(1, dtype=torch.float64)
        for pp, rr, cc in itertools.product([0, 1], repeat=3):
            inputs = list(base)
            inputs[0] = p_alt if pp else base[0]
            inputs[3] = r_alt if rr else base[3]
            inputs[2] = c_alt if cc else base[2]
            sign = (-1) ** (3 - pp - rr - cc)
            total = total + sign * model(*inputs)
        differences[arm] = total
    torch.testing.assert_close(differences['A6'], torch.zeros(1, dtype=torch.float64), atol=1e-12, rtol=0)
    assert differences['A3'].abs().max() > 1e-9
