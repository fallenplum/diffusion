"""Actual pinned-source qualification; unavailable source fails, never skips."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.phase19_attention.baselines import (
    CAMRCore, MGCTCore, MMPCore, MANIFEST, _external_root, _groups,
    build_baseline, official_definitions, panther_tokens, verify_baseline_sources,
)


def membership():
    member = np.zeros((8, 512), bool)
    for k in range(8):
        member[k, k*64:(k+1)*64] = True
    return member


def rows(batch=3):
    torch.manual_seed(191921)
    result = {k: torch.randn(batch, 512) for k in ['R', 'G', 'L']}
    result.update({k: torch.ones(batch, 512) for k in ['RM', 'CM']})
    result.update({k: torch.randn(batch, 32) for k in ['RPCA32', 'CNAPCA32', 'UNI32']})
    result['C'] = torch.randn(batch, 4)
    return result


def config(**extra):
    return {'training': {'dropout': 0.}, **extra}


def test_pinned_sources_verified_and_changed_or_missing_source_fails(tmp_path):
    verified = verify_baseline_sources()
    assert verified['status'] == 'PASS' and verified['files_verified'] == 10
    manifest = json.loads(MANIFEST.read_text())
    root = _external_root()
    for name, spec in manifest['repositories'].items():
        for relative in spec['files']:
            target = tmp_path / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name / relative).read_bytes())
    override = {'baseline_external_root': str(tmp_path)}
    assert verify_baseline_sources(override)['status'] == 'PASS'
    changed = tmp_path/'MGCT/models/MGCT.py'
    changed.write_bytes(changed.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='changed'):
        verify_baseline_sources(override)
    changed.unlink()
    with pytest.raises(FileNotFoundError, match='missing'):
        verify_baseline_sources(override)


def test_mgct_batch_port_matches_native_layer_outputs_and_all_gradients():
    torch.manual_seed(191922)
    env = official_definitions('MGCT')
    native = env['CrossModalityFusionLayer'](drop=0.).double().eval()
    batched = copy.deepcopy(native)
    p = torch.randn(7, 1, 256, dtype=torch.float64, requires_grad=True)
    o = torch.randn(8, 1, 256, dtype=torch.float64, requires_grad=True)
    bp, bo = p.detach().clone().requires_grad_(), o.detach().clone().requires_grad_()
    expected, _ = native(p, o)
    actual = MGCTCore.batch_layer(batched, bp, bo).squeeze(0)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    loss_weights = torch.randn_like(actual)
    (actual * loss_weights).sum().backward()
    (expected * loss_weights).sum().backward()
    torch.testing.assert_close(p.grad, bp.grad, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(o.grad, bo.grad, atol=2e-12, rtol=2e-12)
    for (name, a), (other, b) in zip(native.named_parameters(), batched.named_parameters()):
        assert name == other and (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-12, rtol=2e-12, msg=name)
    # Batched masked outputs must agree with each patient's unpadded native path.
    p2 = torch.randn(9, 2, 256, dtype=torch.float64)
    o2 = torch.randn(8, 2, 256, dtype=torch.float64)
    padding = torch.zeros(2, 9, dtype=torch.bool); padding[0, 5:] = True
    port = MGCTCore.batch_layer(batched, p2, o2, padding)
    for i, length in [(0, 5), (1, 9)]:
        reference, _ = native(p2[:length, i:i+1], o2[:, i:i+1])
        torch.testing.assert_close(port[i], reference, atol=2e-12, rtol=2e-12)


def test_mgct_full_official_core_has_live_gradients_and_native_dead_stage2():
    torch.manual_seed(191923)
    core = build_baseline('MGCT', 4, membership(), config()).eval()
    visual = torch.randn(3, 7, 1536)
    mask = torch.ones(3, 7, dtype=torch.bool); mask[0, -2:] = False
    f = rows()
    first = core(visual, mask, f)['risk']
    changed = visual.clone(); changed[~mask] = 1000.
    torch.testing.assert_close(core(changed, mask, f)['risk'], first, atol=0, rtol=0)
    assert first.shape == (3,) and torch.isfinite(first).all()
    first.square().sum().backward()
    assert core.core.wsi_net[0].weight.grad.norm() > 0
    assert core.core.omic_net[0][0][0].weight.grad.norm() > 0
    assert all(p.grad is None for p in core.core.stage2_layers[0].parameters())
    assert core.core.stage2_layers[1].wsi_omic_attn.in_proj_weight.grad.norm() > 0


def test_camr_exact_third_query_repair_and_retained_native_core():
    torch.manual_seed(191924)
    original = official_definitions('CAMR')['MisaLMFGatedRec'](32, 1, hidden_size1=32, dropout=0.).eval()
    repaired = official_definitions('CAMR', repaired=True)['MisaLMFGatedRec'](32, 1, hidden_size1=32, dropout=0.).eval()
    repaired.load_state_dict(original.state_dict(), strict=True)
    f = rows(); inputs = CAMRCore.inputs(f)
    query_original, query_repaired = [], []
    h1 = original.linear_z1.register_forward_pre_hook(lambda _, x: query_original.append(x[0].detach().clone()))
    h2 = repaired.linear_z1.register_forward_pre_hook(lambda _, x: query_repaired.append(x[0].detach().clone()))
    a, b = original(*inputs), repaired(*inputs)
    h1.remove(); h2.remove()
    assert len(query_original) == len(query_repaired) == 3
    torch.testing.assert_close(query_original[0], query_repaired[0], atol=0, rtol=0)
    torch.testing.assert_close(query_original[1], query_repaired[1], atol=0, rtol=0)
    torch.testing.assert_close(query_original[2], a[8], atol=0, rtol=0)
    torch.testing.assert_close(query_repaired[2], b[9], atol=0, rtol=0)
    assert not torch.allclose(query_original[2], query_repaired[2])
    for i in [1, 3, 5, 7, 8, 9, 10, 11, 12]:
        torch.testing.assert_close(a[i], b[i], atol=0, rtol=0)
    assert a[0].shape == b[0].shape == (3, 1)
    assert len(a) == 13 and all(torch.isfinite(x).all() for x in b)
    assert repaired.lmf.rank == 16


def test_camr_discriminator_and_generator_have_separate_real_gradient_paths():
    torch.manual_seed(191925)
    core = CAMRCore(config()).train()
    f, visual, mask = rows(), torch.randn(3, 4, 1536), torch.ones(3, 4, dtype=torch.bool)
    dparams = list(core.discriminator_parameters())
    gparams = list(core.generator_parameters())
    assert not {id(p) for p in dparams} & {id(p) for p in gparams}
    dopt = torch.optim.Adam(dparams, lr=1e-3)
    before = [p.detach().clone() for p in dparams]
    dloss = core.discriminator_loss(visual, mask, f)
    dloss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in dparams)
    assert all(p.grad is None for p in gparams)
    dopt.step()
    assert any(not torch.equal(p, q) for p, q in zip(dparams, before))
    core.zero_grad(set_to_none=True)
    result = core(visual, mask, f)
    assert result['risk'].shape == (3,) and result['extra_loss'] > 0
    (result['risk'].square().sum() + result['extra_loss']).backward()
    assert all(p.grad is None for p in dparams)
    assert core.core.common[0].weight.grad.norm() > 0
    assert core.core.rec1[0].weight.grad.norm() > 0
    assert core.core.unique3[0].weight.grad.norm() > 0


def test_mmp_official_em_tokenizer_match_and_padding_invariance():
    torch.manual_seed(191926)
    env = official_definitions('MMP')
    x = torch.randn(2, 11, 12) * .2
    mask = torch.ones(2, 11, dtype=torch.bool); mask[0, -3:] = False
    centroids = torch.randn(16, 12) * .1
    actual = panther_tokens(x, mask, centroids)
    official = env['PANTHERBase'](12, p=16, L=3, tau=10., out='allcat', ot_eps=.1,
                                 load_proto=False, fix_proto=True)
    with torch.no_grad():
        official.priors.m.copy_(centroids)
        flat, _ = official(x.masked_fill(~mask[..., None], 0), mask.float())
        prob, mean, cov = env['PrototypeTokenizer']('PANTHER', 'allcat', 16)(flat)
        expected = torch.cat([prob[..., None], mean, cov], -1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[..., 0].sum(-1), torch.ones(2), atol=1e-6, rtol=1e-6)
    changed = x.clone(); changed[~mask] = float('nan')
    torch.testing.assert_close(panther_tokens(changed, mask, centroids), actual, atol=0, rtol=0)
    # Removing padded slots entirely must leave the same per-WSI EM estimate.
    for i in range(2):
        unpadded = panther_tokens(x[i:i+1, mask[i]], torch.ones(1, int(mask[i].sum()), dtype=torch.bool), centroids)
        torch.testing.assert_close(unpadded[0], actual[i], atol=1e-7, rtol=1e-6)


def test_mmp_fit_only_prototypes_private_cache_and_native_forward_parity(tmp_path):
    torch.manual_seed(191927)
    # No survival endpoint exists in this data object: preparation cannot read it.
    visual = torch.randn(6, 10, 1536) * .1
    mask = torch.ones(6, 10, dtype=torch.bool)
    owner, fit = np.arange(6), np.array([0, 1, 2, 3])
    ds = SimpleNamespace(visual=visual, mask=mask, owner=owner, fit=fit,
             global_wsi_indices=np.array([3, 7, 12, 19, 31, 43]),
             fold=1, role='inner', audit={'cache_sha256': 'synthetic-no-labels',
                 'X_blocks': {'C': (0, 4)}, 'membership_sizes': np.array([64]*8)})
    cfg = config(private_root=str(tmp_path))
    core = MMPCore(membership(), cfg).eval()
    with pytest.raises(RuntimeError, match='prepared'):
        core(visual[:3], mask[:3], rows())
    receipt = core.prepare_data(ds)
    assert receipt['labels_used'] is False and receipt['fit_patient_count'] == 4
    assert receipt['fit_patch_pool_count'] == 40 and receipt['prototype_count'] == 16
    assert core.prototype_features.shape == (6, 16, 3073)
    assert 'prototype_features' not in core.state_dict()
    f = rows()
    ids = np.array([43, 7, 31])
    positions = [5, 1, 4]
    actual = core(visual[positions], mask[positions], f, wsi_indices=ids)['risk']
    expected = core.core.forward_no_loss(core.prototype_features[positions],
                         _groups(f, core.membership))['logits'].reshape(-1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    assert core.core.path_proj_net[0].weight.grad.norm() > 0
    assert core.core.sig_networks[0][0][0].weight.grad.norm() > 0
    # A cache load must reuse the exact source/fold/fit-specific frozen prior.
    loaded = MMPCore(membership(), cfg).eval()
    loaded.prepare_data(ds)
    torch.testing.assert_close(loaded.prototype_features, core.prototype_features, atol=0, rtol=0)
    # Mutate held-out visual only: no prototype is allowed to change.
    paths = list((tmp_path/'baseline_features').glob('*.pt'))
    assert len(paths) == 1
    bank_before = torch.load(paths[0], weights_only=False)
    changed = copy.copy(ds); changed.visual = visual.clone(); changed.visual[4:] += .5
    changed.audit = {'cache_sha256': 'different-heldout-visual'}
    other = MMPCore(membership(), cfg).eval(); other.prepare_data(changed)
    paths = list((tmp_path/'baseline_features').glob('*.pt'))
    assert len(paths) == 2
    banks = [torch.load(p, weights_only=False) for p in paths]
    for bank in banks:
        np.testing.assert_array_equal(bank['centroids'], bank_before['centroids'])
    assert not torch.allclose(other.prototype_features[4:], core.prototype_features[4:])
    with pytest.raises(ValueError, match='unknown WSI'):
        loaded(visual[:3], mask[:3], f, wsi_indices=[900, 7, 31])
