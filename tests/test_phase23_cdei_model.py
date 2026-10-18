import copy
import numpy as np
import pytest
import torch
from src.phase23_cdei.model import CompatibilityBackbone, FrozenDualEvidence, module_fingerprint
from src.phase23_cdei.reference import FrozenPatientReferenceBank


@pytest.fixture
def example():
    torch.set_num_threads(2); torch.manual_seed(230018)
    member = torch.zeros(8, 16, dtype=torch.bool)
    for k in range(8): member[k, 2*k:2*k+2] = True
    model = CompatibilityBackbone(4, member)
    c = torch.randn(6, 4); r = torch.randn(6, 16)
    g, l = torch.rand(6, 16), torch.rand(6, 16)
    rm, cm = torch.ones_like(r, dtype=torch.bool), torch.ones_like(r, dtype=torch.bool)
    model.requires_grad_(False); model.eval()
    with torch.no_grad(): o, om = model.encode_omics(r, g, l, rm, cm)
    bank = FrozenPatientReferenceBank(o, om, c, [0, 1, 2],
               provenance={'kernel_sha256': module_fingerprint(model)})
    extractor = FrozenDualEvidence(model, bank)
    visual, mask = torch.randn(2, 5, 1536), torch.ones(2, 5, dtype=torch.bool)
    mask[:, -1] = False; visual[:, -1] = float('nan')
    return extractor, (visual, mask, c, r, g, l, rm, cm)


def call(extractor, f, ids=(3, 4), **kw):
    v, m, c, r, g, l, rm, cm = f
    ix = list(ids)
    return extractor.extract(v, m, c[ix], r[ix], g[ix], l[ix], rm[ix], cm[ix], torch.tensor(ix), **kw)


def test_real_modules_integrate_and_keep_kernel_and_reference_frozen(example):
    ex, f = example; before = module_fingerprint(ex.backbone)
    reference = ex.reference.public_summary()
    result = call(ex, f)
    assert result['relative'].shape == (2, 64) and result['absolute'].shape == (2, 2)
    assert result['absolute_available'].all()
    for key, x in result.items():
        assert not x.requires_grad and torch.isfinite(x).all(), key
    assert module_fingerprint(ex.backbone) == before and ex.reference.public_summary() == reference
    torch.testing.assert_close(result['absolute'][:, 0],
        result['mean_only'][:, 0]+result['centered_only'][:, 0], atol=1e-12, rtol=1e-12)


def test_wrong_clinical_changes_only_absolute_pathway(example):
    ex, f = example; matched = call(ex, f)
    wrong = call(ex, f, absolute_clinical=f[2][[4, 3]])
    torch.testing.assert_close(wrong['relative'], matched['relative'], rtol=0, atol=0)
    torch.testing.assert_close(wrong['relative_p_to_o'], matched['relative_p_to_o'], rtol=0, atol=0)
    assert not torch.equal(wrong['absolute'], matched['absolute'])


def test_paired_wrong_omics_same_reference_union_and_matched_relative(example):
    ex, f = example; v, m, c, r, g, l, rm, cm = f
    ids, donors = torch.tensor([0, 1]), torch.tensor([1, 0])
    exclusions = torch.stack((ids, donors), -1)
    with torch.no_grad():
        _, own_mask = ex.backbone.encode_omics(r[ids], g[ids], l[ids], rm[ids], cm[ids])
        _, donor_mask = ex.backbone.encode_omics(r[donors], g[donors], l[donors], rm[donors], cm[donors])
    shared_mask = own_mask & donor_mask
    matched = call(ex, f, (0, 1), reference_exclusions=exclusions, absolute_token_mask=shared_mask)
    wrong = call(ex, f, (0, 1), absolute_omics=tuple(x[donors] for x in (r, g, l, rm, cm)),
                 omics_source_indices=donors, reference_exclusions=exclusions, absolute_token_mask=shared_mask)
    a = ex.reference.query_reference_mean(ids, additional_exclusions=exclusions)
    b = ex.reference.query_reference_mean(donors, additional_exclusions=exclusions)
    torch.testing.assert_close(a.tokens, b.tokens, atol=0, rtol=0)
    torch.testing.assert_close(a.counts, b.counts, atol=0, rtol=0)
    torch.testing.assert_close(wrong['relative'], matched['relative'], rtol=0, atol=0)
    assert not torch.equal(wrong['absolute'], matched['absolute'])


def test_wrong_omics_without_same_reference_plan_is_rejected(example):
    ex, f = example
    with pytest.raises(ValueError, match='exclusion'):
        call(ex, f, absolute_omics=tuple(x[[4, 3]] for x in f[3:]), omics_source_indices=[4, 3])


def test_kernel_drift_and_train_mode_are_rejected(example):
    ex, f = example
    with pytest.raises(ValueError, match='training'): ex.train()
    with torch.no_grad(): ex.backbone.compatibility.q.weight[0, 0] += 1.
    with pytest.raises(ValueError, match='drift'): call(ex, f)


def test_unfrozen_or_unbound_kernel_is_rejected(example):
    ex, _ = example; model = copy.deepcopy(ex.backbone)
    model.requires_grad_(True)
    with pytest.raises(ValueError, match='freeze'): FrozenDualEvidence(model, ex.reference)
    model.requires_grad_(False); model.eval()
    with torch.no_grad(): model.compatibility.k.weight[0, 0] += 1.
    with pytest.raises(ValueError, match='reference'): FrozenDualEvidence(model, ex.reference)


def test_no_measured_omics_is_explicitly_unavailable(example):
    ex, f = example; f = list(f)
    f[-2] = torch.zeros_like(f[-2]); f[-1] = torch.zeros_like(f[-1])
    result = call(ex, f)
    assert not result['absolute_available'].any()
    torch.testing.assert_close(result['absolute'], torch.zeros_like(result['absolute']))


@pytest.mark.parametrize('tau', [0, -1, np.nan, np.inf])
def test_invalid_temperature(example, tau):
    ex, _ = example
    with pytest.raises(ValueError): FrozenDualEvidence(ex.backbone, ex.reference, tau=tau)
