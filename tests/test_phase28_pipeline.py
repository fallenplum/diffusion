"""Synthetic integration checks; no clinical data, files or fitted weights."""
import copy
import numpy as np
import pytest
import torch

from src.phase28_reread import train
from src.phase28_reread.training_math import risk_gradient_cache
from tests.test_phase28_model import fixture


def test_foreign_own_bank_replays_multiple_slides_and_preserves_original_values():
    model, x, mask, ids, slides = fixture()
    slides.append({'features': torch.randn(5, 12, dtype=x.dtype),
                   'plan': {'region_indices': [[0, 1], [2, 3, 4]]}})
    encoded = model.encode(x, mask, ids, slides)
    expected = model.read_encoded(encoded, .2)
    original = [v.clone() for v in encoded['path_evidence']]
    bank = train._bank(expected, encoded)
    context = train._foreign(bank, encoded, 'cpu', gene=True, path=True)
    actual = model.read_encoded(encoded, .2, context=context)
    assert torch.equal(expected['risk'], actual['risk'])
    assert all(torch.equal(a, b) for a, b in zip(original, encoded['path_evidence']))


def test_foreign_different_region_counts_only_resample_context():
    model, x, mask, ids, slides = fixture()
    own = model.encode(x, mask, ids, slides)
    foreign_slides = [{'features': torch.randn(3, 12, dtype=x.dtype),
                       'plan': {'region_indices': [[0, 1, 2]]}}]
    foreign = model.encode(x, mask, ids, foreign_slides)
    bank = train._bank(model.read_encoded(foreign, 0.), foreign)
    states = train._foreign(bank, own, 'cpu', path=True)
    assert all(v['path'][0].shape == own['path0'][0].shape for v in states)
    assert all(torch.equal(v['path'][0][:2], v['path'][0][2:]) for v in states)
    assert 'gene' not in states[0]


def test_mean_context_weights_patients_and_slides_equally_not_region_counts():
    def state(slide_values, gene_value):
        return {'path': [torch.full((2*n, 128), float(value)) for n, value in slide_values],
                'path_valid': [torch.ones(2*n, dtype=torch.bool) for n, _ in slide_values],
                'gene': torch.full((108, 128), float(gene_value)),
                'gene_valid': torch.ones(108, dtype=torch.bool)}
    a = state([(1, 2), (4, 6)], 3)
    b = state([(3, 12)], 9)
    answer = train._mean_context([[copy.deepcopy(a)]*2, [copy.deepcopy(b)]*2])
    assert len(answer) == 2
    for v in answer:
        assert torch.equal(v['path'][0], torch.full((2, 128), 8.))
        assert torch.equal(v['gene'], torch.full((108, 128), 6.))


@pytest.mark.parametrize('arm', ['C00', 'C11', 'DIRECT'])
def test_synthetic_full_risk_objective_update_preserves_external_offset(arm):
    model, x, mask, ids, slides = fixture(arm)
    model.train()
    class Data:
        molecular = torch.stack([x.detach()+j*.15 for j in range(4)])
        molecular_mask = torch.stack([mask.clone() for _ in range(4)])
        gene_ids = ids
        base = torch.tensor([.1, -.3, .7, .2], dtype=x.dtype)
        def load_patient_slides(self, patient, device=None):
            return [{'features': torch.roll(v['features'], patient, -1), 'plan': v['plan']} for v in slides]
    ds = Data()
    offset = ds.base.clone()
    before = model.path_project[1].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.zero_grad(set_to_none=True)
    audit = risk_gradient_cache(model, lambda patients: train.forward_patients(model, ds, patients),
                                np.arange(4), np.array([2., 2., 4., 6.]), np.array([1, 1, 0, 1]),
                                microbatch=1, regularizer=lambda: train.regularizer(model))
    assert audit['maximum_absolute_component_replay_error'] < 1e-12
    assert audit['parameters_with_nonzero_data_gradient'] > 0
    assert torch.isfinite(model.gene_evidence.identity.weight.grad).all()
    optimizer.step()
    assert torch.equal(ds.base, offset)
    assert not torch.equal(before, model.path_project[1].weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    if arm == 'C00':
        # Initial zero main heads block data gradients to this encoder; its
        # first update alone could come from L2. Check the next data-only path.
        assert audit['data_gradient_L2_by_top_level_module'].get('path_project', 0.) == 0
        optimizer.zero_grad(set_to_none=True)
        second = risk_gradient_cache(model, lambda patients: train.forward_patients(model, ds, patients),
                         np.arange(4), np.array([2., 2., 4., 6.]), np.array([1, 1, 0, 1]), microbatch=1)
        assert second['data_gradient_L2_by_top_level_module']['path_project'] > 0


def test_wrong_map_uses_original_missing_patterns_and_partition_local_ids(monkeypatch):
    from src.phase28_reread import data
    from src.phase23_cdei import data as inherited
    cancer = np.array(['A', 'A', 'B', 'B'])
    raw = np.array([[1., np.nan], [2., np.nan], [3., 4.], [5., 6.]])
    calls = {}
    monkeypatch.setattr(data, 'read_feature_npz', lambda: {'rna_genes': raw})
    def pattern(value):
        assert value is raw
        calls['natural'] = True
        return ['missing', 'missing', 'full', 'full']
    def permutation(c, patterns, indices, seed):
        assert c is cancer and calls['natural'] and list(patterns) == ['missing', 'missing', 'full', 'full']
        assert np.array_equal(indices, [0, 1]) and seed == 28
        return np.array([1, 0, 2, 3]), {'only_outcome_blind_fields': True}
    monkeypatch.setattr(inherited, 'raw_rna_missing_patterns', pattern)
    monkeypatch.setattr(inherited, 'partition_derangement', permutation)
    class Data:
        cancer_type = cancer
    mapping, audit = train._wrong_map(Data(), [0, 1], 28)
    assert np.array_equal(mapping[:2], [1, 0]) and audit['only_outcome_blind_fields']
