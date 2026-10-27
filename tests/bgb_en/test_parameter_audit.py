"""Parameter eligibility audit is not a gradient or prediction experiment."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.bgb_en_parameter_audit import neural_counts, state_parameter_count, tie_break_proof
from src.bgb_en.training import MechanismModel


def make_synthetic_model(arm):
    n, genes = 2, 12
    feature = dict(rna=np.zeros((n, genes), np.float32), rna_mask=np.ones((n, genes), bool),
        cna_gain=np.zeros((n, genes), np.float32), cna_loss=np.zeros((n, genes), np.float32),
        cna_mask=np.ones((n, genes), bool), C=np.zeros((n, 3), np.float32))
    prepared = dict(features=feature, base=np.zeros(n), RNA=np.zeros(n), CS=np.zeros(n), beta=1.,
                    train=np.array([0]), valid=np.array([1]))
    membership = torch.arange(genes)[None, :]//4 == torch.arange(3)[:, None]
    return MechanismModel('R3', arm, prepared, membership,
                          SimpleNamespace(input_dim=768, step=512.), 'cpu', gamma=.1)


def test_four_actual_architectures_cpu_construction_matches_saved_count_contract_without_forward():
    rows = []
    for arm in ('F2', 'F3', 'F5', 'F10'):
        model = make_synthetic_model(arm)
        total = sum(parameter.numel() for parameter in model.parameters())
        active = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        artifact = dict(parameters=total, active_parameters=active,
                        snapshots={80: dict(state_dict=model.state_dict())})
        result = neural_counts(artifact, dict(parameters=total, stage='R3'))
        assert result['total_registered_parameters'] == total
        assert result['active_parameters'] == active
        assert result['frozen_parameters'] == 6336
        assert result['source_frozen_module_counts'] == {
            'path.mil.': 4192, 'path.output.': 2080, 'omics.risk_head.': 32, 'omics.pool_query': 32}
        assert result['active_is_not_observed_gradient_activity'] is True
        rows.extend(dict(stage='R3', arm=arm, inner_partition=part, seed=seed, **result)
                    for part in (0, 1, 2) for seed in (31, 47))
    proof = tie_break_proof(rows, ['F2', 'F3', 'F5', 'F10'])
    assert proof['pairwise_comparisons_verified'] == 36
    assert proof['frozen_difference_values'] == [6336]
    assert proof['constant_frozen_difference_across_all_candidates_and_cells']
    assert all(parameter.grad is None for parameter in model.parameters())


def test_buffer_exclusion_is_explicit_and_unknown_integer_state_fails_closed():
    total, excluded, frozen = state_parameter_count({
        'learned.weight': torch.zeros(2, 3), 'rna.edge_gene': torch.arange(8),
        'sequence.orders': torch.zeros(4, 5, dtype=torch.long)})
    assert total == 6 and excluded == 28 and not any(frozen.values())
    with pytest.raises(ValueError, match='UNKNOWN_INTEGER_BUFFER'):
        state_parameter_count({'new_buffer': torch.zeros(4, dtype=torch.long)})


def test_ineligible_parameter_or_tiebreak_changes_are_not_silently_accepted():
    model = make_synthetic_model('F5')
    total = sum(parameter.numel() for parameter in model.parameters())
    active = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    with pytest.raises(ValueError, match='SOURCE_FROZEN_MODULES'):
        neural_counts(dict(parameters=total, active_parameters=active-1,
                           snapshots={80: dict(state_dict=model.state_dict())}),
                      dict(parameters=total, stage='R3'))
    rows = [dict(stage='R3', arm=arm, inner_partition=part, seed=seed,
                 total_registered_parameters=100+index, active_parameters=90+index)
            for index, arm in enumerate(('F2', 'F3', 'F5', 'F10'))
            for part in (0, 1, 2) for seed in (31, 47)]
    rows[-1]['active_parameters'] -= 1
    with pytest.raises(ValueError, match='DIFFERENT_FROZEN_BUDGETS'):
        tie_break_proof(rows, ['F2', 'F3', 'F5', 'F10'])
