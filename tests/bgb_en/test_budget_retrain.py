import numpy as np
import torch
from scripts.bgb_en_budget_retrain import compare_parent80, initialize_new_model


def test_new_model_same_seed_and_no_visual_bank_match_original_initialization():
    from src.bgb_en.training import MechanismModel
    features = dict(rna=np.ones((8, 12), np.float32), rna_mask=np.ones((8, 12), bool),
        cna_gain=np.zeros((8, 12), np.float32), cna_loss=np.zeros((8, 12), np.float32),
        cna_mask=np.ones((8, 12), bool), C=np.zeros((8, 2), np.float32))
    prepared = dict(features=features, RNA=np.zeros(8), base=np.zeros(8), beta=1., train=np.arange(4), valid=np.arange(4, 8))
    membership = torch.ones(3, 12, dtype=torch.bool)
    for arm in ('O3', 'O4'):
        new = initialize_new_model(arm, prepared, membership, 'cpu', 31, .3)
        torch.manual_seed(31); np.random.seed(31)
        original = MechanismModel('R2', arm, prepared, membership, object(), 'cpu', gamma=.3)
        assert all(torch.equal(value, original.state_dict()[name]) for name, value in new.state_dict().items())


def test_parent80_check_records_real_pair_credit_changes_under_small_error():
    time = np.array([1., 2., 3.]); event = np.ones(3, int)
    original = np.array([.3, .3000001, -.2]); replay = np.array([.3000002, .3000001, -.2])
    result = compare_parent80(time, event, time, event, original, replay)
    assert result['within_locked_tolerance']
    assert result['changed_pair_credits'] == 1
    assert result['Harrell_change'] != 0
