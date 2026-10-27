"""Nontrivial aggregation, saved-membership and provenance guards, no fits."""
import copy

import numpy as np
import pytest
import torch

from scripts.bgb_en_checkpoint_audit import AuditError, sha256
from scripts.bgb_en_report_matched_support import (
    arm_summary, inspect_artifact, inspect_saved_state, paired_comparison, verify_receipt,
)


def metric(num=5, den=10, uno_num=None, uno_den=None):
    un, ud = num if uno_num is None else uno_num, den if uno_den is None else uno_den
    return {'Harrell': num / den, 'Uno60': un / ud, 'Harrell_num': num, 'Harrell_den': den,
            'Uno_num': un, 'Uno_den': ud}


def receipt(part=0, seed=31, metrics=None):
    m = metric() if metrics is None else metrics
    return {'status': 'COMPLETE', 'stage': 'R2', 'arm': 'O4_MATCHED_SUPPORT',
        'inner_partition': part, 'seed': seed, 'outer_fold': 1, 'outer_test_evaluated': False,
        'lineage': {'fixture': 'correct'}, 'epochs': 80, 'actual_optimizer_fits': 1,
        'train_n': 20, 'valid_n': 10, 'parameters': 1234, 'metrics': m,
        'curves': [{'epoch': 20, 'valid': metric(4)}, {'epoch': 80, 'valid': m}]}


def saved_state(member):
    member = torch.as_tensor(member, dtype=torch.bool)
    p, g = torch.where(member)
    return {'omics.rna.edge_path': p, 'omics.rna.edge_gene': g, 'omics.rna.membership_count': member.sum(1),
        'omics.rna.edge_weight': torch.ones(len(g), 4), 'omics.rna.project.weight': torch.ones(32, 5),
        'omics.rna.project.bias': torch.zeros(32), 'omics.rna.identity': torch.zeros(len(member), 32)}


def test_aggregation_uses_counts_per_seed_then_unweighted_seed_mean():
    rows = {(0, 31): receipt(0, 31, metric(1, 2)), (1, 31): receipt(1, 31, metric(9, 10)),
            (0, 47): receipt(0, 47, metric(1, 10)), (1, 47): receipt(1, 47, metric(1, 10))}
    out = arm_summary(rows, arm='synthetic', stage='R2', parts=[0, 1], seeds=[31, 47])
    assert out['Harrell'] == pytest.approx((10 / 12 + 2 / 20) / 2)
    assert out['Harrell'] != pytest.approx(np.mean([.5, .9, .1, .1]))
    assert out['Harrell'] != pytest.approx(12 / 32)


def test_paired_delta_respects_seed_specific_denominators_and_endpoint_weights():
    a = {(p, s): receipt(p, s, metric(3 + p, 10, 2 + p, 20)) for p in range(3) for s in (31, 47)}
    b = {(p, s): receipt(p, s, metric(2, 10, 1, 20)) for p in range(3) for s in (31, 47)}
    out = paired_comparison(a, b, parts=[0, 1, 2], seeds=[31, 47], label='a-b', family='test')
    assert out['status'] == 'COMPLETE'
    assert out['delta_Harrell'] == pytest.approx(.2)
    assert out['delta_Uno60'] == pytest.approx(.1)


def test_partial_group_does_not_emit_full_metric_or_comparison():
    a = {(0, 31): receipt()}; b = {(0, 31): receipt()}
    out = paired_comparison(a, b, parts=[0, 1, 2], seeds=[31, 47], label='a-b', family='test')
    assert out['status'] == 'RUNNING' and out['delta_Harrell'] is None
    assert arm_summary(a, arm='test', stage='R2', parts=[0, 1, 2], seeds=[31, 47])['Harrell'] is None


def test_different_ipcw_populations_are_rejected():
    a = {(0, 31): receipt()}; b = {(0, 31): receipt(metrics=metric(5, 10, 5, 11))}
    with pytest.raises(AuditError, match='POPULATION'):
        paired_comparison(a, b, parts=[0], seeds=[31], label='a-b', family='test')


def test_saved_membership_buffers_are_checked_not_just_parameter_count():
    member = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)
    audit = inspect_saved_state(saved_state(member), member)
    assert audit['actual_membership_buffers_equal'] is True
    other = member[:, [2, 1, 0]]  # same edge count and row sizes, different gene correspondence
    with pytest.raises(AuditError, match='EDGE_GENE'):
        inspect_saved_state(saved_state(member), other)


def test_corrupted_weight_dimension_is_rejected():
    member = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    state = saved_state(member); state['omics.rna.edge_weight'] = torch.ones(2, 3)
    with pytest.raises(AuditError, match='DIMENSION'):
        inspect_saved_state(state, member)


def test_nonfinite_saved_weights_are_rejected():
    member = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    state = saved_state(member); state['omics.rna.edge_weight'][0, 0] = torch.nan
    with pytest.raises(AuditError, match='NONFINITE'):
        inspect_saved_state(state, member)


@pytest.mark.parametrize('field,new_value,code', [
    ('lineage', {'fixture': 'wrong'}, 'LINEAGE'), ('seed', 47, 'SEED'),
    ('arm', 'O9', 'ARCHITECTURE'), ('epochs', 40, 'BUDGET'),
    ('inner_partition', 2, 'PARTITION'), ('outer_test_evaluated', True, 'SEALED')])
def test_receipt_identity_source_and_budget_fail_closed(field, new_value, code):
    row = receipt(); row[field] = new_value
    with pytest.raises(AuditError, match=code):
        verify_receipt(row, arm='O4_MATCHED_SUPPORT', stage='R2', partition=0, seed=31,
                       lineage={'fixture': 'correct'})


def test_checkpoint_metric_identity_and_file_hash_are_both_required(tmp_path):
    member = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    row = receipt(); path = tmp_path / 'model.pt'
    saved = {'lineage': row['lineage'], 'stage': row['stage'], 'arm': row['arm'], 'seed': row['seed'],
        'epochs': 80, 'optimizer_updates': 80, 'full_riskset': True, 'gradient_cache': True,
        'outer_test_evaluated': False, 'train': np.arange(3), 'valid': np.array([3]),
        'snapshots': {80: {'state_dict': saved_state(member), 'metrics': row['metrics']}}}
    torch.save(saved, path); row['artifact_sha256'] = sha256(path)
    assert inspect_artifact(path, row, expected_member=member, train=np.arange(3), valid=np.array([3]))['weight_dimensions_valid']
    bad_hash = copy.deepcopy(row); bad_hash['artifact_sha256'] = '0' * 64
    with pytest.raises(AuditError, match='ARTIFACT_SHA256'):
        inspect_artifact(path, bad_hash)
    saved['snapshots'][80]['metrics'] = metric(4)
    torch.save(saved, path); row['artifact_sha256'] = sha256(path)
    with pytest.raises(AuditError, match='SNAPSHOT_METRICS'):
        inspect_artifact(path, row)
