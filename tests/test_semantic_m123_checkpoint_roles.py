import pytest
import torch
from scripts.semantic_m123_checkpoint_roles import describe_state


def test_head_only_zero_local_does_not_imply_zero_explicit_interaction():
    r=describe_state({'local_out.weight':torch.zeros(64,64),'interaction_head.weight':torch.ones(1,16)},20)
    assert r['actual_joint_encoder_updates']==0 and r['local_output_matrix_exactly_zero']
    assert r['explicit_interaction_head_norm']==4.


def test_joint_and_no_update_schedule_counts_are_distinct():
    assert describe_state({'local_out.weight':torch.zeros(64,64)},0)['actual_head_only_updates']==0
    r=describe_state({'local_out.weight':torch.eye(64)},50)
    assert r['actual_joint_encoder_updates']==30 and not r['local_output_matrix_exactly_zero']


def test_head_only_checkpoint_with_changed_local_branch_fails():
    with pytest.raises(ValueError,match='LOCAL_WEIGHT_CHANGED'):describe_state({'local_out.weight':torch.eye(64)},20)
