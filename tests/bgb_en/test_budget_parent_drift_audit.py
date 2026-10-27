import pytest
import torch

from scripts.bgb_en_budget_parent_drift_audit import state_difference


def test_saved_state_difference_counts_parameters_and_buffers_without_forward():
    left={'weight':torch.zeros(2,2),'orders':torch.tensor([0,1],dtype=torch.long)}
    right={'weight':torch.tensor([[0.,1.],[0.,0.]]),'orders':torch.tensor([1,0],dtype=torch.long)}
    result=state_difference(left,right)
    assert result['state_tensors']==2 and result['changed_tensors']==2
    assert result['state_elements']==6 and result['changed_elements']==3
    assert result['max_abs_state_error']==1.
    assert result['RMS_state_error']==pytest.approx((3/6)**.5)


def test_state_key_or_dtype_mismatch_does_not_produce_a_false_exact_match():
    with pytest.raises(ValueError,match='STATE_KEYS'):
        state_difference({'w':torch.zeros(1)},{'other':torch.zeros(1)})
    with pytest.raises(ValueError,match='TYPE_SHAPE'):
        state_difference({'w':torch.zeros(1)},{'w':torch.zeros(1,dtype=torch.float64)})
