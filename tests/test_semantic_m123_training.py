import numpy as np
import pytest
import torch
from src.semantic_m123.training import efron,efron_log_weights,fit
from src.semantic_m123.models import JointModel
from src.semantic_b0b7.survival import efron_numpy


@pytest.mark.parametrize('times,events',[
    ([1,2,3,4],[1,1,0,0]),([1,1,1,3],[1,1,0,1]),([0,0,3,5],[1,0,1,1])])
def test_full_risk_efron_matches_independent_patient_loop(times,events):
    r=torch.tensor([.5,-.2,.8,.1],dtype=torch.float64,requires_grad=True)
    w,e=efron_log_weights(times,events)
    actual=efron(r,w,e)
    expected=efron_numpy(r.detach().numpy(),times,events)
    assert abs(actual.item()-expected)<1e-12
    grad=torch.autograd.grad(actual,r)[0].numpy()
    for i in range(4):
        plus=r.detach().numpy().copy();minus=plus.copy()
        plus[i]+=1e-5;minus[i]-=1e-5
        finite=(efron_numpy(plus,times,events)-efron_numpy(minus,times,events))/2e-5
        assert abs(grad[i]-finite)<2e-9


def test_mini_batch_risk_cannot_be_substituted():
    w,e=efron_log_weights([1,2,3,4],[1,1,0,0])
    with pytest.raises(ValueError,match='FULL_TRAINING'):
        efron(torch.zeros(2),w,e)


def test_no_event_is_not_fake_success():
    with pytest.raises(ValueError,match='ENDPOINTS'):
        efron_log_weights([1,2],[0,0])


def test_validation_callback_does_not_change_training():
    from tests.test_semantic_m123_models import batch
    b=batch(3);states=[]
    for add in [0,100]:
        torch.manual_seed(31);m=JointModel('M1_SCRATCH',12,18,5)
        def cb(epoch,tr,ev):return add+np.asarray(ev['risk']).sum()
        log=fit(m,b,[1,2,3],[1,0,1],seed=31,l2=.001,max_epochs=3,
                head_epochs=1,snapshot_epochs=(0,1,3),evaluation_batch=b,snapshot_callback=cb)
        states.append({k:v.detach().clone() for k,v in m.state_dict().items()})
        assert log['optimizer_updates']==3 and len(log['history'])==3
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])

