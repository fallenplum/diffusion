import copy
from unittest.mock import patch
import numpy as np
import pytest
import torch
from src.phase17r_ssd.model import RepairS6
from src.phase18a_readout.model import ReadoutS6
from src.phase18a_readout.scan import explicit_local_readout

def example(arm,n=7):
    torch.set_num_threads(2);torch.manual_seed(181101)
    membership=np.zeros((8,512),bool)
    for i in range(8):membership[i,i*64:(i+1)*64]=True
    model=ReadoutS6(11,membership,arm,.25)
    args=[torch.randn(2,n,1536),torch.ones(2,n,dtype=torch.bool),torch.randn(2,11)]
    args += [torch.randn(2,512) for _ in range(3)]
    args += [torch.ones(2,512) for _ in range(2)]+[torch.randn(2)]
    return model,membership,args

@pytest.mark.parametrize('arm,base',[('F0','R1_S6_noDirect'),('F1','R1_A8_noDirect')])
@pytest.mark.parametrize('training',[False,True])
def test_final_model_identical_to_locked_parent(arm,base,training):
    model,membership,args=example(arm)
    torch.manual_seed(181101);parent=RepairS6(11,membership,base,1.,.25)
    for k,v in parent.state_dict().items():assert torch.equal(v,model.state_dict()[k])
    assert len(parent.state_dict())==len(model.state_dict()) and model.direct is None
    model.train(training);parent.train(training)
    a=[x.clone().requires_grad_(x.dtype.is_floating_point) for x in args]
    b=[x.clone().requires_grad_(x.dtype.is_floating_point) for x in args]
    torch.manual_seed(1919);r=model(*a)
    torch.manual_seed(1919);p=parent(*b)
    assert torch.equal(r,p)
    r.square().sum().backward();p.square().sum().backward()
    for (ka,va),(kb,vb) in zip(model.named_parameters(),parent.named_parameters()):
        assert ka==kb and ((va.grad is None)==(vb.grad is None))
        if va.grad is not None:torch.testing.assert_close(va.grad,vb.grad,rtol=0,atol=0)
    for x,y in zip(a,b):
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)

@pytest.mark.parametrize('arm',['L0','L1'])
@pytest.mark.parametrize('n',[1,9])
def test_local_full_model_parameter_and_input_gradients_against_python_oracle(arm,n):
    model,_,args=example(arm,n);model=model.double().eval();oracle=copy.deepcopy(model)
    a=[x.double().requires_grad_(True) if x.dtype.is_floating_point else x for x in args]
    b=[x.detach().clone().requires_grad_(True) if x.dtype.is_floating_point else x.clone() for x in a]
    actual=model(*a)
    def reference(u,dt,B,C,A,m,**kwargs):return explicit_local_readout(u,dt,B,C,A,m,h0=kwargs.get('h0'))
    with patch('src.phase18a_readout.model.local_readout_scan',reference):expected=oracle(*b)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    actual.square().sum().backward();expected.square().sum().backward()
    for (name,x),(_,y) in zip(model.named_parameters(),oracle.named_parameters()):
        assert (x.grad is None)==(y.grad is None),name
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=1e-9,atol=1e-10)
    for x,y in zip(a,b):
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=1e-9,atol=1e-10)

@pytest.mark.parametrize('local,final',[('L0','F0'),('L1','F1')])
def test_single_position_readout_equivalence_and_no_extra_parameters(local,final):
    lm,_,args=example(local,1);fm,_,_=example(final,1);fm.load_state_dict(lm.state_dict());lm.eval();fm.eval()
    torch.testing.assert_close(lm(*args),fm(*args),rtol=0,atol=1e-7)
    assert sum(p.numel() for p in lm.parameters())==sum(p.numel() for p in fm.parameters())
