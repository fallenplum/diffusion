"""Standalone checks; import torch/pytest. Only synthetically generated inputs."""
import numpy as np
import torch
from riskset_core import efron_nll, riskset_information, orthogonalize_interaction


def test_efron_hessian_equals_riskset_covariance_with_ties():
    gen=torch.Generator().manual_seed(119)
    x=torch.randn(14,3,generator=gen,dtype=torch.float64)
    t=torch.tensor([1,1,2,2,2,3,4,4,5,5,5,6,7,8],dtype=torch.float64)
    e=torch.tensor([1,0,1,1,0,1,0,1,1,0,1,0,1,0])
    beta=torch.tensor([.3,-.2,.5],dtype=torch.float64,requires_grad=True)
    def f(p):return efron_nll(x@p,t,e,average=False)
    h=torch.autograd.functional.hessian(f,beta)
    inf=riskset_information((x@beta).detach(),t,e,x)
    assert float((h-inf).abs().max())<1e-9


def test_projection_and_reparameterization_identity():
    gen=torch.Generator().manual_seed(48)
    b=torch.randn(70,3,generator=gen,dtype=torch.float64)
    z=(2*b[:,[0]]-.3*b[:,[1]]+.05*torch.randn(70,1,generator=gen,dtype=torch.float64))
    t=torch.linspace(.1,6,70,dtype=torch.float64)
    e=torch.tensor([int(i%5==0) for i in range(70)])
    r=.5*b[:,0]
    zp,a,info=orthogonalize_interaction(z,b,r,t,e,ridge=1e-9)
    assert (info[:3,3:]-info[:3,:3]@a).abs().max()<1e-7
    betab=torch.tensor([1.,-1.,.5],dtype=torch.float64)
    betaz=torch.tensor([.3],dtype=torch.float64)
    # (B,Z) and (B,Z-BA) span exactly the same function class.
    pred1=b@betab+z@betaz
    pred2=b@(betab+a@betaz)+zp@betaz
    assert torch.allclose(pred1,pred2,atol=1e-10)


def test_no_event_or_invalid_inputs_fail_closed():
    try:
        efron_nll(torch.ones(3),np.array([1,2,3]),np.array([0,0,0]))
        assert False
    except ValueError:
        pass
