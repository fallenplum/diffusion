import numpy as np
import pytest
import torch
from src.rscil.geometry import EfronRiskSets,fit_projection
from src.semantic_m123.training import efron_log_weights,efron


def case():
    torch.manual_seed(231)
    x=torch.randn(17,7,dtype=torch.float64)
    t=np.array([1,1,1,2,2,3,3,4,5,5,7,8,9,9,10,12,13.])
    e=np.array([1,1,0,1,1,1,0,1,0,1,0,1,1,0,0,1,0])
    return x,t,e


def test_tied_efron_hessian_equals_summed_covariances():
    x,t,e=case();rs=EfronRiskSets(t,e);w=torch.randn(7,dtype=torch.float64)*.2
    h=torch.autograd.functional.hessian(lambda q:rs.loss(x@q,normalize=False),w)
    info=rs.information(x@w,x)
    assert (h-info).abs().max()<1e-8
    assert torch.linalg.eigvalsh(info).min()>-1e-10


def test_efron_agrees_with_existing_project_implementation():
    x,t,e=case();risk=x[:,0];rs=EfronRiskSets(t,e)
    lw,ev=efron_log_weights(t,e)
    assert abs(rs.loss(risk)-efron(risk,lw,ev))<1e-12
    assert abs(rs.loss(risk)-rs.loss(risk+10000))<1e-10


def test_projection_ridge_identity_not_false_exact_orthogonality():
    x,t,e=case();rs=EfronRiskSets(t,e);b=x[:,:3];z=x[:,3:]
    a,diag=fit_projection(b,z,geometry='riskset',risk=x[:,0],risksets=rs)
    cross=rs.information(x[:,0],torch.cat([b,z-b@a],1))[:3,3:]
    assert (cross-1e-4*a).abs().max()<1e-10
    assert diag['ridge_identity_error']<1e-10
    assert cross.norm()>0  # ridge projection is approximate, not exact.
    az,_=fit_projection(b,z,geometry='riskset',risk=x[:,0],risksets=rs,rho=0)
    assert rs.information(x[:,0],torch.cat([b,z-b@az],1))[:3,3:].abs().max()<1e-10


def test_function_classes_identical_but_l2_not_invariant():
    x,t,e=case();b=x[:,:3];z=x[:,3:];a,_=fit_projection(b,z,geometry='euclidean')
    wb=torch.tensor([.2,-.4,.7],dtype=torch.float64);wz=torch.arange(4,dtype=torch.float64)*.2
    r1=b@wb+z@wz;r2=b@(wb+a@wz)+(z-b@a)@wz
    assert (r1-r2).abs().max()<1e-12
    assert not torch.isclose(wb.square().sum(),(wb+a@wz).square().sum())


def test_stop_gradient_only_geometry_not_z():
    x,t,e=case();b=x[:,:3].clone().requires_grad_();z=x[:,3:].clone().requires_grad_()
    a,_=fit_projection(b,z,geometry='riskset',risk=b.sum(1),risksets=EfronRiskSets(t,e))
    assert not a.requires_grad
    loss=(z-b.detach()@a.detach()).square().sum();gz,gb=torch.autograd.grad(loss,(z,b),allow_unused=True)
    assert gz.norm()>0 and gb is None


@pytest.mark.parametrize('t,e',[(np.arange(3),[0,0,0]),([-1,1,2],[1,0,0]),([0,float('nan'),2],[1,0,1]),([1,2],[1]),([1,2],[1,.5])])
def test_invalid_endpoints_fail_closed(t,e):
    with pytest.raises(ValueError):EfronRiskSets(t,e)


def test_nonfinite_risk_and_wrong_shapes_fail_closed():
    rs=EfronRiskSets([1,2,3],[1,0,0])
    for r in (torch.tensor([1.,2.,3.]),torch.tensor([1.,2.],dtype=torch.float64),torch.tensor([1.,float('inf'),2.],dtype=torch.float64)):
        with pytest.raises(ValueError):rs.loss(r)
