"""Numerical/gradient checks of budgets and mass-preserving selective updates."""
import numpy as np
import pytest
import torch
from scipy.optimize import brentq, minimize

from src.phase20_selective.operators import (
    capped_simplex,dense_budget_gate,conservative_update,rank_one_energy_projection,
)


@pytest.mark.parametrize('operator',[capped_simplex,dense_budget_gate])
def test_gate_budget_mask_empty_row_and_translation_invariance(operator):
    s = torch.tensor([[.2,-.1,.4,.3,.1,0.],[-1.,2.,0.,4.,-3.,1.]],dtype=torch.float64)
    m = torch.tensor([[True,True,False,True,True,False],[False]*6])
    g = operator(s,m)
    torch.testing.assert_close(g.sum(-1),m.sum(-1).double()*.25,atol=1e-13,rtol=0)
    assert ((g>=0)&(g<=1)).all() and g[~m].count_nonzero()==0
    torch.testing.assert_close(g,operator(s+37.,m),atol=1e-13,rtol=0)
    ss = s.clone().requires_grad_(); out = operator(ss,m)
    (out*torch.arange(6,dtype=out.dtype)).sum().backward()
    assert torch.isfinite(ss.grad).all() and ss.grad[1].count_nonzero()==0


def test_capped_simplex_matches_independent_constrained_qp():
    s = np.array([-.9,.02,.22,.41,.8,1.4])
    answer = minimize(lambda g:.5*np.sum((g-s)**2),np.full(6,.25),jac=lambda g:g-s,
        method='SLSQP',bounds=[(0,1)]*6,constraints={'type':'eq','fun':lambda g:g.sum()-1.5,
        'jac':lambda g:np.ones(6)},options={'ftol':1e-13,'maxiter':200})
    assert answer.success
    g = capped_simplex(torch.tensor(s,dtype=torch.float64),torch.ones(6,dtype=torch.bool))
    np.testing.assert_allclose(g.numpy(),answer.x,atol=1e-8,rtol=1e-8)


def test_dense_gate_matches_independent_scalar_root_and_is_not_uniform():
    s = np.array([-.8,.1,.6,1.2,-.4])
    tau = brentq(lambda x:(1/(1+np.exp(-(s-x)))).sum()-1.25,-40,40)
    expected = 1/(1+np.exp(-(s-tau)))
    actual = dense_budget_gate(torch.tensor(s,dtype=torch.float64),torch.ones(5,dtype=torch.bool))
    np.testing.assert_allclose(actual.numpy(),expected,atol=1e-13,rtol=1e-13)
    assert actual.std()>0 and (actual>0).all() and (actual<1).all()


@pytest.mark.parametrize('operator',[capped_simplex,dense_budget_gate])
def test_gate_custom_backward_matches_finite_differences_and_analytic_jacobian(operator):
    s = torch.tensor([[.07,-.12,.16,.03,.1]],dtype=torch.float64,requires_grad=True)
    mask = torch.ones_like(s,dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda x:operator(x,mask),(s,),eps=1e-6,atol=2e-6,rtol=2e-5)
    g = operator(s,mask).detach()[0]
    jac = torch.autograd.functional.jacobian(lambda x:operator(x,mask),s)[0,:,0,:]
    if operator is capped_simplex:
        f = ((g>0)&(g<1)).double()
        expected = torch.diag(f)-f[:,None]*f[None,:]/f.sum()
    else:
        w = g*(1-g)
        expected = torch.diag(w)-w[:,None]*w[None,:]/w.sum()
    torch.testing.assert_close(jac,expected,atol=1e-12,rtol=1e-12)
    assert jac.norm()>0


@pytest.mark.parametrize('operator',[capped_simplex,dense_budget_gate])
@pytest.mark.parametrize('fraction',[0.,1.])
def test_zero_full_budget_and_single_valid_item(operator,fraction):
    s = torch.tensor([[2.,-1.]],dtype=torch.float64,requires_grad=True)
    mask = torch.tensor([[True,False]])
    g = operator(s,mask,fraction)
    torch.testing.assert_close(g,mask.double()*fraction,atol=0,rtol=0)
    g.sum().backward()
    assert s.grad.count_nonzero()==0
    g = operator(s,mask,.25)
    torch.testing.assert_close(g,torch.tensor([[.25,0.]],dtype=torch.float64),atol=1e-13,rtol=0)


def test_budget_is_mass_not_fixed_number_of_active_edges():
    g = capped_simplex(torch.zeros(12,dtype=torch.float64),torch.ones(12,dtype=torch.bool))
    assert (g>0).sum()==12
    torch.testing.assert_close(g,torch.full_like(g,.25))


@pytest.mark.parametrize('operator',[capped_simplex,dense_budget_gate])
def test_conservative_update_preserves_probability_support_and_tv_bound(operator):
    torch.manual_seed(20)
    a = torch.randn(2,3,7,dtype=torch.float64).softmax(-1)
    mask = torch.ones_like(a,dtype=torch.bool); mask[0,0]=False
    score = torch.randn_like(a)
    gate = operator(score,mask)
    delta = .5*torch.randn_like(a).tanh()
    result,info = conservative_update(a,gate,delta,mask,diagnostic=True)
    torch.testing.assert_close(result.sum(-1),a.sum(-1),atol=1e-13,rtol=1e-13)
    assert (result>=0).all() and torch.isfinite(result).all()
    torch.testing.assert_close(result[gate==0],a[gate==0],atol=0,rtol=0)
    assert (info['TV']<=info['TV_bound']+1e-13).all()
    torch.testing.assert_close(conservative_update(a,gate,torch.zeros_like(delta),mask),a,atol=0,rtol=0)
    torch.testing.assert_close(conservative_update(a,torch.zeros_like(gate),delta,mask),a,atol=0,rtol=0)


def test_conservative_update_matches_literal_formula_and_input_gradients():
    torch.manual_seed(91)
    a = torch.randn(2,5,dtype=torch.float64).softmax(-1).requires_grad_()
    g = torch.rand_like(a,requires_grad=True)
    d = (.4*torch.randn_like(a).tanh()).requires_grad_()
    mask = torch.ones_like(a,dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda x,y,z:conservative_update(x,y,z,mask),(a,g,d),
                                    eps=1e-6,atol=1e-6,rtol=1e-5)
    rows = []
    for b in range(2):
        w = a[b]*g[b]; mass = sum(w)
        z = sum(w[k]*d[b,k].exp() for k in range(5))/mass
        rows.append(torch.stack([a[b,k]+.5*w[k]*(d[b,k].exp()/z-1) for k in range(5)]))
    torch.testing.assert_close(conservative_update(a,g,d,mask),torch.stack(rows),atol=1e-13,rtol=1e-13)


def test_tiny_selected_mass_is_finite_without_epsilon_amplification():
    a = torch.tensor([[1.,1e-40,1e-40]],dtype=torch.float32)
    g = torch.tensor([[0.,.5,.5]],dtype=torch.float32)
    mask = torch.ones_like(a,dtype=torch.bool)
    updated,info = conservative_update(a,g,torch.tensor([[0.,.5,-.5]]),mask,diagnostic=True)
    assert torch.isfinite(updated).all() and updated[0,1:].max()<1e-39
    assert info['TV'].max()<1e-39


def test_rank_one_projection_has_rank_one_and_isotropic_trace_energy_matching():
    identity = torch.eye(16,dtype=torch.float64)
    transform = rank_one_energy_projection(identity)
    assert torch.linalg.matrix_rank(transform)==1
    torch.testing.assert_close((transform.T@transform).trace(),torch.tensor(16.,dtype=torch.float64))
    x = torch.randn(2,24,16,dtype=torch.float64,requires_grad=True)
    y = rank_one_energy_projection(x)
    torch.testing.assert_close(y,4*x.mean(-1,keepdim=True).expand_as(x),atol=0,rtol=0)
    y.square().sum().backward()
    assert (x.grad.abs().sum((0,1))>0).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_gate_and_update_output_gradient_parity():
    torch.manual_seed(201)
    for operator in [capped_simplex,dense_budget_gate]:
        s = torch.randn(2,5,dtype=torch.float64,requires_grad=True)
        sc = s.detach().cuda().requires_grad_()
        m = torch.ones_like(s,dtype=torch.bool)
        a = torch.randn_like(s).softmax(-1)
        delta = .4*torch.randn_like(s).tanh()
        cpu = conservative_update(a,operator(s,m),delta,m)
        gpu = conservative_update(a.cuda(),operator(sc,m.cuda()),delta.cuda(),m.cuda())
        probe = torch.randn_like(cpu)
        (cpu*probe).sum().backward(); (gpu*probe.cuda()).sum().backward()
        torch.testing.assert_close(cpu,gpu.cpu(),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(s.grad,sc.grad.cpu(),atol=1e-12,rtol=1e-12)
