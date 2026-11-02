import copy,io
import numpy as np
import torch
from src.qpr.survival_model import FullRelation
from src.rscil.geometry import EfronRiskSets
from src.semantic_m123r.io import selected_npy_rows


def batch():
    torch.manual_seed(55)
    return [torch.randn(6,50,3,dtype=torch.float64),torch.randn(6,7,dtype=torch.float64),torch.randn(6,32,dtype=torch.float64),
            torch.randn(6,12,32),torch.randn(6,12,64),torch.ones(6,12,dtype=torch.bool),torch.ones(6,50,dtype=torch.bool)]


def test_exact_initial_anchor_and_query_gradient_activation():
    b=batch();model=FullRelation(7,'B3')
    assert torch.equal(model(*b),torch.zeros(6,dtype=torch.float64))
    with torch.no_grad():model.beta.normal_(std=.1)
    model.stage(21);model(*b).square().sum().backward()
    assert model.query.g.weight.grad.norm()>0
    model.stage(20);assert not model.query.g.weight.requires_grad


def test_full_risk_cached_gradient_equals_monolithic_with_ties():
    b=batch();model=FullRelation(7,'B3');model.stage(21)
    with torch.no_grad():model.beta.normal_(std=.1)
    other=copy.deepcopy(model);rs=EfronRiskSets([1,1,2,3,4,5],[1,1,0,1,0,1]);base=torch.randn(6,dtype=torch.float64)
    delta=model(*b);loss=rs.loss(base+delta)+.01*delta.square().mean();loss.backward()
    with torch.no_grad():d=other(*b)
    leaf=d.detach().requires_grad_();g=torch.autograd.grad(rs.loss(base+leaf)+.01*leaf.square().mean(),leaf)[0]
    for start in (0,3):
        out=other(*[x[start:start+3] for x in b]);torch.dot(out,g[start:start+3]).backward()
    for (n,p),(n2,p2) in zip(model.named_parameters(),other.named_parameters()):
        assert torch.allclose(p.grad,p2.grad,atol=1e-10,rtol=1e-9),n


def test_selected_label_rows_do_not_decode_forbidden_nan_values():
    stream=io.BytesIO();np.save(stream,np.array([1.,np.nan,3.,np.nan]));stream.seek(0)
    assert np.array_equal(selected_npy_rows(stream,[2,0],4),[3.,1.])
