import copy
import numpy as np
import pytest
import torch
from src.bgb_en.r5_training import full_risk_step,risk_diagnostics,frozen_attention_probe
from src.phase10_trimodal.math import torch_cox,log_weights


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.device='cpu';self.intervention=None
        self.w=torch.nn.Parameter(torch.tensor([.1,-.2,.3]))
        self.register_buffer('x',torch.tensor([[1.,2.,0.],[0.,1.,2.],[2.,0.,1.],[1.,-1.,1.],[2.,1.,-1.],[0.,2.,1.]]))
    def forward(self,idx):
        main=.2*(self.x[idx]@self.w);cross=.05*(self.x[idx].square()@self.w)
        return dict(risk=main+cross,delta=main+cross,cross_delta=cross)
    def regularization(self):return 1e-4*self.w.square().sum()


def test_cached_update_matches_full_graph_efron_and_both_penalties():
    a=Toy();b=copy.deepcopy(a);indices=torch.arange(6)
    time=np.array([1.,2.,2.,4.,5.,6.]);event=np.array([1,1,1,0,1,0])
    oa=torch.optim.SGD(a.parameters(),lr=.03);ob=torch.optim.SGD(b.parameters(),lr=.03)
    out=a(indices);loss=torch_cox(out['risk'],log_weights(time,event,'cpu'),torch.as_tensor(event))
    loss=loss+.1*out['delta'].square().mean()+.1*out['cross_delta'].square().mean()+a.regularization()
    loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),5.);oa.step()
    receipt=full_risk_step(b,[indices[:2],indices[2:5],indices[5:]],time,event,ob)
    torch.testing.assert_close(a.w,b.w,atol=2e-8,rtol=1e-7)
    assert receipt['full_riskset_patients']==6 and receipt['optimizer_updates']==1


def test_nonfinite_regularizer_stops_before_optimizer_update():
    model=Toy();before=model.w.detach().clone();model.regularization=lambda:torch.tensor(float('inf'))
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    with pytest.raises(RuntimeError,match='NONFINITE_PARAMETER_REGULARIZER'):
        full_risk_step(model,[torch.arange(6)],np.arange(1,7.),np.array([1,1,0,1,0,1]),optimizer)
    assert torch.equal(model.w,before)


def test_disabled_route_is_not_reported_as_saturation():
    values={key:np.zeros(3) for key in ('risk','delta','cross_delta','readout_derivative_z','cross_route_active')}
    answer=risk_diagnostics(values,np.array([0.,1.,2.]))
    assert answer['derivative_below_one_percent_of_initial_max'] is None
    assert answer['derivative_saturation_not_applicable_when_route_disabled']


class AttentionToy(torch.nn.Module):
    device='cpu'
    def refresh_training_query_mean(self,*args):pass
    def forward(self,idx,control=None):
        a=torch.zeros(len(idx),1,1,64)
        if control=='matched_query':a[...,0]=.7;a[...,1]=.3
        else:a[...,1]=.4;a[...,2]=.6
        return {'interaction_details':{'trace':[{'omics_to_patch':{'attention':a}}]}}


def test_positive_topk_overlap_excludes_arbitrary_zero_padding():
    answer=frozen_attention_probe(AttentionToy(),np.arange(2))['directions']['omics_to_patch']
    assert answer['top32_positive_support_Jaccard']==pytest.approx(1/3)
    assert answer['attention_TV']==pytest.approx(.7)
