"""Read-only stage diagnostics: operator identities and independent aggregation."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.phase20_selective.model import SelectiveModel
from src.phase20_selective.operators import conservative_update
from src.phase21_relative.diagnostics import (
    _Moments,_hierarchy,_direction_metrics,diagnose_batch,diagnose_checkpoint)
from src.phase21_relative.operators import direct_update


def direction(delta,values=None,operator=conservative_update):
    shape=delta.shape
    base=torch.full_like(delta,1/shape[-1]);gate=torch.full_like(delta,.25)
    mask=torch.ones_like(delta,dtype=torch.bool);query=mask.any(-1)
    if values is None:
        values=torch.arange(shape[-1]*3,dtype=delta.dtype,device=delta.device).reshape(1,shape[-1],3).expand(shape[0],-1,-1)/5
    updated=operator(base,gate,delta,mask)
    return _direction_metrics(base,updated,gate,delta,mask,values,query,update_operator=operator)


@pytest.mark.parametrize('operator',[conservative_update,direct_update])
def test_common_offset_is_removed_by_both_normalized_operators(operator):
    delta=torch.tensor([[[.3,.3,.3,.3],[-.2,-.2,-.2,-.2]]],dtype=torch.float64)
    out=direction(delta,operator=operator)
    for key in ['raw_delta_row_SD','weighted_centered_variance','weighted_centered_RMS','attention_TV']:
        assert out['rows'][key][0].abs().max()<2e-15
    torch.testing.assert_close(out['rows']['common_energy_fraction'][0],torch.ones((1,2),dtype=delta.dtype),atol=2e-15,rtol=0)
    assert out['message'].abs().max()<2e-15
    assert out['checks']['constant_only_map_max_error']<2e-15


@pytest.mark.parametrize('operator',[conservative_update,direct_update])
def test_weighted_moments_and_taylor_error_have_independent_oracles(operator):
    raw=torch.tensor([[[-2.,-1.,1.,2.],[-1.,1.,-2.,2.]]],dtype=torch.float64)
    large=direction(raw*.01,operator=operator);small=direction(raw*.001,operator=operator)
    torch.testing.assert_close(large['rows']['weighted_centered_variance'][0],torch.full((1,2),.00025,dtype=raw.dtype),atol=1e-15,rtol=0)
    assert large['rows']['common_energy_fraction'][0].abs().max()<1e-15
    assert small['rows']['taylor_absolute_error_L1'][0].max()<large['rows']['taylor_absolute_error_L1'][0].max()*.02
    assert small['checks']['centered_delta_invariance_max_error']<1e-15


def test_nonzero_attention_variation_can_cancel_completely_in_values():
    delta=torch.tensor([[[-.2,-.1,.1,.2]]],dtype=torch.float64)
    values=torch.ones((1,4,3),dtype=delta.dtype)
    out=direction(delta,values)
    assert out['rows']['attention_TV'][0].min()>0
    assert out['rows']['weighted_centered_RMS'][0].min()>0
    assert out['rows']['delta_message_L2_before_query_average'][0].abs().max()<2e-15
    assert out['rows']['weighted_value_centered_RMS'][0].abs().max()<2e-15


def test_query_mean_can_cancel_nonzero_row_messages():
    delta=torch.tensor([[[-.2,.2],[.2,-.2]]],dtype=torch.float64)
    out=direction(delta)
    assert (out['message'].norm(dim=-1)>0).all()
    assert out['message'].mean(1).norm()<2e-15


def test_query_wsi_patient_aggregation_does_not_weight_patients_by_query_count():
    stats=_Moments()
    x=torch.tensor([[0.,0.,0.],[2.,99.,99.],[10.,10.,10.]],dtype=torch.float64)
    valid=torch.tensor([[True]*3,[True,False,False],[True]*3])
    _hierarchy(stats,'x',x,valid,np.array([0,0,1]),np.array([0,1]))
    out=stats.summary()
    assert out['x_query']['mean']==pytest.approx(32/7)
    assert out['x_WSI']['mean']==4.
    assert out['x_patient']['mean']==5.5
    assert out['x_query']['count']==7 and out['x_WSI']['count']==3 and out['x_patient']['count']==2
    copy_stats=_Moments();copy_stats.merge(stats.data)
    assert copy_stats.summary()==out


def test_empty_selected_support_is_counted_without_nan_or_fabricated_energy_ratios():
    delta=torch.zeros((1,2,3),dtype=torch.float64)
    base=torch.full_like(delta,1/3);gate=torch.zeros_like(delta)
    mask=torch.ones_like(delta,dtype=torch.bool);query=mask.any(-1)
    out=_direction_metrics(base,base,gate,delta,mask,torch.ones((1,3,2),dtype=delta.dtype),query)
    assert not out['rows']['common_energy_fraction'][1].any()
    assert out['rows']['zero_selected_mass'][0].sum()==2
    assert all(np.isfinite(q) for q in out['checks'].values())


def synthetic(device):
    torch.manual_seed(2101)
    n=3;g=24
    member=np.zeros((8,g),bool)
    for k in range(8):member[k,k*3:(k+1)*3]=True
    dtype=torch.float64
    ft={key:torch.randn((n,width),dtype=dtype,device=device) for key,width in [('C',3),('R',g),('G',g),('L',g),('X',5)]}
    ft['RM']=torch.ones((n,g),dtype=dtype,device=device);ft['CM']=ft['RM'].clone()
    ft['RM'][0,:3]=0;ft['CM'][1,3:6]=0
    ft['ANCHOR']=torch.tensor([.2,.4,.3],dtype=dtype,device=device)
    ds=SimpleNamespace(ft=ft,owner=np.array([0,0,1]),fit=np.array([0,1]),fold=1,role='inner',seed=31,
        visual=torch.randn((3,4,1536),dtype=dtype,device=device),
        mask=torch.ones((3,4),dtype=torch.bool,device=device),wrong_c=np.array([1,0,2]),wrong_o=np.array([1,0,2]))
    ds.mask[0,3]=False
    ds.check_indices=lambda ix:np.asarray(ix,dtype=int)
    model=SelectiveModel(3,member,np.linspace(-.1,.1,5),np.zeros(3),.8,'S16',
                         {'hidden':32,'training':{'dropout':.1}}).to(device=device,dtype=dtype)
    return model,ds


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_checkpoint_diagnostics_preserve_state_gradients_mode_rng_and_privacy(device):
    model,ds=synthetic(device);model.train()
    before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    result=diagnose_checkpoint(model,ds,chunk_size=1)
    assert result['status']=='PASS' and model.training
    assert result['fit_patients']==2 and result['fit_WSIs']==3 and result['batches']==2
    assert result['additional_model_forward_calls']==2 and result['additional_operator_calls']==12
    assert result['additional_gradient_calls']==result['optimizer_steps']==0
    assert not any(p.grad is not None for p in model.parameters())
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert torch.equal(rng,torch.get_rng_state())
    if device=='cuda':assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert result['metrics']['po_raw_delta_row_SD_patient']['count']==2
    assert result['metrics']['chain_actual_patient_correction_risk']['count']==2
    assert 'moments' not in result
    assert all(value is False for key,value in result['privacy'].items() if key!='new_probe_fits')
    assert len(model.p_value._forward_hooks)==len(model.o_value._forward_hooks)==len(model.correction_head._forward_hooks)==0
    with pytest.raises(ValueError,match='fit patients'):
        diagnose_batch(model,ds,np.array([2]))


def test_checkpoint_chunking_retains_equal_patient_means_and_message_head_chain():
    model,ds=synthetic('cpu')
    one=diagnose_checkpoint(model,ds,chunk_size=1)
    two=diagnose_checkpoint(model,ds,chunk_size=2)
    for key,q in one['metrics'].items():
        r=two['metrics'][key]
        assert q['count']==r['count'],key
        if q['count']:
            assert q['mean']==pytest.approx(r['mean'],rel=2e-8,abs=2e-12),key
    assert two['checks']['head_input_reconstruction_max_error']<1e-12
    assert two['checks']['head_projection_reconstruction_max_error']<1e-12
    assert two['checks']['risk_branch_reconstruction_max_error']<1e-12
