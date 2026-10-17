"""Exact clinical nulls, frozen anchor and comparable Token-capacity changes."""
import numpy as np
import pytest
import torch
from src.phase20_selective.model import SignedPathwayTokens,SelectiveModel,ARMS,normalize_gate_scores


def fixture(arm='S16',sd=.8):
    torch.manual_seed(201020)
    member = np.zeros((8,160),bool)
    for k in range(8):member[k,k*20:(k+1)*20]=True
    w = np.array([.1234567898765,-.4,.09,.02,.1],dtype=np.float64)
    model = SelectiveModel(3,member,w,np.zeros(3),sd,arm,{'hidden':32,'training':{'dropout':0.}}).double().eval()
    torch.manual_seed(920)
    visual = torch.randn(2,5,1536,dtype=torch.float64)
    mask = torch.ones(2,5,dtype=torch.bool);mask[0,-1]=False
    c = torch.randn(2,3,dtype=torch.float64)
    r,g,l = [torch.randn(2,160,dtype=torch.float64) for _ in range(3)]
    rm,cm = torch.ones_like(r),torch.ones_like(r)
    x = torch.randn(2,5,dtype=torch.float64)
    return model,(visual,mask,c,r,g,l,rm,cm,x),w


def test_all_arms_common_initial_state_is_exact_and_main_is_frozen_float64():
    models = {arm:fixture(arm)[0] for arm in ARMS}
    ref = models['P16'].state_dict()
    for arm,model in models.items():
        assert 'main' not in dict(model.named_parameters()) and model.main.dtype==torch.float64
        for key,value in model.state_dict().items():
            if not key.startswith('molecular.'):
                torch.testing.assert_close(value,ref[key],atol=0,rtol=0,msg=arm+':'+key)
        if arm!='L0':
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,ref[key],atol=0,rtol=0,msg=arm+':'+key)
    _,_,w=fixture()
    np.testing.assert_array_equal(models['S16'].main.numpy(),w)


def test_anchor_override_is_exact_and_detached_and_main_only_has_no_pair():
    model,args,_=fixture()
    anchor=torch.tensor([.127623456,.876512345],dtype=torch.float64,requires_grad=True)
    risk,diag=model(*args,anchor_override=anchor,diagnostic=True)
    torch.testing.assert_close(diag['main_risk'],anchor.detach(),atol=0,rtol=0)
    main=model(*args,anchor_override=anchor,main_only=True)
    torch.testing.assert_close(main,anchor.detach(),atol=0,rtol=0)
    risk.sum().backward()
    assert anchor.grad is None and model.main.grad is None


@pytest.mark.parametrize('arm',['S1','S16','D16','F16','WC16','WO16'])
def test_clinical_reference_is_exact_correction_off_and_predictions_are_bounded(arm):
    model,args,_=fixture(arm)
    reference=model.clinical_mean.expand_as(args[2])
    null,dn=model(*args,context_clinical=reference,diagnostic=True)
    off=model(*args,interaction_off=True)
    torch.testing.assert_close(null,off,atol=0,rtol=0)
    assert dn['correction_risk'].count_nonzero()==0
    for updated,base in [('p_to_o','base_p_to_o'),('o_to_p','base_o_to_p')]:
        torch.testing.assert_close(dn[updated],dn[base],atol=0,rtol=0)
    _,diag=model(*args,diagnostic=True)
    assert (diag['correction_risk'].abs()<=model.correction_cap+1e-14).all()
    assert (diag['pair_risk'].abs()<=model.risk_scale+1e-14).all()


def test_zero_anchor_sd_creates_zero_caps_without_epsilon_inflation():
    model,args,_=fixture(sd=0.)
    _,diag=model(*args,diagnostic=True)
    assert diag['pair_risk'].count_nonzero()==diag['correction_risk'].count_nonzero()==0


def test_only_correction_sees_wrong_clinical_or_wrong_omics():
    model,args,_=fixture()
    risk,matched=model(*args,diagnostic=True)
    wrong_c,dc=model(*args,context_clinical=args[2].flip(0),diagnostic=True)
    wrong_o,do=model(*args,context_omics=tuple(v.flip(0) for v in args[3:8]),diagnostic=True)
    for changed in [dc,do]:
        for key in ['main_risk','pair_risk','tokens','base_p_to_o','base_o_to_p']:
            torch.testing.assert_close(changed[key],matched[key],atol=0,rtol=0,msg=key)
    assert not torch.allclose(risk,wrong_c,atol=1e-13,rtol=1e-13)
    assert not torch.allclose(risk,wrong_o,atol=1e-13,rtol=1e-13)


def test_frozen_gate_ignores_clinical_while_delta_still_uses_it():
    model,args,_=fixture('F16')
    _,a=model(*args,diagnostic=True)
    _,b=model(*args,context_clinical=-args[2],diagnostic=True)
    torch.testing.assert_close(a['gate_p_to_o'],b['gate_p_to_o'],atol=0,rtol=0)
    torch.testing.assert_close(a['gate_o_to_p'],b['gate_o_to_p'],atol=0,rtol=0)
    assert not torch.allclose(a['delta'],b['delta'])


def test_donor_missingness_only_removes_correction_eligibility_not_base_support():
    model,args,_=fixture()
    context=[x.clone() for x in args[3:8]]
    context[3].zero_();context[4].zero_()
    risk,diag=model(*args,context_omics=tuple(context),diagnostic=True)
    off,base=model(*args,interaction_off=True,diagnostic=True)
    assert not diag['correction_eligible_mask'].any()
    torch.testing.assert_close(risk,off,atol=0,rtol=0)
    torch.testing.assert_close(diag['base_p_to_o'],base['base_p_to_o'],atol=0,rtol=0)
    assert diag['gate_p_to_o'].count_nonzero()==diag['gate_o_to_p'].count_nonzero()==0


def test_uniform_pair_freezes_only_attention_not_own_values_or_main():
    model,args,_=fixture('P16')
    _,base=model(*args,diagnostic=True)
    _,uniform=model(*args,pair_uniform=True,diagnostic=True)
    torch.testing.assert_close(base['tokens'],uniform['tokens'],atol=0,rtol=0)
    torch.testing.assert_close(base['main_risk'],uniform['main_risk'],atol=0,rtol=0)
    expected=uniform['omics_mask'][:,None].double().expand_as(uniform['base_p_to_o'])/uniform['omics_mask'].sum(-1)[:,None,None]
    expected=expected*uniform['patch_mask'][:,:,None]
    torch.testing.assert_close(uniform['base_p_to_o'],expected,atol=1e-13,rtol=1e-13)
    assert not torch.allclose(base['base_p_to_o'],uniform['base_p_to_o'])


@pytest.mark.parametrize('arm',['P1','P16'])
def test_all_sixteen_filters_participate_in_gradient_and_rank1_capacity_is_explicit(arm):
    model,args,_=fixture(arm)
    risk,diag=model(*args,diagnostic=True)
    risk.square().sum().backward()
    assert diag['summary16_before'].shape==(2,24,16)
    if arm=='P1':
        torch.testing.assert_close(diag['summary16_after'],4*diag['summary16_before'].mean(-1,keepdim=True).expand_as(diag['summary16_before']),atol=0,rtol=0)
    else:
        torch.testing.assert_close(diag['summary16_after'],diag['summary16_before'],atol=0,rtol=0)
    for weight in model.molecular.filters:
        assert weight.grad is not None and (weight.grad.norm(dim=-1)>0).all()


def test_signed_filter_matches_literal_measured_row_norm_and_separates_channels():
    model,args,_=fixture('P16')
    r,g,l,rm,cm=[x.clone() for x in args[3:8]]
    rm[0,:4]=0;r[0,:4]=torch.nan
    cm[0,:20]=0;g[0,:20]=torch.nan;l[0,:20]=torch.nan
    token,valid,diag=model.molecular(r,g,l,rm,cm,diagnostic=True)
    w=model.molecular.filters[0]; ix=model.molecular.gene_indices_0
    measured=rm[:,ix].bool(); x=torch.where(measured,r[:,ix],torch.zeros_like(r[:,ix]))
    expected=torch.stack([torch.stack([(x[b]*row).sum()/torch.sqrt((row.square()*measured[b]).sum()) for row in w]) for b in range(2)])
    torch.testing.assert_close(diag['summary16_before'][:,0],expected,atol=1e-12,rtol=1e-12)
    assert valid[0,:8].all() and not valid[0,8] and not valid[0,16]
    assert token[0,8].count_nonzero()==0 and token[0,16].count_nonzero()==0
    assert torch.isfinite(token).all()


def test_model_partial_gate_and_update_paths_have_real_gradients():
    model,args,_=fixture()
    risk,diag=model(*args,diagnostic=True)
    risk.square().sum().backward()
    for name in ['Up.weight','Uo.weight','Uc.weight','wg.weight','wd.weight','correction_head.weight']:
        grad=dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0,name
    for updated,gate,eligible in [('p_to_o','gate_p_to_o',diag['correction_eligible_mask']),
                                  ('o_to_p','gate_o_to_p',diag['correction_eligible_mask'].transpose(1,2))]:
        torch.testing.assert_close(diag[gate].sum(-1),.25*eligible.to(diag[gate].dtype).sum(-1),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(diag[updated].sum(-1),diag['base_'+updated].sum(-1),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_nonzero_clinical_reference_is_bitwise_null_in_float32(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model,args,_=fixture()
    model=model.float().to(device)
    # Production preserves main as float64 even when all trainable modules use
    # float32; avoid relying on the implicit main path in this null test.
    model.main=model.main.double()
    with torch.no_grad():model.clinical_mean.copy_(torch.tensor([.5173,-.2191,.1777],device=device))
    args=tuple(v.to(device=device,dtype=torch.bool if v.dtype==torch.bool else torch.float32) for v in args)
    reference=model.clinical_mean.expand_as(args[2])
    anchor=torch.tensor([.134,.512],device=device)
    risk,diag=model(*args,context_clinical=reference,anchor_override=anchor,diagnostic=True)
    off=model(*args,anchor_override=anchor,interaction_off=True)
    torch.testing.assert_close(risk,off,atol=0,rtol=0)
    assert diag['delta'].count_nonzero()==diag['correction_risk'].count_nonzero()==0
    torch.testing.assert_close(diag['p_to_o'],diag['base_p_to_o'],atol=0,rtol=0)


def test_signed_filter_input_and_parameter_gradient_matches_literal_oracle():
    model,args,_=fixture('P16')
    r=args[3].clone().requires_grad_(); rm=args[6].clone();rm[0,:7]=0
    _,_,diag=model.molecular(r,*args[4:6],rm,args[7],diagnostic=True)
    w=model.molecular.filters[0]
    measured=rm[:,:20].bool()
    x=torch.where(measured,r[:,:20],torch.zeros_like(r[:,:20]))
    literal=torch.stack([torch.stack([(x[b]*row).sum()/(row.square()*measured[b]).sum().sqrt() for row in w]) for b in range(2)])
    probe=torch.arange(32,dtype=torch.float64).reshape(2,16)/31
    actual_grads=torch.autograd.grad((diag['summary16_before'][:,0]*probe).sum(),(r,w),retain_graph=True)
    oracle_grads=torch.autograd.grad((literal*probe).sum(),(r,w))
    for actual,oracle in zip(actual_grads,oracle_grads):
        torch.testing.assert_close(actual,oracle,atol=2e-12,rtol=2e-12)
    assert actual_grads[0][0,:7].count_nonzero()==0


def test_masked_patch_padding_is_ignored_and_patch_order_is_equivariant():
    model,args,_=fixture()
    risk,diag=model(*args,diagnostic=True)
    padded=list(args);padded[0]=args[0].clone();padded[0][~args[1]]=torch.nan
    padded_risk=model(*padded)
    torch.testing.assert_close(risk,padded_risk,atol=0,rtol=0)
    permutation=torch.tensor([4,1,3,0,2]);permuted=list(args)
    permuted[0]=args[0][:,permutation];permuted[1]=args[1][:,permutation]
    permuted_risk,other=model(*permuted,diagnostic=True)
    torch.testing.assert_close(risk,permuted_risk,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(diag['p_to_o'][:,permutation],other['p_to_o'],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(diag['o_to_p'][:,:,permutation],other['o_to_p'],atol=1e-12,rtol=1e-12)


def test_output_correction_cap_cannot_be_undone_by_a_large_readout():
    model,args,_=fixture()
    with torch.no_grad():model.correction_head.weight.mul_(1e12)
    _,diag=model(*args,diagnostic=True)
    assert (diag['correction_risk'].abs()<=model.correction_cap).all()


def test_gate_normalization_gradcheck_mask_and_positive_scale_shift_invariance():
    score=torch.tensor([[.1,.7,-.2,.3,9.],[.5,.5,.5,.5,8.],[4.,2.,8.,1.,9.]],dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[True,True,True,True,False],[True,True,True,True,False],[False]*5])
    normalized=normalize_gate_scores(score,mask)
    torch.testing.assert_close(normalized[0,:4].mean(),torch.tensor(0.,dtype=score.dtype),atol=1e-13,rtol=0)
    torch.testing.assert_close(normalized[0,:4].square().mean(),torch.tensor(1.,dtype=score.dtype),atol=1e-13,rtol=0)
    assert normalized[~mask].count_nonzero()==0 and normalized[1:].count_nonzero()==0
    torch.testing.assert_close(normalized,normalize_gate_scores(score*7.+37.,mask),atol=1e-12,rtol=1e-12)
    assert torch.autograd.gradcheck(lambda s:normalize_gate_scores(s,mask),(score,),eps=1e-8,atol=1e-4,rtol=1e-4)
    (normalized*torch.arange(5,dtype=score.dtype)).sum().backward()
    assert torch.isfinite(score.grad).all() and score.grad[0].norm()>0
    assert score.grad[~mask].count_nonzero()==0 and score.grad[2].count_nonzero()==0


def test_gate_normalization_padding_and_floor_have_finite_exact_behavior():
    s=torch.tensor([[.1,.4,-.7]],dtype=torch.float64,requires_grad=True)
    mask=torch.ones_like(s,dtype=torch.bool)
    padded=torch.cat([s,torch.full((1,2),torch.nan,dtype=s.dtype)],-1)
    padded_mask=torch.cat([mask,torch.zeros((1,2),dtype=torch.bool)],-1)
    torch.testing.assert_close(normalize_gate_scores(s,mask),normalize_gate_scores(padded,padded_mask)[:,:3],atol=0,rtol=0)
    constant=torch.ones((2,4),dtype=torch.float64,requires_grad=True)
    result=normalize_gate_scores(constant,torch.ones_like(constant,dtype=torch.bool))
    assert result.count_nonzero()==0
    (result*torch.arange(4,dtype=result.dtype)).sum().backward()
    assert torch.isfinite(constant.grad).all()
    tiny=torch.tensor([[0.,1e-8]],dtype=torch.float64)
    torch.testing.assert_close(normalize_gate_scores(tiny,torch.ones_like(tiny,dtype=torch.bool)),torch.tensor([[-.005,.005]],dtype=torch.float64),atol=1e-14,rtol=0)


def test_model_directional_gate_normalization_enables_actual_zero_support():
    model,args,_=fixture()
    _,diag=model(*args,diagnostic=True)
    for normkey,gatekey,eligible in [('normalized_gate_score_p','gate_p_to_o',diag['correction_eligible_mask']),
        ('normalized_gate_score_o','gate_o_to_p',diag['correction_eligible_mask'].transpose(1,2))]:
        normalized=diag[normkey];count=eligible.sum(-1).clamp_min(1)
        torch.testing.assert_close(normalized.sum(-1)/count,torch.zeros_like(count,dtype=normalized.dtype),atol=1e-12,rtol=0)
        valid_rows=eligible.any(-1)
        torch.testing.assert_close(normalized.square().sum(-1)[valid_rows]/count[valid_rows],torch.ones_like(count[valid_rows],dtype=normalized.dtype),atol=1e-12,rtol=1e-12)
        assert (diag[gatekey][eligible]==0).any()
