import pytest
import torch
from src.semantic_m123.models import JointModel,ARMS,slide_mean,masked_softmax


def batch(n=3):
    torch.manual_seed(9)
    mask=torch.ones(n,7,dtype=torch.bool);mask[:,-1]=False
    return {'patch':torch.randn(n,7,768),'mask':mask,
        'slide':torch.tensor([[0,0,1,1,1,1,-1]]).expand(n,-1).clone(),
        'mean':torch.randn(n,768),'P':torch.randn(n,12),'G':torch.randn(n,18),
        'C':torch.randn(n,5),'program':torch.randn(n,50,3),
        'program_mask':torch.ones(n,50,3,dtype=torch.bool),'coverage':torch.ones(n,50,3)}


@pytest.mark.parametrize('arm',ARMS[1:])
def test_every_arm_exact_trainable_warm_risk(arm):
    b={k:(v.double() if v.is_floating_point() else v) for k,v in batch().items()}
    m=JointModel(arm,12,18,5).double()
    wp=torch.randn(12,dtype=torch.float64);wg=torch.randn(18,dtype=torch.float64);wc=torch.randn(5,dtype=torch.float64)
    m.warm(wp,wg,wc,.3,-.2)
    expected=b['P']@wp+b['G']@wg+b['C']@wc+.1
    assert torch.allclose(m(b)['risk'],expected,atol=2e-13,rtol=1e-13)
    assert all(p.requires_grad for p in m.parameters())
    assert not any('offset' in k or 'teacher' in k for k in m.state_dict())


@pytest.mark.parametrize('arm',ARMS)
def test_padding_and_joint_patch_permutation_invariance(arm):
    b=batch();m=JointModel(arm,12,18,5).eval()
    with torch.no_grad():m.local_out.weight.copy_(torch.eye(64))
    r=m(b)['risk'];bad={**b,'patch':b['patch'].clone()}
    bad['patch'][~b['mask']]=float('nan')
    assert torch.equal(m(bad)['risk'],r)
    p=torch.tensor([6,2,0,4,1,5,3])
    perm={**b,**{k:b[k][:,p] for k in ('patch','mask','slide')}}
    assert torch.allclose(m(perm)['risk'],r,atol=2e-6,rtol=1e-5)


def test_equal_slide_weight_not_patch_weight_and_rename_invariance():
    x=torch.tensor([[[2.],[8.],[8.],[8.]]]);m=torch.ones(1,4,dtype=torch.bool)
    assert slide_mean(x,m,torch.tensor([[0,1,1,1]])).item()==5.
    assert slide_mean(x,m,torch.tensor([[9,2,2,2]])).item()==5.


def test_all_masked_softmax_has_zero_output_and_finite_gradient():
    x=torch.randn(2,4,requires_grad=True)
    a=masked_softmax(x,torch.tensor([[False]*4,[True,False,True,False]]))
    assert a[0].sum()==0 and torch.allclose(a[1].sum(),torch.tensor(1.))
    a.square().sum().backward();assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('arm',ARMS[-4:])
def test_no_measured_program_removes_interaction_not_own_main_effect(arm):
    b=batch();m=JointModel(arm,12,18,5)
    with torch.no_grad():m.interaction_head.weight.fill_(.1)
    b['program_mask'].zero_();b['program'].fill_(float('nan'))
    out=m(b)
    assert torch.equal(out['interaction_risk'],torch.zeros(3))
    assert torch.isfinite(out['risk']).all() and not torch.equal(out['omics_risk'],torch.zeros(3))


def test_wrong_query_only_preserves_owner_values_and_all_main_effects():
    b=batch();m=JointModel('M3_FULL',12,18,5)
    with torch.no_grad():m.interaction_head.weight.fill_(1.)
    wrong={k:b[k].roll(1,0) for k in ('program','program_mask','coverage')}
    a=m(b,diagnostics=True);c=m(b,query_context=wrong,diagnostics=True)
    for key in ('pathology_risk','omics_risk','clinical_risk'):assert torch.equal(a[key],c[key])
    assert not torch.equal(a['patch_attention'],c['patch_attention'])
    assert not torch.equal(a['interaction_risk'],c['interaction_risk'])


def test_learned_query_attention_ignores_rna_but_main_effect_does_not():
    b=batch();m=JointModel('M3_LEARNED_QUERY',12,18,5)
    wrong={k:b[k].roll(1,0) for k in ('program','program_mask','coverage')}
    assert torch.equal(m(b,diagnostics=True)['patch_attention'],m(b,query_context=wrong,diagnostics=True)['patch_attention'])


def test_head_stage_then_joint_stage_has_real_cross_modal_gradient():
    b=batch();m=JointModel('M3_FULL',12,18,5)
    m.set_stage(True)
    assert set(id(p) for p in m.parameters() if p.requires_grad)==set(id(p) for p in m.head_parameters())
    m.set_stage(False)
    with torch.no_grad():m.interaction_head.weight.fill_(.1)
    m(b)['interaction_risk'].square().sum().backward()
    for layer in (m.patch[0],m.program[0],m.query_projection,m.attention.q,m.attention.k,m.attention.v):
        assert layer.weight.grad is not None and layer.weight.grad.norm()>0


def test_scratch_cannot_accept_r4_weights():
    with pytest.raises(ValueError,match='SCRATCH'):
        JointModel('M1_SCRATCH',12,18,5).warm(torch.zeros(12),torch.zeros(18),torch.zeros(5),0,0)


def test_mean_context_requires_training_reference():
    with pytest.raises(ValueError,match='TRAIN_ONLY'):
        JointModel('M3_FULL',12,18,5)(batch(),query_mode='mean_context')

