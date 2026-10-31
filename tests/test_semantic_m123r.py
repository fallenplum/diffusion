import copy
import numpy as np
import pytest
import torch
from src.semantic_m123r.models import RepairModel,ARMS,NormalizedAttention
from src.semantic_m123r.training import fit,function_penalty,a_multiplier
from src.semantic_m123r.protocol import inner_jobs,configurations,choose
from src.semantic_m123r.data import patch_scale
from src.semantic_m123r.io import selected_npy_rows,stratified_cycle,save_checkpoint,load_checkpoint,sha

torch.set_num_threads(1)


def batch():
    torch.manual_seed(91);n=6
    b={'P':torch.randn(n,12,dtype=torch.float64),'G':torch.randn(n,18,dtype=torch.float64),
        'C':torch.randn(n,5,dtype=torch.float64),'patch':torch.randn(n,12,768),
        'mask':torch.ones(n,12,dtype=torch.bool),'slide':torch.zeros(n,12,dtype=torch.long),
        'program':torch.randn(n,50,3),'program_mask':torch.ones(n,50,3,dtype=torch.bool),'coverage':torch.ones(n,50,3)}
    b['mean_patch']=b['patch'].mean(1)
    w={'p_coef':torch.randn(12,dtype=torch.float64),'g_coef':torch.randn(18,dtype=torch.float64),
       'c_coef':torch.randn(5,dtype=torch.float64),'p_bias':.1,'g_bias':-.3}
    return b,w


@pytest.mark.parametrize('arm',ARMS)
def test_s0_exact_fp64_anchor_without_fixed_offset(arm):
    b,w=batch();m=RepairModel(arm,12,18,5);m.warm(w)
    expected=b['P']@w['p_coef']+b['G']@w['g_coef']+b['C']@w['c_coef']-.2
    assert torch.allclose(m(b)['risk'],expected,atol=3e-13,rtol=1e-13)
    assert m(b)['risk'].dtype==torch.float64
    assert all('offset' not in n for n,p in m.named_parameters())


def test_schedule_activates_actual_outputs_before_encoders_and_main_effects():
    b,w=batch();m=RepairModel('R6',12,18,5);m.warm(w)
    m.stage(10);assert all(n.startswith(m.output_names()) for n,p in m.named_parameters() if p.requires_grad)
    m.stage(11);assert all(not p.requires_grad for n,p in m.group('base'))
    assert m.patch[0].weight.requires_grad and m.cross_attention.q.weight.requires_grad
    m.stage(91);assert all(p.requires_grad for p in m.parameters())


def test_actual_mil_encoder_updates_and_base_freeze():
    b,w=batch();m=RepairModel('R5',12,18,5);m.warm(w)
    ref=m(b)['risk'].detach().numpy();initial={n:p.detach().clone() for n,p in m.group('base')}
    result=fit(m,b,np.arange(1,7),[1,1,0,1,0,1],ref,{'a':.01,'b':.01,'n':.0005},max_epochs=30)
    assert result['actual_parameter_update_counts']['patch']==20
    assert result['actual_parameter_update_counts']['mil_query']==20
    assert result['actual_parameter_update_counts']['mil_attention']==20
    assert result['actual_parameter_update_counts']['base']==0
    assert all(torch.equal(p,initial[n]) for n,p in m.group('base'))
    assert m.local_out.weight.norm()>0
    assert (m(b)['risk']-m(b,disable_mil=True)['risk']).std()>0


def test_literal_function_sum_shift_invariance_and_not_mean_normalized():
    x=torch.tensor([1.,3.,2.]);r=torch.tensor([3.,1.,2.])
    assert function_penalty(x,r)==8
    assert function_penalty(x+100,r-20)==8
    assert function_penalty(x.repeat(2),r.repeat(2))==16
    assert a_multiplier(90)==1 and a_multiplier(140)==pytest.approx(.1)


def test_group_regularization_is_proximal_to_warm_not_zero():
    b,w=batch();m=RepairModel('R3',12,18,5);m.warm(w)
    pb,pn=m.penalties();assert pb==0 and pn>0
    with torch.no_grad():m.base['p'].head.weight[0,0]+=.1
    assert float(m.penalties()[0].detach())==pytest.approx(.01)


def test_wrong_query_preserves_owner_main_effects_and_mil():
    b,w=batch();m=RepairModel('R6',12,18,5);m.warm(w)
    with torch.no_grad():m.local_out.weight.fill_(.01);m.interaction_head.weight.fill_(.1)
    wrong={k:b[k].roll(1,0) for k in ('program','program_mask','coverage')}
    x=m(b,diagnostics=True);y=m(b,query_context=wrong,diagnostics=True)
    for key in ('base_risk','local_risk','pathology_main_risk','omics_main_risk','clinical_main_risk'):assert torch.equal(x[key],y[key])
    assert not torch.equal(x['cross_attention'],y['cross_attention'])
    assert not torch.equal(x['risk'],y['risk'])


def test_learned_query_projection_is_predictive_and_not_a_dummy_parameter():
    b,w=batch();m=RepairModel('R7',12,18,5);m.warm(w)
    with torch.no_grad():m.interaction_head.weight.fill_(.1)
    m(b)['interaction_risk'].sum().backward()
    assert m.query_projection.weight.grad is not None and m.query_projection.weight.grad.norm()>1e-10
    context={k:b[k].roll(1,0) for k in ('program','program_mask','coverage')}
    assert torch.equal(m(b,diagnostics=True)['cross_attention'],m(b,query_context=context,diagnostics=True)['cross_attention'])


def test_shared_patch_scale_is_train_only_and_missing_safe():
    x=np.arange(4*4*3,dtype=float).reshape(4,4,3);mask=np.ones((4,4),bool);mask[:,-1]=False
    x[:,-1]=np.nan;slide=np.zeros((4,4),int)
    a,st=patch_scale(x,mask,slide,2);x[2:]+=1000;b,ss=patch_scale(x,mask,slide,2)
    assert np.array_equal(st['patch_mean'],ss['patch_mean']) and np.array_equal(st['patch_sd'],ss['patch_sd'])
    assert np.array_equal(a[:2],b[:2]) and np.isfinite(a).all() and np.all(a[~mask]==0)


def test_padding_and_joint_patch_order_invariance():
    b,w=batch();m=RepairModel('R6',12,18,5);m.warm(w)
    with torch.no_grad():m.local_out.weight.fill_(.01);m.interaction_head.weight.fill_(.1)
    p=torch.randperm(12);c={**b,**{k:b[k][:,p] for k in ('patch','mask','slide')}}
    assert torch.allclose(m(b)['risk'],m(c)['risk'],atol=1e-7,rtol=1e-7)


def test_finite_qualification_grid_and_trained_snapshot_exclusion():
    assert len(inner_jobs())==540 and all(j['outer']==1 for j in inner_jobs())
    metric=lambda h:{'Harrell_num':h,'Harrell_den':1,'Uno_num':h,'Uno_den':1}
    c=[{'config':configurations('R3')[0],'epoch':e,'metrics':[metric(h)]*3} for e,h in ((10,.9),(30,.8))]
    assert choose(c,trained=False)['epoch']==10 and choose(c,trained=True)['epoch']==30


def test_normalized_attention_all_missing_and_parameter_matching():
    att=NormalizedAttention();q=torch.randn(2,8,64);x=torch.randn(2,5,64)
    z,a=att(q,x,torch.zeros(2,5,dtype=torch.bool));assert torch.equal(z,torch.zeros_like(z)) and a.sum()==0
    models=[RepairModel(k,12,18,5) for k in ('R6','R7','R8','R9')]
    assert len({sum(p.numel() for p in m.parameters()) for m in models})==1


def test_selected_endpoint_bytes_do_not_decode_heldout_nan():
    import io
    x=np.array([1.,np.nan,3.,np.nan]);stream=io.BytesIO();np.save(stream,x);stream.seek(0)
    assert np.array_equal(selected_npy_rows(stream,[2,0],4),[3.,1.])


def test_wrong_context_cycle_preserves_full_measurement_pattern():
    r=np.array([[1,0],[1,0],[1,1],[0,1]],bool);c=np.ones((4,2),bool)
    order,valid=stratified_cycle(r,c)
    assert np.array_equal(r,r[order]) and list(valid)==[True,True,False,False]
    assert (order[valid]!=np.flatnonzero(valid)).all()


@pytest.mark.parametrize('arm,epoch',[('R3',0),('R3',30),('R6',10),('R6',140)])
def test_real_checkpoint_composition_replays_fp64_and_fp32_modules(tmp_path,arm,epoch):
    import pickle
    b,w=batch();m=RepairModel(arm,12,18,5);m.warm(w)
    p=tmp_path/'preprocessing'/'initial.private.pkl';p.parent.mkdir()
    with p.open('wb') as f:pickle.dump({n:v.detach().numpy().copy() for n,v in m.named_parameters() if n.startswith('base.')},f)
    with torch.no_grad():m.local_out.weight.fill_(.001)
    m.stage(epoch)
    f=tmp_path/'snapshot.private.pt.gz';save_checkpoint(f,m,epoch,{},[12,18,5],{'file':p.name,'sha256':sha(p)},'synthetic')
    other=RepairModel(arm,12,18,5);load_checkpoint(f,tmp_path,other)
    assert {n:p.requires_grad for n,p in m.named_parameters()}=={n:p.requires_grad for n,p in other.named_parameters()}
    assert torch.equal(m(b)['risk'],other(b)['risk'])


def test_failed_mechanism_cannot_launch_formal_controls_as_candidates():
    from src.semantic_m123r.protocol import formal_arms
    g={k:{'passed':False} for k in ('M1_GROUP','M1_FUNCTION','MIL','MOLECULAR')}
    assert formal_arms(g)==[]
    g['MIL']['passed']=True;assert formal_arms(g)==['R3','R4','R5']


def test_direction_requires_both_metrics_and_two_seed_support():
    from src.semantic_m123r.reporting import comparison
    base=[{'seed':0,'Harrell':.8,'Uno60':.8}]
    x=[{'seed':s,'Harrell':.81,'Uno60':.79} for s in (31,47,83)]
    assert not comparison(x,base)['direction_pass']
    x[0]['Uno60']=.81;x[1]['Uno60']=.81
    assert comparison(x,base)['direction_pass']


def test_inactive_encoder_cannot_pass_structure_from_geometry_alone():
    from src.semantic_m123r.reporting import structure_pass
    node={'actual_update_counts':{'patch':0,'mil_query':0,'mil_attention':0},
        'maximum_observed_training_Cox_only_gradients':{},'diagnostics':{'validation':{},'train':{}}}
    assert not structure_pass('R5',node)


def test_missing_geometry_is_failed_qualification_not_a_runtime_error():
    from src.semantic_m123r.reporting import structure_pass
    keys=('patch','mil_query','mil_attention')
    node={'actual_update_counts':{k:20 for k in keys},'maximum_observed_training_Cox_only_gradients':{k:.1 for k in keys},
        'diagnostics':{'validation':{'MIL_slot_effective_rank_mean':None},'train':{}}}
    assert not structure_pass('R5',node)


def test_retention_keeps_both_global_tracks_and_all_metrics(tmp_path,monkeypatch):
    from scripts import semantic_m123r_selection as module
    from scripts.semantic_m123r_common import folder,write
    from src.semantic_m123r.training import SNAPSHOTS
    jobs=module.block_jobs(1,'R2',31,'locked');receipts={}
    for j in jobs:
        f=folder(tmp_path,j);f.mkdir(parents=True);states={};snaps=[]
        for epoch in SNAPSHOTS:
            p=f/f'checkpoint_{epoch}.private.pt.gz';p.write_bytes(str(epoch).encode())
            states[str(epoch)]={'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size}
            h=.9 if epoch==10 else .8 if epoch==30 else .7
            snaps.append({'epoch':epoch,'actual_update_counts':{'base':epoch},'metrics':{'validation':{'risk':{'Harrell_num':h,'Harrell_den':1,'Uno_num':h,'Uno_den':1}}}})
        r={'checkpoints':states,'snapshots':snaps};receipts[str(f)]=r
        write(f/'METRICS_NOT_DELETED.private.json',{'all_snapshots':len(snaps)})
    monkeypatch.setattr(module,'verify_cell',lambda private,job:receipts[str(folder(private,job))])
    chosen=module.select_block(tmp_path,1,'R2',31,'locked')
    assert chosen['choices']['EARLY_ANY']['epoch']==10 and chosen['choices']['ENCODER_TRAINED']['epoch']==30
    assert len(list(tmp_path.rglob('checkpoint_*.private.pt.gz')))==6
    assert len(list(tmp_path.rglob('METRICS_NOT_DELETED.private.json')))==12
