"""Synthetic R7 wrapper/full-risk/replay contracts; no real cohort fits."""
from copy import deepcopy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts import bgb_en_r7_common as c
from scripts import bgb_en_r7_replay as replay
from src.bgb_en.r7_training import full_risk_step,full_train_gradient_probe,checkpoint_diagnostics,predict,saved_state
from src.bgb_en.r6_omics import mapping_from_programs,SparseProgramEncoder,PROGRAM_ORDER
from src.bgb_en.r6_source_model import SourceProgramModel
from src.bgb_en.r6_pathology import source_model_state_sha256
from src.phase10_trimodal.math import torch_cox,log_weights


def fixture(arm='GATE1'):
    torch.manual_seed(612)
    programs=[dict(id=name,up_genes=[chr(65+i),chr(69+i)],down_genes=[]) for i,name in enumerate(PROGRAM_ORDER)]
    mapping=mapping_from_programs(programs)
    source=SourceProgramModel('S7',gene_reader=SparseProgramEncoder(mapping),tissue_prior=[.3,.3,.4],init_seed=31)
    source.requires_grad_(False).eval();n,genes=6,9
    feature=dict(rna=np.random.default_rng(9).normal(size=(n,genes)).astype(np.float32),rna_mask=np.ones((n,genes),bool))
    feature['rna_mask'][1,2]=False;feature['rna'][~feature['rna_mask']]=0.
    prepared=dict(features=feature,base=np.array([.4,-.6,.1,.3,-.2,.5]),CP=np.array([.4,-.6,.1,.3,-.2,.5]),
        RNA=np.linspace(-1.,1.,n),beta=.2,train=np.arange(4),valid=np.array([4,5]))
    membership=torch.zeros(50,genes,dtype=torch.bool)
    for k in range(50):membership[k,k%genes]=True;membership[k,(k+2)%genes]=True
    encoded=dict(states=torch.randn(n,1,4,64),patch_mask=torch.ones(n,1,4,dtype=torch.bool),
        program_score_prior=torch.randn(n,1,4,4),diagnostics=dict(source_state_sha256=source_model_state_sha256(source)))
    neighbor=torch.arange(4)[None,None,:,None].expand(n,1,4,1).clone();neighbor=torch.cat((neighbor,neighbor.roll(-1,2)),-1)
    bank=c.old.EncodedPatchBank(encoded,neighbor,'cpu')
    model=c.R7PatientModel(arm,prepared,membership,['outside']+list(mapping.gene_symbols),source,bank,'cpu',seed=31)
    model.query_map=np.array([1,2,3,0,5,4])
    return model,prepared,dict(t=np.array([1.,2.,2.,4.,5.,6.]),e=np.array([1,1,1,0,1,0]))


def test_all_arms_share_norm_calibration_and_shared_initial_state():
    signatures=[];calibrations=[]
    for arm in c.TARGET_ARMS:
        model,prepared,_=fixture(arm);signatures.append(c.shared_state_hash(model))
        calibration=c.calibrate_training_rms(model,prepared['train'],batch_size=2)
        calibrations.append(calibration)
        assert all(v>=1e-6 for v in calibration['scales'].values())
        assert set(calibration['scales'])=={'context_rms','patch_rms','query_rms','gene_key_rms','evidence_rms','reverse_query_rms'}
        assert calibration['outcomes_accessed'] is False and calibration['new_optimizer_updates']==0
    assert len(set(signatures))==1
    assert all(row==calibrations[0] for row in calibrations)


def test_rms_rejects_nontraining_scope():
    model,prepared,_=fixture()
    with pytest.raises(ValueError,match='CURRENT_TRAIN_ONLY'):c.calibrate_training_rms(model,prepared['valid'])


@pytest.mark.parametrize('arm',['GATE1','GATE2_WRONG_RNA','GATE1_WRONG_PATH'])
def test_query_controls_keep_values_main_offset_and_main_off_keeps_trained_route(arm):
    model,prepared,_=fixture(arm);c.calibrate_training_rms(model,prepared['train']);model.refresh_training_query_mean()
    idx=torch.tensor([0,1,2]);matched=model(idx,control='matched_query')
    for control in ('wrong_query','wrong_path','mean_query','interaction_off'):
        row=model(idx,control=control)
        for key in ('gene_values','gene_measured_mask','patch_values','main_delta','offset'):assert torch.equal(row[key],matched[key])
    trained=model(idx);off=model(idx,control='main_off')
    assert torch.equal(off['cross_delta'],trained['cross_delta'])
    assert torch.count_nonzero(off['main_delta'])==0 and torch.equal(off['risk'],trained['offset']+trained['cross_delta'])
    assert torch.count_nonzero(model(idx,control='interaction_off')['cross_delta'])==0


def test_cached_efron_update_matches_single_full_graph_with_slot_geometry():
    a,prepared,raw=fixture('SLOT');c.calibrate_training_rms(a,prepared['train']);b=deepcopy(a)
    idx=torch.arange(4);time,event=raw['t'][:4],raw['e'][:4]
    oa=torch.optim.SGD([p for p in a.parameters() if p.requires_grad],lr=.03)
    ob=torch.optim.SGD([p for p in b.parameters() if p.requires_grad],lr=.03)
    row=a(idx);loss=torch_cox(row['risk'],log_weights(time,event,'cpu'),torch.as_tensor(event))
    loss=loss+.1*row['delta'].square().mean()+.1*row['cross_delta'].square().mean()+a.regularization()
    loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),5.);oa.step()
    result=full_risk_step(b,[idx[:1],idx[1:3],idx[3:]],time,event,ob)
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other;torch.testing.assert_close(p,q,atol=4e-8,rtol=3e-6,msg=name)
    assert result['optimizer_updates']==1 and result['full_riskset_patients']==4


def test_complete_train_gradient_paths_match_direct_vector_derivatives_and_no_grad_mutation():
    model,prepared,raw=fixture('SLOT');c.calibrate_training_rms(model,prepared['train'])
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
    for _,p in named:p.grad=torch.full_like(p,.123)
    old_grads=[p.grad.clone() for p in params];before=saved_state(model)
    result=full_train_gradient_probe(model,prepared,raw,batch_size=2)
    assert result['patients']==4 and result['microbatches']==2
    assert set(result['groups_by_path'])=={'cross_delta','risk','path_cross_delta','gene_cross_delta','total_delta_L2','cross_delta_L2','parameter_L2'}
    row=model(torch.arange(4));leaf=row['risk'].detach().requires_grad_(True)
    loss=torch_cox(leaf,log_weights(raw['t'][:4],raw['e'][:4],'cpu'),torch.as_tensor(raw['e'][:4]))
    weight=torch.autograd.grad(loss,leaf)[0].detach()
    for path in result['groups_by_path']:
        target=.1*row['delta'].square().mean() if path=='total_delta_L2' else .1*row['cross_delta'].square().mean() if path=='cross_delta_L2' else model.regularization() if path=='parameter_L2' else torch.dot(row[path],weight)
        grads=torch.autograd.grad(target,params,allow_unused=True,retain_graph=True)
        expected={}
        for (name,_),gradient in zip(named,grads):
            key='.'.join(name.split('.')[:2]);expected[key]=expected.get(key,0.)+(0. if gradient is None else float(gradient.double().square().sum()))
        for group,value in expected.items():assert result['groups_by_path'][path][group]['L2_squared']==pytest.approx(value,rel=3e-5,abs=1e-12)
    # Exact regularizer geometry for four independent slot directions.
    reg=model.regularization();grads=torch.autograd.grad(reg,params,allow_unused=True)
    for (name,p),gradient in zip(named,grads):
        factor=.25 if name in ('core.slot_path_weight','core.slot_gene_weight') else 1.
        torch.testing.assert_close(gradient,2e-4*factor*p,atol=1e-10,rtol=2e-6,msg=name)
    assert all(torch.equal(old,p.grad) for old,p in zip(old_grads,params))
    assert all(torch.equal(before[name],value) for name,value in model.state_dict().items())


def test_epoch60_diagnostics_never_access_validation(monkeypatch):
    model,prepared,raw=fixture('GATE2');c.calibrate_training_rms(model,prepared['train'])
    import src.bgb_en.r7_training as training
    original=training.predict;seen=[]
    def guarded(model,indices,*args,**kwargs):
        assert set(indices)<=set(prepared['train']);seen.extend(indices);return original(model,indices,*args,**kwargs)
    monkeypatch.setattr(training,'predict',guarded)
    result=checkpoint_diagnostics(model,prepared,raw,60,include_validation=False)
    assert result['validation']=='NOT_EVALUATED_AT_DIAGNOSTIC_ONLY_EPOCH'
    assert result['query_probe']['scope'].startswith('fixed_first_eight_current_train')
    assert result['query_probe']['controls']['matched_query']['rounds'][1]['round']==2
    assert result['query_probe']['fixed_six_RMS'].keys()=={'context_rms','patch_rms','query_rms','gene_key_rms','evidence_rms','reverse_query_rms'}
    assert set(seen)==set(prepared['train'])


def test_replay_rejects_nan_in_nonrisk_saved_scalar():
    raw=dict(t=np.array([1.,2.]),e=np.array([1,0]));idx=np.array([0,1])
    actual=dict(risk=np.array([1.,0.]),delta=np.array([.1,.2]));expected=deepcopy(actual);expected['delta'][0]=np.nan
    with pytest.raises(ValueError,match='NONFINITE_SAVED_SCALAR'):
        replay.compare_predictions(raw,idx,actual,expected,code='TEST')


def test_qualification_produced_upstream_schema_consumed_without_rewriting(tmp_path,monkeypatch):
    rows=[dict(partition=p,status='PASS',coefficient_models=13,beta_gradient_abs=0.,maximum_risk_error=0.,
        new_optimizer_fits=0,new_preprocessing_fits=0) for p in c.PARTITIONS]
    value=dict(status='PASS_R7_REUSED_UPSTREAM_39_COX_AND_3_BETA',rows=rows,teacher_refits=0,source_refits=0,
        new_optimizer_fits=0,inherited_R6_contract_sha256='a'*64)
    path=tmp_path/'UPSTREAM_REPLAY_AUDIT.json';path.write_text(json.dumps(value),encoding='utf-8');before=path.read_bytes()
    (tmp_path/'QUALIFICATION.json').write_text(json.dumps(dict(upstream_replay_audit_sha256=c.sha256(path))),encoding='utf-8')
    monkeypatch.setattr(replay,'upstream_replay',lambda ctx,bundle:{k:v for k,v in rows[bundle].items() if k!='partition'})
    ctx=dict(r6_ctx={},upstream_r6=dict(contract_sha256='a'*64))
    result,digest=replay.upstream_audit(SimpleNamespace(private=tmp_path),ctx,{p:p for p in c.PARTITIONS},{})
    assert result==value and digest==c.sha256(path) and path.read_bytes()==before
    path.write_bytes(before+b' ')
    with pytest.raises(ValueError,match='QUALIFIED_UPSTREAM_AUDIT_CHANGED'):
        replay.upstream_audit(SimpleNamespace(private=tmp_path),ctx,{p:p for p in c.PARTITIONS},{})


def test_saved_checkpoint_sidecopy_changes_rejected_before_pass_resume(tmp_path):
    metadata=dict(checkpoint_artifact_sha256={},checkpoint_diagnostics={})
    receipt=tmp_path/'COMPLETE.json';receipt.write_text('{}')
    for epoch in (20,40,80):
        path=tmp_path/f'checkpoint_epoch{epoch}.pt';path.write_bytes(b'private synthetic checkpoint')
        metadata['checkpoint_artifact_sha256'][str(epoch)]=c.sha256(path)
    for epoch in (20,40,60,80):
        value=dict(epoch=epoch);metadata['checkpoint_diagnostics'][str(epoch)]=value
        (tmp_path/f'DIAGNOSTIC_epoch{epoch}.json').write_text(json.dumps(value))
    replay.verify_saved_auxiliary_files(receipt,metadata)
    with (tmp_path/'checkpoint_epoch20.pt').open('ab') as stream:stream.write(b' changed')
    with pytest.raises(ValueError,match='CHECKPOINT_BYTES_CHANGED'):replay.verify_saved_auxiliary_files(receipt,metadata)


def test_actual_synthetic_80_update_runner_and_three_checkpoint_replay(tmp_path,monkeypatch):
    """One tiny synthetic unit checks producer/consumer, never a target-data fit."""
    from scripts import bgb_en_r7_run as runner
    c.configure_deterministic_runtime();model,prepared,raw=fixture('GATE1')
    model.calibration=c.calibrate_training_rms(model,prepared['train'])
    raw['t'][4:]=[2.5,3.5]  # Synthetic held-out events stay inside training IPCW support.
    raw.update(pfi_t=raw['t'].copy(),pfi_e=raw['e'].copy(),pfi_valid=np.ones(6,bool))
    pristine=deepcopy(model);source=dict(complete_source_state_sha256=source_model_state_sha256(model.core.foundation))
    model_origin=dict(source_key='MATCHED',source_origin=source,synthetic_only=True)
    config=tmp_path/'config.json';source_config=tmp_path/'source.json';config.write_text('{}');source_config.write_text('{}')
    args=SimpleNamespace(private=tmp_path/'private',arm='GATE1',partition=0,seed=31,resume=False,device='cpu',
        config=config,source_config=source_config)
    args.private.mkdir();ctx=dict(lock=dict(contract_sha256='a'*64),raw=raw,upstream_r6=dict(contract_sha256='b'*64),source_audits={'MATCHED':source})
    origin=dict(prepared_receipt_sha256='c'*64,synthetic_only=True)
    constructor=lambda *a,**k:(deepcopy(pristine),deepcopy(model_origin))
    monkeypatch.setattr(runner,'construct',constructor);monkeypatch.setattr(replay,'construct',constructor)
    monkeypatch.setattr(runner,'validate_qualification',lambda *a:'d'*64)
    monkeypatch.setattr(runner,'verify_target_lock',lambda *a:None)
    monkeypatch.setattr(runner,'public_snapshot_guard',lambda:{})
    result=runner.run_unit(args,ctx,prepared,origin)
    assert result['optimizer_updates']==80 and result['actual_new_optimizer_fits']==1
    directory=args.private/'cells/GATE1_p0_seed31';receipt=directory/'COMPLETE.json'
    artifact=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
    detail,_=replay.replay_one(args,ctx,prepared,origin,receipt,result,artifact,'d'*64)
    assert detail['actual_checkpoint_count']==3 and [r['epoch'] for r in detail['checkpoints']]==[20,40,80]
    assert all(r['actual_state_dict_forward_replayed'] and r['changed_pair_credits']==0 for r in detail['checkpoints'])
    assert result['checkpoint_diagnostics']['60']['validation']=='NOT_EVALUATED_AT_DIAGNOSTIC_ONLY_EPOCH'
    # Keep final80 intact, forge mutually-consistent epoch20 state copies only.
    # Actual replay must detect changed early risks rather than merely hashing80.
    state=artifact['snapshots'][20]['state_dict'];state['core.path_main.weight']=state['core.path_main.weight']+.1
    path=directory/'checkpoint_epoch20.pt';cp=torch.load(path,map_location='cpu',weights_only=False)
    cp['state_dict']=state;cp['snapshot']['state_dict']=state;torch.save(cp,path)
    result['checkpoint_artifact_sha256']['20']=c.sha256(path)
    with pytest.raises(ValueError,match='CHECKPOINT_TRAIN_NUMERIC_TOLERANCE_FAILED'):
        replay.replay_one(args,ctx,prepared,origin,receipt,result,artifact,'d'*64)


def test_worker_increase_requires_actual_GPU_RAM_headroom_and_largest_arm(tmp_path):
    from scripts.bgb_en_r7_schedule import qualified_workers
    args=SimpleNamespace(private=tmp_path,max_workers=6);ctx=dict(lock=dict(contract_sha256='a'*64))
    assert qualified_workers(args,ctx,'b'*64)==4
    value=dict(status='PASS_R7_CONCURRENT_RESOURCE_QUALIFICATION',contract_sha256='a'*64,qualification_sha256='b'*64,
        observed_real_forward_backward=True,observed_concurrent_workers=4,qualified_max_workers=6,
        largest_two_round_arm_measured=True,resource_measurements={kind+'_'+suffix:amount for kind in ('GPU','RAM')
            for suffix,amount in (('total_bytes',1000),('peak_used_bytes_at_observed_workers',400))})
    value['resource_measurements'].update(per_extra_worker_GPU_bytes_upper_bound=200,per_extra_worker_RAM_bytes_upper_bound=200)
    path=tmp_path/'RESOURCE_QUALIFICATION.json';path.write_text(json.dumps(value))
    assert qualified_workers(args,ctx,'b'*64)==6
    value['resource_measurements']['RAM_peak_used_bytes_at_observed_workers']=600;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='RAM_HEADROOM_INSUFFICIENT'):qualified_workers(args,ctx,'b'*64)


def test_private_norm_cache_shared_across_arm_and_hash_change_rejected(tmp_path,monkeypatch):
    seed_model,prepared,_=fixture('NORM');source=seed_model.core.foundation;bank=seed_model.patch_bank
    membership=torch.zeros(50,9,dtype=torch.bool)
    for k in range(50):membership[k,k%9]=True;membership[k,(k+2)%9]=True
    config=tmp_path/'recipe.json';config.write_text('{}')
    args=SimpleNamespace(private=tmp_path/'private',partition=0,device='cpu',config=config)
    ctx=dict(lock=None,member=membership,raw=dict(rna_symbols=['outside']+list(source.gene_reader.mapping.gene_symbols)),
        sources={'MATCHED':source},source_audits={'MATCHED':dict(complete_source_state_sha256=source_model_state_sha256(source))},
        r6_args=None,r6_ctx={})
    origin=dict(prepared_receipt_sha256='c'*64)
    monkeypatch.setattr(c,'qualified_prepared',lambda *a:(prepared,origin))
    monkeypatch.setattr(c.old,'patch_bank',lambda *a,**k:(bank,'d'*64))
    monkeypatch.setattr(c.old,'pairing',lambda *a:(seed_model.query_map,dict(synthetic_only=True)))
    result=c.create_calibration_cache(args,ctx,prepared,31)
    path=args.private/'calibration/p0_seed31.json';digest=c.sha256(path)
    assert result['origin']['calibration_model']=='untrained_NORM'
    ctx['lock']=dict(calibration_receipt_sha256={'p0_seed31':digest})
    raw,raw_origin=c.construct(args,ctx,prepared,'RAW',31)
    gate,gate_origin=c.construct(args,ctx,prepared,'GATE2_WRONG_RNA',31)
    assert raw_origin['calibration']==gate_origin['calibration']==result['calibration']
    assert c.shared_state_hash(raw)==c.shared_state_hash(gate)
    assert raw_origin['calibration_receipt_sha256']==gate_origin['calibration_receipt_sha256']==digest
    with path.open('ab') as stream:stream.write(b' ')
    with pytest.raises(ValueError,match='LOCKED_CALIBRATION_RECEIPT_CHANGED'):
        c.construct(args,ctx,prepared,'GATE1',31)
