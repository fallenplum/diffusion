"""Independent Phase21 qualification, 90-checkpoint replay and paired metrics.

Only DIRECT is new training. P16/S16 and all preprocessing/anchor artifacts are
read-only qualified Phase20 controls. No solver, teacher or new fit is called.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import validate_phase20 as parent
from scripts.validate_phase19 import load_checkpoint,history_audit
from src.phase21_relative.evaluation import patient_shared_transition_bootstrap,pair_transition_summary

ARMS=['P16','S16','DIRECT']
MODES=[('matched','matched_risk'),('wrongC','wrongC_risk'),('wrongO','wrongO_risk'),
       ('constantC','constantC_risk'),('off','interaction_off_risk'),('anchor','anchor_risk')]
SUFFIX={'wrongC':'WRONG_C','wrongO':'WRONG_O','constantC':'CONSTANT_C','off':'OFF'}
P20_SOURCE='e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548'
PRIMARY={'DIRECT-S16':('S16','DIRECT'),'DIRECT-P16':('P16','DIRECT')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)
def api():
    from src.phase21_relative import data,train
    return data,train
def write(path,value):parent.write(path,value)

def gates(require_done=True):
    data,_=api();identity=data.source();config=data.cfg()
    lock=read(data.R/'PROTOCOL_LOCK.json');qualification=read(data.R/'QUALIFICATION.json')
    require(lock['source']==qualification['source']==identity and qualification['status']=='PASS','new source qualification differs')
    require(lock['training_files']==data.source_files() and lock['configuration']==config,'new protocol/configuration bytes changed')
    require(lock['qualification_sha256']==sha(data.R/'QUALIFICATION.json'),'new qualification receipt changed')
    require(all(sha(ROOT/name)==value for name,value in qualification['test_files'].items()),'qualified test file changed')
    require(data.prior.source()==P20_SOURCE,'frozen Phase20 training/data source changed')
    parent._provenance().require_record(data.prior.R/'PROTOCOL_LOCK.json')
    publication=read(data.prior.R/'ARTIFACT_MANIFEST.json')
    require(publication['status']=='PASS' and publication['source']==P20_SOURCE,'Phase20 publication reference missing')
    require(config['seeds']==[31,47,83] and config['folds']==[1,2,3,4,5],'locked patient/seed matrix changed')
    require(config['bootstrap']['draws']==20000 and config['bootstrap']['primary_family_tests']==4,'bootstrap/family differs')
    require(config['primary_contrasts']=={name:{new:1,old:-1} for name,(old,new) in PRIMARY.items()},'primary contrasts differ')
    if require_done:
        done=read(data.P/'TRAINING_DONE.json')
        require(done['status']=='PASS' and done['source']==identity and done['new_formal_fits']==30 and done['reused_control_models']==60,'formal training incomplete')
    return data,identity,config

def checkpoint(arm,seed,fold,role):
    data,_=api()
    if arm=='DIRECT':
        path=data.P/role/f'{arm}_seed{seed}_fold{fold}.pt'
        ck,receipt=load_checkpoint(path,arm,seed,fold,role,data.source())
    else:
        path=data.prior.P/role/f'{arm}_seed{seed}_fold{fold}.pt'
        ck,receipt=parent.qualified_checkpoint(path,arm,seed,fold,role)
    return ck,receipt,path

def epoch_selections():
    data,_=api();identity=data.source();config=data.cfg()
    lock=read(data.R/'EPOCH_SELECTION_LOCK.json')
    require(lock['source']==identity and lock['status']=='ALL15_INNER_EPOCHS_LOCKED'
            and lock['no_outer_results_used'],'inner epoch lock differs')
    expected={(s,f) for s in config['seeds'] for f in config['folds']}
    rows=lock['selections'];keys=[(q['seed'],q['fold']) for q in rows]
    require(len(keys)==len(set(keys))==15 and set(keys)==expected,'epoch lock must contain all15 unique seed/fold selections')
    return dict(zip(keys,rows))

def qualify_outer_epoch(ck,receipt,selections):
    inner,inner_receipt,path=checkpoint('DIRECT',ck['seed'],ck['fold'],'inner')
    selection=selections[ck['seed'],ck['fold']];actual=sha(path)
    require(selection['selected_epoch']==inner['selected_epoch']==inner_receipt['selected_epoch']==ck['selected_epoch']==receipt['selected_epoch'],
            'outer epoch is not the exact locked inner epoch')
    require(selection['checkpoint_sha256']==inner_receipt['checkpoint_sha256']==actual
            ==ck['inner_checkpoint_sha256']==receipt['inner_checkpoint_sha256'],
            'outer, inner receipt/file and epoch-lock SHA lineage differs')
    return {'seed':ck['seed'],'fold':ck['fold'],'selected_epoch':selection['selected_epoch'],
        'inner_checkpoint_sha256':actual,'inner_receipt_sha256':sha(path.with_suffix('.json')),
        'outer_checkpoint_inner_reference_identical':True,'outer_receipt_inner_reference_identical':True}

def qualify_direct(ck,receipt,ds,config,simple,selections):
    data,_=api()
    require(np.array_equal(ck['fit'],ds.fit) and np.array_equal(ck['validation_or_test'],ds.valid),'DIRECT patient partitions differ')
    require(not np.intersect1d(ds.fit,ds.valid).size and not (ds.role=='inner' and np.intersect1d(ds.active,ds.part['outer_test']).size),'patient leakage')
    parent.assert_equal_structure(ck['preprocess_audit'],ds.audit,'DIRECT read-only Phase20 features')
    history_audit(ck,config)
    require(ck['assigned_mode']=='matched' and np.array_equal(ck['risk'],ck['matched_risk']),'DIRECT primary assignment differs')
    for _,field in MODES:
        require(np.asarray(ck[field]).shape==(len(ds.valid),) and np.isfinite(ck[field]).all(),'invalid saved DIRECT prediction '+field)
    require(np.array_equal(ck['constantC_risk'],ck['interaction_off_risk']),'reference Clinical correction is not exact zero')
    q=simple['models']['J0']
    require(np.array_equal(ck['anchor_w'],q['w']) and np.array_equal(ck['anchor_risk'],q['risk']),'frozen anchor identity differs')
    fit_anchor=np.asarray(ds.f['X'][ds.fit],np.float64)@q['w']
    require(abs(np.std(fit_anchor,ddof=0)-ck['anchor_sd_fit'])<=1e-12,'fit-only anchor scale differs')
    require(np.array_equal(ds.f['ANCHOR'][ds.fit],fit_anchor.astype(np.float32)) and np.array_equal(ds.f['ANCHOR'][ds.valid],q['risk']),'exact anchor override differs')
    require(np.array_equal(ck['state']['main'].cpu().numpy(),np.asarray(q['w'],np.float64)),'frozen main buffer changed')
    for mapping in [ds.wrong_c,ds.wrong_o]:
        for indices in [ds.fit,ds.valid]:
            require(not np.any(mapping[indices]==indices) and set(mapping[indices])==set(indices),'donor crosses partition or contains self')
    if ds.role=='outer':
        qualify_outer_epoch(ck,receipt,selections)

def flatten_aggregate(value,prefix=''):
    result={}
    for key,item in value.items():
        name=prefix+str(key)
        if isinstance(item,dict):result.update(flatten_aggregate(item,name+'.'))
        elif not isinstance(item,(list,tuple)):result[name]=item
    return result

def direct_diagnosis(model,ds,ck,receipt,path,identity,evaluator):
    from scripts.audit_phase21_saved import diagnose_partitions,model_digest,assert_public_aggregate
    from src.phase21_relative.diagnostics import diagnose_checkpoint
    from src.phase21_relative.operators import direct_update
    data,_=api();before=model_digest(model);begin=time.monotonic()
    for parameter in model.parameters():parameter.requires_grad_(False)
    diagnostic=lambda m,d,chunk_size:diagnose_checkpoint(m,d,chunk_size=chunk_size,update_operator=direct_update)
    partitions=diagnose_partitions(model,ds,diagnostic)
    require(before==model_digest(model) and all(q.grad is None for q in model.parameters()),'DIRECT diagnosis mutated model or created gradients')
    for partition in ['fit','heldout']:
        q=partitions[partition];q['registered_parent_arm']=q['arm'];q['arm']='DIRECT'
    record={'status':'PASS','source':identity,'evaluator_sha256':evaluator,'actual_fit_source':ck['source'],
        'arm':'DIRECT','seed':ck['seed'],'fold':ck['fold'],'role':ck['role'],
        'checkpoint_sha256':receipt['checkpoint_sha256'],'receipt_sha256':sha(path.with_suffix('.json')),
        'update_operator':'direct_update','selected_epoch':ck['selected_epoch'],'diagnostics':partitions,
        'model_values_unchanged':True,'model_gradients_created':False,'new_training_fits':0,'optimizer_calls':0,
        'sampling_uses_DSS_outcomes':False,'existing_Data_loads_protected_endpoints':True,
        'diagnostic_routine_reads_endpoints':False,'raw_risks_indices_maps_public':False,
        'seconds':time.monotonic()-begin}
    assert_public_aggregate(record)
    target=data.R/'direct_checkpoints'/f'DIRECT_{ck["role"]}_seed{ck["seed"]}_fold{ck["fold"]}.json'
    write(target,record)
    return {'direct_diagnostic_sha256':sha(target),
        'diagnostic_model_forward_calls':partitions['additional_model_forward_calls'],
        'diagnostic_operator_calls':partitions['additional_operator_calls']}

def replay(shard=0,shards=1):
    import pandas as pd
    import torch
    data,identity,config=gates();_,train=api();_,oldtrain=parent._api()
    require(0<=shard<shards,'invalid replay shard')
    evaluator=sha(Path(__file__));rows=[];features=[];seen=set();selections=epoch_selections()
    jobs=[(arm,seed,fold,role) for arm in ARMS for role in ['inner','outer'] for seed in config['seeds'] for fold in config['folds']]
    for arm,seed,fold,role in jobs[shard::shards]:
        ck,receipt,path=checkpoint(arm,seed,fold,role)
        train.seed_all(seed+100*fold);ds=data.Data(fold,role,seed)
        simple,_=parent.inherited_simple(ds)
        if arm=='DIRECT':qualify_direct(ck,receipt,ds,config,simple,selections);factory,forward=train.make_model,train.forward
        else:parent.qualify(ck,ds,data.prior.cfg(),simple);factory,forward=oldtrain.make_model,oldtrain.forward
        if (role,fold) not in seen:
            features.append(parent.independent_feature_audit(ds));seen.add((role,fold))
        model=factory(ds,arm,ck['anchor_w']);model.load_state_dict(ck['state'],strict=True);model.eval()
        require('main' not in dict(model.named_parameters()),'main became trainable')
        layout=[(name,list(value.shape)) for name,value in model.named_parameters()]
        errors={}
        with torch.no_grad():
            for mode,field in MODES:
                actual=forward(model,ds,ds.valid,mode=mode,training=False).cpu().numpy()
                errors[mode]=float(abs(actual-ck[field]).max())
                require(np.array_equal(actual,ck[field]),'prediction replay is not elementwise identical: '+str((arm,seed,fold,role,mode,errors[mode])))
        pair=float(abs(ck['interaction_off_risk']-ck['anchor_risk']).max())
        correction=max(float(abs(ck[field]-ck['interaction_off_risk']).max()) for mode,field in MODES if mode!='anchor')
        require(pair<=ck['anchor_sd_fit']+5e-6 and correction<=.25*ck['anchor_sd_fit']+5e-6,'bounded final risk exceeded anchor-scale cap')
        if arm=='P16':require(all(np.array_equal(ck[field],ck['interaction_off_risk']) for mode,field in MODES if mode!='anchor'),'P16 has a correction-context effect')
        diagnostic=direct_diagnosis(model,ds,ck,receipt,path,identity,evaluator) if arm=='DIRECT' else {}
        rows.append({'arm':arm,'seed':seed,'fold':fold,'role':role,'source':identity,'actual_fit_source':ck['source'],
            'fit_classification':'new_DIRECT_fit' if arm=='DIRECT' else 'qualified_reused_Phase20_control',
            'checkpoint_sha256':receipt['checkpoint_sha256'],'receipt_sha256':sha(path.with_suffix('.json')),
            'maximum_replay_error':max(errors.values()),'anchor_elementwise_identical':True,'prediction_vectors_replayed':6,
            'selected_epoch':ck['selected_epoch'],'registered_parameters':sum(q.numel() for q in model.parameters()),
            'parameter_layout_sha256':hashlib.sha256(json.dumps(layout).encode()).hexdigest(),**diagnostic})
        print('PHASE21_REPLAY',arm,seed,fold,role,max(errors.values()),flush=True)
        del ds,model;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(data.R/f'CHECKPOINT_REPLAY_{shard}.csv',index=False)
    pd.DataFrame(features).to_csv(data.R/f'FEATURE_RECONSTRUCTION_{shard}.csv',index=False)
    require(data.source()==identity and sha(Path(__file__))==evaluator,'source changed during replay')
    write(data.P/f'REPLAY_SHARD_{shard}.json',{'status':'PASS','source':identity,'evaluator_sha256':evaluator,'rows':len(rows),'shards':shards})

def collect():
    import pandas as pd
    import torch
    data,identity,config=gates();evaluator=sha(Path(__file__))
    paths=sorted(data.P.glob('REPLAY_SHARD_*.json'));require(paths,'no qualified prediction replay shards')
    markers=[read(q) for q in paths];shards=markers[0]['shards']
    require(len(markers)==shards and {q.name for q in paths}=={f'REPLAY_SHARD_{i}.json' for i in range(shards)},'replay shard set incomplete')
    rows=[]
    for i,marker in enumerate(markers):
        require(marker['status']=='PASS' and marker['source']==identity and marker['evaluator_sha256']==evaluator,'replay/source identity differs')
        current=pd.read_csv(data.R/f'CHECKPOINT_REPLAY_{i}.csv').to_dict('records')
        require(len(current)==marker['rows'],'replay row count differs');rows+=current
    expected={(a,s,f,r) for a in ARMS for s in config['seeds'] for f in config['folds'] for r in ['inner','outer']}
    keys=[(q['arm'],q['seed'],q['fold'],q['role']) for q in rows]
    require(len(keys)==90 and set(keys)==expected and sum(q['prediction_vectors_replayed'] for q in rows)==540,'90-checkpoint/540-vector replay incomplete')
    require(all(q['source']==identity and q['maximum_replay_error']==0 and q['anchor_elementwise_identical'] for q in rows),'replay is not exact')
    frame=pd.DataFrame(rows)
    require(all(group.parameter_layout_sha256.nunique()==1 and group.registered_parameters.nunique()==1
                for _,group in frame.groupby(['seed','fold','role'])),'registered parameter budget/layout differs across controls')
    upstream,_=parent._upstream();cohort=upstream.parent.load();endpoint=upstream.parent.endpoints(cohort)['DSS']
    require(len(cohort['patient_ids'])==512 and int(endpoint['event'].sum())==65,'DSS cohort changed')
    banks={};anchors={};curves=[];manifest=[];summaries=[];risks={};epochs={};epoch_rows=[];direct_diagnostics=[]
    selections=epoch_selections()
    for role in ['inner','outer']:
        for fold in config['folds']:
            path=upstream.P/'preprocessing'/f'fold{fold}_{role}.pt';bank=torch.load(path,map_location='cpu',weights_only=False)
            receipt=read(path.with_suffix('.json'));require(sha(path)==receipt['sha256'],'read-only feature bank changed')
            parts=upstream.parent.splits(cohort,'DSS',fold)
            fit,valid=parts['inner_train' if role=='inner' else 'outer_train'],parts['inner_val' if role=='inner' else 'outer_test']
            require(np.array_equal(bank['fit'],fit) and np.array_equal(bank['valid'],valid),'read-only feature partition differs')
            ds=SimpleNamespace(f=bank['features'],fit=fit,valid=valid,t=endpoint['time'],e=endpoint['event'],role=role,fold=fold)
            simple,_=parent.inherited_simple(ds);banks[role,fold]=bank;anchors[role,fold]=simple
    lookup=dict(zip(keys,rows))
    for arm in ARMS:
        for role in ['inner','outer']:
            for seed in config['seeds']:
                for fold in config['folds']:
                    ck,receipt,path=checkpoint(arm,seed,fold,role);q=lookup[arm,seed,fold,role];bank=banks[role,fold]
                    require(q['checkpoint_sha256']==sha(path) and q['receipt_sha256']==sha(path.with_suffix('.json')) and q['actual_fit_source']==ck['source'],'checkpoint changed after replay')
                    require(np.array_equal(ck['fit'],bank['fit']) and np.array_equal(ck['validation_or_test'],bank['valid']),'saved patient partition differs')
                    require(np.array_equal(ck['anchor_risk'],anchors[role,fold]['models']['J0']['risk']),'shared exact anchor changed')
                    history_audit(ck,config if arm=='DIRECT' else data.prior.cfg())
                    if arm=='DIRECT':
                        if role=='outer':epoch_rows.append(qualify_outer_epoch(ck,receipt,selections))
                        target=data.R/'direct_checkpoints'/f'DIRECT_{role}_seed{seed}_fold{fold}.json';diagnostic=read(target)
                        require(q['direct_diagnostic_sha256']==sha(target) and diagnostic['status']=='PASS'
                            and diagnostic['source']==identity and diagnostic['evaluator_sha256']==evaluator
                            and diagnostic['actual_fit_source']==ck['source'] and diagnostic['arm']=='DIRECT'
                            and diagnostic['seed']==seed and diagnostic['fold']==fold and diagnostic['role']==role
                            and diagnostic['checkpoint_sha256']==q['checkpoint_sha256'] and diagnostic['receipt_sha256']==q['receipt_sha256'],
                            'DIRECT independent diagnostic identity differs')
                        part=diagnostic['diagnostics']
                        require(part['optimizer_steps']==part['additional_gradient_calls']==diagnostic['new_training_fits']==diagnostic['optimizer_calls']==0
                            and diagnostic['model_values_unchanged'] and not diagnostic['model_gradients_created'],
                            'DIRECT diagnostic is not read-only')
                        require(q['diagnostic_model_forward_calls']==part['additional_model_forward_calls']
                            and q['diagnostic_operator_calls']==part['additional_operator_calls'],'DIRECT diagnostic call ledger differs')
                        direct_diagnostics.append(diagnostic)
                    if role=='inner':epochs[arm,seed,fold]=ck['selected_epoch']
                    else:require(ck['selected_epoch']==epochs[arm,seed,fold],'outer selected epoch differs')
                    meta={k:q[k] for k in ['arm','seed','fold','role','actual_fit_source','fit_classification']}
                    manifest.append({**meta,'source':identity,'checkpoint_sha256':q['checkpoint_sha256'],
                        'selected_epoch':ck['selected_epoch'],'actual_epochs':len(ck['history'])})
                    summaries.append({**meta,**{k:v for k,v in ck['summary'].items() if isinstance(v,(int,float,bool,str,np.generic))}})
                    curves.extend({**{k:meta[k] for k in ['arm','seed','fold','role']},**r} for r in ck['history'])
                    if role=='outer':
                        for mode,field in MODES:
                            if mode=='anchor':continue
                            label=arm if mode=='matched' else arm+'__'+SUFFIX[mode]
                            vector=risks.setdefault((label,seed),np.full(512,np.nan));ix=bank['valid']
                            require(not np.isfinite(vector[ix]).any(),'duplicate OOF patient prediction');vector[ix]=ck[field]
    for seed in config['seeds']:
        vector=np.full(512,np.nan)
        for fold in config['folds']:vector[banks['outer',fold]['valid']]=anchors['outer',fold]['models']['J0']['risk']
        risks['J0',seed]=vector
    require(len(risks)==48 and all(q.shape==(512,) and np.isfinite(q).all() for q in risks.values()),'48 complete OOF vectors required')
    require(len(epoch_rows)==15 and len(direct_diagnostics)==30,'epoch lineage or DIRECT diagnosis incomplete')
    write(data.R/'EPOCH_LOCK_AUDIT.json',{'status':'PASS','source':identity,'rows':epoch_rows,
        'epoch_selection_lock_sha256':sha(data.R/'EPOCH_SELECTION_LOCK.json'),'unique_inner_selections':15,
        'outer_epoch_and_inner_file_receipt_lock_references_identical':True,'no_outer_results_used':True})
    pd.DataFrame([flatten_aggregate(q) for q in direct_diagnostics]).to_csv(data.R/'DIRECT_DIAGNOSTICS.csv',index=False)
    write(data.R/'DIRECT_DIAGNOSTIC_AUDIT.json',{'status':'PASS','source':identity,'evaluator_sha256':evaluator,
        'checkpoints':30,'partition_views':60,'fit_views':30,'heldout_views':30,
        'diagnostic_model_forward_calls':sum(q['diagnostics']['additional_model_forward_calls'] for q in direct_diagnostics),
        'diagnostic_operator_calls':sum(q['diagnostics']['additional_operator_calls'] for q in direct_diagnostics),
        'additional_gradient_calls':0,'new_training_fits':0,'optimizer_calls':0,
        'model_values_unchanged':True,'sampling_uses_DSS_outcomes':False,'diagnostic_routine_reads_endpoints':False,
        'raw_risks_maps_indices_public':False,'diagnostic_CSV_sha256':sha(data.R/'DIRECT_DIAGNOSTICS.csv')})
    for name,values in [('CHECKPOINT_REPLAY',rows),('CELL_MANIFEST',manifest),('TRAINING_CURVES',curves),('COMPUTE_PROFILE',summaries)]:
        pd.DataFrame(values).to_csv(data.R/(name+'.csv'),index=False)
    pd.DataFrame([q for i in range(shards) for q in pd.read_csv(data.R/f'FEATURE_RECONSTRUCTION_{i}.csv').to_dict('records')]).to_csv(data.R/'FEATURE_RECONSTRUCTION.csv',index=False)
    np.savez_compressed(data.P/'OOF_PRIVATE.npz',**{name+'__seed'+str(seed):value for (name,seed),value in risks.items()})
    write(data.R/'CHECKPOINT_REPLAY.json',{'status':'PASS','source':identity,'evaluator_sha256':evaluator,'checkpoints':90,
        'new_DIRECT_checkpoints':30,'qualified_reused_Phase20_controls':60,'prediction_vectors':540,'maximum_replay_error':0.,
        'all_modes_elementwise_identical':True,'anchor_elementwise_identical':True,'registered_parameter_layouts_matched':True})
    direct=pd.DataFrame(summaries);inner=direct[(direct.arm=='DIRECT')&(direct.role=='inner')]
    fraction=float(inner.hit_maximum.mean())
    write(data.R/'BUDGET_VERIFICATION.json',{'status':'PASS','source':identity,'new_inner_cells':15,'new_outer_cells':15,
        'DIRECT_hit_maximum_fraction':fraction,'maximum_cap_fraction':.2,'budget_gate_pass':fraction<=.2,
        'selected_epoch_median':float(inner.selected_epoch.median()),'actual_epoch_median':float(inner.actual_epochs.median()),
        'controls':'original independent Phase20 inner selections; not reselected or refitted','global_convergence_proved':False})
    counts=pd.DataFrame(manifest).groupby(['fit_classification','actual_fit_source']).size().reset_index(name='models')
    write(data.R/'MODEL_SOURCE_AUDIT.json',{'status':'PASS','source':identity,'rows':counts.to_dict('records'),
        'new_neural_fits':30,'reused_neural_models':60,'new_convex_or_feature_fits':0,
        'phase20_source':P20_SOURCE,'phase20_publication_manifest_sha256':sha(data.prior.R/'ARTIFACT_MANIFEST.json')})
    return cohort,risks

def statistics(cohort,risks,endpoint_data,seed_rows,config):
    import pandas as pd
    data,_=api();t,e,valid,ipcw=endpoint_data
    definitions={**PRIMARY,'DIRECT-J0':('J0','DIRECT'),**{arm+'-OWN_OFF':(arm+'__OFF',arm) for arm in ARMS},
        **{arm+'-'+arm+'__'+tag:(arm+'__'+tag,arm) for arm in ['DIRECT','S16'] for tag in ['WRONG_C','WRONG_O']}}
    require(len(definitions)==10,'ten fixed diagnostic and primary pair comparisons required')
    comparisons={name:{s:(risks[old,s],risks[new,s]) for s in config['seeds']} for name,(old,new) in definitions.items()}
    started=time.monotonic()
    result=patient_shared_transition_bootstrap(t,e,comparisons,folds=cohort['folds'],valid=valid,ipcw=ipcw,
        draws=config['bootstrap']['draws'],seed=config['bootstrap']['seed'],family_tests=4)
    byseed=pd.DataFrame(seed_rows).set_index(['arm','seed']);points=pd.DataFrame(result['point_rows'])
    intervals=pd.DataFrame(result['interval_rows']);pairs=[];identities=[]
    for q in points.to_dict('records'):
        old,new=definitions[q['comparison']];metric=q['metric']
        expected=np.array([byseed.loc[new,s][metric]-byseed.loc[old,s][metric] for s in config['seeds']])
        error=abs(expected.mean()-q['net']);require(error<1e-12,'pair-credit net differs from complete seed metric contrast')
        identities.append({'comparison':q['comparison'],'metric':metric,'net_delta':q['net'],'complete_seed_C_difference':float(expected.mean()),'error':error})
        ci=intervals[(intervals.comparison==q['comparison'])&(intervals.metric==metric)&(intervals.statistic=='net')].iloc[0]
        primary=q['comparison'] in PRIMARY
        pairs.append({'comparison':q['comparison'],'metric':metric,'family':'primary' if primary else 'diagnostic',
            'delta':q['net'],'CI95_low':ci.CI_low,'CI95_high':ci.CI_high,
            'Bonferroni_low':ci.corrected_low if primary else None,'Bonferroni_high':ci.corrected_high if primary else None,
            'family_tests':4 if primary else None,'positive_seeds':int((expected>0).sum()),'seeds':len(expected)})
    # Conditional credit-capacity rates are point diagnostics, not intervals
    # reconstructed incorrectly from marginal repair/damage confidence bounds.
    conditional=[]
    for q in result['point_rows']:
        conditional.append({**q,'repair_over_baseline_credit_deficit':q['repair']/(1-q['C0']) if q['C0']<1 else None,
            'damage_over_baseline_available_credit':q['damage']/q['C0'] if q['C0']>0 else None})
    pd.DataFrame(pairs).to_csv(data.R/'PAIRED_COMPARISONS.csv',index=False)
    pd.DataFrame(conditional).to_csv(data.R/'REPAIR_DAMAGE.csv',index=False)
    pd.DataFrame(result['seed_rows']).to_csv(data.R/'SEED_REPAIR_DAMAGE.csv',index=False)
    primary_mask=intervals.comparison.isin(PRIMARY)&(intervals.statistic=='net')
    intervals.loc[~primary_mask,['family_tests','corrected_low','corrected_high']]=np.nan
    intervals.to_csv(data.R/'PAIR_TRANSITION_INTERVALS.csv',index=False)
    write(data.R/'PAIR_CREDIT_IDENTITY_AUDIT.json',{'status':'PASS','source':data.source(),'rows':identities,
        'risk_tie_tolerance':1e-8,'repair_damage_include_tie_half_credit':True,
        'denominator':'all comparable pair weight; conditional point rates are ratios of mean-seed credit masses, not means of seed-specific ratios or strict-wrong/correct-only rates',
        'conditional_rate_intervals_provided':False,'pair_comparisons':10,
        'training_used_outer_error_groups':False,'no_patient_pair_details_public':True})
    write(data.R/'BOOTSTRAP_AUDIT.json',{'status':'PASS','source':data.source(),**result['audit'],
        'primary_family_tests':4,'primary_comparisons':list(PRIMARY),'seconds':time.monotonic()-started})

def evaluate():
    import pandas as pd
    data,identity,config=gates();evaluator=sha(Path(__file__));began=time.monotonic()
    cohort,risks=collect()
    t,e,valid,ipcw,seeds=parent.metrics(cohort,risks,'DSS',data.R)
    statistics(cohort,risks,(t,e,valid,ipcw),seeds,config)
    parent.metrics(cohort,risks,'PFI',data.R)
    # Ancestor metrics intentionally retain their original MultiIndex export.
    # Normalize only these new Phase21 presentation tables to unique columns.
    for name in ['SUMMARY.csv','PFI_SUMMARY.csv']:
        path=data.R/name
        summary=pd.read_csv(path,header=[0,1],index_col=0)
        require(list(summary.columns)==[('Harrell','mean'),('Harrell','std'),('Uno60','mean'),('Uno60','std')],
                'unexpected inherited summary table schema')
        summary.columns=['Harrell_mean','Harrell_std','Uno60_mean','Uno60_std']
        summary.rename_axis('arm').reset_index().to_csv(path,index=False)
    require(identity==data.source() and evaluator==sha(Path(__file__)),'source changed during evaluation')
    write(data.R/'INDEPENDENT_VALIDATION.json',{'status':'PASS','source':identity,'evaluator_sha256':evaluator,
        'new_neural_fits':30,'qualified_reused_controls':60,'qualified_checkpoints':90,'prediction_replays':540,
        'complete_model_seed_OOF_vectors':48,'DSS_fold_rows':240,'PFI_fold_rows':240,'bootstrap_draws':20000,
        'primary_family_tests':4,'pair_comparisons':10,'independent_sksurv_every_fold':True,'repair_damage_net_identity':True,
        'outer_epoch_and_inner_file_receipt_lock_references_identical':True,'DIRECT_diagnostic_checkpoints':30,'DIRECT_diagnostic_partition_views':60,
        'PFI':'frozen DSS model risk transfer; no PFI training, selection or primary testing',
        'within_checkpoint_baseline':'same checkpoint correction off; joint-fitted backbone retained',
        'between_model_baseline':'independent whole-pipeline fits, including backbone differences',
        'no_new_feature_convex_or_teacher_fits':True,'seconds':time.monotonic()-began})
    write(data.P/'EVALUATION_DONE.json',{'status':'PASS','source':identity,'evaluator_sha256':evaluator,
        'checkpoints':90,'prediction_replays':540,'bootstrap_draws':20000,'OOF_vectors':48,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print('PHASE21_INDEPENDENT_EVALUATION_PASS',90,540,20000,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();replay(args.shard,args.shards) if args.replay else evaluate()
