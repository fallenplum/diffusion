"""Independent Phase20 replay and patient-paired factorial evaluation.

Old feature/convex artifacts are read-only, qualified references. No fitting is
performed here. All current neural predictions must be fresh outer predictions.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_phase18a import (metrics, resampled_c_indices, linear_c_contrast,
    components, assert_equal_structure)
from scripts.validate_phase19 import (independent_feature_audit, history_audit,
    load_checkpoint, flatten_diagnostics, check_model_seed_coverage, point_tensor)

ARMS = ['L0','P1','P16','S1','S16','D16','F16','WC16','WO16']
DID_NAME='DID=(S16-P16)-(S1-P1)'
PRIMARY = {
    'P16-P1': {'P16':1.,'P1':-1.},
    **{'S16-'+arm:{'S16':1.,arm:-1.} for arm in ['P16','D16','F16','WC16','WO16']},
    DID_NAME: {'S16':1.,'P16':-1.,'S1':-1.,'P1':1.},
}
PRIMARY_FAMILY_TESTS = 14
MODES = [('matched','matched_risk'),('wrongC','wrongC_risk'),('wrongO','wrongO_risk'),
         ('constantC','constantC_risk'),('off','interaction_off_risk')]
REPRESENTATION_MODES=[('fullOmicsWrong','full_omics_wrong_risk'),('uniformPair','uniform_pair_risk')]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    def cast(item):
        if isinstance(item,np.ndarray): return item.tolist()
        if isinstance(item,np.generic): return item.item()
        raise TypeError(type(item).__name__)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=cast,allow_nan=False)+'\n')
    temporary.replace(path)

def _api():
    from src.phase20_selective import data, train
    return data,train

def _provenance():
    from src.phase20_selective import provenance
    return provenance

def qualified_checkpoint(path,arm,seed,fold,role):
    """A legacy source is eligible only through its exact record and PT hashes."""
    path=Path(path)
    record=_provenance().require_record(path.with_suffix('.json'),checkpoint_path=path)
    ck,receipt=load_checkpoint(path,arm,seed,fold,role,record['source'])
    return ck,receipt

def _upstream():
    from src.phase19_attention import data
    identity=data.source()
    if json.loads((data.R/'PROTOCOL_LOCK.json').read_text())['source']!=identity:
        raise ValueError('read-only upstream source changed')
    return data,identity

def update_coverage(coverage,ck,owner):
    item=coverage.setdefault((ck['arm'],ck['role']),{'checkpoints':0,'risk_patients':set(),'risk_WSIs':set(),
        'gradient_patients':set(),'gradient_WSIs':set(),'risk_counts':[],'risk_WSI_counts':[],
        'gradient_counts':[],'gradient_WSI_counts':[],
        'isolated_input_gradient_available':ck['mechanism']['isolated_correction_input_gradient_available']})
    item['checkpoints']+=1
    expected={'risk_attention':np.asarray(ck['fit']),
        'isolated_gradient':np.asarray(ck['fit'])[np.unique(np.linspace(0,len(ck['fit'])-1,min(8,len(ck['fit']))).astype(int))]}
    for label,prefix in [('risk_attention','risk'),('isolated_gradient','gradient')]:
        patients=np.asarray(ck['probe_indices'][label])
        if not np.array_equal(patients,expected[label]): raise ValueError('probe sampling differs from locked fit-only rule')
        slides=np.flatnonzero(np.isin(owner,patients))
        item[prefix+'_patients'].update(patients.tolist());item[prefix+'_WSIs'].update(slides.tolist())
        item[prefix+'_counts'].append(len(patients));item[prefix+'_WSI_counts'].append(len(slides))

def coverage_rows(coverage):
    result=[]
    for (arm,role),item in sorted(coverage.items()):
        q={'arm':arm,'role':role,'checkpoint_count':item['checkpoints'],
            'isolated_input_gradient_available':item['isolated_input_gradient_available']}
        for prefix,label in [('risk','risk_attention_probe'),('gradient','isolated_input_gradient_probe')]:
            q[label+'_unique_query_patients']=len(item[prefix+'_patients']);q[label+'_unique_query_WSIs']=len(item[prefix+'_WSIs'])
            q[label+'_per_checkpoint_patients_min']=min(item[prefix+'_counts']);q[label+'_per_checkpoint_patients_max']=max(item[prefix+'_counts'])
            q[label+'_per_checkpoint_WSIs_min']=min(item[prefix+'_WSI_counts']);q[label+'_per_checkpoint_WSIs_max']=max(item[prefix+'_WSI_counts'])
        result.append(q)
    return result

def selectivity_behavior_audit(mechanisms,identity):
    selected=[q for q in mechanisms if q['arm']=='S16' and q['role']=='inner']
    if len(selected)!=15 or len({(q['seed'],q['fold']) for q in selected})!=15:
        raise ValueError('15 selected-inner S16 behavior probes required')
    rows=[];directions={}
    for q in selected:
        row={key:q[key] for key in ['arm','role','seed','fold','selected_epoch']}
        for direction in ['po','op']:
            for suffix in ['support_patient_mean','all_rows_support_patient_mean','eligible_lt2_patient_mean',
                           'raw_score_degenerate_patient_mean','eligible_ge2_query_count',
                           'support_query_count','support_WSI_count','support_patient_count']:
                key=direction+'_selectivity_'+suffix
                value=q[key]
                if not np.isfinite(value): raise ValueError('nonfinite behavior diagnostic '+key)
                if suffix.endswith('_count') and value<=0: raise ValueError('no eligible choice rows/WSIs/patients: '+key)
                if not suffix.endswith('_count') and not 0<=value<=1: raise ValueError('invalid behavior fraction '+key)
                row[key]=value
        rows.append(row)
    for direction in ['po','op']:
        median=float(np.median([q[direction+'_selectivity_support_patient_mean'] for q in rows]))
        directions[direction]={'median_selected_inner_checkpoint_patient_mean_support':median,
            'strict_maximum':.75,'behavior_gate_pass':median<.75}
    return {'status':'PASS','source':identity,'rows':rows,'directions':directions,
        'behavior_gate_pass':all(q['behavior_gate_pass'] for q in directions.values()),
        'threshold':.75,'nonzero_threshold':1e-6,'eligible_candidates_minimum':2,
        'sampling':'15 selected inner S16 checkpoints; all fit, label-blind probes; no outer data in threshold/decision',
        'aggregation':'query mean within WSI, equal WSI mean within patient, equal fit-patient mean, median across checkpoints',
        'interpretation':'operational sparse-selection behavior only, not clinical validity or biological causality; degenerate/all-row support reported separately'}

def definitions(risks,seeds):
    labels=check_model_seed_coverage(risks,seeds)
    required={arm for coefficients in PRIMARY.values() for arm in coefficients}
    if not required<=set(labels): raise ValueError('factorial primary matrix incomplete')
    result=[(name,coefficient,'primary') for name,coefficient in PRIMARY.items()]
    for arm in ['S1','L0','J0','J_CS',
                'S16__WRONG_C','S16__WRONG_O','S16__CONSTANT_C','S16__CORRECTION_OFF']:
        if arm in labels: result.append(('S16-'+arm,{'S16':1.,arm:-1.},'secondary'))
    extra={arm+'-'+arm+'__'+suffix:{arm:1.,arm+'__'+suffix:-1.}
           for arm in ['P1','P16'] for suffix in ['FULL_OMICS_WRONG','UNIFORM_PAIR']}
    extra['DID_uniform']={'P16':1.,'P1':-1.,'P16__UNIFORM_PAIR':-1.,'P1__UNIFORM_PAIR':1.}
    for name,coefficients in extra.items():
        if set(coefficients)<=set(labels): result.append((name,coefficients,'exploratory_representation'))
    return result

def bootstrap(d,risks,endpoint_data,seed_rows,config,report_root):
    import pandas as pd
    time,event,valid,ipcw=endpoint_data;bc=config['bootstrap'];seeds=config['seeds']
    if bc['draws']!=20000 or bc['family_tests']!=14: raise ValueError('locked bootstrap/family14 differs')
    if config['primary_contrast_coefficients']!=PRIMARY: raise ValueError('prelocked factorial contrast coefficients differ')
    pairs=definitions(risks,seeds);count_audit=[]
    if {name for name,_,family in pairs if family=='secondary'}!=set(config['secondary_comparisons']):
        raise ValueError('locked secondary comparison family differs')
    extra={name:coefficient for name,coefficient,family in pairs if family=='exploratory_representation'}
    if extra!=config['exploratory_representation_contrast_coefficients']:
        raise ValueError('locked representation exploration coefficients differ')
    keys,values,den,good=resampled_c_indices(time,event,valid,d['folds'],risks,ipcw,
        draws=bc['draws'],seed=bc['seed'],count_audit=count_audit)
    if int(good.sum())<.99*bc['draws']: raise ValueError('too many zero-denominator draws')
    points=point_tensor(seed_rows,keys);index={key:i for i,key in enumerate(keys)}
    folds=pd.read_csv(Path(report_root)/'FOLD_METRICS.csv').set_index(['arm','seed','fold'])
    rows=[];seed_deltas=[];audits=[]
    family_tests={family:2*sum(tag==family for _,_,tag in pairs) for family in {row[2] for row in pairs}}
    if family_tests.get('secondary')!=bc['secondary_family_tests']: raise ValueError('secondary multiplicity differs')
    if family_tests.get('exploratory_representation')!=bc['exploratory_representation_family_tests']:
        raise ValueError('representation exploration multiplicity differs')
    for name,coefficients,family in pairs:
        draws=linear_c_contrast(keys,values,coefficients,seeds)
        point=linear_c_contrast(keys,points,coefficients,seeds)[0]
        tests=14 if family=='primary' else family_tests[family]
        for m,metric in enumerate(['Harrell','Uno60']):
            per_seed=[sum(coefficient*points[0,index[arm,seed],m] for arm,coefficient in coefficients.items()) for seed in seeds]
            if abs(np.mean(per_seed)-point[m])>1e-12: raise ValueError('point/bootstrap statistical object differs')
            cell=sum(coefficient*folds.loc[arm,metric].sort_index() for arm,coefficient in coefficients.items())
            low,high=np.quantile(draws[:,m],[.025,.975])
            corrected_low,corrected_high=np.quantile(draws[:,m],[.025/tests,1-.025/tests])
            rows.append({'comparison':name,'metric':metric,'family':family,'delta':point[m],
                'CI95_low':low,'CI95_high':high,'Bonferroni_low':corrected_low,'Bonferroni_high':corrected_high,
                'family_tests':tests,'positive_seeds':int((np.asarray(per_seed)>0).sum()),'seeds':len(seeds),
                'positive_fold_cells':int((cell>0).sum()),'fold_cells':len(cell),
                'draws_requested':bc['draws'],'draws_valid':int(good.sum()),
                'statistical_object':'mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw'})
            seed_deltas.extend({'comparison':name,'metric':metric,'family':family,'seed':seed,'delta':delta}
                               for seed,delta in zip(seeds,per_seed))
            audits.append({'comparison':name,'metric':metric,'point_from_complete_seed_C':float(np.mean(per_seed)),
                'reported_point':point[m],'same_statistic_in_each_bootstrap_draw':True,'coefficients':coefficients})
    pd.DataFrame(rows).to_csv(Path(report_root)/'PAIRED_COMPARISONS.csv',index=False)
    pd.DataFrame(seed_deltas).to_csv(Path(report_root)/'SEED_PAIRED_CONTRASTS.csv',index=False)
    identity=_api()[0].source()
    write(Path(report_root)/'CONTRAST_POINT_AUDIT.json',{'status':'PASS','source':identity,'primary_family_tests':14,'rows':audits,
        'DID':'(C_S16-C_P16)-(C_S1-C_P1), complete four-C contrast in point and every shared-patient resample'})
    write(Path(report_root)/'BOOTSTRAP_AUDIT.json',{'status':'PASS','source':identity,'draws':bc['draws'],'valid':int(good.sum()),
        'zero_denominator_draws':int((~good).sum()),'primary_family_tests':14,'shared_patient_count_matrices':count_audit,
        'minimum_H_denominator':float(den[:,0].min()),'minimum_U_denominator':float(den[:,1].min()),
        'sampling':'one within-fold patient multinomial count matrix shared all models and algorithm seeds',
        'pair_weight':'count_i*count_j','IPCW':'frozen corresponding outer-train reverse KM, tau60',
        'risk_ensembling':False,'uncertainty_scope':'fixed fitted models; excludes refitting and repeated researcher selection',
        'previous_neural_models_counted_as_current_OOF':False})

def inherited_simple(ds):
    """Read and independently qualify ancestor artifacts; never call a solver."""
    import torch
    from scripts.validate_phase17_independent import anchor_gradient
    upstream,identity=_upstream();path=upstream.P/'simple'/f'{ds.role}_fold{ds.fold}.pt'
    receipt=json.loads(path.with_suffix('.json').read_text());saved=torch.load(path,map_location='cpu',weights_only=False)
    if receipt['status']!='PASS' or receipt['source']!=identity or receipt['checkpoint_sha256']!=sha(path) or saved['source']!=identity:
        raise ValueError('inherited convex source/hash mismatch')
    if (saved['role']!=ds.role or saved['fold']!=ds.fold or
        not np.array_equal(saved['fit'],ds.fit) or not np.array_equal(saved['valid'],ds.valid)):
        raise ValueError('inherited convex partition differs')
    if hasattr(ds,'audit'):
        if ds.audit['upstream_source']!=identity: raise ValueError('new data cites wrong upstream source')
        original=ds.audit['upstream_audit']
        normalized=json.loads(json.dumps({k:v for k,v in original.items() if k!='wrong_controls'},
            default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),sort_keys=True))
        assert_equal_structure(normalized,saved['preprocess_audit'],'inherited_simple_features')
    checks=[]
    for arm,key in [('J0','X'),('J_CS','CS')]:
        q=saved['models'][arm];actual=np.asarray(ds.f[key][ds.valid]@q['w'],np.float32)
        error=float(abs(actual-q['risk']).max())
        gradient=float(abs(anchor_gradient(ds.f[key][ds.fit].astype(float),ds.t[ds.fit],ds.e[ds.fit],q['w'],q['lambda'])).max())
        if error>=1e-6 or gradient>=2e-5 or q['lambda']!=10.: raise ValueError('inherited convex numerical audit failed')
        checks.append({'arm':arm,'role':ds.role,'fold':ds.fold,'checkpoint_sha256':sha(path),
            'upstream_source':identity,'risk_reconstruction_max_error':error,'independent_penalized_Efron_stationarity':gradient,
            'lambda':q['lambda'],'new_convex_fit':False})
    return saved,checks

def historical_reference_identity(cached,stored,expected_indices,stored_indices,artifact_sha,receipt_sha):
    """Use the published Phase19 historical replay rule, never current-risk rules."""
    if artifact_sha!=receipt_sha: raise ValueError('historical LOCAL artifact hash mismatch')
    if not np.array_equal(expected_indices,stored_indices): raise ValueError('historical LOCAL patient fold differs')
    cached=np.asarray(cached);stored=np.asarray(stored)
    if cached.shape!=stored.shape or cached.shape!=(len(expected_indices),): raise ValueError('historical LOCAL shape differs')
    if cached.dtype!=np.dtype('float64') or stored.dtype!=np.dtype('float64'):
        raise ValueError('historical LOCAL requires original float64 arrays without casting')
    if not np.isfinite(cached).all() or not np.isfinite(stored).all(): raise ValueError('historical LOCAL nonfinite risk')
    error=float(abs(cached-stored).max())
    if error>1e-9: raise ValueError('historical LOCAL exceeds published Phase19 replay tolerance')
    a=cached[:,None]-cached[None,:];b=stored[:,None]-stored[None,:]
    order_equal=np.array_equal(np.sign(a),np.sign(b))
    ties_equal=np.array_equal(abs(a)<=1e-8,abs(b)<=1e-8)
    if not order_equal or not ties_equal: raise ValueError('historical LOCAL pair ordering or tie classification differs')
    return {'artifact_sha256':artifact_sha,'receipt_artifact_sha256':receipt_sha,'patient_fold_elementwise_identical':True,
        'shape_identical':True,'finite':True,'stored_dtype':str(stored.dtype),'OOF_dtype':str(cached.dtype),
        'maximum_error':error,'nonidentical_float64_values':int((cached!=stored).sum()),
        'published_Phase19_replay_absolute_tolerance':1e-9,'all_pair_strict_order_identical':True,
        'all_pair_tie_classification_identical':True,'pair_tie_tolerance':1e-8,'OOF_values_changed_or_cast':False}

def qualify(ck,ds,config,simple):
    data,train=_api()
    if not np.array_equal(ck['fit'],ds.fit) or not np.array_equal(ck['validation_or_test'],ds.valid): raise ValueError('patient partitions differ')
    if np.intersect1d(ds.fit,ds.valid).size or (ds.role=='inner' and np.intersect1d(ds.active,ds.part['outer_test']).size):
        raise ValueError('patient leakage')
    path=data.P/ds.role/f"{ck['arm']}_seed{ck['seed']}_fold{ck['fold']}.pt"
    record=_provenance().require_record(path.with_suffix('.json'),checkpoint_path=path)
    if ck['source']!=record['source']: raise ValueError('qualified source differs from checkpoint')
    expected_audit=copy.deepcopy(ds.audit)
    expected_audit['source']=record['source']
    assert_equal_structure(ck['preprocess_audit'],expected_audit,'current_preprocessing')
    history_audit(ck,config)
    assigned=train.assigned_mode(ck['arm'])
    if ck['assigned_mode']!=assigned: raise ValueError('primary assignment differs')
    key=dict(MODES)[assigned]
    if not np.array_equal(ck['risk'],ck[key]): raise ValueError('primary prediction not training-assigned mode')
    for key in ['risk','anchor_risk',*[key for _,key in MODES]]+([key for _,key in REPRESENTATION_MODES] if ck['arm'] in ['P1','P16'] else []):
        if np.asarray(ck[key]).shape!=(len(ds.valid),) or not np.isfinite(ck[key]).all(): raise ValueError('invalid prediction '+key)
    if not np.array_equal(ck['constantC_risk'],ck['interaction_off_risk']): raise ValueError('Cref correction is not zero')
    if not np.array_equal(ck['anchor_w'],simple['models']['J0']['w']): raise ValueError('frozen anchor coefficient identity differs')
    if not np.array_equal(ck['anchor_risk'],simple['models']['J0']['risk']): raise ValueError('frozen anchor prediction is not elementwise inherited J0')
    fit_anchor=np.asarray(ds.f['X'][ds.fit],np.float64)@simple['models']['J0']['w']
    if abs(float(np.std(fit_anchor,ddof=0))-ck['anchor_sd_fit'])>1e-12: raise ValueError('anchor fit-only scale differs')
    if not np.array_equal(ds.f['ANCHOR'][ds.fit],fit_anchor.astype(np.float32)) or not np.array_equal(ds.f['ANCHOR'][ds.valid],ck['anchor_risk']):
        raise ValueError('anchor override is not exact fit/valid offset')
    if not np.array_equal(ck['state']['main'].cpu().numpy(),np.asarray(ck['anchor_w'],dtype=ck['state']['main'].cpu().numpy().dtype)):
        raise ValueError('frozen main buffer changed')
    for mapping in [ds.wrong_c,ds.wrong_o]:
        for indices in [ds.fit,ds.valid]:
            if np.any(mapping[indices]==indices) or set(mapping[indices])!=set(indices): raise ValueError('mismatch crosses partition/contains identity')
    if ds.role=='outer':
        path=data.P/'inner'/f"{ck['arm']}_seed{ck['seed']}_fold{ck['fold']}.pt"
        inner,_=qualified_checkpoint(path,ck['arm'],ck['seed'],ck['fold'],'inner')
        if inner['selected_epoch']!=ck['selected_epoch']: raise ValueError('outer epoch not inner-selected')

def replay(shard=0,shards=1):
    import pandas as pd
    import torch
    data,train=_api();config=data.cfg();identity=data.source()
    _provenance().require_record(data.R/'PROTOCOL_LOCK.json')
    rows=[];features=[];convex=[];seen=set()
    jobs=[(arm,seed,fold,role) for arm in ARMS for role in ['inner','outer'] for seed in config['seeds'] for fold in config['folds']]
    for arm,seed,fold,role in jobs[shard::shards]:
        path=data.P/role/f'{arm}_seed{seed}_fold{fold}.pt'
        ck,receipt=qualified_checkpoint(path,arm,seed,fold,role)
        train.seed_all(seed+100*fold);ds=data.Data(fold,role,seed)
        simple,checks=inherited_simple(ds);qualify(ck,ds,config,simple)
        if (role,fold) not in seen:
            features.append(independent_feature_audit(ds));convex.extend(checks);seen.add((role,fold))
        model=train.make_model(ds,arm,ck['anchor_w']);model.load_state_dict(ck['state'],strict=True);model.eval()
        if any(name=='main' for name,_ in model.named_parameters()): raise ValueError('main is trainable')
        errors={}
        with torch.no_grad():
            modes=MODES+[('anchor','anchor_risk')]+(REPRESENTATION_MODES if arm in ['P1','P16'] else [])
            for mode,key in modes:
                actual=train.forward(model,ds,ds.valid,mode=mode,training=False).cpu().numpy()
                errors[mode]=float(abs(actual-ck[key]).max())
                if mode=='anchor' and not np.array_equal(actual,ck[key]): raise ValueError('anchor replay not bitwise identical')
        if max(errors.values())>=2e-6: raise ValueError(('mode replay failed',arm,seed,fold,role,errors))
        pair_max=float(abs(ck['interaction_off_risk']-ck['anchor_risk']).max())
        correction_max=max(float(abs(ck[key]-ck['interaction_off_risk']).max()) for _,key in MODES)
        if pair_max>ck['anchor_sd_fit']+5e-6 or correction_max>.25*ck['anchor_sd_fit']+5e-6:
            raise ValueError('final risk correction exceeded locked anchor-scale cap')
        if arm in ['P1','P16'] and any(float(abs(ck[key]-ck['anchor_risk']).max())>ck['anchor_sd_fit']+5e-6 for _,key in REPRESENTATION_MODES):
            raise ValueError('frozen representation control exceeded base-risk cap')
        if arm in ['L0','P1','P16'] and any(not np.array_equal(ck[key],ck['interaction_off_risk']) for _,key in MODES):
            raise ValueError('no-correction baseline depends on correction context')
        rows.append({'arm':arm,'seed':seed,'fold':fold,'role':role,'source':identity,
            'actual_fit_source':ck['source'],
            'checkpoint_sha256':receipt['checkpoint_sha256'],'max_replay_error':max(errors.values()),
            'anchor_replay_max_error':errors['anchor'],'anchor_elementwise_identical':True,
            'extra_representation_modes':len(REPRESENTATION_MODES) if arm in ['P1','P16'] else 0,
            'anchor_sd_fit':ck['anchor_sd_fit'],'maximum_pair_risk_magnitude':pair_max,
            'maximum_correction_risk_magnitude':correction_max,
            'constantC_equals_correction_off':True,'frozen_main_buffer':True,'selected_epoch':ck['selected_epoch']})
        print('PHASE20_REPLAY',arm,seed,fold,role,max(errors.values()),flush=True)
        del model,ds;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(data.R/f'CHECKPOINT_REPLAY_{shard}.csv',index=False)
    pd.DataFrame(features).to_csv(data.R/f'FEATURE_RECONSTRUCTION_{shard}.csv',index=False)
    pd.DataFrame(convex).to_csv(data.R/f'CONVEX_REFERENCE_AUDIT_{shard}.csv',index=False)
    write(data.P/f'REPLAY_SHARD_{shard}.json',{'status':'PASS','source':identity,'rows':len(rows),'shards':shards})

def collect():
    import pandas as pd
    import torch
    from types import SimpleNamespace
    data,train=_api();config=data.cfg();identity=data.source();upstream,parent_identity=_upstream()
    done=_provenance().require_record(data.P/'TRAINING_DONE.json')
    _provenance().require_record(data.R/'PROTOCOL_LOCK.json')
    if done['status']!='PASS' or done['source']!=identity: raise ValueError('training incomplete/stale')
    if config['arms']!=ARMS: raise ValueError('locked nine-arm matrix changed')
    cohort=upstream.parent.load();event=upstream.parent.endpoints(cohort)['DSS']
    if len(cohort['patient_ids'])!=512 or int(event['event'].sum())!=65: raise ValueError('cohort changed')
    replay_rows=[q for path in sorted(data.R.glob('CHECKPOINT_REPLAY_*.csv')) for q in pd.read_csv(path).to_dict('records')]
    expected={(arm,seed,fold,role) for arm in ARMS for role in ['inner','outer'] for seed in config['seeds'] for fold in config['folds']}
    keys=[(q['arm'],q['seed'],q['fold'],q['role']) for q in replay_rows]
    if len(keys)!=270 or set(keys)!=expected: raise ValueError('270 replay records incomplete/duplicated')
    if sum(q['extra_representation_modes'] for q in replay_rows)!=120: raise ValueError('120 frozen representation replays missing')
    lookup=dict(zip(keys,replay_rows))
    if any(q['source']!=identity or q['max_replay_error']>=2e-6 or q['anchor_replay_max_error']!=0 for q in replay_rows): raise ValueError('failed replay')
    feature_rows=[q for path in sorted(data.R.glob('FEATURE_RECONSTRUCTION_*.csv')) for q in pd.read_csv(path).to_dict('records')]
    if {(q['role'],q['fold']) for q in feature_rows}!={(r,f) for r in ['inner','outer'] for f in config['folds']}:
        raise ValueError('feature replay incomplete')
    banks={};simple_banks={};convex=[];manifest=[]
    for role in ['inner','outer']:
        for fold in config['folds']:
            path=upstream.P/'preprocessing'/f'fold{fold}_{role}.pt';receipt=json.loads(path.with_suffix('.json').read_text())
            bank=torch.load(path,map_location='cpu',weights_only=False)
            if sha(path)!=receipt['sha256'] or bank['provenance']['source']!=parent_identity: raise ValueError('upstream feature bank changed')
            if any(q['status']!='PASS' or q['feature_cache_sha256']!=sha(path) for q in feature_rows if q['role']==role and q['fold']==fold):
                raise ValueError('independent feature audit stale')
            parts=upstream.parent.splits(cohort,'DSS',fold)
            fit,valid=parts['inner_train' if role=='inner' else 'outer_train'],parts['inner_val' if role=='inner' else 'outer_test']
            if not np.array_equal(fit,bank['fit']) or not np.array_equal(valid,bank['valid']): raise ValueError('upstream feature partition differs')
            ds=SimpleNamespace(f=bank['features'],fit=fit,valid=valid,t=event['time'],e=event['event'],role=role,fold=fold)
            simple,checks=inherited_simple(ds);convex.extend(checks);banks[role,fold]=bank;simple_banks[role,fold]=simple
            for arm in ['J0','J_CS']:
                manifest.append({'arm':arm,'seed':'deterministic','fold':fold,'role':role,
                    'fit_classification':'qualified_inherited_convex_reference','source':parent_identity,
                    'checkpoint_sha256':sha(upstream.P/'simple'/f'{role}_fold{fold}.pt'),'selected_epoch':0,'actual_epochs':0})
    risks={};curves=[];mechanisms=[];summaries=[];inner_epochs={};coverage={}
    owner=np.load(upstream.VISUAL/'owner.npy')
    for arm in ARMS:
        for role in ['inner','outer']:
            for seed in config['seeds']:
                for fold in config['folds']:
                    path=data.P/role/f'{arm}_seed{seed}_fold{fold}.pt';ck,receipt=qualified_checkpoint(path,arm,seed,fold,role)
                    if lookup[arm,seed,fold,role]['checkpoint_sha256']!=sha(path): raise ValueError('checkpoint changed after replay')
                    bank=banks[role,fold];simple=simple_banks[role,fold]
                    if not np.array_equal(ck['fit'],bank['fit']) or not np.array_equal(ck['validation_or_test'],bank['valid']): raise ValueError('stored partitions changed')
                    if ck['preprocess_audit']['upstream_source']!=parent_identity or ck['preprocess_audit']['source']!=ck['source']: raise ValueError('current/upstream provenance differs')
                    if lookup[arm,seed,fold,role]['actual_fit_source']!=ck['source']: raise ValueError('replayed actual fit source differs')
                    if not np.array_equal(ck['anchor_w'],simple['models']['J0']['w']) or not np.array_equal(ck['anchor_risk'],simple['models']['J0']['risk']): raise ValueError('shared frozen anchor changed')
                    history_audit(ck,config)
                    if role=='inner': inner_epochs[arm,seed,fold]=ck['selected_epoch']
                    elif ck['selected_epoch']!=inner_epochs[arm,seed,fold]: raise ValueError('outer schedule differs')
                    if ck['assigned_mode']!=train.assigned_mode(arm) or not np.array_equal(ck['risk'],ck[dict(MODES)[ck['assigned_mode']]]): raise ValueError('assigned pairing differs')
                    if not np.array_equal(ck['constantC_risk'],ck['interaction_off_risk']): raise ValueError('zero correction reference differs')
                    update_coverage(coverage,ck,owner)
                    meta={'arm':arm,'seed':seed,'fold':fold,'role':role,'selected_epoch':ck['selected_epoch']}
                    curves.extend({**meta,**q} for q in ck['history']);mechanisms.append({**meta,**ck['mechanism']})
                    summaries.append({**meta,**{k:v for k,v in ck['summary'].items() if not isinstance(v,(dict,list,np.ndarray))}})
                    manifest.append({**meta,'fit_classification':'new_neural_fit','source':identity,'actual_fit_source':ck['source'],'checkpoint_sha256':receipt['checkpoint_sha256'],
                        'actual_epochs':len(ck['history']),'assigned_mode':ck['assigned_mode']})
                    if role=='outer':
                        test=bank['valid'];vector=risks.setdefault((arm,seed),np.full(512,np.nan))
                        if np.isfinite(vector[test]).any(): raise ValueError('duplicate outer prediction')
                        vector[test]=ck['risk']
                        if arm in ['P1','P16']:
                            for label,field in [('FULL_OMICS_WRONG','full_omics_wrong_risk'),('UNIFORM_PAIR','uniform_pair_risk')]:
                                risks.setdefault((arm+'__'+label,seed),np.full(512,np.nan))[test]=ck[field]
                        if arm=='S16':
                            for label,field in [('WRONG_C','wrongC_risk'),('WRONG_O','wrongO_risk'),('CONSTANT_C','constantC_risk'),('CORRECTION_OFF','interaction_off_risk')]:
                                risks.setdefault(('S16__'+label,seed),np.full(512,np.nan))[test]=ck[field]
    for arm in ['J0','J_CS']:
        for seed in config['seeds']:
            vector=np.full(512,np.nan)
            for fold in config['folds']: vector[banks['outer',fold]['valid']]=simple_banks['outer',fold]['models'][arm]['risk']
            risks[arm,seed]=vector
    # Qualified historical simple comparator, explicitly different visual inputs.
    historic=np.load('/root/phase16r_light_rna_private/OOF_PRIVATE.npz');historic_rows=[]
    for seed in config['seeds']:
        vector=historic['LOCAL__CV3__seed'+str(seed)]
        for fold in config['folds']:
            p=Path('/root/phase16r_light_rna_private/artifacts')/f'seed{seed}_fold{fold}.pt'
            receipt=json.loads((Path('/root/phase16r_light_rna_private/cells')/f'seed{seed}_fold{fold}.json').read_text())
            old=torch.load(p,map_location='cpu',weights_only=False);ix=banks['outer',fold]['valid']
            historical_audit=historical_reference_identity(vector[ix],old['models']['LOCAL__CV3']['risk'],
                ix,old['part']['outer_test'],sha(p),receipt['sha256'])
            historic_rows.append({'arm':'HIST_CS_S32_CONCH32','seed':seed,'fold':fold,**historical_audit,
                'scope':'qualified historical reference, different inputs/training; no new fit'})
        risks['HIST_CS_S32_CONCH32',seed]=vector.copy()
    check_model_seed_coverage(risks,config['seeds'])
    if len(risks)!=60 or any(q.shape!=(512,) or not np.isfinite(q).all() for q in risks.values()): raise ValueError('60-vector OOF incomplete')
    frame=pd.DataFrame(summaries);budget=[]
    prior=_provenance().require_record(data.R/'BUDGET_AUDIT.json')
    if not prior['no_outer_results_used']: raise ValueError('preouter budget evidence differs')
    for arm,group in frame[frame.role=='inner'].groupby('arm'):
        fraction=float(group.hit_maximum.mean());maximum=config['engineering_gate']['maximum_cap_fraction_per_arm']
        if abs(prior['inner_cap_fractions'][arm]-fraction)>1e-12: raise ValueError('budget differs from preouter decision')
        budget.append({'arm':arm,'inner_cells':len(group),'cap_fraction':fraction,'maximum_cap_fraction':maximum,
            'budget_gate_pass':fraction<=maximum,'selected_epoch_median':float(group.selected_epoch.median()),'actual_epoch_median':float(group.actual_epochs.median())})
    write(data.R/'BUDGET_VERIFICATION.json',{'status':'PASS','source':identity,'rows':budget,
        'mechanism_budget_sufficient':all(q['budget_gate_pass'] for q in budget),'original_preouter_audit_preserved':True,
        'interpretation':'prelocked patience/cap gate only, not global convergence'})
    for name,values in [('CELL_MANIFEST',manifest),('TRAINING_CURVES',curves),('COMPUTE_PROFILE',summaries),
                       ('ATTENTION_MECHANISM',flatten_diagnostics(mechanisms)),('CHECKPOINT_REPLAY',replay_rows),
                       ('FEATURE_RECONSTRUCTION',feature_rows),('CONVEX_REFERENCE_AUDIT',convex),('HISTORICAL_LOCAL_REFERENCE_AUDIT',historic_rows)]:
        pd.DataFrame(values).to_csv(data.R/(name+'.csv'),index=False)
    np.savez_compressed(data.P/'OOF_PRIVATE.npz',**{arm+'__seed'+str(seed):q for (arm,seed),q in risks.items()})
    write(data.R/'CHECKPOINT_REPLAY.json',{'status':'PASS','source':identity,'new_neural_checkpoints':270,'modes_per_checkpoint':5,
        'additional_anchor_replays':270,'maximum_error':max(q['max_replay_error'] for q in replay_rows),
        'additional_representation_replays':sum(q['extra_representation_modes'] for q in replay_rows),
        'anchor_elementwise_identical':True,'constantC_equals_correction_off':True,'frozen_main_buffer':True,
        'off_semantics':'retains fixed J0 plus patient-matched pairwise risk; only new Clinical correction disabled'})
    neural=[q for q in manifest if q['fit_classification']=='new_neural_fit']
    fit_sources={key:sum(q['actual_fit_source']==key for q in neural) for key in sorted({q['actual_fit_source'] for q in neural})}
    compatibility=json.loads((data.P/'ANCHOR_AUDIT_REPAIR_COMPATIBILITY.json').read_text())
    legacy_count=sum(q['actual_fit_source']!=identity for q in neural)
    if compatibility['source']!=identity or compatibility['status']!='PASS' or legacy_count!=len(compatibility['legacy_checkpoints']) or legacy_count!=189:
        raise ValueError('audit-only repair retained checkpoint inventory differs')
    if fit_sources.get(identity)!=81: raise ValueError('post-repair fresh outer count differs')
    write(data.R/'FIT_AND_REFERENCE_COUNTS.json',{'status':'PASS','source':identity,'new_inner_neural_fits':135,'new_outer_neural_fits':135,
        'actual_fit_source_counts':fit_sources,'retained_pre_audit_repair_formal_fits':legacy_count,'post_audit_repair_formal_fits':81,
        'new_unique_convex_models':0,'qualified_inherited_unique_convex_models':20,'convex_seed_role_references':60,
        'historical_LOCAL_outer_references':15,'previous_neural_models_counted_as_new_fit_or_OOF':0,
        'new_native_method_fits':0,'new_PCA_or_KMeans_fits':0})
    coverage_table=coverage_rows(coverage)
    if len(coverage_table)!=18 or any(q['checkpoint_count']!=15 for q in coverage_table): raise ValueError('probe coverage incomplete')
    write(data.R/'PROBE_COVERAGE_AUDIT.json',{'status':'PASS','source':identity,'rows':coverage_table,
        'additional_forward_or_gradient_calls':0,'sampling':'risk/attention: all fit patients; gradients: eight fixed equally-spaced fit positions',
        'scope':'unique query patients/WSIs, not independent checkpoint count; wrong context donor is not the queried risk patient',
        'raw_patient_indices_public':False})
    write(data.R/'SELECTIVITY_BEHAVIOR_AUDIT.json',selectivity_behavior_audit(mechanisms,identity))
    return cohort,risks

def formal_evaluation():
    data,_=_api();began=time.monotonic();identity=data.source();evaluator_hash=sha(Path(__file__))
    math=json.loads((data.R/'EVALUATION_MATH_AUDIT.json').read_text())
    if (math['status']!='PASS' or math.get('source',math.get('training_source'))!=identity or
        math['server_actual_tests']!=7 or math['failures_errors_skips']!=0 or
        math['evaluator_sha256']!=evaluator_hash or math['test_source_sha256']!=sha(ROOT/'tests/test_phase20_evaluation.py')):
        raise ValueError('actual independent math tests/source identity must pass before formal evaluation')
    cohort,risks=collect();config=data.cfg()
    t,e,valid,ipcw,seeds=metrics(cohort,risks,'DSS',data.R)
    bootstrap(cohort,risks,(t,e,valid,ipcw),seeds,config,data.R)
    metrics(cohort,risks,'PFI',data.R)
    write(data.R/'INDEPENDENT_VALIDATION.json',{'status':'PASS','source':data.source(),'model_seed_vectors':len(risks),
        'outer_fold_metric_rows':len(risks)*5,'independent_sksurv_Harrell_Uno60_each_fold':True,
        'primary_tests':14,'risk_ensembling':False,'previous_neural_models_counted_as_current_OOF':False,
        'PFI':'frozen DSS predictions only; no PFI fitting/selection','native_methods':'historical background only, not current-protocol leaderboard',
        'retained_pre_audit_repair_checkpoints':189,'post_audit_repair_checkpoints':81,
        'legacy_eligibility':'exact receipt/PT hashes and all-formal-mode bitwise compatibility proof, never a source-string allowlist'})
    if data.source()!=identity or sha(Path(__file__))!=evaluator_hash: raise ValueError('source changed during evaluation')
    write(data.P/'EVALUATION_DONE.json',{'status':'PASS','source':identity,'evaluator_sha256':evaluator_hash,
        'test_source_sha256':sha(ROOT/'tests/test_phase20_evaluation.py'),'actual_new_checkpoint_replays':270,
        'context_modes_per_checkpoint':5,'additional_anchor_replays':270,'additional_representation_replays':120,
        'actual_total_prediction_vectors_replayed':1740,'model_seed_vectors':len(risks),'outer_fold_metric_rows':len(risks)*5,
        'retained_pre_audit_repair_checkpoints':189,'post_audit_repair_checkpoints':81,
        'bootstrap_draws':20000,'seconds_formal_collection_metrics_bootstrap':time.monotonic()-began,
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print('PHASE20_INDEPENDENT_EVALUATION_PASS',len(risks)*5,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    replay(args.shard,args.shards) if args.replay else formal_evaluation()
