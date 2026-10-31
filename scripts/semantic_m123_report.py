"""Post-training M1--M3 audit, official metrics and anonymous summaries.

Consumes complete registered runs only. No optimizer, validation-driven new
candidate, confidence interval, bootstrap or patient array publication exists.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,csv,gzip,hashlib,json,pickle,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123_worker import check_lock,read,sha,write
from scripts.semantic_m123_schedule import inner_jobs,final_jobs,verify_complete,choose,cell_path
from scripts.semantic_b0b7_p2_matrix import raw_inputs
from src.semantic_m123.models import JointModel,ARMS
from src.semantic_m123.training import subset
from src.semantic_m123.replay import load_saved_inputs,repair_damage
from src.semantic_b0b7.p2_data import read_dss_only
from src.semantic_b0b7.p2_survival import evaluate_safe
from src.semantic_b0b7.survival import aggregate_folds,uno_weights
from types import SimpleNamespace

CONTRASTS=[
 ('M0PLUS','M0'),('M1_SCRATCH','M0PLUS'),('M1_WARM','M0PLUS'),('M1_WARM','M0'),
 ('M1_WARM','M1_SCRATCH'),('M2_MEAN_MATCHED','M1_WARM'),('M2_SINGLE','M2_MEAN_MATCHED'),
 ('M2_EIGHT','M2_SINGLE'),('M2_EIGHT','M2_MEAN_MATCHED'),('M2_EIGHT_MEAN','M2_EIGHT'),
 ('M2_EIGHT_MEAN','M2_MEAN_MATCHED'),('M2_EIGHT_MEAN','M1_WARM'),
 ('M3_FULL','M2_EIGHT_MEAN'),('M3_FULL','M3_LEARNED_QUERY'),
 ('M3_FULL','M3_NO_PRODUCT'),('M3_FULL','M3_BIDIRECTIONAL'),
 ('M3_FULL','M0'),('M3_FULL','M0PLUS'),
 ('M3_FULL','M3_FULL_WRONG_QUERY'),('M3_FULL','M3_FULL_MEAN_QUERY'),
 ('M3_FULL','M3_FULL_NO_INTERACTION')]


def aggregate(rows):
    hnum=sum(x['Harrell_num'] for x in rows);hden=sum(x['Harrell_den'] for x in rows)
    supported=all(x['Uno60'] is not None for x in rows)
    unum=sum(x['Uno_num'] for x in rows) if supported else None
    uden=sum(x['Uno_den'] for x in rows) if supported else None
    return {'Harrell':hnum/hden if hden else None,'Uno60':unum/uden if supported and uden>0 else None,
        'Harrell_num':hnum,'Harrell_den':hden,'Uno_num':unum,'Uno_den':uden,
        'folds':len(rows),'aggregation':'within_fold_pair_components_only'}


def seed_summary(foldrows):
    results=[]
    for arm in sorted({r['arm'] for r in foldrows}):
        seeds=sorted({r['seed'] for r in foldrows if r['arm']==arm})
        byseed=[]
        for seed in seeds:
            cells=[r for r in foldrows if r['arm']==arm and r['seed']==seed]
            if sorted(r['outer'] for r in cells)!=[1,2,3,4,5]:raise ValueError('FIVE_OOF_FOLD_COVERAGE_REQUIRED')
            byseed.append({'seed':seed,**aggregate(cells)})
        def mean(key):
            vals=[x[key] for x in byseed]
            return float(np.mean(vals)) if all(x is not None for x in vals) else None
        results.append({'arm':arm,'Harrell':mean('Harrell'),'Uno60':mean('Uno60'),
            'seed_results':byseed,'aggregation':'five_within_fold_components_per_seed_then_equal_seed_mean',
            'seed_independent_cohorts_claim':False,'frozen_intervention':arm.endswith(('_WRONG_QUERY','_MEAN_QUERY','_NO_INTERACTION'))})
    return results


def paired(summaries):
    index={x['arm']:x for x in summaries};result=[]
    for a,b in CONTRASTS:
        if a not in index or b not in index:raise ValueError('MISSING_PRELOCKED_COMPARISON')
        aa,bb=index[a],index[b];deltas=[]
        for row in aa['seed_results']:
            reference=bb['seed_results'][0] if len(bb['seed_results'])==1 else next(x for x in bb['seed_results'] if x['seed']==row['seed'])
            deltas.append({'seed':row['seed'],'Harrell':row['Harrell']-reference['Harrell'],
                'Uno60':row['Uno60']-reference['Uno60'] if row['Uno60'] is not None and reference['Uno60'] is not None else None})
        result.append({'comparison':a+' minus '+b,'Harrell_delta':aa['Harrell']-bb['Harrell'],
            'Uno60_delta':aa['Uno60']-bb['Uno60'] if aa['Uno60'] is not None and bb['Uno60'] is not None else None,
            'positive_Harrell_seeds':sum(x['Harrell']>0 for x in deltas),
            'positive_Uno60_seeds':sum(x['Uno60'] is not None and x['Uno60']>0 for x in deltas),
            'positive_both_seeds':sum(x['Harrell']>0 and x['Uno60'] is not None and x['Uno60']>0 for x in deltas),
            'seed_deltas':deltas,'CI_bootstrap_pvalues':False})
    return result


def csv_shards(root,stem,rows,limit=1_800_000):
    import io
    if not rows:return []
    fields=sorted(set().union(*(x.keys() for x in rows)))
    if len(fields)>149:raise ValueError('PUBLIC_CSV_WIDTH_EXCEEDED')
    def textrow(row,header=False):
        s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n')
        if header:w.writeheader()
        else:w.writerow(row)
        return s.getvalue()
    header=textrow({},True);chunks=[];data=header;count=0;size=len(header.encode('utf-8'))
    for row in rows:
        line=textrow(row)
        line_size=len(line.encode('utf-8'))
        if size+line_size>limit or count>=18000:
            p=root/(stem+'_'+str(len(chunks)+1).zfill(2)+'.csv');p.write_text(data,encoding='utf-8',newline='')
            chunks.append(p.name);data=header;count=0;size=len(header.encode('utf-8'))
        data+=line;count+=1;size+=line_size
    if count:
        p=root/(stem+'_'+str(len(chunks)+1).zfill(2)+'.csv');p.write_text(data,encoding='utf-8',newline='');chunks.append(p.name)
    return chunks


def official_metrics(tt,te,t,e,risk):
    from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
    def structured(time,event):
        x=np.empty(len(time),dtype=[('event','?'),('time','<f8')]);x['event']=event.astype(bool);x['time']=time
        return x
    risk=np.asarray(risk,np.float64)
    h=float(concordance_index_censored(np.asarray(e,bool),np.asarray(t,np.float64),risk,tied_tol=1e-8)[0])
    u=float(concordance_index_ipcw(structured(tt,te),structured(t,e),risk,tau=60.,tied_tol=1e-8)[0])
    return h,u


def run(args):
    lock=check_lock(args.lock,args.expected_lock_sha256);private=Path(args.private)
    if sha(args.analysis_lock)!=args.expected_analysis_sha256:raise ValueError('ANALYSIS_LOCK_CHANGED')
    analysis=read(args.analysis_lock)
    if analysis['master_lock_sha256']!=args.expected_lock_sha256:raise ValueError('ANALYSIS_MASTER_BINDING')
    for name,value in analysis['analysis_sources'].items():
        if sha(ROOT/name)!=value:raise ValueError('ANALYSIS_SOURCE_CHANGED:'+name)
    if (private/'OWNER.private.json').exists() or not (private/'STUDY_COMPLETE.private.json').exists():
        raise ValueError('ALL_FORMAL_TRAINING_MUST_FINISH_BEFORE_OUTER_SCORING')
    complete=read(private/'STUDY_COMPLETE.private.json')
    if complete['lock_sha256']!=args.expected_lock_sha256:raise ValueError('COMPLETION_LOCK')
    if not args.formal_complete_authorized:raise ValueError('EXPLICIT_COMPLETED_RUN_FLAG_REQUIRED')
    barrier=read(private/'ALL_OUTER_SELECTIONS_FROZEN.private.json')
    bsha=sha(private/'ALL_OUTER_SELECTIONS_FROZEN.private.json')
    if bsha!=complete['selection_barrier_sha256']:raise ValueError('BARRIER_HASH')
    for name in lock['sources']:
        if sha(ROOT/name)!=lock['sources'][name]:raise ValueError('FROZEN_SOURCE_CHANGED')
    official_path=Path(args.official_scorer_path)
    sys.path.insert(0,str(official_path))
    import sksurv
    if sksurv.__version__!='0.23.1':raise ValueError('PINNED_OFFICIAL_SCORER_VERSION')
    out=Path(args.output_newdir);out.mkdir(parents=True,exist_ok=False)
    oofdir=private/'PRIVATE_OOF_v1';oofdir.mkdir(exist_ok=False)
    ijobs=inner_jobs(args.expected_lock_sha256);ojobs=final_jobs(args.expected_lock_sha256,barrier,bsha)
    inner=[];curve={};train_route=[];budget=[];totals={'neural_initializations':0,'neural_fits_with_positive_updates':0,'Cox_fits':0,
        'new_optimizer_updates':0,'worker_elapsed_seconds_sum':0.,'new_teacher_fits':0,'postprocess_optimizer_fits':0,
        'postprocess_fitted_preprocessors':0}
    allreceipts={}
    for job in ijobs+ojobs:
        r=verify_complete(private,job)
        if r is None:raise ValueError('MISSING_REGISTERED_COMPLETION')
        allreceipts[(job['split'],job['arm'],job['seed'],job['lambda'])]=r
        totals['new_optimizer_updates']+=r['optimizer_updates'];totals['worker_elapsed_seconds_sum']+=r['elapsed_seconds']
        totals['Cox_fits']+=int(job['arm']=='M0PLUS');totals['neural_initializations']+=int(job['arm']!='M0PLUS')
        totals['neural_fits_with_positive_updates']+=int(job['arm']!='M0PLUS' and r['optimizer_updates']>0)
        if r['route_audit'].get('applicable'):
            row={'split':job['split'],'role':job['role'],'arm':job['arm'],'seed':job['seed'],'L2_lambda':job['lambda'],
                **{k:v for k,v in r['route_audit'].items() if k not in ('isolated_interaction_risk_gradient_norms_without_L2','applicable')}}
            row.update({'isolated_gradient_'+k.replace('.','_'):v for k,v in r['route_audit']['isolated_interaction_risk_gradient_norms_without_L2'].items()})
            train_route.append(row)
        if job['role']!='inner':continue
        for snap in r['snapshots']:
            m=snap['evaluation_metrics'];inner.append({'split':job['split'],'arm':job['arm'],'seed':job['seed'],
                'L2_lambda':job['lambda'],'epoch':snap['epoch'],'parameters':r['parameters'],
                **{k:m.get(k) for k in ('Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den')}})
        if job['arm']!='M0PLUS':
            byepoch={s['epoch']:s['evaluation_metrics'] for s in r['snapshots']}
            budget.append({'split':job['split'],'arm':job['arm'],'seed':job['seed'],'L2_lambda':job['lambda'],
                'Harrell_400_minus200':byepoch[400]['Harrell']-byepoch[200]['Harrell'],
                'Uno60_400_minus200':byepoch[400]['Uno60']-byepoch[200]['Uno60'] if byepoch[400]['Uno60'] is not None and byepoch[200]['Uno60'] is not None else None})
        with gzip.open(cell_path(private,job)/'HISTORY.private.json.gz','rt') as f:history=json.load(f)
        for row in history:
            key=(job['arm'],job['lambda'],row['epoch'])
            acc=curve.setdefault(key,{'n':0,'values':{}})
            acc['n']+=1
            for k,v in row.items():
                if k in ('epoch','stage'):continue
                acc['values'].setdefault(k,[]).append(v)
    if totals['neural_initializations']!=1050 or totals['Cox_fits']!=50:raise ValueError('FIT_LEDGER_CLOSURE')
    csv_shards(out,'03_ALL_INNER_FIXED_CANDIDATES',inner)
    csv_shards(out,'08_TRAINING_ONLY_ROUTE_DIAGNOSTICS',train_route)
    csv_shards(out,'11_INNER_400_VS200_BUDGET',budget)
    curves=[]
    for (arm,lam,epoch),acc in sorted(curve.items()):
        row={'arm':arm,'L2_lambda':lam,'epoch':epoch,'contributing_correlated_fits':acc['n'],'timing':'pre_optimizer_update'}
        for k,values in acc['values'].items():
            row[k+'_mean']=float(np.mean(values));row[k+'_min']=float(np.min(values));row[k+'_max']=float(np.max(values))
        curves.append(row)
    csv_shards(out,'07_INNER_TRAINING_CURVES',curves)
    raw,_,assets=raw_inputs(SimpleNamespace(assets=lock['assets_path'],legacy_root=lock['legacy_root']),endpoints=False)
    endpoints,_=read_dss_only(assets['endpoints'],raw['patient_ids'])
    frozen=torch.load(lock['frozen_images'],map_location='cpu',weights_only=False,mmap=True)
    plan=read(lock['parent_split_plan'])['partitions']
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    foldrows=[];cases=[];repairs=[];oof={};seen={};route=[];loaded_preprocessors=0
    def score(arm,seed,outer,risk,reference,extra):
        part=plan[f'outer{outer}_final'];tr=np.asarray(part['fit']);ev=np.asarray(part['evaluation'])
        tt,te=endpoints['t'][tr],endpoints['e'][tr];t,e=endpoints['t'][ev],endpoints['e'][ev]
        metrics=evaluate_safe(tt,te,t,e,np.asarray(risk,np.float64))
        oh,ou=official_metrics(tt,te,t,e,risk)
        if abs(oh-metrics['Harrell'])>1e-12 or metrics['Uno60'] is None or abs(ou-metrics['Uno60'])>1e-10:
            raise ValueError('OFFICIAL_SCORER_MISMATCH')
        ref_sd=float(np.std(reference));risk=np.asarray(risk,np.float64)
        foldrows.append({'arm':arm,'seed':seed,'outer':outer,**{k:metrics[k] for k in ('Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den')},
            'risk_sd':float(np.std(risk)),'M0_risk_sd':ref_sd,'risk_minus_M0_sd':float(np.std(risk-reference)),
            'delta_sd_over_M0_sd':float(np.std(risk-reference)/ref_sd) if ref_sd>1e-12 else None,**extra})
        cases.append({'arm':arm,'seed':seed,'outer':outer,'Harrell_API_error':abs(oh-metrics['Harrell']),
            'Uno60_API_error':abs(ou-metrics['Uno60']),'official_API_calls':2})
        key=(arm,seed)
        oof.setdefault(key,np.full(512,np.nan));seen.setdefault(key,np.zeros(512,int))
        oof[key][ev]=risk;seen[key][ev]+=1
        h=repair_damage(t,e,risk,reference)
        u=repair_damage(t,e,risk,reference,uno_weights(tt,te,t,e,tau=60.))
        repairs.append({'arm':arm,'seed':seed,'outer':outer,'reference':'M0_exact',
            'Harrell_repair':h['repair_num'],'Harrell_damage':h['damage_num'],'Harrell_den':h['den'],
            'Uno_repair':u['repair_num'],'Uno_damage':u['damage_num'],'Uno_den':u['den']})
    for outer in range(1,6):
        split=f'outer{outer}_final';part=plan[split];current=[j for j in ojobs if j['split']==split]
        first=allreceipts[(split,current[0]['arm'],current[0]['seed'],current[0]['lambda'])]
        ps=private/'preprocessing'/first['shared_preprocessing']['file']
        if sha(ps)!=first['shared_preprocessing']['sha256']:raise ValueError('SAVED_PREPROCESSOR_CHANGED')
        with ps.open('rb') as f:state=pickle.load(f)
        batch,flat,rows=load_saved_inputs(raw,frozen,state,part['fit'],part['evaluation'],
            expected_input_sha256=lock['partitions'][split]['input_sha256'])
        loaded_preprocessors+=1;nt=len(part['fit']);trix=np.arange(nt);evix=np.arange(nt,len(rows))
        train=subset(batch,trix,'cuda');evaluation=subset(batch,evix,'cuda')
        ap=Path(lock['parent_private'])/'anchors'/(split+'.private.pkl')
        if sha(ap)!=lock['partitions'][split]['anchor_sha256']:raise ValueError('M0_STATE_CHANGED')
        with ap.open('rb') as f:anchor=pickle.load(f)
        reference=anchor.b0_exact_evaluation
        score('M0',0,outer,reference,reference,{'epochs':0,'parameters':None,'L2_lambda':None,'frozen_intervention':False})
        for job in current:
            r=allreceipts[(split,job['arm'],job['seed'],job['lambda'])];folder=cell_path(private,job)
            if r['shared_preprocessing']!=first['shared_preprocessing']:raise ValueError('SAME_PARTITION_MUST_SHARE_PREPROCESSING')
            with np.load(folder/'PREDICTIONS.private.npz',allow_pickle=False) as z:
                saved=np.array(z[f"{job['epochs']}_evaluation_risk"],np.float64)
                if not np.array_equal(z['rows'],rows):raise ValueError('PREDICTION_ROW_ORDER')
            if job['arm']=='M0PLUS':
                with gzip.open(folder/r['checkpoint_file'],'rb') as f:cp=pickle.load(f)
                replay=flat[evix]@cp['coef'];error=float(np.max(np.abs(replay-saved)))
            else:
                cp=torch.load(folder/r['checkpoint_file'],map_location='cpu',weights_only=False)
                model=JointModel(cp['arm'],*cp['dims']).cuda();model.load_state_dict(cp['state_dict']);model.eval()
                with torch.no_grad():matched=model(evaluation)
                replay=matched['risk'].cpu().numpy();error=float(np.max(np.abs(replay-saved)))
            if error>2e-6:raise ValueError('INDEPENDENT_SAVED_MODEL_FORWARD_REPLAY_FAILED')
            score(job['arm'],job['seed'],outer,saved,reference,{'epochs':job['epochs'],'parameters':r['parameters'],
                'L2_lambda':job['lambda'],'frozen_intervention':False})
            cases[-1]['saved_model_forward_max_error']=error
            if job['arm']=='M3_FULL':
                main_only=(matched['pathology_risk']+matched['omics_risk']+matched['clinical_risk']).cpu().numpy()
                score('M3_FULL_NO_INTERACTION',job['seed'],outer,main_only,reference,{'epochs':job['epochs'],
                    'parameters':r['parameters'],'L2_lambda':job['lambda'],'frozen_intervention':True})
                order=torch.arange(len(evix),device='cuda').roll(1)
                wrong={k:evaluation[k][order] for k in ('program','program_mask','coverage')}
                mask=train['program_mask'];count=mask.sum(0).clamp_min(1)
                mean={'program':(torch.where(mask,train['program'],0.).sum(0)/count)[None].expand(len(evix),-1,-1),
                    'program_mask':mask.any(0)[None].expand(len(evix),-1,-1),
                    'coverage':train['coverage'].mean(0,keepdim=True).expand(len(evix),-1,-1)}
                for label,context in [('WRONG_QUERY',wrong),('MEAN_QUERY',mean)]:
                    with torch.no_grad():intervention=model(evaluation,query_context=context,query_mode='mean_context' if label=='MEAN_QUERY' else 'matched')
                    for k in ('pathology_risk','omics_risk','clinical_risk'):
                        if not torch.equal(intervention[k],matched[k]):raise ValueError('INTERVENTION_CHANGED_MAIN_EFFECT')
                    risk=intervention['risk'].cpu().numpy()
                    score('M3_FULL_'+label,job['seed'],outer,risk,reference,{'epochs':job['epochs'],
                        'parameters':r['parameters'],'L2_lambda':job['lambda'],'frozen_intervention':True})
                    route.append({'outer':outer,'seed':job['seed'],'kind':label,'owner_main_effect_max_error':0.,
                        'matched_minus_intervened_risk_sd':float(np.std(replay-risk)),
                        'new_fits':0,'mean_uses_current_training_only':label=='MEAN_QUERY'})
            if job['arm']!='M0PLUS':del model,cp,matched;torch.cuda.empty_cache()
        del batch,train,evaluation,state,anchor;torch.cuda.empty_cache()
    for (arm,seed),risk in oof.items():
        if not np.isfinite(risk).all() or not np.all(seen[arm,seed]==1):raise ValueError('PRIVATE_OOF_COVERAGE')
        with (oofdir/(arm+'_seed'+str(seed)+'.private.npz')).open('xb') as f:
            np.savez_compressed(f,risk=risk,folds=raw['folds'],canonical_rows=np.arange(512))
    csv_shards(out,'04_OUTER_FOLD_METRICS',foldrows);csv_shards(out,'08_PAIR_REPAIR_DAMAGE',repairs)
    csv_shards(out,'08_FROZEN_QUERY_INTERVENTIONS',route)
    sums=seed_summary(foldrows);pairs=paired(sums)
    write(out/'05_SEED_METRICS.json',sums);write(out/'06_PRELOCKED_PAIRED_COMPARISONS.json',pairs)
    required=['M3_FULL minus '+b for b in ('M2_EIGHT_MEAN','M3_LEARNED_QUERY','M3_NO_PRODUCT','M3_BIDIRECTIONAL')]
    idx={x['comparison']:x for x in pairs}
    def pass_test(row):
        return row['Harrell_delta']>0 and row['Uno60_delta'] is not None and row['Uno60_delta']>0 and row['positive_both_seeds']>=2
    gate=all(pass_test(idx[k]) for k in required)
    anchors=all(pass_test(idx['M3_FULL minus '+b]) for b in ('M0','M0PLUS'))
    pairing=all(pass_test(idx['M3_FULL minus M3_FULL_'+b]) for b in ('WRONG_QUERY','MEAN_QUERY'))
    decision={'schema':'m123_scientific_decision_v1','status':'COMPLETE_LOCKED_M1_M2_M3','M3_four_prespecified_control_gate':gate,
        'M3_strong_anchor_gate':anchors,'M3_frozen_patient_query_dependency_gate':pairing,
        'M4_M5_trained':False,'no_posthoc_outer_winner_selected':True,'CI_bootstrap_pvalues':False,
        'independent_external_confirmation':False,'all_baselines_and_negative_results_retained':True,
        'next_stage':'CONDITIONAL_CANDIDATE_REVIEW' if gate and anchors and pairing else 'DO_NOT_EXPAND_M4_M5_FROM_THIS_EVIDENCE',
        'mechanism_biological_truth_claim':False,'all_fixed_paired_comparisons':pairs}
    write(out/'decision.json',decision)
    selected=[]
    for arm in ARMS:
        choices=[v for k,v in barrier['choices'].items() if k.split('/')[1]==arm]
        selected.append({'arm':arm,'final_configurations':len(choices),
            'epoch_counts':[{'epoch':e,'count':sum(x['epoch']==e for x in choices)} for e in (0,20,50,100,200,400)],
            'L2_counts':[{'lambda':l,'count':sum(x['lambda']==l for x in choices)} for l in (.001,.01)],
            '400_is_only_budget_boundary_not_convergence_proof':True})
    write(out/'11_SELECTION_AND_BUDGET_SCOPE.json',{'schema':'m123_selected_budget_description_v1',
        'selections':selected,'no_automatic_extension_or_architecture_search':True,
        'zero_epoch_is_initialization_not_newly_trained_network':True,'CI_bootstrap_pvalues':False})
    totals.update(private_OOF_vectors=len(oof),outer_saved_NN_forward_replays=150,
        frozen_query_intervention_forwards=30,derived_frozen_main_only_vectors=15,saved_preprocessor_reconstructions=loaded_preprocessors)
    observer=read(private/'RESOURCE_OBSERVER.private.json') if (private/'RESOURCE_OBSERVER.private.json').exists() else None
    if observer:totals['scheduler_wall_seconds_from_process_create_to_completion_marker']=(private/'STUDY_COMPLETE.private.json').stat().st_mtime-observer['scheduler_create_epoch']
    totals['worker_elapsed_sum_is_not_wall_clock']=True
    write(out/'10_COMPUTE_LEDGER.json',totals)
    audit={'schema':'m123_actual_post_training_metric_replay_audit_v1','status':'PASS',
        'official_scorer_version':sksurv.__version__,'actual_official_API_calls':sum(x['official_API_calls'] for x in cases),
        'outer_metric_cases':len(cases),'registered_private_fit_artifacts_verified':len(ijobs)+len(ojobs),
        'frozen_source_files_verified':len(lock['sources']),'cases':cases,'no_new_optimizer_fits':True,
        'all_model_inputs_reconstructed_from_saved_transform_state':True,'new_preprocessor_fits':0,
        'new_independent_raw_FM_extraction':False,'CI_bootstrap_pvalues':False}
    write(out/'09_ACTUAL_REPLAY_OFFICIAL_SCORER_AUDIT.json',audit)
    lines=['# M1–M3 联合优化与条件区域读取：锁定协议结果','',
        '本轮复用既有队列与患者分区；历史外折结果已经被查看，不能称为新的独立外部验证。未计算置信区间、bootstrap或p值。','',
        '|模型|Harrell|Uno60|','|---|---:|---:|']
    for row in sums:
        lines.append('|'+row['arm']+'|'+f"{row['Harrell']:.6f}"+'|'+(f"{row['Uno60']:.6f}" if row['Uno60'] is not None else 'NA')+'|')
    lines+=['','所有神经网络的主效应和交互层联合参与DSS优化，使用完整训练风险集Efron；没有固定r0偏移。M0+与新模型共享GRID32、REGIONS Patch、TITAN、S32、原R4 RNA512与Hallmark50资产。GRID32是精确warm所需的显式共同输入补充。','',
        'M2的全局均值残差与已有Patch投影共享参数。M3双向控制使用权重共享的普通双向注意力；四个M3对照有效参数量相同，不能称为某篇论文的官方复现。M3_NO_PRODUCT仅删除显式乘积，其条件读取仍具有跨模态非线性。','',
        '参数量匹配不等于统计自由度或优化难度完全相同。M0+本身也是联合拟合的凸加性Cox；M1相对M0+的比较同时涉及非线性、参数化和正则，不能单独归因为联合优化。','',
        'M3相对M2替换区域查询和局部读出，比较中同时存在查询数与读出方式差异；同结构的Learned-Query对照才更直接地隔离患者分子查询的作用。已有S32对所有模型共同提供文本来源的全局数值，本轮不据此主张新增文本语义创新。未加入随机基因集语义控制，因此M3即使有效，也不能单独证明真实Hallmark医学语义的必要性。','',
        '选择仅使用当前外层训练中的三个内折，全部五个外折的配置先封存再训练/评分。每seed按五折内部可比分子/分母汇总，然后对三个seed等权平均；不进行跨折风险配对。私有OOF已完整保存。','',
        '部分模型可能选择epoch0。这表示保留初始化函数，没有发生新的DSS优化，不能宣称为训练后的网络增益。FP32初始化是数学配方映射，未宣称与原FP64风险位级一致。','',
        '400轮是统一预算上限而非收敛证明。内层400相对200的逐单元变化、完整描述性训练曲线及最终选轮分布均保留；若选到边界且仍改善，应报告预算尚未闭合，不能直接将负结果推广为idea无效。','',
        '未新增任何配对辅助损失，因此Wrong-pair Auxiliary为不适用；错误患者和训练均值Query仅为冻结干预，自己的原始证据与所有主效应保持不变，不等价于真实重训消融。','',
        'M3_FULL_NO_INTERACTION由同一检查点的三个主效应按原精度相加，关闭交互风险读出；它只测已训练模型的依赖，不是真实重训的加性基线。','',
        f"预锁四项M3控制门槛：{gate}；相对M0/M0+强锚点门槛：{anchors}；冻结患者Query依赖门槛：{pairing}。", '',
        '门槛按预定双指标均值和至少两个seed同为正方向判定，不是统计显著性或医学因果证据。完整配对结果和推进决定见decision.json。M4/M5未训练。','',
        '优势：共同资产、精确训练分区warm、完整主效应、近容量控制与全风险集优化。风险：65个事件限制、历史队列反复开发、bulk RNA取材不对应局部Patch、冻结图文分数缺少本轮独立组织学真值，基础模型预训练队列重叠未被排除。','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    files=[{'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()]
    write(out/'REPORT_INDEX.json',{'schema':'m123_anonymous_report_index_v1','status':'COMPLETE','lock_sha256':args.expected_lock_sha256,
        'files':files,'reporter_source_sha256':sha(__file__),'private_arrays_or_weights_published':False})
    print(json.dumps({'status':'COMPLETE','outer_cases':len(cases),'official_API_calls':audit['actual_official_API_calls'],
        'M3_control_gate':gate,'M3_anchor_gate':anchors,'patient_query_gate':pairing,'public_files':len(files)+1}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','analysis-lock','expected-analysis-sha256','private','output-newdir','official-scorer-path'):p.add_argument('--'+k,required=True)
    p.add_argument('--formal-complete-authorized',action='store_true')
    run(p.parse_args())

