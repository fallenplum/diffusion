"""Score gated final models only after all five outer choices are frozen.

This derived reporter does not change the training master, search grid or gates.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123r_common import verify_lock,verify_cell,build,folder,read,write,sha
from scripts.semantic_m123r_schedule import final_jobs
from src.semantic_m123r.io import load_checkpoint,partition_endpoints
from src.semantic_m123r.models import RepairModel
from src.semantic_m123r.protocol import formal_arms
from src.semantic_m123r.reporting import aggregate,comparison,csv_shards
from src.semantic_m123.training import subset
from src.semantic_m123.replay import repair_damage
from src.semantic_b0b7.p2_survival import evaluate_safe
from src.semantic_b0b7.survival import uno_weights


def complete_coverage(rows):
    if len(rows)!=512 or not np.array_equal(np.sort(rows),np.arange(512)):
        raise ValueError('EXACT_FIVE_FOLD_512_PATIENT_COVERAGE_REQUIRED')


def final_barrier(private,decision,lock_sha):
    if (private/'OWNER.private.json').exists():raise ValueError('TRAINING_STILL_ACTIVE')
    d=read(decision);fc=read(private/'FORMAL_COMPLETE.private.json')
    bp=private/'ALL_OUTER_CHOICES_FROZEN.private.json';barrier=read(bp)
    if d['lock_sha256']!=lock_sha or not d['formal_arms'] or d['formal_arms']!=formal_arms(d['gates']):raise ValueError('FORMAL_QUALIFICATION_REQUIRED')
    if (fc['status']!='ALL_FORMAL_FITS_COMPLETE_NO_OUTER_SCORING_YET' or fc['lock_sha256']!=lock_sha
        or fc['formal_arms']!=d['formal_arms'] or fc['selection_sha256']!=sha(bp)
        or barrier['qualification_decision_sha256']!=sha(decision) or barrier['lock_sha256']!=lock_sha
        or d['training_complete_sha256']!=sha(private/'QUALIFICATION_COMPLETE.private.json')):
        raise ValueError('FORMAL_BARRIER_LINEAGE_MISMATCH')
    jobs=final_jobs(barrier,sha(bp));receipts=[]
    for j in jobs:
        r=verify_cell(private,j)
        if r is None:raise ValueError('ALL_SELECTED_FINAL_FITS_REQUIRED_BEFORE_LABEL_ACCESS')
        receipts.append((j,r))
    return d,barrier,receipts


def main():
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','private','qualification-decision','output-newdir','official-scorer-path'):
        p.add_argument('--'+k,required=True)
    a=p.parse_args();lock,parent=verify_lock(a.lock,a.expected_lock_sha256);private=Path(a.private)
    decision,barrier,receipts=final_barrier(private,a.qualification_decision,a.expected_lock_sha256)
    out=Path(a.output_newdir);out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,a.official_scorer_path)
    import sksurv
    if sksurv.__version__!='0.23.1':raise ValueError('OFFICIAL_VERSION')
    from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
    def official(t,e,vt,ve,r):
        def arr(t,e):
            s=np.empty(len(t),dtype=[('e','?'),('t','<f8')]);s['e']=e.astype(bool);s['t']=t;return s
        return float(concordance_index_censored(ve.astype(bool),vt,r,tied_tol=1e-8)[0]),float(concordance_index_ipcw(arr(t,e),arr(vt,ve),r,tau=60.,tied_tol=1e-8)[0])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    metrics=[];mechanisms=[];pairs=[];replays=[];oof={};coverage={};index={};official_errors=[]
    for job,r in receipts:index[(job['outer'],job['arm'],job['seed'],json.dumps(job['config'],sort_keys=True))]=(job,r)
    tracks=('EARLY_ANY','ENCODER_TRAINED')
    for outer in range(1,6):
        b,data,scale,raw,assets,part,digest=build(parent,f'outer{outer}_final');nt=len(data.train)
        # Endpoint access is below the verified all-final-fits/global-selection barrier.
        tt,ee=partition_endpoints(assets['endpoints'],raw['patient_ids'],data.rows,range(512))
        t,e,vt,ve=tt[:nt],ee[:nt],tt[nt:],ee[nt:];rows=data.rows[nt:];r0=data.baseline[nt:]
        batch=subset(b,data.evaluation,'cuda')
        def score(arm,seed,track,risk,extra=None):
            m=evaluate_safe(t,e,vt,ve,risk);hh,uu=official(t,e,vt,ve,risk)
            dh=abs(hh-m['Harrell']);du=abs(uu-m['Uno60']) if m['Uno60'] is not None else float('inf')
            if dh>1e-12 or du>1e-10:raise ValueError('OFFICIAL_METRICS_DISAGREE')
            official_errors.append((dh,du));key=(track,arm,seed)
            oof.setdefault(key,np.full(512,np.nan))[rows]=risk;coverage.setdefault(key,[]).extend(rows.tolist())
            metrics.append({'outer':outer,'arm':arm,'seed':seed,'track':track,**m,**(extra or {})})
            for label,weights in (('Harrell',None),('Uno60',uno_weights(t,e,vt,ve,tau=60.))):
                pair=repair_damage(vt,ve,risk,r0,weights)
                if abs(pair['net_delta']-(m[label]-evaluate_safe(t,e,vt,ve,r0)[label]))>1e-10:raise ValueError('REPAIR_DAMAGE_IDENTITY_FAILED')
                pairs.append({'outer':outer,'arm':arm,'seed':seed,'track':track,'metric':label,**pair})
        for track in tracks:score('R0',0,track,r0)
        for seed in (31,47,83):
            old=list((Path(lock['previous_study_private'])/'fits'/f'outer{outer}_final').glob(f'M1_WARM_seed{seed}_lambda*/COMPLETE.private.json'))
            if len(old)!=1:raise ValueError('ORIGINAL_R1_FINAL_REFERENCE_REQUIRED')
            rr=read(old[0]);pred=old[0].parent/'PREDICTIONS.private.npz'
            if sha(pred)!=rr['prediction_sha256']:raise ValueError('ORIGINAL_R1_PREDICTIONS_CHANGED')
            with np.load(pred,allow_pickle=False) as z:
                if not np.array_equal(z['rows'],data.rows):raise ValueError('ORIGINAL_R1_ORDER_CHANGED')
                risk=np.array(z[f"{rr['job']['epochs']}_evaluation_risk"],np.float64)
            for track in tracks:score('R1',seed,track,risk,{'epoch':rr['job']['epochs'],'new_optimizer_updates':0})
        for arm in decision['formal_arms']:
            for seed in (31,47,83):
                choices=barrier['selections'][f'outer{outer}/{arm}/{seed}']['choices']
                for track,c in choices.items():
                    job,r=index[(outer,arm,seed,json.dumps(c['config'],sort_keys=True))]
                    if r['input_sha256']!=digest:raise ValueError('CURRENT_FIT_PREPROCESSING_REBUILD_CHANGED')
                    node=next(n for n in r['snapshots'] if n['epoch']==c['epoch'])
                    with np.load(folder(private,job)/'PREDICTIONS.private.npz',allow_pickle=False) as z:
                        risk=np.array(z[f"{c['epoch']}_validation_risk"],np.float64)
                        frozen={s:np.array(z[f"{c['epoch']}_validation_{k}"],np.float64) for s,k in
                          (('NO_MIL','no_mil_risk'),('WRONG_QUERY','wrong_query_risk'),('MEAN_QUERY','mean_query_risk'),('NO_INTERACTION','no_interaction_risk'))
                          if (arm=='R5' and s=='NO_MIL') or (arm=='R6' and s!='NO_MIL')}
                    model=RepairModel(arm,*[b[k].shape[1] for k in ('P','G','C')]).cuda()
                    load_checkpoint(folder(private,job)/node['checkpoint']['file'],private,model)
                    with torch.no_grad():replay=model(batch)['risk'].cpu().numpy()
                    error=float(np.max(np.abs(risk-replay)))
                    if error>1e-9:raise ValueError('FORMAL_SELECTED_CHECKPOINT_REPLAY_FAILED')
                    replays.append({'outer':outer,'arm':arm,'seed':seed,'track':track,'error':error})
                    score(arm,seed,track,risk,{'epoch':c['epoch'],**c['config'],'parameters':r['parameters']})
                    for suffix,values in frozen.items():score(arm+'_'+suffix,seed,track,values,{'frozen_intervention':True})
                    mechanisms.append({'outer':outer,'arm':arm,'seed':seed,'track':track,'epoch':c['epoch'],
                        **node['diagnostics']['validation'],**{'actual_updates_'+k:v for k,v in node['actual_update_counts'].items()}})
                    del model;torch.cuda.empty_cache()
        del b,data,raw,batch;torch.cuda.empty_cache()
    privout=private/'FORMAL_OOF.private';privout.mkdir(exist_ok=False)
    for key,risk in oof.items():
        complete_coverage(coverage[key])
        if not np.isfinite(risk).all():raise ValueError('NONFINITE_FORMAL_OOF')
        np.savez_compressed(privout/('_'.join(map(str,key))+'.private.npz'),risk=risk)
    summaries={track:{} for track in tracks}
    for track in tracks:
        for arm in sorted({x['arm'] for x in metrics if x['track']==track}):
            ss=[]
            for seed in sorted({x['seed'] for x in metrics if x['track']==track and x['arm']==arm}):
                rr=[x for x in metrics if x['track']==track and x['arm']==arm and x['seed']==seed]
                if sorted(x['outer'] for x in rr)!=[1,2,3,4,5]:raise ValueError('FIVE_FOLD_METRIC_COVERAGE')
                ss.append({'seed':seed,**aggregate(rr)})
            summaries[track][arm]=ss
    requested={(arm,'R0') for arm in decision['formal_arms']}|{(arm,'R1') for arm in decision['formal_arms']}
    requested|={('R3','R2'),('R5','R4'),('R5','R3'),('R6','R5'),('R6','R7'),('R6','R8'),('R6','R9'),
      ('R5','R5_NO_MIL'),('R6','R6_WRONG_QUERY'),('R6','R6_MEAN_QUERY'),('R6','R6_NO_INTERACTION')}
    comparisons={track:{aa+' minus '+bb:comparison(v[aa],v[bb]) for aa,bb in sorted(requested) if aa in v and bb in v} for track,v in summaries.items()}
    csv_shards(out,'01_FORMAL_FOLD_METRICS',metrics);csv_shards(out,'02_FORMAL_MECHANISMS',mechanisms);csv_shards(out,'03_PAIR_REPAIR_DAMAGE',pairs)
    write(out/'04_FORMAL_SEED_METRICS.json',summaries);write(out/'05_FORMAL_COMPARISONS.json',comparisons)
    audit={'status':'PASS','lock_sha256':a.expected_lock_sha256,'derived_reporter_sha256':sha(__file__),
       'global_selection_sha256':sha(private/'ALL_OUTER_CHOICES_FROZEN.private.json'),
       'qualification_decision_sha256':sha(a.qualification_decision),'actual_official_API_calls':2*len(official_errors),
       'actual_selected_checkpoint_replays':len(replays),'maximum_checkpoint_replay_error':max(r['error'] for r in replays),
       'maximum_Harrell_API_error':max(r[0] for r in official_errors),'maximum_Uno60_API_error':max(r[1] for r in official_errors),
       'private_512_patient_OOF_vectors':len(oof),'all_five_selections_frozen_before_outer_endpoint_decoding':True,
       'new_teacher_fits':0,'new_optimizer_fits':0,'CI_bootstrap_pvalues':False,'patient_arrays_or_weights_public':False,
       'aggregation':'within_fold_components_per_seed_then_equal_seed_average','history_outer_scores_previously_seen':True}
    write(out/'06_FORMAL_ACTUAL_AUDIT.json',audit)
    result={'status':'COMPLETE_GATED_FORMAL_VALIDATION','formal_arms':decision['formal_arms'],'CI_bootstrap_pvalues':False,
       'M4_M5_trained':False,'new_external_independent_test_claim':False,'qualification_decision_sha256':sha(a.qualification_decision),
       'lock_sha256':a.expected_lock_sha256,'trained_vs_M0':{arm:comparisons['ENCODER_TRAINED'][arm+' minus R0'] for arm in decision['formal_arms']}}
    write(out/'decision.json',result)
    lines=['# M1–M3 修复：通过资格模型的五外折验证','','每个seed先合并五折内部可比对分子分母，再等权平均三个seed。没有跨折风险直接拼接计算C-index。早停与充分更新轨道并列；不做置信区间。','','|轨道|模型|Harrell|Uno60|','|---|---|---:|---:|']
    for track,v in summaries.items():
        for arm,rr in v.items():lines.append(f"|{track}|{arm}|{np.mean([r['Harrell'] for r in rr]):.6f}|{np.mean([r['Uno60'] for r in rr]):.6f}|")
    lines+=['','优势：全五折选择先冻结，保留原R4及RNA主效应，所选模型逐一从真实磁盘检查点回放。风险：原队列长期反复开发、样本与事件量有限；冻结干预和真实重训消融回答不同问题；slot与attention不能视为医学真值。','','M4/M5未开展；只对资格通过模型及其预定必要对照评分。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='')
    files=[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]
    write(out/'REPORT_INDEX.json',{'schema':'m123r_formal_report_index_v1','status':'COMPLETE','files':files,'lock_sha256':a.expected_lock_sha256})
    print(json.dumps({'status':result['status'],'formal_arms':result['formal_arms'],'actual_official_API_calls':audit['actual_official_API_calls']}))


if __name__=='__main__':main()
