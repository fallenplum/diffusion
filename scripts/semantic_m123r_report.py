"""Completed outer1 development qualification; no outer-test label access."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
from pathlib import Path
import argparse,collections,gzip,json,math,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123r_common import verify_lock,verify_cell,folder,build,read,write,sha
from scripts.semantic_m123r_selection import block_jobs
from src.semantic_m123r.models import RepairModel,ARMS
from src.semantic_m123r.reporting import aggregate,comparison,structure_pass,csv_shards
from src.semantic_m123r.protocol import formal_arms
from src.semantic_m123r.io import load_checkpoint
from src.semantic_m123.training import subset
from src.semantic_b0b7.p2_survival import evaluate_safe
from src.semantic_m123.replay import repair_damage


def main():
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','private','output-newdir','official-scorer-path'):p.add_argument('--'+k,required=True)
    a=p.parse_args();lock,parent=verify_lock(a.lock,a.expected_lock_sha256);private=Path(a.private)
    if (private/'OWNER.private.json').exists() or not (private/'QUALIFICATION_COMPLETE.private.json').exists():raise ValueError('QUALIFICATION_NOT_COMPLETE')
    sys.path.insert(0,a.official_scorer_path)
    import sksurv
    if sksurv.__version__!='0.23.1':raise ValueError('OFFICIAL_VERSION')
    from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
    def official(t,e,vt,ve,r):
        def arr(t,e):
            s=np.empty(len(t),dtype=[('e','?'),('t','<f8')]);s['e']=e.astype(bool);s['t']=t;return s
        return float(concordance_index_censored(ve.astype(bool),vt,r,tied_tol=1e-8)[0]),float(concordance_index_ipcw(arr(t,e),arr(vt,ve),r,tau=60.,tied_tol=1e-8)[0])
    out=Path(a.output_newdir);out.mkdir(parents=True,exist_ok=False)
    fullmetrics=[];curves={};receipts={};params={};updates=0;cases=[]
    for arm in ARMS:
        for seed in (31,47,83):
            for job in block_jobs(1,arm,seed,a.expected_lock_sha256):
                r=verify_cell(private,job)
                if r is None:raise ValueError('INCOMPLETE_REGISTERED_FIT')
                key=(arm,seed,json.dumps(job['config'],sort_keys=True),job['inner']);receipts[key]=(job,r)
                params[arm]=r['parameters'];updates+=r['optimizer_updates']
                for s in r['snapshots']:
                    row={'arm':arm,'seed':seed,'inner':job['inner'],'epoch':s['epoch'],'parameters':r['parameters'],**job['config']}
                    for side in ('train','validation'):
                        row.update({side+'_'+k:v for k,v in s['metrics'][side]['risk'].items() if k in ('Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den')})
                        row.update({side+'_'+k:s['diagnostics'][side][k] for k in ('base_drift_ratio','full_drift_sd','local_risk_sd','interaction_risk_sd')})
                    row.update({'actual_updates_'+k:v for k,v in s['actual_update_counts'].items()});fullmetrics.append(row)
                with gzip.open(folder(private,job)/'HISTORY.private.json.gz','rt') as f:history=json.load(f)
                for h in history:
                    if h['epoch'] not in (1,10,11,20,30,40,50,60,70,80,90,91,100,110,120,130,140):continue
                    key=(arm,json.dumps(job['config'],sort_keys=True),h['epoch']);acc=curves.setdefault(key,{'rows':[],'config':job['config']});acc['rows'].append(h)
    if len(receipts)!=540 or updates!=75600:raise ValueError('QUALIFICATION_LEDGER_COUNT')
    csv_shards(out,'01_ALL_FIXED_CANDIDATE_METRICS',fullmetrics)
    curve_rows=[]
    for (arm,_,ep),acc in sorted(curves.items()):
        row={'arm':arm,'epoch':ep,**acc['config'],'correlated_cells':len(acc['rows'])}
        for name in sorted(set().union(*(h.keys() for h in acc['rows']))-{'epoch','stage'}):
            vals=[h[name] for h in acc['rows'] if name in h];row.update({name+'_mean':float(np.mean(vals)),name+'_min':float(np.min(vals)),name+'_max':float(np.max(vals))})
        curve_rows.append(row)
    csv_shards(out,'02_TRAINING_CURVES',curve_rows)
    selected=[];geometry=[];structural={};bytrack={'EARLY_ANY':{},'ENCODER_TRAINED':{}};snapshot_replays=0;max_replay=0.
    statecache={}
    oof={};covered={};privOOF=private/'QUALIFICATION_OOF.private';privOOF.mkdir(exist_ok=False)
    for inner in range(3):
        b,data,scale,raw,assets,part,digest=build(parent,f'outer1_inner{inner}')
        val=subset(b,data.evaluation,'cuda')
        first=next((j,r) for (arm,seed,_,ii),(j,r) in receipts.items() if arm=='R2' and seed==31 and ii==inner)
        with np.load(folder(private,first[0])/'PREDICTIONS.private.npz',allow_pickle=False) as z:
            t=np.array(z['train_time']);e=np.array(z['train_event']);vt=np.array(z['evaluation_time']);ve=np.array(z['evaluation_event']);r0=np.array(z['r0_validation'])
        def score(arm,seed,risk,track,extra=None):
            m=evaluate_safe(t,e,vt,ve,risk);h,u=official(t,e,vt,ve,risk)
            if abs(h-m['Harrell'])>1e-12 or m['Uno60'] is None or abs(u-m['Uno60'])>1e-10:raise ValueError('OFFICIAL_METRIC_DISAGREEMENT')
            row={'arm':arm,'seed':seed,'inner':inner,'track':track,**m,**(extra or {})}
            bytrack[track].setdefault(arm,{}).setdefault(seed,[]).append(row)
            key=(track,arm,seed);oof.setdefault(key,np.full(512,np.nan));covered.setdefault(key,np.zeros(512,int));oof[key][data.rows[len(data.train):]]=risk;covered[key][data.rows[len(data.train):]]+=1
            pair=repair_damage(vt,ve,risk,r0);selected.append({k:row[k] for k in ('arm','seed','inner','track','Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den') }|{k:v for k,v in (extra or {}).items() if k!='config'}|{'Harrell_repair':pair['repair_num'],'Harrell_damage':pair['damage_num']})
            cases.append({'arm':arm,'seed':seed,'inner':inner,'track':track,'Harrell_API_error':abs(h-m['Harrell']),'Uno_API_error':abs(u-m['Uno60'])})
        for track in bytrack:score('R0',0,r0,track)
        for seed in (31,47,83):
            finals=list((Path(lock['previous_study_private'])/'fits'/'outer1_final').glob(f'M1_WARM_seed{seed}_lambda*/COMPLETE.private.json'))
            if len(finals)!=1:raise ValueError('EXACT_ONE_ORIGINAL_R1_CHOICE')
            prior=read(finals[0])['job'];old=Path(lock['previous_study_private'])/'fits'/f'outer1_inner{inner}'/f"M1_WARM_seed{seed}_lambda{prior['lambda']}"
            rr=read(old/'COMPLETE.private.json')
            binding=lock['R1_reference_bindings'][f'{seed}/{inner}']
            if (prior['lambda']!=binding['lambda'] or prior['epochs']!=binding['epoch']
                or sha(old/'COMPLETE.private.json')!=binding['receipt_sha256']
                or rr['input']['input_sha256']!=binding['input_sha256']
                or sha(old/'PREDICTIONS.private.npz')!=binding['predictions_sha256']):raise ValueError('ORIGINAL_R1_PREDICTIONS_CHANGED')
            with np.load(old/'PREDICTIONS.private.npz',allow_pickle=False) as z:
                if not np.array_equal(z['rows'],data.rows):raise ValueError('ORIGINAL_R1_ROW_SCOPE')
                prediction=np.array(z[f"{prior['epochs']}_evaluation_risk"],np.float64)
            for track in bytrack:score('R1',seed,prediction,track,{'epoch':prior['epochs'],'new_optimizer_updates':0})
        for arm in ARMS:
            for seed in (31,47,83):
                s=read(private/'selections'/f'outer1_{arm}_seed{seed}.private.json')
                for track,choice in s['choices'].items():
                    job,r=receipts[(arm,seed,json.dumps(choice['config'],sort_keys=True),inner)]
                    node=next(n for n in r['snapshots'] if n['epoch']==choice['epoch'])
                    with np.load(folder(private,job)/'PREDICTIONS.private.npz',allow_pickle=False) as z:
                        risks={k:np.array(z[f"{choice['epoch']}_validation_{k}"],np.float64) for k in node['metrics']['validation']}
                    model=RepairModel(arm,*[b[k].shape[1] for k in ('P','G','C')]).cuda()
                    load_checkpoint(folder(private,job)/node['checkpoint']['file'],private,model)
                    with torch.no_grad():replayed=model(val)['risk'].cpu().numpy()
                    err=float(np.max(np.abs(replayed-risks['risk'])));max_replay=max(max_replay,err);snapshot_replays+=1
                    if err>1e-9:raise ValueError('SELECTED_RETAINED_STATE_REPLAY_FAILED')
                    score(arm,seed,risks['risk'],track,{'epoch':choice['epoch'],'a':choice['config']['a'],'b':choice['config']['b'],'n':choice['config']['n']})
                    for suffix,key in (('NO_MIL','no_mil_risk'),('WRONG_QUERY','wrong_query_risk'),('MEAN_QUERY','mean_query_risk'),('NO_INTERACTION','no_interaction_risk')):
                        if (arm=='R5' and suffix=='NO_MIL') or (arm=='R6' and suffix!='NO_MIL'):score(arm+'_'+suffix,seed,risks[key],track,{'epoch':choice['epoch'],'frozen_intervention':True})
                    if track=='ENCODER_TRAINED':structural.setdefault(arm,{}).setdefault(seed,[]).append(structure_pass(arm,node))
                    for side in ('train','validation'):
                        geometry.append({'arm':arm,'seed':seed,'inner':inner,'track':track,'side':side,'epoch':choice['epoch'],
                            **node['diagnostics'][side],**{'actual_updates_'+k:v for k,v in node['actual_update_counts'].items()},
                            **{'maximum_observed_Cox_gradient_'+k:v for k,v in node['maximum_observed_training_Cox_only_gradients'].items()}})
                    del model;torch.cuda.empty_cache()
        del b,data,raw,val;torch.cuda.empty_cache()
    allowed=set(read(parent['parent_split_plan'])['partitions']['outer1_final']['fit'])
    for key,values in oof.items():
        expected=np.array(sorted(allowed));seen=covered[key]
        if not np.all(seen[expected]==1) or np.any(seen[np.array(sorted(set(range(512))-allowed))]):raise ValueError('409_PATIENT_QUALIFICATION_OOF_COVERAGE')
        np.savez_compressed(privOOF/('_'.join(map(str,key))+'.private.npz'),risk=values[expected],canonical_rows=expected)
    csv_shards(out,'03_SELECTED_INNER_FOLD_METRICS',selected);csv_shards(out,'04_SELECTED_STRUCTURE_AND_GRADIENTS',geometry)
    summaries={track:{arm:[{'seed':seed,**aggregate(rows)} for seed,rows in sorted(seeds.items())] for arm,seeds in arms.items()} for track,arms in bytrack.items()}
    write(out/'05_SEED_METRICS.json',summaries)
    controls={'M1_GROUP':('R2',('R1',)),'M1_FUNCTION':('R3',('R1','R2')),
        'MIL':('R5',('R3','R4')),'MOLECULAR':('R6',('R5','R7','R8','R9','R6_WRONG_QUERY','R6_MEAN_QUERY'))}
    gates={};pairs={}
    for track,values in summaries.items():
        pairs[track]={}
        requested={(arm,'R0') for arm in ARMS}|{(arm,c) for arm,cc in controls.values() for c in cc}|{('R5','R5_NO_MIL'),('R6','R6_NO_INTERACTION')}
        for aa,bb in sorted(requested):pairs[track][aa+' minus '+bb]=comparison(values[aa],values[bb])
    for name,(arm,cc) in controls.items():
        support=[{'seed':seed,'passing_inner_cells':sum(v),'all_three_cell_checks':v} for seed,v in sorted(structural[arm].items())]
        structure=sum(x['passing_inner_cells']>=2 for x in support)>=2
        direction=all(pairs['ENCODER_TRAINED'][arm+' minus '+c]['direction_pass'] for c in cc)
        gates[name]={'passed':structure and direction,'structure_pass':structure,'direction_pass':direction,'structure_support':support,'candidate':arm,'controls':list(cc)}
    decision={'schema':'m123r_development_qualification_decision_v1','status':'COMPLETE_DEVELOPMENT_QUALIFICATION',
        'lock_sha256':a.expected_lock_sha256,'training_complete_sha256':sha(private/'QUALIFICATION_COMPLETE.private.json'),
        'gates':gates,'formal_arms':formal_arms(gates),'M4_M5_trained':False,'CI_bootstrap_pvalues':False,
        'independent_test_evidence':False,'same_inner_results_used_for_selection_and_qualification_optimistic':True,
        'no_posthoc_gate_or_grid_change':True}
    write(out/'06_PAIRED_COMPARISONS.json',pairs);write(out/'decision.json',decision)
    audit={'status':'PASS','registered_real_fits':540,'new_optimizer_updates':updates,'actual_official_API_calls':2*len(cases),
        'scheduler_wall_seconds':read(private/'QUALIFICATION_COMPLETE.private.json')['scheduler_wall_seconds'],
        'worker_elapsed_seconds_sum_not_wall':sum(r['elapsed_seconds'] for _,r in receipts.values()),
        'actual_selected_checkpoint_replays_after_retention':snapshot_replays,'maximum_selected_replay_error':max_replay,
        'all_snapshot_disk_replays_before_retention':540*6,'patient_OOF_vectors_private':len(oof),'OOF_patients_per_vector':409,
        'outer1_test_endpoint_elements_decoded':0,'new_teacher_fits':0,'new_postprocess_optimizer_fits':0,
        'cases':cases,'source_sha256':sha(__file__),'original_R1_reference':'saved_predictions_at_its_original_locked_choice;no_early_weight_replay_claim'}
    write(out/'07_ACTUAL_AUDIT.json',audit)
    table=['# M1–M3 修复：outer1 内部开发资格结果','','本轮只在原 outer1 训练患者内部进行；三内折用于选择及门槛判断，结果不是独立测试证据。没有置信区间、bootstrap 或 p 值。','',
        '|轨道|模型|Harrell|Uno60|','|---|---|---:|---:|']
    for track,arms in summaries.items():
        for arm,rows in arms.items():table.append(f"|{track}|{arm}|{np.mean([r['Harrell'] for r in rows]):.6f}|{np.mean([r['Uno60'] for r in rows]):.6f}|")
    table+=['','所有真实候选均完成140次更新；原始任意快照早停与至少20次真实新增编码器更新的轨道分别保留。达到某个快照不自动等于该结构对风险有效，额外检查实际参数变化、Cox专属梯度、风险关闭干预及注意力几何。','',
        '主效应采用原64维Branch参数化的双精度精确Warm；新模块为单精度。函数保持使用中心化风险平方和，不做风险方差或样本数量归一化。补充组分包括每头Q/K规范化与固定余弦尺度5、匿名正交八查询初始化；未加入强迫分散或强迫错患者风险变化的损失。','',
        '优势是明确激活新增输出路径、保留强初始化、实测更新并提供同容量控制。风险是小事件样本、开发选择乐观、函数保持可能压制真实新信号、量化结构门槛属于工程规则；注意力与slot不是组织学真值。','',
        '未通过门槛的候选不进入正式外折训练；控制模型可能因通过候选的必要比较而保留。R8仍有条件读取，仅移除显式乘积；R9是普通权重共享双向注意力控制，不是任何论文的官方复现。','',
        'R0复用同分区R4；R1复用原M1-Warm固定选择的已保存预测，不冒充新训练或早期权重重放。真正的全五外折验证由门槛决定。','',f"正式阶段允许的模型：{', '.join(decision['formal_arms']) or '无'}。M4/M5未开展。",'']
    (out/'RESULTS.md').write_text('\n'.join(table),encoding='utf-8',newline='')
    files=[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]
    write(out/'REPORT_INDEX.json',{'schema':'m123r_qualification_report_index_v1','status':'COMPLETE','files':files,'lock_sha256':a.expected_lock_sha256})
    print(json.dumps({'status':'COMPLETE','formal_arms':decision['formal_arms'],'gates':{k:v['passed'] for k,v in gates.items()},'actual_official_calls':audit['actual_official_API_calls']}))


if __name__=='__main__':main()
