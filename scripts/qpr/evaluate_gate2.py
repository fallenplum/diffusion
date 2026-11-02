"""Independent official scoring and within-partition paired G2 decisions."""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from src.bgb_en.survival import evaluate,combine_metrics
from src.qpr.protocol import require_stage

ARMS=('B0','B1','B2','B3','B4','B5','B3_QUERY_OFF')


def official(train_time,train_event,time,event,risk):
    from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
    def surv(t,e):return np.array(list(zip(np.asarray(e,bool),np.asarray(t,float))),dtype=[('event','?'),('time','<f8')])
    h=concordance_index_censored(np.asarray(event,bool),time,risk,tied_tol=1e-8)[0]
    u=concordance_index_ipcw(surv(train_time,train_event),surv(time,event),risk,tau=60.,tied_tol=1e-8)[0]
    return float(h),float(u)


def scorer_qualification():
    maximum=0.;cases=0
    for seed in range(8):
        rng=np.random.default_rng(seed)
        tt=np.r_[rng.integers(1,101,80),120.];te=np.r_[rng.integers(0,2,80),0]
        t=rng.integers(1,91,45);e=rng.integers(0,2,45);r=rng.integers(-3,4,45).astype(float)
        own=evaluate(tt,te,t,e,r);h,u=official(tt,te,t,e,r)
        maximum=max(maximum,abs(h-own['Harrell']),abs(u-own['Uno60']));cases+=1
    if maximum>1e-10:raise ValueError('OFFICIAL_SCORER_DISAGREEMENT')
    return {'synthetic_cases_with_event_time_and_risk_ties':cases,'maximum_absolute_difference':maximum,'status':'PASS'}


def run(a):
    require_stage('gate2',a.reports);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    locksha=hashlib.sha256(Path(a.lock).read_bytes()).hexdigest();rows=[];official_max=0.;repairs=[];cost=[]
    qualification=scorer_qualification()
    for inner in (1,2,3):
        with (Path(a.ram_root)/f'qpr_G2_inner{inner}_20261102.private.pkl').open('rb') as f:data=pickle.load(f)
        args=[data['train_time'],data['train_event'],data['evaluation_time'],data['evaluation_event']]
        base=data['r0_evaluation']
        for seed in (31,47):
            risks={'B0':base}
            for arm in ARMS[1:]:
                cell=Path(a.source)/f'inner{inner}_{arm}_seed{seed}';meta=json.loads((cell/'COMPLETE.private.json').read_text())
                if meta['formal_lock_sha256']!=locksha or meta['optimizer_updates']!=80:raise ValueError('INCOMPLETE_OR_DIFFERENT_G2_RUN')
                epoch=meta['selected_epoch'];pred=np.load(cell/f'epoch{epoch:03d}_predictions.private.npz')
                risk=np.asarray(pred['evaluation_risk']);risks[arm]=risk
                for snap in meta['snapshots']:
                    if snap['replay_max_abs']>1e-10:raise ValueError('CHECKPOINT_REPLAY_FAILURE')
                cost.append({'inner':inner,'seed':seed,'arm':arm,'selected_epoch':epoch,'epochs':80,
                    'total_parameters':meta['total_parameters'],'effective_parameters':meta['effective_parameters'],
                    'query_updates':meta['actual_parameter_updates']['query'],'head_updates':meta['actual_parameter_updates']['head'],
                    'elapsed_seconds':meta['elapsed_seconds'],'peak_cuda_bytes':meta['peak_cuda_bytes'],
                    'selected_train_delta_SD':meta['selected']['train_delta_SD'],'selected_validation_delta_SD':meta['selected']['validation_delta_SD'],
                    'wrong_query_mapping':meta['wrong_query_mapping']})
            for arm,risk in risks.items():
                own=evaluate(*args,risk,base);h,u=official(*args,risk)
                error=max(abs(own['Harrell']-h),abs(own['Uno60']-u));official_max=max(official_max,error)
                if error>1e-10:raise ValueError('OFFICIAL_G2_SCORE_DISAGREEMENT')
                rows.append({'inner':inner,'seed':seed,'arm':arm,**own,'official_Harrell':h,'official_Uno60':u})
            for control in ('B0','B1','B2','B4','B5','B3_QUERY_OFF'):
                m=evaluate(*args,risks['B3'],risks[control]);ref=evaluate(*args,risks[control])
                if abs(m['repair']-m['damage']-(m['Harrell']-ref['Harrell']))>1e-10:raise ValueError('PAIRED_REPAIR_ACCOUNTING')
                repairs.append({'inner':inner,'seed':seed,'comparison':'B3-'+control,
                                'delta_Harrell':m['Harrell']-ref['Harrell'],'delta_Uno60':m['Uno60']-ref['Uno60'],
                                **{k:m[k] for k in ('repair','damage','Uno_repair','Uno_damage','risk_shift_sd_ratio')}})
    byseed={(s,arm):combine_metrics([r for r in rows if r['seed']==s and r['arm']==arm]) for s in (31,47) for arm in ARMS}
    mean={arm:{metric:float(np.mean([byseed[s,arm][metric] for s in (31,47)])) for metric in ('Harrell','Uno60')} for arm in ARMS}
    diffs={arm:{k:mean['B3'][k]-mean[arm][k] for k in ('Harrell','Uno60')} for arm in ARMS if arm!='B3'}
    innerpos=sum(np.mean([r['delta_Harrell'] for r in repairs if r['comparison']=='B3-B0' and r['inner']==i])>0 for i in (1,2,3))
    seedpos=sum(byseed[s,'B3']['Harrell']>byseed[s,'B0']['Harrell'] for s in (31,47))
    basepass=diffs['B0']['Harrell']>=.005 and diffs['B0']['Uno60']>=.003
    controlpass=all(diffs[x]['Harrell']>0 and diffs[x]['Uno60']>=0 for x in ('B2','B4','B5'))
    specificity=diffs['B1']['Harrell']>0 and diffs['B3_QUERY_OFF']['Harrell']>0
    updatepass=all(c['query_updates']==60 and c['head_updates']>=79 for c in cost if c['arm'] not in ('B1',))
    passed=basepass and controlpass and specificity and innerpos>=2 and seedpos==2 and updatepass
    status='RELATION_GO' if passed else ('RELATION_NO_GO' if not basepass else ('NO_PATIENT_RNA_QUERY_EVIDENCE' if not specificity or diffs['B5']['Harrell']<=0 else 'NO_TOPOLOGY_SPECIFIC_INCREMENT'))
    decision={'relation_go':bool(passed),'status':status,'role':'INTERNAL_OUTER1_TRAIN_SIDE_DEVELOPMENT_ONLY',
        'mean_metrics':mean,'paired_mean_differences_B3_minus':diffs,'positive_inner_splits':int(innerpos),'positive_seeds':int(seedpos),
        'actual_training_updates_qualified':bool(updatepass),'formal_neural_fits':36,'formal_neural_optimizer_steps':2880,
        'fresh_anchor_Cox_fits':156,'fresh_scalar_anchor_fits':12,'outer_test_labels_decoded':0,
        'official_scorer_maximum_absolute_difference':official_max,'scorer_qualification':qualification,
        'no_CI_bootstrap_pvalues':True,'gate3_allowed':False}
    for filename,items in [('fold_metrics.csv',rows),('paired_repair_damage.csv',repairs)]:
        with (out/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    (out/'TRAINING_COST.json').write_text(json.dumps(cost,indent=2)+'\n')
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(decision))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('source','output','reports','lock','ram-root'):p.add_argument('--'+x,required=True)
    run(p.parse_args())
