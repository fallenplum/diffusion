"""Raw private OOF evaluation; imports NO training features/math/metrics."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from validate_phase11_independent import pairs,ipcw,components
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/phase12_c4_confirmation';PRIVATE=Path('/root/phase12_c4_private/formal')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def check_manifest(m,d,fold,spec):
    tr=np.asarray(m['outer_train'],int);te=np.asarray(m['outer_test'],int);it=np.asarray(m['inner_train'],int);iv=np.asarray(m['inner_val'],int)
    if not np.array_equal(tr,np.flatnonzero(d['folds']!=fold)) or not np.array_equal(te,np.flatnonzero(d['folds']==fold)) or set(it)&set(iv) or set(it)|set(iv)!=set(tr):raise ValueError('patient/split isolation')
    if m['patient_hash']!=ahash(d['patient_ids']) or m['fold_hash']!=ahash(d['folds']):raise ValueError('patient/fold hash')
    for role,fit in [('inner',it),('outer',tr)]:
        a=m['preprocess_'+role]
        if a['fit_indices']!=fit.tolist() or a['fit_hash']!=ahash(fit):raise ValueError('global/scaler/TopK fit')
        if a['pca']['fit_hash']!=ahash(fit):raise ValueError('global PCA')
        if set(a['source_fields'])!=set(['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE']) or a['selection_source']!='RNA expression only; no outcome':raise ValueError('forbidden outcome alias')
        extra=a['phase12']
        if extra['selection_source']!='frozen annotation + TRAIN observation>=.9 and variance; NO survival labels':raise ValueError('outcome selected enriched genes')
        for mode in ['enriched','matched']:
            ix=np.asarray(extra[mode+'_indices'],np.int64)
            if len(ix)!=512 or len(set(ix))!=512 or ahash(ix)!=extra[mode+'_hash']:raise ValueError('gene identity/hash')
    if not spec.get('reuse') and spec['kind'] not in ['ridgeRNA','ridge']:
        mapping=m['effective_modality_mapping']
        for name,ix in [('inner_train',it),('inner_val',iv),('outer_train',tr),('outer_test',te)]:
            source=np.array(mapping[name]['RNA'],int)
            if len(source)!=len(ix) or set(source)!=set(ix):raise ValueError('cross-split RNA')
            if spec['kind']=='wrong' and np.any(source==ix):raise ValueError('wrong RNA fixed point')
            if spec['kind']!='wrong' and not np.array_equal(source,ix):raise ValueError('real RNA wrong patient')
    return tr,te,it,iv


def check_record(cell,z,m,d,arm,seed,fold,spec,source,protocol):
    if cell['status']!='COMPLETE' or cell['arm']!=arm or cell['seed']!=seed or cell['fold']!=fold or str(z['arm'])!=arm or int(z['seed'])!=seed or int(z['fold'])!=fold:raise ValueError('arm/seed/fold container identity')
    if cell['source_hash']!=source or cell['protocol_sha256']!=protocol or cell['spec']!=spec or cell['outer_test_selection'] is not False:raise ValueError('source/protocol/test selection')
    tr,te,it,iv=check_manifest(m,d,fold,spec)
    if not np.array_equal(z['test_indices'],te) or not np.array_equal(z['time'],d['DSS_time'][te]) or not np.array_equal(z['event'],d['DSS_event'][te]) or not np.isfinite(z['risk']).all():raise ValueError('OOF patient/label/direction/finite')
    if cell.get('actual_fit') and 'outer' in cell and (cell['outer']['actual_epochs']!=cell['selected_epoch'] or cell['inner']['actual_epochs']!=80):raise ValueError('training schedule/refit mismatch')
    return tr,te


def validate(arms=None,statistics=False):
    cfg=json.loads((ROOT/'configs/phase12_c4_confirmation.json').read_text());gate=json.loads((REPORT/'HARD_GATES.json').read_text());d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));arms=list(cfg['arms']) if arms is None else arms
    t=d['DSS_time'];e=d['DSS_event'];folds=d['folds'];weights=np.zeros(len(t));rows=[];failure=[];risks={}
    from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
    from sksurv.util import Surv
    for fold in cfg['folds']:weights[folds==fold]=ipcw(t[folds!=fold],e[folds!=fold],t[folds==fold],e[folds==fold])
    for arm in arms:
        for seed in cfg['seeds']:
            risk=np.full(len(t),np.nan)
            for fold in cfg['folds']:
                stem=f'{arm}_seed{seed}_fold{fold}'
                try:
                    cp=PRIVATE/'cells'/f'{stem}.json';pp=PRIVATE/'oof'/f'{stem}.npz';mp=PRIVATE/'manifests'/f'{stem}.json';cell=json.loads(cp.read_text());z=dict(np.load(pp));m=json.loads(mp.read_text());spec=cfg['arms'][arm]
                    tr,te=check_record(cell,z,m,d,arm,seed,fold,spec,gate['source_hash'],digest(ROOT/'configs/phase12_c4_confirmation.json'))
                    if digest(pp)!=cell['prediction_sha256'] or digest(mp)!=cell['manifest_sha256']:raise ValueError('artifact changed')
                    if spec.get('reuse'):
                        parent=Path('/root/phase11_semantic_anchor_private/formal');original=spec['reuse'];pz=parent/'oof'/f'{original}_seed{seed}_fold{fold}.npz'
                        if not np.array_equal(np.load(pz)['risk'],z['risk']) or digest(pz)!=cell['parent_prediction_sha256']:raise ValueError('parent not exactly reused')
                    risk[te]=z['risk'];hn,hd=components(t[te],e[te],risk[te]);un,ud=components(t[te],e[te],risk[te],weights[te])
                    sh=concordance_index_censored(e[te].astype(bool),t[te],risk[te])[0];su=concordance_index_ipcw(Surv.from_arrays(e[tr].astype(bool),t[tr]),Surv.from_arrays(e[te].astype(bool),t[te]),risk[te],tau=60)[0]
                    if abs(sh-hn/hd)>1e-10 or abs(su-un/ud)>1e-10:raise ValueError('independent/sksurv mismatch')
                    rows.append({'arm':arm,'seed':seed,'fold':fold,'Harrell':hn/hd,'Uno60':un/ud,'Harrell_num':hn,'Harrell_den':hd,'Uno_num':un,'Uno_den':ud,'test_n':len(te),'test_events':int(e[te].sum()),'selected_epoch':cell['selected_epoch'],'parameters':cell['parameters'],'actual_fit':cell['actual_fit'],'seconds':cell['seconds']})
                except Exception as ex:failure.append({'arm':arm,'seed':seed,'fold':fold,'error':str(ex)})
            if np.isfinite(risk).all():risks[arm,seed]=risk
    status={'status':'FAIL' if failure else 'PASS','expected_records':len(arms)*15,'verified_records':len(rows),'failures':failure,'method':'independent event pairs/reverseKM vs scikit-survival0.23.1'}
    tag='ALL' if len(arms)==len(cfg['arms']) else '_'.join(arms);write=REPORT/('ALL_INDEPENDENT_VALIDATION.json' if statistics else 'STAGE_'+tag+'_VALIDATION.json');write.write_text(json.dumps(status,indent=2));pd.DataFrame(rows).to_csv(REPORT/('FOLD_METRICS.csv' if statistics else 'STAGE_'+tag+'_FOLDS.csv'),index=False)
    if failure:print(json.dumps(status));return 2
    if statistics:
        seeds=[]
        for (arm,seed),risk in risks.items():
            cells=[r for r in rows if r['arm']==arm and r['seed']==seed];sums={k:sum(r[k] for r in cells) for k in ['Harrell_num','Harrell_den','Uno_num','Uno_den']}
            seeds.append({'arm':arm,'seed':seed,'Harrell':sums['Harrell_num']/sums['Harrell_den'],'Uno60':sums['Uno_num']/sums['Uno_den'],**sums})
        pd.DataFrame(seeds).to_csv(REPORT/'SEED_OOF_METRICS.csv',index=False);bootstrap(cfg,d,weights,risks,seeds)
    print(json.dumps(status));return 0


def bootstrap(cfg,d,weights,risks,seedrows):
    t=d['DSS_time'];e=d['DSS_event'];f=d['folds'];pi,pj=pairs(t,e);keep=f[pi]==f[pj];pi,pj=pi[keep],pj[keep]
    contrasts={**cfg['primary'],**cfg['secondary']};arm_credit={}
    for (arm,seed),risk in risks.items():
        delta=risk[pi]-risk[pj];arm_credit[arm,seed]=np.where(abs(delta)<=1e-8,.5,delta>0)
    credits=[]
    for name,terms in contrasts.items():
        if abs(sum(c for _,c in terms))>1e-12:raise RuntimeError('contrast not zero-sum')
        credits.append(np.mean([sum(c*arm_credit[arm,seed] for arm,c in terms) for seed in cfg['seeds']],axis=0))
    credits=np.stack(credits,axis=1);rng=np.random.default_rng(cfg['bootstrap']['seed']);hash_counts=hashlib.sha256();values={m:[] for m in ['Harrell','Uno60']}
    for start in range(0,cfg['bootstrap']['draws'],250):
        counts=rng.multinomial(len(t),np.ones(len(t))/len(t),size=min(250,cfg['bootstrap']['draws']-start));hash_counts.update(counts.tobytes());q=(counts[:,pi]*counts[:,pj]).astype(float);qu=q*weights[pi]
        values['Harrell'].append((q@credits)/q.sum(1)[:,None]);values['Uno60'].append((qu@credits)/qu.sum(1)[:,None])
    from scipy.stats import binom
    output=[]
    for metric,arrays in values.items():
        samples=np.concatenate(arrays)
        for j,(name,terms) in enumerate(contrasts.items()):
            delta=[sum(c*next(r[metric] for r in seedrows if r['arm']==arm and r['seed']==seed) for arm,c in terms) for seed in cfg['seeds']];primary=name in cfg['primary'];alpha=.05/12 if primary else .05
            ordinary=np.quantile(samples[:,j],[.025,.975]);corrected=np.quantile(samples[:,j],[alpha/2,1-alpha/2]);sorted_vals=np.sort(samples[:,j]);mc={}
            for label,prob in [('low',alpha/2),('high',1-alpha/2)]:
                ranks=np.clip(binom.ppf([.025,.975],len(sorted_vals),prob).astype(int),0,len(sorted_vals)-1);mc['sim_'+label+'_MC95_low']=sorted_vals[ranks[0]];mc['sim_'+label+'_MC95_high']=sorted_vals[ranks[1]]
            output.append({'comparison':name,'metric':metric,'family':'primary' if primary else 'exploratory_secondary','mean_delta':np.mean(delta),**{'seed'+str(seed)+'_delta':v for seed,v in zip(cfg['seeds'],delta)},'positive_seeds':sum(v>0 for v in delta),'CI95_low':ordinary[0],'CI95_high':ordinary[1],'simultaneous_low':corrected[0],'simultaneous_high':corrected[1],'draws':len(samples),**mc})
    frame=pd.DataFrame(output);frame.to_csv(REPORT/'ALL_PAIRED.csv',index=False);frame[frame.family=='primary'].to_csv(REPORT/'PRIMARY_PAIRED.csv',index=False)
    (REPORT/'BOOTSTRAP_AUDIT.json').write_text(json.dumps({'draws':cfg['bootstrap']['draws'],'seed':cfg['bootstrap']['seed'],'counts_sha256':hash_counts.hexdigest(),'primary_family_tests':12,'extreme_tail_expected_draws':cfg['bootstrap']['draws']*.05/24,'shared_patient_resampling_all_arms_seeds_groups':True,'DID_is_within_patient_linear_contrast':True,'seed_or_group_as_independent_patients':False,'risk_averaging':False,'retraining_variance_included':False,'IPCW_training_weights_fixed':True},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arms');ap.add_argument('--statistics',action='store_true');a=ap.parse_args();raise SystemExit(validate(a.arms.split(',') if a.arms else None,a.statistics))
