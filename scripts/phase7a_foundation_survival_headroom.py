"""Locked, patient-level frozen-representation survival headroom audit."""
import argparse
import json
import re
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedShuffleSplit
from phase7_p0_biology_prompt_audit import (REPO, BASE, CACHE, digest, ahash, write_json,
    check_lock, gene_z, raw_top512, group_scores, zscale)

sys.path.insert(0, str(REPO))
from src.phase_m2.data import _load_patient_table, _merge_clinical, _status, build_h5_index

CONFIG = REPO / 'configs/phase7a_foundation_survival_headroom.json'
SELF = REPO / 'scripts/phase7a_foundation_survival_headroom.py'
DEFAULT = Path('/root/phase7a_foundation_headroom_v1')
ROOT = Path('/root/autodl-tmp/diffusion_data')
CAT = ['SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 'PATH_T_STAGE', 'PATH_N_STAGE', 'PATH_M_STAGE']
BLOCKS = {'C':['C'], 'R':['R'], 'U':['U'], 'T':['T'], 'S':['S'], 'M':['M'], 'MC':['MC'],
          'CR':['C','R'], 'CU':['C','U'], 'CT':['C','T'], 'CS':['C','S'], 'CM':['C','M'],
          'CMC':['C','MC'], 'CRU':['C','R','U'], 'CRT':['C','R','T'], 'CRS':['C','R','S'], 'CRMC':['C','R','MC']}


def pool_patient(task):
    pid, paths = task; slides = []; provenance = []
    for path in paths:
        path = Path(path)
        with h5py.File(path, 'r') as f:
            a = np.asarray(f['features'], dtype=np.float32).reshape(-1, 1536)
        assert len(a) and np.isfinite(a).all()
        slides.append(a.mean(0, dtype=np.float64))
        provenance.append({'path':str(path), 'sha256':digest(path), 'patches':len(a), 'size':path.stat().st_size})
    return pid, np.mean(slides,axis=0), provenance


def cache(out):
    out.mkdir(parents=True,exist_ok=True)
    parent = Path(json.loads(CONFIG.read_text())['parent_run']); check_lock(parent)
    d = np.load(parent/'private_dataset.npz'); ids = d['patient_ids'].astype(str)
    target = out/'uni2_patient_means.npz'
    if target.exists():
        a=np.load(target); assert np.array_equal(a['patient_ids'],ids)
        print('UNI2_CACHE_ALREADY_COMPLETE', flush=True); return
    index = build_h5_index(ROOT, ids.tolist()); results={}; provenance=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for pid, mean, records in pool.map(pool_patient, index.items()):
            results[pid]=mean; provenance.extend(records)
            if len(results)%64==0: print('UNI2_PATIENTS',len(results),flush=True)
    a=np.stack([results[p] for p in ids]); assert a.shape==(512,1536) and np.isfinite(a).all()
    np.savez_compressed(target,patient_ids=ids,features=a)
    write_json(out/'private_uni2_provenance.json',provenance)
    write_json(out/'uni2_cache_audit.json',{'status':'PASS','patients':512,'slides':len(provenance),
        'patches':sum(x['patches'] for x in provenance),'dimension':1536,'feature_sha256':ahash(a),
        'source_provenance_sha256':digest(out/'private_uni2_provenance.json'),
        'pooling':'all patches mean per slide; equal slide mean per patient','labels_used':False})
    print('UNI2_CACHE_DONE',flush=True)


def efron_matrix(time,event):
    rows=[]
    for t in np.unique(time[event==1]):
        die=(time==t)&(event==1); count=int(die.sum()); risk=(time>=t).astype(float)
        for l in range(count): rows.append(risk-(l/count)*die)
    return np.stack(rows)


def fit_cox(x,time,event,lam):
    a=efron_matrix(time,event); n=int(event.sum()); event_x=x[event==1].sum(0)
    def objective(w):
        risk=x@w; shift=risk.max(); exp=np.exp(risk-shift); den=a@exp
        assert np.all(den>0)
        loss=((np.log(den)+shift).sum()-risk[event==1].sum())/n + .5*lam*(w@w)
        grad=(x.T@(exp*(a.T@(1/den)))-event_x)/n+lam*w
        return float(loss),grad
    result=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                    options={'maxiter':5000,'ftol':1e-13,'gtol':1e-7,'maxls':50})
    loss,gradient=objective(result.x); g=float(np.max(np.abs(gradient)))
    assert np.isfinite(loss) and g<=1e-5, f'Cox not converged: {result.message}; gradient={g}'
    return result.x,{'iterations':int(result.nit),'gradient_max':g,'objective':loss,'optimizer_message':str(result.message)}


def km_left(train_time,train_event,query):
    unique=np.unique(train_time); surv=[]; value=1.
    for t in unique:
        n=(train_time>=t).sum()-((train_time==t)&(train_event==1)).sum()
        censor=((train_time==t)&(train_event==0)).sum()
        value*=1-censor/n if censor else 1.; surv.append(value)
    pos=np.searchsorted(unique,query,side='left')-1
    return np.where(pos>=0,np.asarray(surv)[np.maximum(pos,0)],1.)


def concordance(time,event,risk,weights=None,folds=None):
    eligible=(event[:,None]==1)&((time[:,None]<time[None,:])|((time[:,None]==time[None,:])&(event[None,:]==0)))
    if folds is not None: eligible &= folds[:,None]==folds[None,:]
    weight=np.ones(len(time)) if weights is None else weights
    pairs=eligible*weight[:,None]
    credit=(risk[:,None]>risk[None,:]).astype(float)+.5*(risk[:,None]==risk[None,:])
    return float((pairs*credit).sum()/pairs.sum())


def mathematical_check():
    # Independent slow definition with tied event times; finite differences also
    # validate analytic gradients used by the actual convex optimizer.
    rng=np.random.default_rng(341); x=rng.normal(size=(11,7));t=np.array([1,1,2,2,2,3,4,4,5,6,7.]);e=np.array([1,1,1,0,1,0,1,0,0,1,0])
    w=rng.normal(size=7)*.1; a=efron_matrix(t,e);r=x@w;exp=np.exp(r)
    fast=np.log(a@exp).sum()-r[e==1].sum(); slow=0.
    for tt in np.unique(t[e==1]):
        die=(t==tt)&(e==1); count=die.sum();sr=np.exp(r[t>=tt]).sum();sd=np.exp(r[die]).sum()
        slow+=sum(np.log(sr-l/count*sd) for l in range(count))-r[die].sum()
    analytic=x.T@(exp*(a.T@(1/(a@exp))))-x[e==1].sum(0)
    def f(v): rr=x@v;return np.log(a@np.exp(rr)).sum()-rr[e==1].sum()
    numerical=np.asarray([(f(w+np.eye(7)[j]*1e-6)-f(w-np.eye(7)[j]*1e-6))/2e-6 for j in range(7)])
    assert abs(fast-slow)<1e-12 and np.max(np.abs(analytic-numerical))<1e-7
    w2,details=fit_cox(x,t,e,.1)
    return {'status':'PASS','tied_efron_loss_error':abs(fast-slow),'gradient_finite_difference_error':float(np.max(np.abs(analytic-numerical))), 'optimizer_check':details}


def prepare(out):
    p=json.loads(CONFIG.read_text());parent=Path(p['parent_run']);pm=check_lock(parent)
    assert json.loads((Path(p['p0r_run'])/'DONE.json').read_text())['status']=='PASS'
    assert json.loads((Path(p['p0r_run'])/'independent_validation.json').read_text())['status']=='PASS'
    assert not (out/'manifest.json').exists()
    d=dict(np.load(parent/'private_dataset.npz'));ids=d['patient_ids'].astype(str)
    clinical=_load_patient_table(BASE/'data_clinical_patient.txt',ids.tolist())
    sup_path=BASE.parent/'coadread_tcga'/'data_clinical_patient.txt'
    sup=_load_patient_table(sup_path,ids.tolist());cov=_merge_clinical(clinical,sup,ids.tolist())
    for k in ['AGE',*CAT]:d['clinical_'+k]=cov[k].fillna('UNKNOWN').astype(str).to_numpy(dtype='U80')
    d['cancer_type']=clinical['CANCER_TYPE_ACRONYM'].fillna('UNKNOWN').to_numpy(dtype='U20')
    labelaudit={}
    for ep in ['DSS','PFS']:
        t=pd.to_numeric(clinical[ep+'_MONTHS'],errors='coerce').to_numpy(float);e=clinical[ep+'_STATUS'].map(_status).to_numpy(float)
        valid=np.isfinite(t)&(t>0)&np.isin(e,[0,1]);d[ep+'_time']=t;d[ep+'_event']=e;d[ep+'_valid']=valid
        labelaudit[ep]={'patients':int(valid.sum()),'events':int(e[valid].sum()),'excluded_invalid_time_or_status':int((~valid).sum()),
            'source_columns':[ep+'_MONTHS',ep+'_STATUS'],'status_dictionary':clinical[ep+'_STATUS'].value_counts().to_dict()}
        for fold in range(1,6):
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold))
            g=km_left(t[tr],e[tr],t[te]);active=(e[te]==1)&(t[te]<=60)
            assert np.all(g[active]>0),'IPCW60 has no censoring support'
    native=np.load(CACHE/'patient_representations.npz');assert np.array_equal(ids,native['patient_ids'].astype(str))
    uni=np.load(out/'uni2_patient_means.npz');assert np.array_equal(ids,uni['patient_ids'].astype(str))
    d['titan']=native['native'].astype(float);d['uni2']=uni['features'].astype(float)
    assert np.allclose(d['static'],native['concept_cosine'],rtol=0,atol=0)
    np.savez_compressed(out/'private_dataset.npz',**d)
    write_json(out/'protocol.json',p)
    sources=[CONFIG,SELF,REPO/'scripts/phase7_p0_biology_prompt_audit.py',REPO/'src/phase_m2/data.py',
        BASE/'data_clinical_patient.txt',sup_path,CACHE/'patient_representations.npz',out/'uni2_patient_means.npz',
        parent/'private_dataset.npz',REPO/'outputs/phase_m1r/fixed_patient_folds.csv',Path(p['p0r_run'])/'decision.json']
    m={'protocol_sha256':digest(out/'protocol.json'),'locked_utc':pd.Timestamp.now(tz='UTC').isoformat(),
       'sources':[{'path':str(f),'sha256':digest(f)} for f in sources], 'private_dataset_sha256':digest(out/'private_dataset.npz'),
       'parent_protocol_sha256':pm['protocol_sha256'],'groups':pm['groups'],'label_audit':labelaudit,
       'order':'P0R independently validated before survival preparation'}
    write_json(out/'manifest.json',m);write_json(out/'endpoint_source_audit.json',labelaudit)
    write_json(out/'mathematical_checks.json',mathematical_check())
    for sub in ['oof','cells','preprocessing','coefficients']:(out/sub).mkdir(exist_ok=True)
    print('PHASE7A_LOCKED',labelaudit,flush=True)


def locked(out):
    m=json.loads((out/'manifest.json').read_text())
    assert digest(out/'protocol.json')==m['protocol_sha256'] and digest(out/'private_dataset.npz')==m['private_dataset_sha256']
    for s in m['sources']:assert digest(s['path'])==s['sha256']
    return m


def features(d,idx,groups):
    result={};audit={'fit_indices_sha256':ahash(idx),'fit_n':len(idx)};private={'fit_idx':idx}
    age=pd.to_numeric(pd.Series(d['clinical_AGE']),errors='coerce').to_numpy(float)
    med=float(np.nanmedian(age[idx]));age=np.where(np.isfinite(age),age,med)[:,None];clinical=[age]
    levels={}
    for k in CAT:
        vals=d['clinical_'+k];choices=np.unique(vals[idx]);levels[k]=choices.tolist()
        clinical.append(np.stack([vals==v for v in choices],axis=1).astype(float))
    result['C']=np.concatenate(clinical,axis=1);audit['clinical_levels']=levels;audit['age_median']=med
    result['R'],keep,stats=raw_top512(d['raw_rna'],idx);private['raw_top512']=keep
    audit['raw_top512_sha256']=ahash(keep);audit['rna_eligible_genes']=int(stats['usable'].sum());audit['RNA_selection_before_zscore']=True
    rz,rs=gene_z(d['rna_genes'],idx);cz,cs=gene_z(d['cna_genes'],idx)
    result['M'],rsets=group_scores(rz,d['rna_gene_symbols'],groups,rs['usable'])
    cn,csets=group_scores(cz,d['cna_gene_symbols'],groups,cs['usable']);result['MC']=np.concatenate([result['M'],cn],axis=1)
    audit['molecular_gene_counts']={'rna':[len(i) for i in rsets],'cna':[len(i) for i in csets]}
    result.update(U=d['uni2'],T=d['titan'],S=d['static'])
    for k in result:
        result[k],mean,sd=zscale(result[k],idx);private[k+'_mean']=mean;private[k+'_sd']=sd
        assert np.isfinite(result[k]).all()
    audit['dimensions']={k:v.shape[1] for k,v in result.items()}
    return result,audit,private


def worker(out,worker_id):
    m=locked(out);p=json.loads((out/'protocol.json').read_text());d=dict(np.load(out/'private_dataset.npz'))
    ep=['DSS','PFS'][worker_id//3];seed=p['seeds'][worker_id%3];t=d[ep+'_time'];e=d[ep+'_event'];valid=d[ep+'_valid']
    for fold in range(1,6):
        tag=f'{ep}_seed{seed}_fold{fold}';tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold))
        strat=np.asarray([str(c)+'_'+str(int(ev)) for c,ev in zip(d['cancer_type'][tr],e[tr])]);_,cnt=np.unique(strat,return_counts=True)
        if cnt.min()<2:strat=e[tr]
        a,b=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=seed+fold*100).split(tr,strat));it=tr[a];iv=tr[b]
        assert not set(tr)&set(te) and not set(it)&set(iv) and set(it)|set(iv)==set(tr)
        fi,ai,pi=features(d,it,m['groups']);fo,ao,po=features(d,tr,m['groups'])
        np.savez_compressed(out/'preprocessing'/f'{tag}_inner.npz',**pi,tr=tr,te=te,it=it,iv=iv)
        np.savez_compressed(out/'preprocessing'/f'{tag}_outer.npz',**po,tr=tr,te=te,it=it,iv=iv)
        for method in p['methods']:
            celltag=tag+'_'+method;cp=out/'cells'/f'{celltag}.json';op=out/'oof'/f'{celltag}.npz'
            if cp.exists() and op.exists():continue
            xi=np.concatenate([fi[k] for k in BLOCKS[method]],axis=1);xo=np.concatenate([fo[k] for k in BLOCKS[method]],axis=1)
            trials=[]
            for lam in p['ridge_grid']:
                w,detail=fit_cox(xi[it],t[it],e[it],lam);c=concordance(t[iv],e[iv],xi[iv]@w)
                trials.append({'ridge':lam,'inner_harrell':c,**detail})
            chosen=max(trials,key=lambda a:(a['inner_harrell'],a['ridge']))
            w,detail=fit_cox(xo[tr],t[tr],e[tr],chosen['ridge']);trainrisk=xo[tr]@w;raw=xo[te]@w
            mean=float(trainrisk.mean());sd=float(trainrisk.std());assert sd>1e-10
            pred=(raw-mean)/sd;g=km_left(t[tr],e[tr],t[te]);active=(e[te]==1)&(t[te]<=60);assert np.all(g[active]>0)
            weights=np.zeros(len(te));weights[active]=1/g[active]**2
            np.savez_compressed(op,test_indices=te,patient_ids=d['patient_ids'][te],time=t[te],event=e[te],risk=pred,raw_risk=raw,
                censor_g=g,uno_weight=weights,seed=seed,fold=fold,endpoint=ep,method=method,protocol_sha256=m['protocol_sha256'])
            np.savez_compressed(out/'coefficients'/f'{celltag}.npz',coefficients=w,training_risk_mean=mean,training_risk_sd=sd)
            write_json(cp,{'status':'PASS','endpoint':ep,'seed':seed,'fold':fold,'method':method,'dimension':xo.shape[1],
                'parameter_count':len(w),'selected_ridge':chosen['ridge'],'trials':trials,'outer_fit':detail,
                'inner_preprocessing':ai,'outer_preprocessing':ao,'outer_test_used_for_selection':False,'stacked_predictions':False,
                'n_train':len(tr),'events_train':int(e[tr].sum()),'n_test':len(te),'events_test':int(e[te].sum()),
                'train_risk_mean':mean,'train_risk_sd':sd,'min_censor_g_active':float(g[active].min()),
                'oof_sha256':digest(op),'protocol_sha256':m['protocol_sha256']})
        print('SURVIVAL_FOLD_DONE',ep,seed,fold,flush=True)
    locked(out);write_json(out/f'worker_{worker_id}_DONE.json',{'status':'PASS'})


def aggregate(out):
    m=locked(out);p=json.loads((out/'protocol.json').read_text());d=dict(np.load(out/'private_dataset.npz'));rows=[];pairrows=[];decisions={}
    cells=[json.loads(f.read_text()) for f in (out/'cells').glob('*.json')];assert len(cells)==510
    for w in range(6):assert json.loads((out/f'worker_{w}_DONE.json').read_text())['status']=='PASS'
    assert all(c['status']=='PASS' and c['outer_fit']['gradient_max']<=1e-5 for c in cells)
    for ep in ['DSS','PFS']:
        idx=np.flatnonzero(d[ep+'_valid']);n=len(idx);t=d[ep+'_time'][idx];e=d[ep+'_event'][idx];folds=d['folds'][idx]
        risk={};raw={};weight={}
        for seed in p['seeds']:
            for method in p['methods']:
                full=np.full(512,np.nan);rr=full.copy();uw=full.copy();count=np.zeros(512,int)
                for fold in range(1,6):
                    tag=f'{ep}_seed{seed}_fold{fold}_{method}';op=out/'oof'/f'{tag}.npz';a=np.load(op)
                    assert digest(op)==json.loads((out/'cells'/f'{tag}.json').read_text())['oof_sha256']
                    te=a['test_indices'];assert np.array_equal(a['patient_ids'],d['patient_ids'][te]);assert np.all(d['folds'][te]==fold)
                    full[te]=a['risk'];rr[te]=a['raw_risk'];uw[te]=a['uno_weight'];count[te]+=1
                assert np.all(count[idx]==1) and np.all(count[~d[ep+'_valid']]==0) and np.isfinite(full[idx]).all()
                risk[seed,method]=full[idx];raw[seed,method]=rr[idx];weight[seed]=uw[idx]
                rows.append({'endpoint':ep,'seed':seed,'method':method,'patients':n,'events':int(e.sum()),
                    'oof_harrell':concordance(t,e,full[idx]),'oof_uno60':concordance(t,e,full[idx],uw[idx]),
                    'raw_oof_harrell':concordance(t,e,rr[idx]),'within_fold_pair_weighted_harrell':concordance(t,e,full[idx],folds=folds),
                    'within_fold_pair_weighted_uno60':concordance(t,e,full[idx],uw[idx],folds)})
        # Same stratified multiplicity draws for every paired comparison.
        rng=np.random.default_rng(8762 if ep=='DSS' else 8763);wboot=np.zeros((2000,n))
        for f in range(1,6):
            for ev in [0,1]:
                group=np.flatnonzero((folds==f)&(e==ev));draw=rng.multinomial(len(group),np.ones(len(group))/len(group),size=2000)
                wboot[:,group]=draw
        eligible=(e[:,None]==1)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&(e[None,:]==0)));distributions={}
        for method in p['methods']:
            for metric in ['harrell','uno60']:
                vals=[]
                for seed in p['seeds']:
                    r=risk[seed,method];cred=(r[:,None]>r[None,:]).astype(float)+.5*(r[:,None]==r[None,:])
                    den=eligible.astype(float)*(np.ones(n) if metric=='harrell' else weight[seed])[:,None]
                    num=den*cred
                    vals.append(np.sum((wboot@num)*wboot,axis=1)/np.sum((wboot@den)*wboot,axis=1))
                distributions[method,metric]=np.mean(vals,axis=0)
        family=p['primary_conditional_family'];pairs=family+p['strong_anchor_sensitivity'];decisions[ep]={}
        for pair in pairs:
            a,b=pair.split('-');primary=pair in family;row={'endpoint':ep,'pair':pair,'family':'clinical_primary' if primary else 'clinical_rna_sensitivity'}
            seed_delta=[]
            for metric in ['harrell','uno60']:
                boot=distributions[a,metric]-distributions[b,metric];alpha=.05/(len(family) if primary else len(p['strong_anchor_sensitivity']))
                selected=[r for r in rows if r['endpoint']==ep];dif=[]
                for seed in p['seeds']:
                    get=lambda met:next(r['oof_'+metric] for r in selected if r['method']==met and r['seed']==seed)
                    dif.append(get(a)-get(b))
                row['delta_'+metric]=float(np.mean(dif));row[metric+'_ci95_low'],row[metric+'_ci95_high']=np.quantile(boot,[.025,.975]).tolist()
                row[metric+'_family_ci_low'],row[metric+'_family_ci_high']=np.quantile(boot,[alpha/2,1-alpha/2]).tolist()
                row[metric+'_seed_deltas']=json.dumps(dif)
                if metric=='harrell':seed_delta=dif
            row['robust_positive_increment']=row['harrell_family_ci_low']>0 and row['delta_uno60']>0 and min(seed_delta)>0
            decisions[ep][pair]=bool(row['robust_positive_increment']);pairrows.append(row)
    pd.DataFrame(rows).to_csv(out/'seed_oof_metrics.csv',index=False)
    means=pd.DataFrame(rows).groupby(['endpoint','method'],sort=False).mean(numeric_only=True).drop(columns='seed').reset_index()
    means.to_csv(out/'three_seed_oof_means.csv',index=False);pd.DataFrame(pairrows).to_csv(out/'paired_conditional_increments.csv',index=False)
    write_json(out/'decision.json',{'status':'PASS','primary_endpoint':'DSS','secondary_endpoint':'PFS','robust_increment_by_endpoint':decisions,
        'scope':'conditional frozen representations with linear ridge Cox, no proof that absent significance equals absent signal','bootstrap':'conditional on fitted OOF; three seeds share patients',
        'PFS_not_pooled_with_DSS':True})
    write_json(out/'convergence_audit.json',{'status':'PASS','outer_fits':len(cells),'inner_fits':len(cells)*4,
        'maximum_outer_iterations':max(c['outer_fit']['iterations'] for c in cells),
        'maximum_inner_iterations':max(t['iterations'] for c in cells for t in c['trials']),
        'maximum_gradient':max([c['outer_fit']['gradient_max'] for c in cells]+[t['gradient_max'] for c in cells for t in c['trials']]),
        'schedule_same_inner_outer':True,'no_early_stopping':True})
    write_json(out/'DONE.json',{'status':'PASS','protocol_sha256':m['protocol_sha256'],'outer_fits':510,'seed_oof_models':102})
    print('SURVIVAL_AGGREGATED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['cache','prepare','worker','aggregate']);parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--worker-id',type=int,default=0)
    a=parser.parse_args();globals()[a.mode](a.out,a.worker_id) if a.mode=='worker' else globals()[a.mode](a.out)
