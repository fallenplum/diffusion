"""Independent raw-OOF evaluator. NO import of training math/metrics/features.

Checkpoint replay is delegated to a separate executable and never used to
define the independent concordance/IPCW implementation below.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs(time,event):
    t=np.asarray(time,float);e=np.asarray(event)
    if t.ndim!=1 or e.shape!=t.shape or not np.isfinite(t).all() or not np.isin(e,[0,1]).all():raise ValueError('invalid survival')
    i=[];j=[]
    for a in range(len(t)):
        if e[a]==1:
            for b in range(len(t)):
                if t[a]<t[b] or (t[a]==t[b] and e[b]==0):i.append(a);j.append(b)
    return np.array(i,int),np.array(j,int)


def ipcw(tt,ee,t,e,tau=60):
    unique=np.unique(tt);values=[];survival=1.
    for u in unique:
        censor=((tt==u)&(ee==0)).sum();risk=(tt>=u).sum()-((tt==u)&(ee==1)).sum()
        if censor: survival*=1-censor/risk
        values.append(survival)
    out=np.zeros(len(t))
    for k in np.flatnonzero((e==1)&(t<tau)):
        ix=np.searchsorted(unique,t[k],side='right')-1;g=1 if ix<0 else values[ix]
        if g<=0:raise ValueError('IPCW support failure')
        out[k]=1/(g*g)
    return out


def components(t,e,r,w=None,counts=None):
    r=np.asarray(r,float)
    if r.shape!=np.asarray(t).shape or not np.isfinite(r).all():raise ValueError('invalid risk')
    i,j=pairs(t,e);weights=np.ones(len(t)) if w is None else np.asarray(w)
    q=np.ones(len(i)) if counts is None else np.asarray(counts)[i]*np.asarray(counts)[j]
    delta=r[i]-r[j];credit=np.where(abs(delta)<=1e-8,.5,delta>0);q=q*weights[i]
    return float(q@credit),float(q.sum())


def check_manifest(m,folds,fold):
    tr=np.asarray(m['outer_train'],int);te=np.asarray(m['outer_test'],int);it=np.asarray(m['inner_train'],int);iv=np.asarray(m['inner_val'],int)
    if not np.array_equal(te,np.flatnonzero(folds==fold)) or not np.array_equal(tr,np.flatnonzero(folds!=fold)) or set(it)&set(iv) or set(it)|set(iv)!=set(tr):raise ValueError('split isolation')
    for role,fit in [('inner',it),('outer',tr)]:
        p=m['preprocess_'+role]
        if p['fit_indices']!=fit.tolist() or p['fit_hash']!=hashlib.sha256(fit.astype(np.int64).tobytes()).hexdigest():raise ValueError('preprocessing fit mismatch')
        if set(p['source_fields'])!=set(['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE']) or p['selection_source']!='RNA expression only; no outcome':raise ValueError('forbidden label source')
        if p['pca']['fit_hash']!=p['fit_hash']:raise ValueError('PCA fit mismatch')
    for role,fit in [('inner_train',it),('inner_val',iv),('outer_train',tr),('outer_test',te)]:
        for modality,source in m.get(role+'_modality_indices',{}).items():
            if len(source)!=len(fit) or set(source)!=set(fit):raise ValueError('cross split wrong pairing')
    return tr,te,it,iv


def check_record(cell,z,m,d,arm,seed,fold,source,protocol):
    if cell['status']!='COMPLETE' or cell['arm']!=arm or cell['seed']!=seed or cell['fold']!=fold or str(z['arm'])!=arm or int(z['seed'])!=seed or int(z['fold'])!=fold:raise ValueError('container identity')
    if cell['source_hash']!=source or cell['protocol_sha256']!=protocol or cell['outer_test_selection'] is not False:raise ValueError('source/selection mismatch')
    tr,te,it,iv=check_manifest(m,d['folds'],fold)
    if m['patient_hash']!=hashlib.sha256(np.ascontiguousarray(d['patient_ids']).tobytes()).hexdigest() or m['fold_hash']!=hashlib.sha256(np.ascontiguousarray(d['folds']).tobytes()).hexdigest():raise ValueError('patient/fold identity hash')
    if not np.array_equal(z['test_indices'],te) or not np.array_equal(z['time'],d['DSS_time'][te]) or not np.array_equal(z['event'],d['DSS_event'][te]):raise ValueError('OOF label/direction identity')
    if not np.isfinite(z['risk']).all():raise ValueError('risk finite')
    if cell['selected_epoch'] and cell['outer']['actual_epochs']!=cell['selected_epoch']:raise ValueError('refit budget mismatch')
    return tr,te


def validate(family=None,statistics=False):
    root=Path(__file__).resolve().parents[1];report=root/'reports/phase11_semantic_anchored_trimodal';out=Path('/root/phase11_semantic_anchor_private/formal')
    cfg=json.loads((root/'configs/phase11_semantic_anchored.json').read_text(encoding='utf-8-sig'));gate=json.loads((report/'HARD_GATES.json').read_text())
    d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True))
    arms=[a for a in cfg['arms'] if family is None or a.startswith(family)]
    if family=='P':arms=list(cfg['patch_arms'])+['P0','P3']
    if family is None:arms+=list(cfg['patch_arms'])
    from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
    from sksurv.util import Surv
    t=d['DSS_time'];e=d['DSS_event'];f=d['folds'];weights=np.zeros(len(t));rows=[];fail=[];risks={}
    for fold in cfg['folds']:
        tr=f!=fold;te=f==fold;weights[te]=ipcw(t[tr],e[tr],t[te],e[te])
    for arm in arms:
        for seed in cfg['seeds']:
            risk=np.full(len(t),np.nan)
            for fold in cfg['folds']:
                stem=f'{arm}_seed{seed}_fold{fold}';cp=out/'cells'/f'{stem}.json';pp=out/'oof'/f'{stem}.npz';mp=out/'manifests'/f'{stem}.json'
                try:
                    cell=json.loads(cp.read_text());z=dict(np.load(pp));m=json.loads(mp.read_text())
                    expected_source=gate['source_hash']
                    amendment=report/'P2_SOURCE_AMENDMENT.json'
                    if amendment.exists() and arm.startswith('A'):
                        lineage=json.loads(amendment.read_text())
                        if cell['source_hash']==lineage['prior_source_hash'] and expected_source==lineage['new_source_hash'] and lineage['pre_repair_A_checkpoint_audit']=='PASS':expected_source=lineage['prior_source_hash']
                    tr,te=check_record(cell,z,m,d,arm,seed,fold,expected_source,digest(root/'configs/phase11_semantic_anchored.json'))
                    if cell['prediction_sha256']!=digest(pp) or cell['manifest_sha256']!=digest(mp):raise ValueError('file hash')
                    base=cell.get('reuse_of',arm)
                    if 'reuse_of' in cell:
                        other=np.load(out/'oof'/f'{base}_seed{seed}_fold{fold}.npz')['risk']
                        if not np.array_equal(other,z['risk']):raise ValueError('alias nonexact')
                    risk[te]=z['risk'];hn,hd=components(t[te],e[te],risk[te]);un,ud=components(t[te],e[te],risk[te],weights[te])
                    hr=concordance_index_censored(e[te].astype(bool),t[te],risk[te])[0]
                    ur=concordance_index_ipcw(Surv.from_arrays(e[tr].astype(bool),t[tr]),Surv.from_arrays(e[te].astype(bool),t[te]),risk[te],tau=60)[0]
                    if abs(hn/hd-hr)>1e-10 or abs(un/ud-ur)>1e-10:raise ValueError('metric library mismatch')
                    rows.append({'arm':arm,'seed':seed,'fold':fold,'Harrell':hn/hd,'Uno60':un/ud,'Harrell_num':hn,'Harrell_den':hd,'Uno_num':un,'Uno_den':ud,
                                 'test_n':len(te),'test_events':int(e[te].sum()),'parameters':cell['parameters'],'selected_epoch':cell['selected_epoch'],'actual_fit':cell['actual_fit'],'seconds':cell['seconds']})
                except Exception as ex:fail.append({'arm':arm,'seed':seed,'fold':fold,'error':str(ex)})
            if np.isfinite(risk).all():risks[arm,seed]=risk
    tag=family or 'ALL';pd.DataFrame(rows).to_csv(report/f'{tag}_FOLD_METRICS.csv',index=False)
    result={'status':'FAIL' if fail else 'PASS','scope':tag,'expected_records':len(arms)*15,'verified_records':len(rows),'failures':fail,'metrics':'independent event-pair/reverseKM versus sksurv0.23.1'}
    (report/f'{tag}_INDEPENDENT_VALIDATION.json').write_text(json.dumps(result,indent=2))
    if fail:print(json.dumps(result));return 2
    if statistics:
        pd.DataFrame(rows).to_csv(report/'FOLD_METRICS.csv',index=False)
        seedrows=[]
        for (arm,seed),risk in risks.items():
            selected=[r for r in rows if r['arm']==arm and r['seed']==seed]
            sums={k:sum(r[k] for r in selected) for k in ['Harrell_num','Harrell_den','Uno_num','Uno_den']}
            hn,hd=components(t,e,risk);un,ud=components(t,e,risk,weights)
            seedrows.append({'arm':arm,'seed':seed,'Harrell':sums['Harrell_num']/sums['Harrell_den'],'Uno60':sums['Uno_num']/sums['Uno_den'],
                             'pooled_Harrell':hn/hd,'pooled_Uno60':un/ud,**sums})
        for alias,base in [('P0','A6'),('P3','B0')]:
            for seed in cfg['seeds']:
                risks[alias,seed]=risks[base,seed]
                original=next(r for r in seedrows if r['arm']==base and r['seed']==seed)
                seedrows.append({**original,'arm':alias})
        pd.DataFrame(seedrows).to_csv(report/'SEED_OOF_METRICS.csv',index=False)
        bootstraps(cfg,d,weights,risks,seedrows,report)
    print(json.dumps(result));return 0


def bootstraps(cfg,d,weights,risks,seedrows,report):
    t=d['DSS_time'];e=d['DSS_event'];f=d['folds'];pi,pj=pairs(t,e);keep=f[pi]==f[pj];pi,pj=pi[keep],pj[keep]
    # Linearity in CREDIT permits averaging the 3 already-trained seeds' pair
    # credits, not averaging risk arrays before a nonlinear concordance metric.
    comparisons=cfg['primary_comparisons']+cfg['secondary_comparisons'];credit=[]
    points={}
    for a,b in comparisons:
        delta=[]
        for seed in cfg['seeds']:
            ra=risks[a,seed];rb=risks[b,seed]
            ca=np.where(abs(ra[pi]-ra[pj])<=1e-8,.5,ra[pi]>ra[pj]);cb=np.where(abs(rb[pi]-rb[pj])<=1e-8,.5,rb[pi]>rb[pj]);delta.append(ca-cb)
        credit.append(np.mean(delta,0))
    credits=np.stack(credit,1);rng=np.random.default_rng(cfg['bootstrap']['seed']);allv={'Harrell':[],'Uno60':[]};drawhash=hashlib.sha256()
    for start in range(0,cfg['bootstrap']['draws'],250):
        counts=rng.multinomial(len(t),np.full(len(t),1/len(t)),size=min(250,cfg['bootstrap']['draws']-start));drawhash.update(counts.tobytes())
        q=(counts[:,pi]*counts[:,pj]).astype(float);qu=q*weights[pi]
        allv['Harrell'].append((q@credits)/q.sum(1)[:,None]);allv['Uno60'].append((qu@credits)/qu.sum(1)[:,None])
    result=[]
    for metric in allv:
        vals=np.concatenate(allv[metric]);
        for j,(a,b) in enumerate(comparisons):
            delta=[next(r[metric] for r in seedrows if r['arm']==a and r['seed']==s)-next(r[metric] for r in seedrows if r['arm']==b and r['seed']==s) for s in cfg['seeds']]
            main=j<len(cfg['primary_comparisons']);alpha=.05/(2*len(cfg['primary_comparisons'])) if main else .05
            ordinary=np.quantile(vals[:,j],[.025,.975]);sim=np.quantile(vals[:,j],[alpha/2,1-alpha/2])
            from scipy.stats import binom
            sorted_values=np.sort(vals[:,j]);mc={}
            for name,prob in [('low',alpha/2),('high',1-alpha/2)]:
                ranks=binom.ppf([.025,.975],len(sorted_values),prob).astype(int)
                ranks=np.clip(ranks,0,len(sorted_values)-1)
                mc['sim_'+name+'_quantile_MC95_low']=sorted_values[ranks[0]]
                mc['sim_'+name+'_quantile_MC95_high']=sorted_values[ranks[1]]
            result.append({'comparison':a+'-'+b,'family':'primary' if main else 'exploratory_secondary','metric':metric,'mean_delta':np.mean(delta),
                           **{'seed'+str(s)+'_delta':v for s,v in zip(cfg['seeds'],delta)},'positive_seeds':sum(v>0 for v in delta),
                           'CI95_low':ordinary[0],'CI95_high':ordinary[1],'simultaneous_low':sim[0],'simultaneous_high':sim[1],'draws':len(vals),**mc})
    pd.DataFrame(result).to_csv(report/'PAIRED_COMPARISONS.csv',index=False)
    (report/'BOOTSTRAP_AUDIT.json').write_text(json.dumps({'draws':cfg['bootstrap']['draws'],'seed':cfg['bootstrap']['seed'],'counts_sha256':drawhash.hexdigest(),
        'primary_extreme_tail_expected_draws':cfg['bootstrap']['draws']*.05/32,'tail_probability_MC_SE':float(np.sqrt((.05/32)*(1-.05/32)/cfg['bootstrap']['draws'])),
        'refit_variance_included':False,'patient_counts_shared':True,'averaged_seed_pair_credit_not_risk':True},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['A','B','C','D','P']);p.add_argument('--statistics',action='store_true');a=p.parse_args();raise SystemExit(validate(a.family,a.statistics))
