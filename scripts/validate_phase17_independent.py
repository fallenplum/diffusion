"""Independent event pairs, reverse KM, anchor reconstruction and shared bootstrap.

Does not import training loss, preprocessing or concordance functions.
Checkpoint replay is performed by a separate executable.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];P=Path('/root/autodl-tmp/phase17_ssd_private');R=ROOT/'reports/phase17_ssd_fusion'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def pairs(t,e):
    a=[];b=[]
    for i in range(len(t)):
        if e[i]:
            for j in range(len(t)):
                if t[i]<t[j] or (t[i]==t[j] and not e[j]):a.append(i);b.append(j)
    return np.asarray(a,int),np.asarray(b,int)

def weights(train_t,train_e,test_t,test_e,tau=60.):
    times=np.unique(train_t);g=[];v=1.
    for u in times:
        nc=np.sum((train_t==u)&(~train_e.astype(bool)));at=np.sum(train_t>=u)-np.sum((train_t==u)&train_e.astype(bool))
        if nc:v*=1-nc/at
        g.append(v)
    out=np.zeros(len(test_t))
    for i in range(len(test_t)):
        if test_e[i] and test_t[i]<tau:
            k=np.searchsorted(times,test_t[i],side='right')-1;s=1. if k<0 else g[k]
            if s<=0:raise ValueError('unsupported IPCW event')
            out[i]=s**-2
    return out

def components(t,e,r,w):
    i,j=pairs(t,e);delta=r[i]-r[j];credit=np.where(abs(delta)<=1e-8,.5,delta>0)
    return float(np.sum(credit*w[i])),float(np.sum(w[i])),i,j,credit

def independent_cs(d,fit):
    fields=['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE'];xs=[]
    for name in fields:
        v=d['clinical_'+name].astype(str)
        if name=='AGE':
            age=np.array([float(a) if a.replace('.','',1).isdigit() else np.nan for a in v]);missing=~np.isfinite(age);median=np.nanmedian(age[fit]);xs.append(np.where(missing,median,age)[:,None])
        else:xs.append(np.column_stack([v==a for a in sorted(set(v[fit]))]))
    c=np.c_[np.concatenate(xs,1).astype(float),missing.astype(float)]
    for key,x in [('C',c),('S',np.asarray(d['static'],float))]:
        mean=x[fit].mean(0);sd=x[fit].std(0);sd[sd<1e-8]=1.
        if key=='C':c=((x-mean)/sd).astype(np.float32)
        else:s=((x-mean)/sd).astype(np.float32)
    return c,s

def anchor_gradient(x,t,e,w,lam=10.):
    r=x@w;shift=r.max();ex=np.exp(r-shift);g=-x[e.astype(bool)].sum(0)
    for u in sorted(set(t[e.astype(bool)])):
        ev=(t==u)&e.astype(bool);rs=t>=u;ne=ev.sum()
        for k in range(ne):
            p=ex*rs-(k/ne)*ex*ev;g+=(x*p[:,None]).sum(0)/p.sum()
    return g/e.sum()+lam*w

def collect(d,cfg,lock):
    t=d['DSS_time'];e=d['DSS_event'];folds=d['folds'];rows=[];risks={};anchor_checks=[];models=cfg['arms']+['MGCT','MambaMIL']
    # Paper methods can be blocked, but no placeholders or invented risk arrays.
    blocked=[]
    for arm in models:
        if arm in ['A0','A1']:
            for seed in cfg['seeds']:risks[arm,seed]=np.full(512,np.nan)
            for fold in cfg['folds']:
                p=P/'simple'/f'fold{fold}.pt'
                if not p.exists():raise ValueError('missing simple '+str(p))
                q=torch.load(p,map_location='cpu',weights_only=False);tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);entry=q['models'][arm]
                assert q['source']==lock['training_source'] and np.array_equal(q['parts']['outer_test'],te)
                for seed in cfg['seeds']:risks[arm,seed][te]=entry['risk']
            continue
        paper=arm in ['MGCT','MambaMIL'];base=P/('paper_outer' if paper else 'outer')
        if paper and not any(base.glob(arm+'*.pt')):blocked.append(arm);continue
        for seed in cfg['seeds']:
            risk=np.full(512,np.nan)
            for fold in cfg['folds']:
                stem=f'{arm}_seed{seed}_fold{fold}';p=base/(stem+'.pt');cell=json.loads(p.with_suffix('.json').read_text());q=torch.load(p,map_location='cpu',weights_only=False)
                assert cell['checkpoint_sha256']==sha(p) and q['source']==lock['training_source']
                assert q['arm']==arm and q['seed']==seed and q['fold']==fold and q['role']=='outer' and q['budget']==512
                tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
                assert np.array_equal(q['fit'],tr) and np.array_equal(q['validation_or_test'],te) and len(q['history'])==q['selected_epoch']
                c,s=independent_cs(d,tr);x=s if arm=='A11_NO_CLIN' else c if arm=='A11_NO_S32' else np.c_[c,s]
                expected=(x@q['anchor_w']).astype(np.float32);err=float(abs(expected-q['anchor_risk']).max())
                grad=float(abs(anchor_gradient(x[tr].astype(float),t[tr],e[tr],q['anchor_w'])).max())
                if err>1e-6 or grad>2e-5:raise ValueError(('anchor failed',stem,err,grad))
                anchor_checks.append({'arm':arm,'seed':seed,'fold':fold,'elementwise_max_error':err,'independent_penalized_gradient':grad})
                assert np.isfinite(q['risk']).all();risk[te]=q['risk']
            risks[arm,seed]=risk
    pd.DataFrame(anchor_checks).to_csv(R/'ANCHOR_ELEMENTWISE_AUDIT.csv',index=False)
    # Required pre-existing strong references; no rerun and no risk averaging.
    old=np.load('/root/phase16r_light_rna_private/OOF_PRIVATE.npz')
    historical_checks=[]
    for arm,key in [('HIST_PHASE15_CS','CS_P15_REF'),('HIST_CS_S32_CONCH32','LOCAL__CV3')]:
        for seed in cfg['seeds']:
            risk=old[key+'__seed'+str(seed)]
            if risk.shape!=(512,) or not np.isfinite(risk).all():raise ValueError('invalid frozen historical reference')
            for fold in cfg['folds']:
                te=np.flatnonzero(folds==fold);ap=Path('/root/phase16r_light_rna_private/artifacts')/f'seed{seed}_fold{fold}.pt';cp=Path('/root/phase16r_light_rna_private/cells')/f'seed{seed}_fold{fold}.json';cell=json.loads(cp.read_text());assert sha(ap)==cell['sha256']
                saved=torch.load(ap,map_location='cpu',weights_only=False);assert np.array_equal(saved['part']['outer_test'],te);error=float(abs(saved['models'][key]['risk']-risk[te]).max())
                # Historical independent replay recomputed a float64 dot product;
                # persisted original fit and replay can differ by machine epsilon.
                if error>1e-9:raise ValueError(('historical reference replay',arm,seed,fold,error))
                historical_checks.append({'arm':arm,'seed':seed,'fold':fold,'maximum_OOF_difference':error,'reference_artifact_sha256':sha(ap)})
            risks[arm,seed]=risk.copy()
    pd.DataFrame(historical_checks).to_csv(R/'HISTORICAL_REFERENCE_AUDIT.csv',index=False)
    write(R/'AVAILABLE_MODEL_AUDIT.json',{'trained_models':[a for a in models if (a,cfg['seeds'][0]) in risks],'blocked_models':blocked,'TransSurv':'BLOCKED;no verifiable official code/multiscale input','anchor_records':len(anchor_checks)})
    return risks

def metrics(d,risks,cfg,endpoint='DSS'):
    from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
    from sksurv.util import Surv
    z=dict(np.load('/root/phase13_rna_private/endpoints.npz'));t=z[endpoint+'_time'];e=z[endpoint+'_event'];valid=z[endpoint+'_valid'].astype(bool);f=d['folds'];foldrows=[];seedrows=[];w=np.zeros(512)
    for fold in cfg['folds']:
        tr=np.flatnonzero((f!=fold)&valid);te=np.flatnonzero((f==fold)&valid);w[te]=weights(t[tr],e[tr],t[te],e[te])
        for (arm,seed),risk in risks.items():
            hn,hd,*_=components(t[te],e[te],risk[te],np.ones(len(te)));un,ud,*_=components(t[te],e[te],risk[te],w[te])
            hr=concordance_index_censored(e[te].astype(bool),t[te],risk[te])[0];ur=concordance_index_ipcw(Surv.from_arrays(e[tr].astype(bool),t[tr]),Surv.from_arrays(e[te].astype(bool),t[te]),risk[te],tau=60)[0]
            if abs(hn/hd-hr)>1e-10 or abs(un/ud-ur)>1e-10:raise ValueError(('sksurv disagreement',endpoint,arm,seed,fold))
            foldrows.append({'endpoint':endpoint,'arm':arm,'seed':seed,'fold':fold,'n':len(te),'events':int(e[te].sum()),'Harrell':hn/hd,'Uno60':un/ud,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud})
    for (arm,seed) in risks:
        rr=[r for r in foldrows if r['arm']==arm and r['seed']==seed];sums={key:sum(r[key] for r in rr) for key in ['H_num','H_den','U_num','U_den']}
        seedrows.append({'endpoint':endpoint,'arm':arm,'seed':seed,'Harrell':sums['H_num']/sums['H_den'],'Uno60':sums['U_num']/sums['U_den'],**sums})
    pd.DataFrame(foldrows).to_csv(R/('FOLD_METRICS.csv' if endpoint=='DSS' else 'PFI_FOLD_METRICS.csv'),index=False)
    pd.DataFrame(seedrows).to_csv(R/('SEED_OOF_METRICS.csv' if endpoint=='DSS' else 'PFI_SEED_OOF_METRICS.csv'),index=False)
    return t,e,valid,w,seedrows

def bootstrap(d,risks,cfg,t,e,valid,w,seedrows):
    # One bootstrap count vector per patient, reused across all methods/seeds.
    draws=cfg['bootstrap']['draws'];rng=np.random.default_rng(cfg['bootstrap']['seed']);keys=sorted(risks);folds=d['folds'];num=np.zeros((draws,len(keys),2));den=np.zeros((draws,2))
    for fold in cfg['folds']:
        ix=np.flatnonzero((folds==fold)&valid);n=len(ix);counts=rng.multinomial(n,np.full(n,1/n),size=draws).astype(np.float64);i,j=pairs(t[ix],e[ix]);ws=np.stack([np.ones(n),w[ix]])
        matrices=np.zeros((len(keys),2,n,n));denmat=np.zeros((2,n,n))
        for k,key in enumerate(keys):
            delta=risks[key][ix][i]-risks[key][ix][j];credit=np.where(abs(delta)<=1e-8,.5,delta>0)
            for m in range(2):matrices[k,m,i,j]=credit*ws[m,i]
        for m in range(2):denmat[m,i,j]=ws[m,i]
        # CPU BLAS on per-fold100x100 matrices; bounded batches, no risk ensembling.
        for start in range(0,draws,256):
            cc=counts[start:start+256];products=cc[:,:,None]*cc[:,None,:]
            num[start:start+len(cc)]+=np.einsum('bij,kmij->bkm',products,matrices,optimize=True)
            den[start:start+len(cc)]+=np.einsum('bij,mij->bm',products,denmat,optimize=True)
    good=(den>0).all(1);values=num[good]/den[good,None,:];index={key:i for i,key in enumerate(keys)}
    secondary=[('A7','A5'),('A7','A6'),('A8','A2'),('A8','A1'),('A8','A0'),('A8','A9'),('A8','A10'),('A8','MGCT'),('A8','MambaMIL'),('A8','HIST_PHASE15_CS'),('A8','HIST_CS_S32_CONCH32')]
    primary=[tuple(v) for v in cfg['primary_comparisons']];rows=[]
    for a,b in primary+secondary:
        if (a,31) not in risks or (b,31) not in risks:continue
        aa=np.mean(values[:,[index[a,s] for s in cfg['seeds']]],axis=1);bb=np.mean(values[:,[index[b,s] for s in cfg['seeds']]],axis=1);delta=aa-bb
        for m,metric in enumerate(['Harrell','Uno60']):
            family='primary' if (a,b) in primary else 'secondary';tests=4 if family=='primary' else len(secondary)*2;alpha=.05/tests
            point=np.mean([r[metric] for r in seedrows if r['arm']==a])-np.mean([r[metric] for r in seedrows if r['arm']==b]);lo,hi=np.quantile(delta[:,m],[.025,.975]);cl,ch=np.quantile(delta[:,m],[alpha/2,1-alpha/2])
            rows.append({'comparison':a+'-'+b,'metric':metric,'family':family,'delta':point,'CI95_low':lo,'CI95_high':hi,'Bonferroni_low':cl,'Bonferroni_high':ch,'family_tests':tests,'draws_requested':draws,'draws_valid':int(good.sum()),'zero_denominator_draws':int((~good).sum()),'point_definition':'mean of3 seed within-fold pair-weighted C;NOT C(mean risk)'})
    pd.DataFrame(rows).to_csv(R/'PAIRED_COMPARISONS.csv',index=False)
    write(R/'BOOTSTRAP_AUDIT.json',{'draws':draws,'valid':int(good.sum()),'zero_denominators':int((~good).sum()),'sampling':'multinomial counts inside each outer fold;patient cluster;shared models/seeds','duplicate_pair_weight':'count[i]*count[j];no self comparable pair','IPCW':'frozen outer train reverse KM','minimum_H_denominator':float(den[:,0].min()),'minimum_U_denominator':float(den[:,1].min()),'risk_ensemble_used':False})

def main():
    p=argparse.ArgumentParser();p.add_argument('--statistics',action='store_true');a=p.parse_args();cfg=json.loads((ROOT/'configs/phase17_ssd_fusion.json').read_text());lock=json.loads((R/'PROTOCOL_LOCK.json').read_text());d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True))
    assert len(d['patient_ids'])==512 and d['DSS_event'].sum()==65
    risks=collect(d,cfg,lock);t,e,valid,w,seedrows=metrics(d,risks,cfg)
    np.savez_compressed(P/'OOF_PRIVATE.npz',**{arm+'__seed'+str(seed):r for (arm,seed),r in risks.items()})
    if a.statistics:bootstrap(d,risks,cfg,t,e,valid,w,seedrows)
    metrics(d,risks,cfg,'PFI')
    write(R/'INDEPENDENT_VALIDATION.json',{'status':'PASS','trained_model_seed_vectors':len(risks),'fold_metric_rows':len(risks)*5,'metrics_checked_against_sksurv':True,'anchor_recomputed_independently':True,'checkpoint_replay':'separate CHECKPOINT_REPLAY.json required','PFI':'same frozen DSS risks;no PFI training or selection'})
    print('INDEPENDENT_METRICS_PASS',len(risks)*5,flush=True)

if __name__=='__main__':main()
