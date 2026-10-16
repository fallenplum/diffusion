"""Independent within-fold metrics, anchor reconstruction and paired bootstrap."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,pandas as pd
from src.phase17r_ssd.data import P,R,cfg,source,write,old
from scripts import validate_phase17_independent as independent
from scripts.run_phase17r import stem
from src.phase10_trimodal.data import digest

def collect():
    cs=cfg();dec=json.loads((R/'STAGE_DECISION.json').read_text());done=json.loads((P/'TRAINING_DONE.json').read_text())
    assert done['status']=='PASS' and done['source']==source() and dec['outer_files_present_at_decision']==0
    d=old.parent.load();risks={};audits=[];mechanisms=[];curves=[];summaries=[];expected=[];parent_replays=[]
    for arm in cs['R1']:
        expected += [(arm,s,f,1.,'forward') for s in cs['seeds'] for f in cs['folds']]
    if dec['R2']['run']:expected += [(q['arm'],q['seed'],q['fold'],q['scale'],'forward') for q in dec['R2']['selected']]
    if dec['R3']['run']:
        arm=dec['R3_base_arm'];rows=[q for q in dec['R2']['selected'] if q['arm']==arm] if arm.startswith('R2') else [{'arm':arm,'seed':s,'fold':f,'scale':1.} for s in cs['seeds'] for f in cs['folds']]
        expected += [(arm,q['seed'],q['fold'],q['scale'],'reverse') for q in rows]
    assert len(expected)==done['outer_fits']
    for arm,s,f,scale,direction in expected:
        suffix='' if direction=='forward' else '_reverse';name=stem(arm,s,f,scale);path=P/('outer'+suffix)/(name+'.pt');cell=json.loads(path.with_suffix('.json').read_text());z=torch.load(path,map_location='cpu',weights_only=False)
        assert digest(path)==cell['checkpoint_sha256'] and z['source']==source() and z['role']=='outer'
        inner=torch.load(P/('inner'+suffix)/(name+'.pt'),map_location='cpu',weights_only=False)
        assert inner['selected_epoch']==z['selected_epoch']==len(z['history'])
        if arm=='R1_A8_full' and direction=='forward':
            prior=torch.load(old.P/'outer'/f'A8_seed{s}_fold{f}.pt',map_location='cpu',weights_only=False)
            parameter_error=max(float((z['state'][k]-prior['state'][k]).abs().max()) for k in z['state'] if z['state'][k].is_floating_point())
            risk_error=float(abs(z['risk']-prior['risk']).max())
            assert parameter_error<1e-6 and risk_error<1e-6 and z['selected_epoch']==prior['selected_epoch'],('full A8 parent drift',s,f,parameter_error,risk_error)
            parent_replays.append({'seed':s,'fold':f,'max_parameter_error':parameter_error,'max_risk_error':risk_error,'selected_epoch_identical':True})
        tr=np.flatnonzero(d['folds']!=f);te=np.flatnonzero(d['folds']==f);assert np.array_equal(z['fit'],tr) and np.array_equal(z['validation_or_test'],te)
        c,ss=independent.independent_cs(d,tr);x=np.c_[c,ss];expected_anchor=(x@z['anchor_w']).astype(np.float32)
        err=float(abs(expected_anchor-z['anchor_risk']).max());gradient=float(abs(independent.anchor_gradient(x[tr].astype(float),d['DSS_time'][tr],d['DSS_event'][tr],z['anchor_w'])).max());assert err<1e-6 and gradient<2e-5
        audits.append({'arm':arm,'seed':s,'fold':f,'scale':scale,'direction':direction,'anchor_max_error':err,'anchor_stationarity_error':gradient,'checkpoint_sha256':cell['checkpoint_sha256']})
        label=arm if direction=='forward' else 'R3_reverse'
        for out,key in [(label,'risk'),('ANCHOR_CS','anchor_risk')]:
            a=np.asarray(z[key])[te] if key=='anchor_risk' else np.asarray(z[key]);target=risks.setdefault((out,s),np.full(512,np.nan))
            if np.isfinite(target[te]).any():assert np.array_equal(target[te],a)
            target[te]=a
        if 'A8' in arm and direction=='forward':
            for tag,key in [('WRONG','wrong_risk'),('SSM_OFF','SSM_off_risk'),('GAMMA_ONE','gamma_one_risk')]:risks.setdefault((arm+'__'+tag,s),np.full(512,np.nan))[te]=z[key]
        mechanisms.append({'stage':'outer','arm':label,'seed':s,'fold':f,'scale':scale,**z['mechanism']})
        summaries.append({'arm':label,'seed':s,'fold':f,'scale':scale,'selected_epoch':z['selected_epoch'],**{k:v for k,v in z['summary'].items() if k!='gradient_norms_last_step'}})
    for base,stage in [('inner','inner'),('inner_reverse','inner_reverse'),('outer','outer'),('outer_reverse','outer_reverse')]:
        for path in sorted((P/base).glob('*.pt')):
            z=torch.load(path,map_location='cpu',weights_only=False)
            assert z['source']==source()
            for row in z['history']:curves.append({'stage':stage,'arm':z['arm'],'seed':z['seed'],'fold':z['fold'],'scale':z['scale'],'selected_epoch':z['selected_epoch'],**row})
    for risk in risks.values():assert np.isfinite(risk).all()
    pd.DataFrame(audits).to_csv(R/'ANCHOR_AND_CHECKPOINT_AUDIT.csv',index=False);pd.DataFrame(mechanisms).to_csv(R/'OUTER_MECHANISM.csv',index=False);pd.DataFrame(curves).to_csv(R/'TRAINING_CURVES.csv',index=False);pd.DataFrame(summaries).to_csv(R/'MODEL_CELL_SUMMARY.csv',index=False)
    assert len(parent_replays)==15
    write(R/'PARENT_FULL_A8_PARITY.json',{'status':'PASS','actual_independent_retraining_cells':15,'maximum_parameter_error':max(x['max_parameter_error'] for x in parent_replays),'maximum_risk_error':max(x['max_risk_error'] for x in parent_replays),'rows':parent_replays,'note':'new full A8 was actually retrained; parent outputs were not copied'})
    pd.DataFrame(summaries).groupby('arm').agg(cells=('fold','count'),parameters=('registered_parameters','mean'),gradient_parameters=('last_step_gradient_parameters','mean'),outer_seconds_mean=('seconds','mean'),peak_GPU_MB=('peak_GPU_MB','max'),gamma_mean=('gamma','mean'),selected_epoch_median=('selected_epoch','median')).reset_index().to_csv(R/'MODEL_SUMMARY.csv',index=False)
    np.savez_compressed(P/'OOF_PRIVATE.npz',**{a+'__seed'+str(s):r for (a,s),r in risks.items()})
    return d,risks

def bootstrap(d,risks,ev,w,seedrows):
    cs=cfg();draws=cs['bootstrap']['draws'];rng=np.random.default_rng(cs['bootstrap']['seed']);keys=sorted(risks);num=np.zeros((draws,len(keys),2));den=np.zeros((draws,2));t,e,valid=ev
    for f in cs['folds']:
        ix=np.flatnonzero((d['folds']==f)&valid);n=len(ix);counts=rng.multinomial(n,np.full(n,1/n),size=draws).astype(float);i,j=independent.pairs(t[ix],e[ix]);ws=np.stack([np.ones(n),w[ix]]);mat=np.zeros((len(keys),2,n,n));dm=np.zeros((2,n,n))
        for k,key in enumerate(keys):
            delta=risks[key][ix][i]-risks[key][ix][j];credit=np.where(abs(delta)<=1e-8,.5,delta>0)
            for m in range(2):mat[k,m,i,j]=credit*ws[m,i]
        for m in range(2):dm[m,i,j]=ws[m,i]
        for start in range(0,draws,256):
            cc=counts[start:start+256];products=cc[:,:,None]*cc[:,None,:];num[start:start+len(cc)]+=np.einsum('bij,kmij->bkm',products,mat,optimize=True);den[start:start+len(cc)]+=np.einsum('bij,mij->bm',products,dm,optimize=True)
    good=(den>0).all(1);values=num[good]/den[good,None,:];idx={k:j for j,k in enumerate(keys)};rows=[]
    primary=cs['primary_comparisons'];secondary=[['R1_A8_noDirect','ANCHOR_CS'],['R1_A8_noDirect','R1_A8_noDirect__WRONG'],['R1_S6_noDirect','ANCHOR_CS'],['R1_A3_noDirect','R1_S6_noDirect'],['R2_A8','ANCHOR_CS'],['R2_A8','R2_A8__WRONG'],['R3_reverse',json.loads((R/'STAGE_DECISION.json').read_text())['R3_base_arm']]]
    for a,b in primary+secondary:
        if (a,31) not in risks or (b,31) not in risks:continue
        delta=values[:,[idx[a,s] for s in cs['seeds']]].mean(1)-values[:,[idx[b,s] for s in cs['seeds']]].mean(1)
        for m,metric in enumerate(['Harrell','Uno60']):
            family='primary' if [a,b] in primary else 'secondary';tests=cs['bootstrap']['family_tests_maximum'] if family=='primary' else 14;alpha=.05/tests
            point=np.mean([r[metric] for r in seedrows if r['arm']==a])-np.mean([r[metric] for r in seedrows if r['arm']==b]);lo,hi=np.quantile(delta[:,m],[.025,.975]);cl,ch=np.quantile(delta[:,m],[alpha/2,1-alpha/2])
            rows.append({'comparison':a+'-'+b,'metric':metric,'family':family,'delta':point,'CI95_low':lo,'CI95_high':hi,'Bonferroni_low':cl,'Bonferroni_high':ch,'family_tests':tests,'draws_valid':int(good.sum()),'draws_requested':draws})
    pd.DataFrame(rows).to_csv(R/'PAIRED_COMPARISONS.csv',index=False)
    write(R/'BOOTSTRAP_AUDIT.json',{'status':'PASS','draws':draws,'valid':int(good.sum()),'zero_denominator_draws':int((~good).sum()),'minimum_H_denominator':float(den[:,0].min()),'minimum_U_denominator':float(den[:,1].min()),'sampling':'patient within-fold multinomial; same counts all models and seeds','duplicate_pair_weights':'count_i*count_j; not independent seeds','primary_family_reserved_tests':12,'IPCW':'fixed original outer-train reverse Kaplan Meier'})

def main():
    d,risks=collect();independent.R=R;cs=cfg()
    t,e,valid,w,rows=independent.metrics(d,risks,cs)
    bootstrap(d,risks,(t,e,valid),w,rows)
    independent.metrics(d,risks,cs,'PFI')
    for name in ['SEED_OOF_METRICS','PFI_SEED_OOF_METRICS']:
        q=pd.read_csv(R/(name+'.csv'));q.groupby('arm')[['Harrell','Uno60']].agg(['mean','std']).to_csv(R/('SUMMARY.csv' if name=='SEED_OOF_METRICS' else 'PFI_SUMMARY.csv'))
    write(R/'INDEPENDENT_VALIDATION.json',{'status':'PASS','new_source':source(),'seed_vectors':len(risks),'fold_metric_rows':len(risks)*5,'sksurv_each_fold_Harrell_and_Uno60_checked':True,'independent_Clinical_S32_anchor_reconstruction':True,'checkpoint_replay':'separate replay audit required','PFI':'frozen DSS risk only','no_risk_ensembling':True})
    print('INDEPENDENT_EVALUATION_PASS',len(risks)*5,flush=True)

if __name__=='__main__':main()
