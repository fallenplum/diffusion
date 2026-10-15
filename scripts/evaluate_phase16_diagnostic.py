"""Independent endpoint/metric checks, literal gradients and paired intervals."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
from sksurv.util import Surv
from src.phase16_diagnostic.core import *
from run_phase16_visual_diagnostic import vsource
from validate_phase11_independent import pairs,ipcw,components
from validate_phase13_independent import raw_endpoints

def gradient(x,t,e,w,lam):
    r=x@w;g=-e.astype(float)
    for u in np.unique(t[e==1]):
        deaths=(t==u)&(e==1);risk=t>=u;n=int(deaths.sum());v=np.exp(r-r[risk].max())
        for k in range(n):a=risk.astype(float)-k/n*deaths;g+=a*v/(a@v)
    return x.T@g/e.sum()+lam*w

def rna_arithmetic(d,z):
    maximum=0
    for role in ['inner','outer']:
        state=z['states'][role];fit=state['fit'];raw=np.asarray(d['rna_genes'],float);median=np.nanmedian(raw[fit],axis=0);median=np.where(np.isfinite(median),median,0);filled=np.where(np.isfinite(raw),raw,median);mu=filled[fit].mean(0);sd=filled[fit].std(0);zz=(filled-mu)/np.where(sd>1e-8,sd,1)
        meta,old=parent.parent_features(z['fold'],role);top=np.asarray(meta['top_indices']);hall=np.asarray(meta['phase12']['enriched_indices']);h=zz[:,hall].astype(np.float32).astype(float);c=[];cover=[];states=[]
        for g in state['molecular']['groups']:
            ix=np.asarray(g['indices']);a=zz[:,ix].mean(1);cv=np.isfinite(raw[:,ix]).mean(1);q=np.quantile(a[fit][cv[fit]>=.7],[1/3,2/3]);s=np.full(512,'typical',dtype='U16');s[a<q[0]]='low';s[a>q[1]]='high';s[cv<.7]='unmeasured';c.append(a);cover.append(1-cv);states.append(s)
            if not np.allclose(q,g['thresholds'],atol=1e-12):raise ValueError('RNA train quantile arithmetic')
        c=np.stack(c,1);states=np.stack(states,1);cover=np.stack(cover,1);o=np.zeros_like(c);o[states=='low']=-1;o[states=='high']=1
        if not np.array_equal(states,state['RNA_states']):raise ValueError('quantization coding mismatch')
        expected={'TOP512':zz[:,top].astype(np.float32).astype(float),'HALL512':h,'HALL_POOL8':h@state['same512_membership'].T/state['same512_membership'].sum(1),
                  'HALL_PCA8':(h-state['pca_mean'])@state['pca_components'].T,'PATH8_CONT':c,'PATH8_QUANT_ORD':o,
                  'PATH8_QUANT_ONEHOT':np.stack([states==s for s in ['low','typical','high','unmeasured']],2).reshape(512,-1).astype(float),
                  'PATH8_CONT_COV':np.c_[c,cover],'PATH8_QUANT_COV':np.c_[o,cover]}
        for name,x in expected.items():
            mean=x[fit].mean(0);std=x[fit].std(0);std=np.where(std<1e-8,1,std);err=float(abs((x-mean)/std-z['features'][role][name]).max());maximum=max(maximum,err)
            if err>3e-6:raise ValueError('independent RNA transform mismatch '+name+' '+str(err))
    return maximum

def main():
    d=parent.load();ep=raw_endpoints(d,{});rows=[];selection=[];audit=[];risk={};max_grad=0.;max_replay=0.;max_preprocess=0.;info=[]
    for domain,path,expected in [('RNA',P/'folds',source()),('VISUAL',P/'visual_folds',vsource())]:
        for fold in cfg()['folds']:
            target=path/f'fold{fold}.pt';cell=json.loads(target.with_suffix('.json').read_text());z=torch.load(target,map_location='cpu',weights_only=False)
            if cell['sha256']!=sha(target) or z['source']!=expected:raise ValueError('producer artifact/source changed')
            parts=parent.splits(d,'DSS',fold)
            for k,v in parts.items():
                if not np.array_equal(v,z['parts'][k]):raise ValueError('patient partition changed')
            fit,test=parts['outer_train'],parts['outer_test'];t,e=ep['DSS']['time'],ep['DSS']['event']
            if domain=='RNA':max_preprocess=max(max_preprocess,rna_arithmetic(d,z))
            for key,m in z['models'].items():
                name,schedule=key.rsplit('__',1);label=domain+'__'+key;x=z['features']['outer'][name];r=x[test]@m['w'];err=float(abs(r-m['risk']).max());max_replay=max(max_replay,err)
                if err>1e-10:raise ValueError('risk replay mismatch')
                lam=m['lambda'];gmax=0 if lam=='OFF' else float(abs(gradient(x[fit],t[fit],e[fit],m['w'],float(lam))).max());max_grad=max(max_grad,gmax)
                if gmax>3e-5:raise ValueError('nonstationary linear probe')
                choices=z['trials'][name];best=max(c['inner_Harrell'] for c in choices);tied=[c for c in choices if abs(c['inner_Harrell']-best)<1e-12];expected_lam='OFF' if any(c['lambda']=='OFF' for c in tied) else max(c['lambda'] for c in tied)
                if schedule=='SELECTED' and lam!=expected_lam:raise ValueError('inner selection tie rule')
                if schedule=='FIXED1' and lam!=1:raise ValueError('fixed schedule changed')
                if label not in risk:risk[label]=np.full(512,np.nan)
                risk[label][test]=r
                info.append({'domain':domain,'model':name,'schedule':schedule,'fold':fold,'lambda':lam,'parameters':m['parameters'],'independent_gradient_max':gmax,'checkpoint_replay_error':err})
                for c in choices:selection.append({'domain':domain,'model':name,'schedule':schedule,'fold':fold,'candidate_lambda':c['lambda'],'inner_Harrell':c['inner_Harrell'],'selected':c['lambda']==lam})
                for endpoint in ['DSS','PFI']:
                    tt,ee,valid=[ep[endpoint][k] for k in ['time','event','valid']];tr=fit[valid[fit]];te=test[valid[test]];rr=r[valid[test]];weights=ipcw(tt[tr],ee[tr],tt[te],ee[te],60);hn,hd=components(tt[te],ee[te],rr);un,ud=components(tt[te],ee[te],rr,weights)
                    h=concordance_index_censored(ee[te].astype(bool),tt[te],rr)[0];u=concordance_index_ipcw(Surv.from_arrays(ee[tr].astype(bool),tt[tr]),Surv.from_arrays(ee[te].astype(bool),tt[te]),rr,tau=60)[0]
                    if abs(h-hn/hd)>1e-10 or abs(u-un/ud)>1e-10:raise ValueError('independent metric crosscheck')
                    for seed in cfg()['seeds']:rows.append({'domain':domain,'model':name,'schedule':schedule,'endpoint':endpoint,'fold':fold,'seed':seed,'deterministic_reference':seed!=31,'n':len(te),'events':int(ee[te].sum()),'Harrell':h,'Uno60':u,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud})
            if domain=='RNA':
                st=z['states']['outer'];s=st['RNA_states'];counts=np.unique(s[test],axis=0,return_counts=True)[1];cont=st['continuous'];variance=[]
                for j in range(8):
                    q=np.array([cont[fit,j][s[fit,j]==v].mean() if np.any(s[fit,j]==v) else 0 for v in ['low','typical','high','unmeasured']]);recon=np.array([q[['low','typical','high','unmeasured'].index(v)] for v in s[test,j]]);variance.append(float(np.mean((cont[test,j]-recon)**2)/max(np.var(cont[test,j]),1e-12)))
                audit.append({'fold':fold,'test_patients':len(test),'unique_joint_quantized_states':len(counts),'patients_in_repeated_states':int(counts[counts>1].sum()),'quantized_scalar_reconstruction_relative_MSE_mean':np.mean(variance),'same512_uncovered_genes':st['same512_uncovered']})
    table=pd.DataFrame(rows);table.to_csv(R/'FOLD_METRICS.csv',index=False);pd.DataFrame(info).to_csv(R/'OPTIMIZATION_AUDIT.csv',index=False);pd.DataFrame(selection).to_csv(R/'INNER_SELECTION.csv',index=False);pd.DataFrame(audit).to_csv(R/'QUANTIZATION_DIAGNOSTICS.csv',index=False)
    out=[]
    for keys,g in table.groupby(['domain','model','schedule','endpoint','seed']):out.append({**dict(zip(['domain','model','schedule','endpoint','seed'],keys)),'Harrell':g.H_num.sum()/g.H_den.sum(),'Uno60':g.U_num.sum()/g.U_den.sum()})
    seed_table=pd.DataFrame(out);seed_table.to_csv(R/'SEED_OOF_METRICS.csv',index=False);seed_table.groupby(['domain','model','schedule','endpoint'])[['Harrell','Uno60']].mean().reset_index().to_csv(R/'SUMMARY.csv',index=False)
    np.savez_compressed(P/'OOF_PRIVATE.npz',**risk)
    comparisons=[]
    for schedule in ['FIXED1','SELECTED']:
        for a,b in cfg()['primary_contrasts']:comparisons.append(('RNA',a,b,schedule,'RNA_Bonferroni12'))
        for context in ['VISUAL_ONLY','CS_PLUS_VISUAL']:
            for a,b in json.loads((ROOT/'configs/phase16_visual_diagnostic.json').read_text())['primary_contrasts']:comparisons.append(('VISUAL',context+'__'+a,context+'__'+b,schedule,'VISUAL_Bonferroni12'))
    paired=[];transitions=[]
    for endpoint in ['DSS','PFI']:
        t,e,valid=[ep[endpoint][k] for k in ['time','event','valid']];local=np.flatnonzero(valid);i,j=pairs(t[local],e[local]);i,j=local[i],local[j];keep=d['folds'][i]==d['folds'][j];i,j=i[keep],j[keep];weights=np.zeros(512)
        for f in cfg()['folds']:
            tr=np.flatnonzero(valid&(d['folds']!=f));te=np.flatnonzero(valid&(d['folds']==f));weights[te]=ipcw(t[tr],e[tr],t[te],e[te],60)
        credit={}
        for k,r in risk.items():delta=r[i]-r[j];credit[k]=np.where(abs(delta)<=1e-8,.5,delta>0)
        differences=np.stack([credit[domain+'__'+a+'__'+schedule]-credit[domain+'__'+b+'__'+schedule] for domain,a,b,schedule,_ in comparisons],1);rng=np.random.default_rng(cfg()['bootstrap_seed']+(endpoint=='PFI'));draws={m:[] for m in ['Harrell','Uno60']}
        for start in range(0,cfg()['bootstrap_draws'],250):
            counts=np.zeros((250,512),int);counts[:,local]=rng.multinomial(len(local),np.ones(len(local))/len(local),250);mass=(counts[:,i]*counts[:,j]).astype(float)
            for metric in draws:q=mass*(np.ones(len(i)) if metric=='Harrell' else weights[i]);draws[metric].append(q@differences/q.sum(1)[:,None])
        for metric in draws:
            sample=np.concatenate(draws[metric]);w=np.ones(len(i)) if metric=='Harrell' else weights[i]
            for col,(domain,a,b,schedule,family) in enumerate(comparisons):
                lo,hi=np.quantile(sample[:,col],[.025,.975]);cl,ch=np.quantile(sample[:,col],[.025/12,1-.025/12]);delta=float(w@differences[:,col]/w.sum());paired.append({'domain':domain,'comparison':a+'-'+b,'schedule':schedule,'endpoint':endpoint,'metric':metric,'delta':delta,'CI_low':lo,'CI_high':hi,'corrected_low':cl,'corrected_high':ch,'family':family,'bootstrap_draws':cfg()['bootstrap_draws']})
                ca,cb=credit[domain+'__'+a+'__'+schedule],credit[domain+'__'+b+'__'+schedule];transitions.append({'domain':domain,'comparison':a+'-'+b,'schedule':schedule,'endpoint':endpoint,'metric':metric,'gained_pair_mass':float(w@np.maximum(ca-cb,0)/w.sum()),'lost_pair_mass':float(w@np.maximum(cb-ca,0)/w.sum()),'total_pairs':len(i)})
    pd.DataFrame(paired).to_csv(R/'PAIRED_COMPARISONS.csv',index=False);pd.DataFrame(transitions).to_csv(R/'PAIR_TRANSITIONS.csv',index=False)
    correlations=[]
    for name in cfg()['representations']:
        for schedule in ['FIXED1','SELECTED']:
            for fold in cfg()['folds']:
                ix=np.flatnonzero(d['folds']==fold);a=risk['RNA__'+name+'__'+schedule][ix];b=risk['RNA__HALL512__'+schedule][ix];rho=float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None
                correlations.append({'representation':name,'schedule':schedule,'fold':fold,'heldout_risk_spearman_vs_HALL512':rho,'scope':'within-fold descriptive risk-rank association'})
    pd.DataFrame(correlations).to_csv(R/'RISK_CORRELATIONS.csv',index=False)
    actual=sum(row['lambda']!='OFF' for row in info);constant=len(info)-actual
    result={'status':'PASS','unique_outer_model_evaluations':len(info),'actual_outer_optimizer_fits':actual,'constant_OFF_references':constant,'deterministic_seed_references':2*len(info),'seed_fold_metric_records':len(table),'independent_gradient_max':max_grad,'risk_replay_error':max_replay,'RNA_preprocessing_independent_max_error':max_preprocess,'all_metrics_crosschecked_sksurv':True,'seeds_not_independent':True,'raw_endpoint_verified':True,'source':source(),'visual_source':vsource()};write(R/'EVALUATION_AUDIT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    torch.set_num_threads(2)
    while not (P/'VISUAL_TRAINING_DONE.json').exists():time.sleep(2)
    main()
