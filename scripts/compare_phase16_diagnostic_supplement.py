"""Posthoc direct OOF contrasts answering dual64/single and conditional headroom."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd
from src.phase16_diagnostic.core import *
from validate_phase11_independent import pairs,ipcw
from validate_phase13_independent import raw_endpoints

def main():
    comparisons=[]
    for context in ['VISUAL_ONLY','CS_PLUS_VISUAL']:
        for single in ['CONCH32','UNI32']:comparisons.append((context+'__DUAL64',context+'__'+single))
    for model in ['CONCH32','UNI32','DUAL32','DUAL64']:comparisons.append(('CS_PLUS_VISUAL__'+model,'CS'))
    lock={'comparisons':comparisons,'family_tests':16,'primary':False,'timing':'defined after inspecting core point estimates;no model fitting or choice;all requested direct contrasts reported','bootstrap_draws':30000}
    write(R/'SUPPLEMENTARY_CONTRAST_SCOPE.json',lock);d=parent.load();ep=raw_endpoints(d,{});risk=np.load(P/'OOF_PRIVATE.npz');out=[]
    for endpoint in ['DSS','PFI']:
        t,e,valid=[ep[endpoint][k] for k in ['time','event','valid']];local=np.flatnonzero(valid);i,j=pairs(t[local],e[local]);i,j=local[i],local[j];keep=d['folds'][i]==d['folds'][j];i,j=i[keep],j[keep];weights=np.zeros(512)
        for fold in cfg()['folds']:
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));weights[te]=ipcw(t[tr],e[tr],t[te],e[te],60)
        for schedule in ['FIXED1','SELECTED']:
            credit={}
            for model in sorted(set(v for pair in comparisons for v in pair)):
                r=risk['VISUAL__'+model+'__'+schedule];delta=r[i]-r[j];credit[model]=np.where(abs(delta)<=1e-8,.5,delta>0)
            delta=np.stack([credit[a]-credit[b] for a,b in comparisons],1);rng=np.random.default_rng(1610991+(endpoint=='PFI'));draws={m:[] for m in ['Harrell','Uno60']}
            for start in range(0,30000,250):
                counts=np.zeros((250,512),int);counts[:,local]=rng.multinomial(len(local),np.ones(len(local))/len(local),250);mass=(counts[:,i]*counts[:,j]).astype(float)
                for metric in draws:w=np.ones(len(i)) if metric=='Harrell' else weights[i];q=mass*w;draws[metric].append(q@delta/q.sum(1)[:,None])
            for metric in draws:
                sample=np.concatenate(draws[metric]);w=np.ones(len(i)) if metric=='Harrell' else weights[i]
                for col,(a,b) in enumerate(comparisons):
                    lo,hi=np.quantile(sample[:,col],[.025,.975]);cl,ch=np.quantile(sample[:,col],[.025/16,1-.025/16]);out.append({'endpoint':endpoint,'schedule':schedule,'comparison':a+'-'+b,'metric':metric,'delta':float(w@delta[:,col]/w.sum()),'CI_low':lo,'CI_high':hi,'corrected_low':cl,'corrected_high':ch,'family':'posthoc_direct_Bonferroni16'})
    pd.DataFrame(out).to_csv(R/'SUPPLEMENTARY_PAIRED_COMPARISONS.csv',index=False);write(R/'SUPPLEMENTARY_AUDIT.json',{'status':'PASS','new_fits':0,'not_primary':True,'models_selected_from_results':False});print('SUPPLEMENTARY_DIRECT_PASS',flush=True)

if __name__=='__main__':main()
