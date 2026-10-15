import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase11_independent import pairs,ipcw
from validate_phase13_independent import raw_endpoints
from src.phase16_semantic.omics import *
from run_phase16_mil_two_stage import LP

def main():
    d=parent.load();endpoints=raw_endpoints(d,{});pairs_to_test=[]
    for m in [128,512]:
        for kind in ['RNA','RC_UNGATED','RC_GATED']:pairs_to_test.append((f'G5_CLIN_RNA_TEXT_{m}_{kind}',f'G5_FIXED_TEXT_{m}_{kind}'))
        pairs_to_test.append((f'CLIN_RNA_TEXT_{m}',f'MIL_TWO_STAGE_{m}'))
    pairs_to_test += [('G5_RNA_ONLY','NULL_CONSTANT'),('G5_CNA_ONLY','NULL_CONSTANT')];risks={}
    for arm in sorted(set(v for pair in pairs_to_test for v in pair)):
        for seed in [31,47,83]:
            if arm=='NULL_CONSTANT':risks[arm,seed]=np.zeros(512);continue
            directory=MP if arm.startswith('G5_') else LP if arm.startswith('MIL_TWO_STAGE') else P;risk=np.full(512,np.nan)
            for fold in [1,2,3,4,5]:
                c=json.loads((directory/'cells'/f'{label(arm,seed,fold)}.json').read_text())
                if sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('direct contrast artifact changed')
                z=torch.load(c['artifact'],map_location='cpu',weights_only=False);risk[z['test_indices']]=z['risk']
            if not np.isfinite(risk).all():raise ValueError('direct contrast incomplete')
            risks[arm,seed]=risk
    out=[]
    for ep in ['DSS','PFI']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];local=np.flatnonzero(valid);i,j=pairs(t[local],e[local]);i,j=local[i],local[j];keep=d['folds'][i]==d['folds'][j];i,j=i[keep],j[keep];weight=np.zeros(512)
        for fold in [1,2,3,4,5]:
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));weight[te]=ipcw(t[tr],e[tr],t[te],e[te],60)
        credits={}
        for key,risk in risks.items():delta=risk[i]-risk[j];credits[key]=np.where(abs(delta)<=1e-8,.5,delta>0)
        delta=np.stack([np.mean([credits[a,s]-credits[b,s] for s in [31,47,83]],0) for a,b in pairs_to_test],1);rng=np.random.default_rng(1608531+(ep=='PFI'));draws={m:[] for m in ['Harrell','Uno60']}
        for start in range(0,30000,250):
            counts=np.zeros((250,512),int);counts[:,local]=rng.multinomial(len(local),np.ones(len(local))/len(local),250);q=(counts[:,i]*counts[:,j]).astype(float)
            for metric in draws:w=np.ones(len(i)) if metric=='Harrell' else weight[i];qq=q*w;draws[metric].append(qq@delta/qq.sum(1)[:,None])
        for metric in draws:
            samples=np.concatenate(draws[metric]);w=np.ones(len(i)) if metric=='Harrell' else weight[i]
            for k,(a,b) in enumerate(pairs_to_test):
                points=[float(w@(credits[a,s]-credits[b,s])/w.sum()) for s in [31,47,83]];lo,hi=np.quantile(samples[:,k],[.025,.975]);cl,ch=np.quantile(samples[:,k],[.025/20,1-.025/20]);out.append({'endpoint':ep,'comparison':a+'-'+b,'metric':metric,'delta':np.mean(points),'CI_low':lo,'CI_high':hi,'corrected_low':cl,'corrected_high':ch,'positive_seeds':sum(v>0 for v in points),'family':'supplementary_direct_Bonferroni20',**{f'seed{s}':v for s,v in zip([31,47,83],points)}})
    pd.DataFrame(out).to_csv(R/'DIRECT_PAIRED_COMPARISONS.csv',index=False);write(R/'DIRECT_COMPARISON_AUDIT.json',{'status':'PASS','pairs':10,'metrics':2,'bootstrap_draws':30000,'constant_reference_new_fits':0,'patient_shared_weights':True,'primary_family_unchanged':True,'protocol_sha256':sha(ROOT/'configs/phase16_direct_comparisons.json')});print('DIRECT_COMPARISONS_PASS',flush=True)
if __name__=='__main__':
    while not (MP/'EVALUATION_DONE.json').exists() or not (LP/'EVALUATION_DONE.json').exists():time.sleep(2)
    main()
