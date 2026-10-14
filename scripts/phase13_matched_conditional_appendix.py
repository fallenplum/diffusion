"""Post-inspection matched RAW512 control; never changes the locked core matrix.

Lambda1 is the existing probe penalty, not chosen from new outer performance.
All three gene sets and both endpoints are reported. The scope is explicitly
exploratory; no result is promoted to a pre-registered primary comparison.
"""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd
from src.phase13_rna.data import load,config,PRIVATE,REPORT,cached_features,splits,endpoints,digest,write_json,ahash
from src.phase13_rna.probes import fixed_probe
from validate_phase11_independent import pairs,ipcw,components
from replay_phase13 import coefficient_gradient

cfg=config();d=load();p=PRIVATE/'matched_conditional_appendix';p.mkdir(exist_ok=True);r=REPORT
lock={'scope':'POSTHOC_EXPLORATORY_MATCHED_CAPACITY_CONTROL','reason':'coreRAWconditional tunesRNA penalty butlatentprobes usefixed1;matchRAW512 conditional toexistingprobe,not outerselectedwinner','RNA_penalty':1.0,'baseline_penalty':'sameinner-only selected CS lambda fromlockedcore','genes':list(cfg['genes']),'endpoints':cfg['endpoints'],'primary_upgrade':False,'parent_core_training_source_hash':json.loads((r/'HARD_GATES.json').read_text())['source_hash'],'appendix_source_sha256':digest(__file__)}
write_json(r/'APPENDIX_PROTOCOL_LOCK.json',lock)
model_risks={};rows=[];verification=[]
for ep in cfg['endpoints']:
    ev=endpoints(d)[ep];t,e,valid=[ev[k] for k in ['time','event','valid']];weight=np.zeros(512)
    for f in cfg['folds']:
        part=splits(d,ep,f);tr,te=part['outer_train'],part['outer_test'];weight[te]=ipcw(t[tr],e[tr],t[te],e[te])
    for gene in cfg['genes']:
        risk=np.full(512,np.nan);sums=np.zeros(4)
        for fold in cfg['folds']:
            part=splits(d,ep,fold);tr,te=part['outer_train'],part['outer_test'];fo,_,audit=cached_features(d,ep,fold,'outer');base=np.c_[fo['C'],fo['S']].astype(float);baseline=json.loads((PRIVATE/'formal/cells'/f'{ep}_CS_seed31_fold{fold}.json').read_text());lam=baseline['lambda']
            prediction,w,state=fixed_probe(fo[gene][tr],fo[gene][te],t[tr],e[tr],1.0,base[tr],base[te],lam);risk[te]=prediction
            x=(fo[gene][tr].astype(float)-np.array(state['mean']))/np.array(state['sd']);x=np.c_[base[tr],x];gradient=float(abs(coefficient_gradient(x,t[tr],e[tr],w,np.array(state['penalty']))).max())
            xx=np.c_[base[te],(fo[gene][te].astype(float)-np.array(state['mean']))/np.array(state['sd'])];error=float(abs(xx@w-prediction).max())
            if gradient>2e-5 or error>2e-6:raise RuntimeError('appendix independent gradient/replay')
            stem=f'{ep}_{gene}_fixed1_fold{fold}';file=p/f'{stem}.npz';np.savez_compressed(file,test_indices=te,risk=prediction,w=w,time=t[te],event=e[te],normalization_mean=state['mean'],normalization_sd=state['sd'])
            write_json(p/f'{stem}.json',{'protocol':lock,'outer_train':tr.tolist(),'outer_test':te.tolist(),'preprocessing':audit,'base_lambda':lam,'RNA_lambda':1,'file_sha256':digest(file),'parameters':len(w),'gradient_max':gradient,'actual_fit':True,'seeds':'same unique convexfit,not independentinitializations'})
            hn,hd=components(t[te],e[te],prediction);un,ud=components(t[te],e[te],prediction,weight[te]);sums+=np.array([hn,hd,un,ud]);verification.append({'endpoint':ep,'gene':gene,'fold':fold,'independent_gradient_max':gradient,'replay_max_error':error,'artifact_sha256':digest(file)})
        model_risks[ep,gene]=risk;rows.append({'endpoint':ep,'gene':gene,'Harrell':sums[0]/sums[1],'Uno60':sums[2]/sums[3],'unique_outer_fits':5,'scope':lock['scope']})
pd.DataFrame(rows).to_csv(r/'APPENDIX_FIXED_RAW_SUMMARY.csv',index=False);pd.DataFrame(verification).to_csv(r/'APPENDIX_CHECKPOINT_AUDIT.csv',index=False)
contrasts={};paired=[]
for ep in cfg['endpoints']:
    ev=endpoints(d)[ep];t,e,valid=[ev[k] for k in ['time','event','valid']];indices=np.flatnonzero(valid);weights=np.zeros(512)
    for fold in cfg['folds']:
        tr=valid&(d['folds']!=fold);te=valid&(d['folds']==fold);weights[te]=ipcw(t[tr],e[tr],t[te],e[te])
    pi,pj=pairs(t,e);keep=valid[pi]&valid[pj]&(d['folds'][pi]==d['folds'][pj]);pi,pj=pi[keep],pj[keep];credit=[];names=[]
    def ranking(r):
        diff=r[pi]-r[pj];return np.where(abs(diff)<=1e-8,.5,diff>0)
    def original(model,seed):
        r=np.full(512,np.nan)
        for f in cfg['folds']:
            z=np.load(PRIVATE/'formal/oof'/f'{ep}_{model}_seed{seed}_fold{f}.npz');r[z['test_indices']]=z['risk']
        return r
    for gene in cfg['genes']:
        fixed=ranking(model_risks[ep,gene]);base=ranking(original('CS',31))
        for name,values in [(gene+'_fixedRAW-CS',fixed-base),
                            (gene+'_CS_Fused-fixRAW',np.mean([ranking(original('CS_F_'+gene+'_after_processed_full',s)) for s in cfg['seeds']],0)-fixed),
                            (gene+'_CS_RNAonly-fixRAW',np.mean([ranking(original('CS_RO_'+gene+'_after_processed_full',s)) for s in cfg['seeds']],0)-fixed)]:names.append(name);credit.append(values)
    credit=np.stack(credit,1);rng=np.random.default_rng(139812+(0 if ep=='DSS' else 1));boot={m:[] for m in ['Harrell','Uno60']}
    for start in range(0,cfg['bootstrap']['draws'],250):
        n=min(250,cfg['bootstrap']['draws']-start);count=np.zeros((n,512),int);count[:,indices]=rng.multinomial(len(indices),np.ones(len(indices))/len(indices),size=n);q=(count[:,pi]*count[:,pj]).astype(float);qu=q*weights[pi];boot['Harrell'].append(q@credit/q.sum(1)[:,None]);boot['Uno60'].append(qu@credit/qu.sum(1)[:,None])
    for metric,blocks in boot.items():
        samples=np.concatenate(blocks);weight=np.ones(len(pi)) if metric=='Harrell' else weights[pi];point=weight@credit/weight.sum()
        for j,name in enumerate(names):
            lo,hi=np.quantile(samples[:,j],[.025,.975]);cl,ch=np.quantile(samples[:,j],[.05/72,1-.05/72]);paired.append({'endpoint':ep,'comparison':name,'metric':metric,'mean_delta':point[j],'CI95_low':lo,'CI95_high':hi,'family36_low':cl,'family36_high':ch,'draws':len(samples),'scope':'POSTHOC,36 contrasts corrected,not primary'})
pd.DataFrame(paired).to_csv(r/'APPENDIX_PAIRED.csv',index=False);write_json(r/'APPENDIX_VALIDATION.json',{'status':'PASS','unique_new_outer_fits':30,'formal_core_records_changed':False,'source_hash_core_unchanged':lock['parent_core_training_source_hash'],'independent_gradient_and_replay':True,'both_endpoints_all3sets_reported':True,'posthoc':True});print('MATCHED_CONDITIONAL_APPENDIX_COMPLETE')
