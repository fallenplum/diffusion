"""Independent evaluation of a disclosed post-inspection budget sensitivity."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from budget_phase14 import APP,ACONFIG,appendix_hash
from run_phase14 import *
from validate_phase11_independent import pairs,components,ipcw
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
from sksurv.util import Surv


def main():
    configuration=cfg();d=parent.load();endpoints=parent.endpoints(d);risks={};rows=[];budget=[];maximum=0;trajectory_max=0;weights={};artifact_rows=[]
    torch.set_num_threads(2)
    for ep in configuration['endpoints']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];weights[ep]=np.zeros(512)
        for gene in configuration['genes']:
            for seed in configuration['seeds']:
                for arm in ['CS','COX','ADDITIVE','INTERACT','WRONG_ADDITIVE','WRONG_INTERACT']:risks[ep,gene,seed,arm]=np.full(512,np.nan)
                for fold in configuration['folds']:
                    name=unit(ep,gene,seed,fold);cell=json.loads((APP/'cells'/f'{name}.json').read_text());z=torch.load(cell['artifact'],map_location='cpu',weights_only=False)
                    if z['appendix_hash']!=appendix_hash() or cell['appendix_hash']!=appendix_hash() or sha(cell['artifact'])!=cell['sha256']:raise RuntimeError('appendix artifact')
                    artifact_rows.append({'pipeline':name,'artifact_sha256':cell['sha256'],'artifact_bytes':Path(cell['artifact']).stat().st_size})
                    pcell=json.loads((PRIVATE/'cells'/f'{name}.json').read_text());p=torch.load(pcell['artifact'],map_location='cpu',weights_only=False)
                    f,m,a,tr,te=role_data(ep,gene,fold,'outer');weights[ep][te]=ipcw(t[tr],e[tr],t[te],e[te],60);cs=np.c_[f['C'],f['S']]
                    if not np.array_equal(te,z['partitions']['outer_test']) or not np.array_equal(tr,z['partitions']['outer_train']):raise RuntimeError('appendix patients')
                    producer=RNAOnly(m['clinical_mask'],m[gene],seed).cuda();producer.load_state_dict(p['states']['outer_RANK']);_,a=extract_rna(producer,f[gene],tr);_,b=extract_rna(producer,f[gene],te);del producer
                    _,r,transform=reduce_rep(a['processed_full'],b['processed_full'],64)
                    for key in ['input_mean','input_sd','pca_mean','components','output_mean','output_sd']:
                        if not np.array_equal(transform[key],p['transforms']['outer_RANK64'][key]):raise RuntimeError('exact representation drift')
                    for arm,interaction,wrong in [('ADDITIVE',False,False),('INTERACT',True,False),('WRONG_ADDITIVE',False,True),('WRONG_INTERACT',True,True)]:
                        rr=r[index_map(te,p['permutations']['outer_test'])] if wrong else r;q=Conditional(cs.shape[1],64,np.zeros(cs.shape[1]+64),interaction,configuration,seed).cuda();q.load_state_dict(z['states']['outer_'+arm]);q.eval()
                        with torch.no_grad():replay=q(torch.tensor(cs[te],device='cuda'),torch.tensor(rr,device='cuda')).cpu().numpy()
                        error=float(abs(replay-z['risks'][arm]).max());maximum=max(maximum,error)
                        if error>3e-6:raise RuntimeError('appendix replay')
                        inner=z['infos']['inner_'+arm];outer=z['infos']['outer_'+arm];logs=inner['logs'];selected=min(logs,key=lambda v:v['validation_EMA'])['epoch'];trajectory_max=max(trajectory_max,inner['first80_EMA_max_error'])
                        if inner['actual_epochs']!=1280 or outer['actual_epochs']!=selected or selected!=inner['selected_epoch']:raise RuntimeError('appendix schedule')
                        slope=float(np.polyfit(np.arange(10),[row['validation_EMA'] for row in logs[-10:]],1)[0])
                        budget.append({'endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'arm':arm,'selected_epoch':selected,'inner_epochs':1280,'outer_epochs':outer['actual_epochs'],'first80_EMA_max_error':inner['first80_EMA_max_error'],'ceiling_and_improving':selected==1280 and slope<0,'last10_EMA_slope':slope});del q
                    for arm in ['CS','COX','ADDITIVE','INTERACT','WRONG_ADDITIVE','WRONG_INTERACT']:
                        r=z['risks'][arm] if arm in z['risks'] else p['risks'][arm];risks[ep,gene,seed,arm][te]=r
                        hn,hd=components(t[te],e[te],r);un,ud=components(t[te],e[te],r,weights[ep][te]);hh=concordance_index_censored(e[te].astype(bool),t[te],r)[0];uu=concordance_index_ipcw(Surv.from_arrays(e[tr].astype(bool),t[tr]),Surv.from_arrays(e[te].astype(bool),t[te]),r,tau=60)[0]
                        if abs(hh-hn/hd)>1e-10 or abs(uu-un/ud)>1e-10:raise RuntimeError('appendix metric')
                        rows.append({'endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'model':arm,'Harrell':hh,'Uno60':uu,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud})
    table=pd.DataFrame(rows);table.to_csv(REPORT/'BUDGET_APPENDIX_FOLD_METRICS.csv',index=False);pd.DataFrame(budget).to_csv(REPORT/'BUDGET_APPENDIX_TRAINING.csv',index=False);pd.DataFrame(artifact_rows).to_csv(REPORT/'BUDGET_APPENDIX_ARTIFACT_HASHES.csv',index=False)
    seedrows=[]
    for (ep,gene,seed,model),group in table.groupby(['endpoint','gene','seed','model']):seedrows.append({'endpoint':ep,'gene':gene,'seed':seed,'model':model,'Harrell':group.H_num.sum()/group.H_den.sum(),'Uno60':group.U_num.sum()/group.U_den.sum()})
    pd.DataFrame(seedrows).to_csv(REPORT/'BUDGET_APPENDIX_SEED_METRICS.csv',index=False)
    comparisons=[('INTERACT','ADDITIVE'),('INTERACT','COX'),('INTERACT','WRONG_INTERACT'),('INTERACT','CS')];paired=[]
    for ep in configuration['endpoints']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];local=np.flatnonzero(valid);ii,jj=pairs(t[local],e[local]);ii,jj=local[ii],local[jj];keep=d['folds'][ii]==d['folds'][jj];ii,jj=ii[keep],jj[keep];inventory=[(gene,a,b) for gene in configuration['genes'] for a,b in comparisons];credit=[]
        for gene,a,b in inventory:
            values=[]
            for seed in configuration['seeds']:
                da=risks[ep,gene,seed,a][ii]-risks[ep,gene,seed,a][jj];db=risks[ep,gene,seed,b][ii]-risks[ep,gene,seed,b][jj];values.append(np.where(abs(da)<=1e-8,.5,da>0)-np.where(abs(db)<=1e-8,.5,db>0))
            credit.append(np.mean(values,axis=0))
        credit=np.stack(credit,1);rng=np.random.default_rng(configuration['bootstrap_seed']+(ep=='PFI'));samples={'Harrell':[],'Uno60':[]}
        for start in range(0,30000,250):
            counts=np.zeros((250,512),int);counts[:,local]=rng.multinomial(len(local),np.ones(len(local))/len(local),size=250);q=(counts[:,ii]*counts[:,jj]).astype(float);qw=q*weights[ep][ii];samples['Harrell'].append(q@credit/q.sum(1)[:,None]);samples['Uno60'].append(qw@credit/qw.sum(1)[:,None])
        for metric in samples:
            draws=np.concatenate(samples[metric]);ww=np.ones(len(ii)) if metric=='Harrell' else weights[ep][ii]
            for j,(gene,a,b) in enumerate(inventory):
                lo,hi=np.quantile(draws[:,j],[.025,.975]);alpha=.05/24 if ep=='DSS' else .05;cl,ch=np.quantile(draws[:,j],[alpha/2,1-alpha/2]);point=float(ww@credit[:,j]/ww.sum());paired.append({'endpoint':ep,'gene':gene,'comparison':a+'-'+b,'metric':metric,'delta':point,'CI_low':lo,'CI_high':hi,'corrected_low':cl,'corrected_high':ch,'post_inspection_appendix':True})
    pd.DataFrame(paired).to_csv(REPORT/'BUDGET_APPENDIX_PAIRED.csv',index=False)
    result={'status':'PASS','pipelines':len(artifact_rows),'new_readout_fold_records':360,'total_fold_records_including_reused_baselines':len(rows),'max_replay_error':maximum,'first80_EMA_max_error':trajectory_max,'remaining_ceiling_improving':sum(v['ceiling_and_improving'] for v in budget),'appendix_hash':appendix_hash()}
    write(REPORT/'BUDGET_APPENDIX_AUDIT.json',result);write(APP/'DONE.json',result)
    print(pd.DataFrame(seedrows).query('gene=="HALL"').groupby(['endpoint','model'])[['Harrell','Uno60']].mean().to_string());print(pd.DataFrame(paired).query('gene=="HALL" and endpoint=="DSS"').to_string(index=False));print(json.dumps(result))


if __name__=='__main__':main()
