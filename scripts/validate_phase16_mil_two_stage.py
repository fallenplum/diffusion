import sys,gc,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase11_independent import pairs,components,ipcw
from validate_phase13_independent import raw_endpoints,test_loss
from validate_phase16_independent import cs_independent,literal_pool
from run_phase16_mil_two_stage import LP,lsource,base,parent,R,SemanticBags,write,sha
LR=R/'MIL'
def main():
    base.seed_all(1991);LR.mkdir(exist_ok=True);done=json.loads((LP/'TRAINING_DONE.json').read_text())
    if done['status']!='PASS' or done['source']!=lsource():raise ValueError('MIL module incomplete')
    d=parent.load();ev=raw_endpoints(d,{});risks={};models=[];maximum=0
    for budget in [128,512]:
        for seed in [31,47,83]:
            r=np.full(512,np.nan);old=np.full(512,np.nan)
            for fold in [1,2,3,4,5]:
                name=f'MIL_TWO_STAGE_{budget}_seed{seed}_fold{fold}';c=json.loads((LP/'cells'/f'{name}.json').read_text())
                if c['tag']!={'budget':budget,'seed':seed,'fold':fold,'source':lsource()} or c['outer_test_selection'] or sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('MIL checkpoint identity')
                z=torch.load(c['artifact'],map_location='cpu',weights_only=False);part=parent.splits(d,'DSS',fold)
                for role,st in z['states'].items():
                    fit=part['inner_train' if role=='inner' else 'outer_train'];test=part['inner_val' if role=='inner' else 'outer_test']
                    if not np.array_equal(fit,st['fit']) or not np.array_equal(test,st['test']) or not np.array_equal(st['CS_preprocessing']['fit_indices'],fit) or len(st['StageA_history'])!=80:raise ValueError('MIL stage partition/budget')
                    if any('val_cox' in row for row in st['StageA_history']):raise ValueError('StageA upstream used validation labels')
                    A={k:v.cuda() for k,v in st['StageA_state'].items()};B={k:v.cuda() for k,v in st['StageB_state'].items()}
                    if not torch.equal(A['score'],B['score']) or any(row['gradient']['score']!=0 for row in st['StageB_history']):raise ValueError('StageB changed frozen upstream selector')
                    cs=torch.tensor(cs_independent(d,fit),dtype=torch.float32,device='cuda');bagsA=base.Bags('MIL',budget,seed)
                    with torch.no_grad():pred=literal_pool(bagsA,A['score'],A['roi'],cs,A['base']).cpu().numpy()
                    maximum=max(maximum,float(abs(pred-st['StageA_all_risk']).max()));selected=np.load(st['ROI_path'])
                    if sha(st['ROI_path'])!=st['ROI_sha256']:raise ValueError('MIL selected ROI changed')
                    for x,mask,quota,ids,order,k,qn in bagsA.buckets:
                        values=(x@A['score']).masked_fill(~mask,-torch.inf);ix=values.topk(k,dim=1,sorted=True).indices.cpu().numpy()
                        for j,o in enumerate(order):
                            expected=ids[j,ix[j,:qn[j]]]
                            if not np.array_equal(expected,selected[o,:qn[j]]):raise ValueError('StageA topM ROI does not replay')
                    del bagsA;gc.collect();torch.cuda.empty_cache();bags=SemanticBags(st['ROI_path'],budget)
                    with torch.no_grad():pred=literal_pool(bags,B['score'],B['roi'],cs,B['base']).cpu().numpy()
                    if role=='outer':maximum=max(maximum,float(abs(pred[test]-z['risk']).max()))
                    else:
                        values=[];ema=None
                        for row in st['StageB_history']:ema=row['val_cox'] if ema is None else .3*row['val_cox']+.7*ema;values.append(ema)
                        epoch=int(np.argmin(values))+1
                        if epoch!=z['selected_epoch'] or abs(test_loss(ev['DSS']['time'][test],ev['DSS']['event'][test],pred[test])-st['StageB_history'][epoch-1]['val_cox'])>2e-6:raise ValueError('MIL StageB epoch replay')
                    if float(B['base'][-32:].norm())<=0:raise ValueError('S32 disconnected')
                    del bags;gc.collect();torch.cuda.empty_cache()
                te=part['outer_test'];r[te]=z['risk'];cell=json.loads((base.P/'cells'/f'MIL_{budget}_seed{seed}_fold{fold}.json').read_text());zz=torch.load(cell['artifact'],map_location='cpu',weights_only=False);old[te]=zz['risk']
                if z['parameters']!=zz['parameters']:raise ValueError('end-to-end deployment capacity not matched')
                stageA_parameters=sum(v.numel() for v in z['states']['outer']['StageA_state'].values());models.append({'arm':f'MIL_TWO_STAGE_{budget}','seed':seed,'fold':fold,'parameters':z['parameters'],'StageB_trainable_parameters':z['StageB_trainable_parameters'],'StageA_trainable_parameters':stageA_parameters,'unique_staged_training_parameters':stageA_parameters+z['StageB_trainable_parameters'],'extra_selector_epochs_per_role':80,'extra_StageA_role_fits':2,'seconds':c['seconds'],'selected_epoch':z['selected_epoch']});print('MIL_REPLAY',name,flush=True)
            risks[f'MIL_TWO_STAGE_{budget}',seed]=r;risks[f'MIL_{budget}',seed]=old
    if maximum>3e-6:raise ValueError('MIL prediction mismatch')
    foldrows=[];weights={};seedrows=[];ci=[]
    for ep in ['DSS','PFI']:
        t,e,valid=[ev[ep][k] for k in ['time','event','valid']];weights[ep]=np.zeros(512)
        for fold in [1,2,3,4,5]:
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));iw=ipcw(t[tr],e[tr],t[te],e[te],60);weights[ep][te]=iw
            for (arm,seed),r in risks.items():
                hn,hd=components(t[te],e[te],r[te]);un,ud=components(t[te],e[te],r[te],iw);foldrows.append({'endpoint':ep,'arm':arm,'seed':seed,'fold':fold,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud,'Harrell':hn/hd,'Uno60':un/ud})
    df=pd.DataFrame(foldrows)
    for (ep,arm,seed),g in df.groupby(['endpoint','arm','seed']):seedrows.append({'endpoint':ep,'arm':arm,'seed':seed,'Harrell':g.H_num.sum()/g.H_den.sum(),'Uno60':g.U_num.sum()/g.U_den.sum()})
    for ep in ['DSS','PFI']:
        t,e,valid=[ev[ep][k] for k in ['time','event','valid']];local=np.flatnonzero(valid);i,j=pairs(t[local],e[local]);i,j=local[i],local[j];use=d['folds'][i]==d['folds'][j];i,j=i[use],j[use];delta=[]
        for budget in [128,512]:
            seedcredits=[]
            for seed in [31,47,83]:
                a=risks[f'MIL_TWO_STAGE_{budget}',seed];b=risks[f'MIL_{budget}',seed];dd=a[i]-a[j];ca=np.where(abs(dd)<=1e-8,.5,dd>0);dd=b[i]-b[j];cb=np.where(abs(dd)<=1e-8,.5,dd>0);seedcredits.append(ca-cb)
            delta.append(np.mean(seedcredits,0))
        delta=np.stack(delta,1);rng=np.random.default_rng(1608531+(ep=='PFI'));draw={m:[] for m in ['Harrell','Uno60']}
        for start in range(0,30000,250):
            n=np.zeros((250,512),int);n[:,local]=rng.multinomial(len(local),np.ones(len(local))/len(local),250);q=(n[:,i]*n[:,j]).astype(float)
            for metric in draw:w=np.ones(len(i)) if metric=='Harrell' else weights[ep][i];qq=q*w;draw[metric].append(qq@delta/qq.sum(1)[:,None])
        for metric in draw:
            samples=np.concatenate(draw[metric])
            for k,budget in enumerate([128,512]):
                points=[next(r[metric] for r in seedrows if (r['endpoint'],r['arm'],r['seed'])==(ep,f'MIL_TWO_STAGE_{budget}',s))-next(r[metric] for r in seedrows if (r['endpoint'],r['arm'],r['seed'])==(ep,f'MIL_{budget}',s)) for s in [31,47,83]];lo,hi=np.quantile(samples[:,k],[.025,.975]);cl,ch=np.quantile(samples[:,k],[.025/4,1-.025/4]);ci.append({'endpoint':ep,'comparison':f'MIL_TWO_STAGE_{budget}-MIL_{budget}','metric':metric,'delta':np.mean(points),'CI_low':lo,'CI_high':hi,'corrected_low':cl,'corrected_high':ch,'positive_seeds':sum(p>0 for p in points),'scope':'same deployed capacity;extra selector optimization budget;not compute matched'})
    df.to_csv(LR/'FOLD_METRICS.csv',index=False);pd.DataFrame(seedrows).to_csv(LR/'SEED_OOF_METRICS.csv',index=False);pd.DataFrame(models).to_csv(LR/'MODEL_SUMMARY.csv',index=False);pd.DataFrame(ci).to_csv(LR/'PAIRED_COMPARISONS.csv',index=False);result={'status':'PASS','source':lsource(),'new_outer_replays':30,'extra_StageA_role_fits':60,'deployment_capacity_matched':True,'StageA_validation_labels_used':False,'StageB_attention_frozen_verified':True,'max_replay_error':maximum};write(LR/'INDEPENDENT_AUDIT.json',result);write(LP/'EVALUATION_DONE.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    while not (LP/'TRAINING_DONE.json').exists():time.sleep(2)
    main()
