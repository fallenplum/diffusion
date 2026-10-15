"""Independent endpoint, preprocessing, checkpoint/ROI and paired OOF audit."""
import sys,json,hashlib,gc,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase11_independent import pairs,components,ipcw
from validate_phase13_independent import raw_endpoints,test_loss
from validate_phase16_independent import cs_independent,literal_pool
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
from sksurv.util import Surv
from src.phase16_semantic.core import *

def independent_molecular(d,audit,fit):
    if not np.array_equal(audit['fit'],fit):raise ValueError('RNA preprocessing partition')
    raw=np.asarray(d['rna_genes'],float);median=np.nanmedian(raw[fit],0);median=np.where(np.isfinite(median),median,0);value=np.where(np.isfinite(raw),raw,median);mean=value[fit].mean(0);sd=value[fit].std(0);usable=(np.isfinite(raw[fit]).mean(0)>=.9)&(sd>1e-8);z=(value-mean)/np.where(sd>1e-8,sd,1);st=audit['gene_preprocessing']
    for key,expected in [('median',median),('mean',mean),('sd',sd),('usable',usable)]:
        if not np.array_equal(st[key],expected):raise ValueError('RNA median/scale/eligibility not train-only')
    groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text());lookup={s:j for j,s in enumerate(d['rna_gene_symbols'].astype(str))};states=[]
    for row,(name,genes) in zip(audit['groups'],groups.items()):
        ix=np.array([lookup[g] for g in genes if g in lookup and usable[lookup[g]]]);coverage=np.isfinite(raw[:,ix]).mean(1);score=z[:,ix].mean(1);good=coverage>=.7;q=np.quantile(score[fit][good[fit]],[1/3,2/3]);label=np.where(score<q[0],'low',np.where(score>q[1],'high','typical'));label[~good]='unmeasured';states.append(label)
        if row['group']!=name or not np.array_equal(ix,row['indices']) or not np.array_equal(q,row['thresholds']) or not np.array_equal(row['fit'],fit):raise ValueError('pathway membership/threshold mismatch')
    return np.stack(states,1)

def audit_prompt(d,seed,fold,role,st):
    path=Path(st['prompt_preprocessing_path'])
    if sha(path)!=st['prompt_preprocessing_sha256']:raise ValueError('prompt preprocessing corruption')
    audit=torch.load(path,map_location='cpu',weights_only=False);fit,test=st['fit'],st['test'];states=independent_molecular(d,audit['RNA'],fit)
    if not np.array_equal(states,audit['RNA_states']):raise ValueError('RNA discretization mismatch')
    for name in ['clinical_permutation','RNA_permutation']:
        perm=audit[name]
        for part in [fit,test]:
            if set(perm[part])!=set(part) or np.any(perm[part]==part):raise ValueError('patient permutation identity/leak')
    age=np.array([float(v) if str(v).replace('.','',1).isdigit() else np.nan for v in d['clinical_AGE']]);med=np.nanmedian(age[fit])
    if med!=audit['clinical']['age_median'] or not np.array_equal(audit['clinical']['fit'],fit):raise ValueError('clinical caption imputation leak')
    # Reconstruct exact actual texts only after the input operations above were
    # independently checked. SHA verifies that the locked captions were used.
    from src.phase16_semantic.prompts import captions
    query,regenerated=captions(d,fit,test,seed,fold);done=json.loads((directory(seed,fold,role)/'DONE.json').read_text())
    for selector,items in query.items():
        text_sha=hashlib.sha256('\n'.join(v for row in items for v in row).encode()).hexdigest()
        if text_sha!=done['metadata'][selector]['prompt_sha256']:raise ValueError('actual query mismatch')
    return audit

def replay(d,z,arm,seed,fold,checked):
    if z['tag']!=tag(arm,seed,fold):raise ValueError('checkpoint tag mismatch')
    part=parent.splits(d,'DSS',fold);ev=parent.endpoints(d)['DSS'];maximum=0;budget=int(arm.rsplit('_',1)[1]);owner=np.load(base.P/'row_owners.npy',mmap_mode='r');slides=base.manifest()['slides']
    for role,st in z['states'].items():
        fit=part['inner_train' if role=='inner' else 'outer_train'];test=part['inner_val' if role=='inner' else 'outer_test']
        if not np.array_equal(st['fit'],fit) or not np.array_equal(st['test'],test) or not np.array_equal(st['preprocessing']['fit_indices'],fit):raise ValueError('model fit/validation order')
        key=(seed,fold,role)
        if key not in checked:audit_prompt(d,seed,fold,role,st);checked.add(key)
        indices=Path(st['ROI_path'])
        if sha(indices)!=st['ROI_sha256']:raise ValueError('ROI changed')
        selected=np.load(indices);quota=np.load(base.P/'selectors'/f'quota_{budget}.npy');total=np.zeros(512,int)
        for i,s in enumerate(slides):ids=selected[i,:quota[i]];base.check_rows(ids,owner,s['patient']);total[s['patient']]+=len(ids)
        available=np.bincount(owner.astype(int),minlength=512)
        if not np.array_equal(total,np.minimum(budget,available)):raise ValueError('wrong patient budget')
        bags=SemanticBags(indices,budget);cs=torch.tensor(cs_independent(d,fit),dtype=torch.float32,device='cuda');state={k:v.cuda() for k,v in st['model'].items()}
        with torch.no_grad():risk=literal_pool(bags,state['score'],state['roi'],cs,state['base']).cpu().numpy()
        if role=='outer':maximum=max(maximum,float(np.max(abs(risk[test]-z['risk']))))
        else:
            rows=z['history']['inner'];ema=None;trace=[]
            for row in rows:ema=row['val_cox'] if ema is None else .3*row['val_cox']+.7*ema;trace.append(ema)
            epoch=int(np.argmin(trace))+1
            if epoch!=z['selected_epoch'] or abs(test_loss(ev['time'][test],ev['event'][test],risk[test])-rows[epoch-1]['val_cox'])>2e-6:raise ValueError('inner checkpoint/epoch selection replay')
        del bags;gc.collect();torch.cuda.empty_cache()
    if maximum>3e-6:raise ValueError('prediction replay mismatch')
    return maximum

def comparisons():
    primary=json.loads(base.CONFIG.read_text())['primary_contrasts'];secondary=[]
    for m in cfg()['budgets']:
        secondary += [(f'{a}_{m}',f'{b}_{m}') for a,b in [('CLIN_TEXT','WRONG_CLIN'),('RNA_TEXT','WRONG_RNA'),('CLIN_RNA_TEXT','JOINT_WRONG_RNA'),('CLIN_RNA_TEXT','JOINT_WRONG_CLIN'),('CLIN_RNA_TEXT','SHUFFLED_FIELDS'),('CLIN_RNA_TEXT','GRID'),('CLIN_RNA_TEXT','MIL')]]+[(f'CLIN_RNA_TEXT_{m}','C4_LEGACY')]
    if len(primary)!=8 or len(secondary)!=16:raise ValueError('comparison registry')
    return [(a,b,'primary',16) for a,b in primary]+[(a,b,'mechanism_secondary',32) for a,b in secondary]

def statistics(d,endpoints,weights,risks,seedrows):
    output=[];checks=[]
    for ep in ['DSS','PFI']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];local=np.flatnonzero(valid);i,j=pairs(t[local],e[local]);i,j=local[i],local[j];ok=d['folds'][i]==d['folds'][j];i,j=i[ok],j[ok];credits={}
        for arm,seed in risks:
            rr=risks[arm,seed];delta=rr[i]-rr[j];credits[arm,seed]=np.where(abs(delta)<=1e-8,.5,delta>0)
        compare=comparisons();delta=np.stack([np.mean([credits[a,s]-credits[b,s] for s in cfg()['seeds']],0) for a,b,_,_ in compare],1);rng=np.random.default_rng(cfg()['bootstrap_seed']+(ep=='PFI'));samples={m:[] for m in ['Harrell','Uno60']};counts_hash=hashlib.sha256()
        for start in range(0,cfg()['bootstrap_draws'],250):
            count=np.zeros((min(250,cfg()['bootstrap_draws']-start),512),int);count[:,local]=rng.multinomial(len(local),np.ones(len(local))/len(local),len(count));counts_hash.update(count.tobytes());q=(count[:,i]*count[:,j]).astype(float)
            for metric in samples:
                ww=np.ones(len(i)) if metric=='Harrell' else weights[ep][i];weighted=q*ww;samples[metric].append(weighted@delta/weighted.sum(1)[:,None])
                if start==0:
                    a,b,_,_=compare[0]
                    for row in range(3):
                        literal=0
                        for seed in cfg()['seeds']:
                            num=0;den=0
                            for fold in cfg()['folds']:
                                ids=np.flatnonzero(valid&(d['folds']==fold));ex=np.repeat(ids,count[row,ids]);w=None if metric=='Harrell' else weights[ep][ex];an,ad=components(t[ex],e[ex],risks[a,seed][ex],w);bn,bd=components(t[ex],e[ex],risks[b,seed][ex],w);num+=an-bn;den+=ad
                            literal+=num/den/3
                        err=abs(literal-samples[metric][-1][row,0]);checks.append(err)
                        if err>1e-10:raise ValueError('literal patient bootstrap mismatch')
        for metric in samples:
            values=np.concatenate(samples[metric])
            for k,(a,b,family,n) in enumerate(compare):
                points=[next(r[metric] for r in seedrows if (r['endpoint'],r['arm'],r['seed'])==(ep,a,s))-next(r[metric] for r in seedrows if (r['endpoint'],r['arm'],r['seed'])==(ep,b,s)) for s in cfg()['seeds']];lo,hi=np.quantile(values[:,k],[.025,.975]);cl,ch=np.quantile(values[:,k],[.025/n,1-.025/n]);output.append({'endpoint':ep,'comparison':a+'-'+b,'family':family if ep=='DSS' else 'PFI_secondary_'+family,'metric':metric,'delta':np.mean(points),'CI_low':lo,'CI_high':hi,'corrected_low':cl,'corrected_high':ch,'positive_seeds':sum(v>0 for v in points),**{f'seed{s}':v for s,v in zip(cfg()['seeds'],points)}})
        write(R/f'BOOTSTRAP_{ep}.json',{'status':'PASS','draws':cfg()['bootstrap_draws'],'count_sha256':counts_hash.hexdigest(),'shared_weights':True,'within_fold_pairs_only':True,'averaged_seed_credits_not_risks':True,'refit_variance':False,'literal_max_error':max(checks)})
    pd.DataFrame(output).to_csv(R/'PAIRED_COMPARISONS.csv',index=False)

def main():
    torch.set_num_threads(2);d=parent.load();done=json.loads((P/'TRAINING_DONE.json').read_text())
    if done['status']!='PASS' or done['source']!=source() or done['cells']!=270:raise ValueError('semantic fits not complete')
    endpoints=raw_endpoints(d,{});risks={};modelrows=[];checked=set();max_error=0;trace=[]
    for arm in base.arm_names():
        for seed in cfg()['seeds']:
            r=np.full(512,np.nan)
            for fold in cfg()['folds']:
                c=json.loads((base.P/'cells'/f'{arm}_seed{seed}_fold{fold}.json').read_text())
                if sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('original reference changed')
                z=torch.load(c['artifact'],map_location='cpu',weights_only=False);r[z['test_indices']]=z['risk'];modelrows.append({'arm':arm,'seed':seed,'fold':fold,'new_in_this_batch':False,'parameters':c['parameters'],'selected_epoch':c['selected_epoch'],'seconds':c['seconds']})
            risks[arm,seed]=r
    for arm in arms():
        for seed in cfg()['seeds']:
            r=np.full(512,np.nan)
            for fold in cfg()['folds']:
                name=label(arm,seed,fold);c=json.loads((P/'cells'/f'{name}.json').read_text())
                if c['tag']!=tag(arm,seed,fold) or c['outer_test_selection'] or sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('semantic integrity')
                z=torch.load(c['artifact'],map_location='cpu',weights_only=False);err=replay(d,z,arm,seed,fold,checked);max_error=max(max_error,err);te=parent.splits(d,'DSS',fold)['outer_test']
                if not np.array_equal(te,z['test_indices']) or not np.array_equal(endpoints['DSS']['time'][te],z['time']) or not np.array_equal(endpoints['DSS']['event'][te],z['event']):raise ValueError('OOF endpoint identity')
                r[te]=z['risk'];modelrows.append({'arm':arm,'seed':seed,'fold':fold,'new_in_this_batch':True,'parameters':z['parameters'],'selected_epoch':z['selected_epoch'],'seconds':c['seconds'],'peak_GPU_MiB':z['peak_GPU_MiB'],'replay_error':err})
                for role,values in z['history'].items():
                    for row in values:trace.append({'arm':arm,'seed':seed,'fold':fold,'role':role,**{k:v for k,v in row.items() if k!='gradient'},**{k+'_gradient':v for k,v in row['gradient'].items()}})
                print('SEMANTIC_REPLAY',name,flush=True)
            risks[arm,seed]=r
    weights={};rows=[]
    for ep in ['DSS','PFI']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];weights[ep]=np.zeros(512)
        for fold in cfg()['folds']:
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));iw=ipcw(t[tr],e[tr],t[te],e[te],60);weights[ep][te]=iw
            for (arm,seed),risk in risks.items():
                rr=risk[te];hn,hd=components(t[te],e[te],rr);un,ud=components(t[te],e[te],rr,iw);hh=concordance_index_censored(e[te].astype(bool),t[te],rr)[0];uu=concordance_index_ipcw(Surv.from_arrays(e[tr].astype(bool),t[tr]),Surv.from_arrays(e[te].astype(bool),t[te]),rr,tau=60)[0]
                if abs(hh-hn/hd)>1e-10 or abs(uu-un/ud)>1e-10:raise ValueError('sksurv metric mismatch')
                rows.append({'endpoint':ep,'arm':arm,'seed':seed,'fold':fold,'n':len(te),'events':int(e[te].sum()),'Harrell':hh,'Uno60':uu,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud,'scope':'candidate-pool-limited;PFI frozen transfer' if ep=='PFI' else 'candidate-pool-limited;DSS'})
    table=pd.DataFrame(rows);table.to_csv(R/'FOLD_METRICS.csv',index=False);seeds=[]
    for (ep,arm,seed),g in table.groupby(['endpoint','arm','seed']):seeds.append({'endpoint':ep,'arm':arm,'seed':seed,'Harrell':g.H_num.sum()/g.H_den.sum(),'Uno60':g.U_num.sum()/g.U_den.sum()})
    pd.DataFrame(seeds).to_csv(R/'SEED_OOF_METRICS.csv',index=False);pd.DataFrame(modelrows).to_csv(R/'MODEL_SUMMARY.csv',index=False);pd.DataFrame(trace).to_csv(R/'TRAINING_DIAGNOSTICS.csv',index=False);statistics(d,endpoints,weights,risks,seeds)
    result={'status':'PASS','source':source(),'new_checkpoint_replays':270,'unchanged_prior_fold_records':210,'unified_fold_records':len(rows),'max_replay_error':max_error,'independent_role_preprocessing_checked':len(checked),'raw_endpoints_verified':True,'all_fold_metrics_sksurv_verified':True};write(R/'INDEPENDENT_AUDIT.json',result);write(P/'EVALUATION_DONE.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
