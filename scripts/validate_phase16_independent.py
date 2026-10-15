"""Raw endpoint, independent concordance/IPCW and literal checkpoint replay."""
import sys,json,hashlib,argparse,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase11_independent import pairs,ipcw,components
from validate_phase13_independent import raw_endpoints,test_loss
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
from sksurv.util import Surv
from src.phase16_routing.core import P,R,CONFIG,cfg,source,sha,write,arm_names,parent,check_tag,Bags,check_rows,manifest

def cs_independent(d,fit):
    parts=[]
    for field in ['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE']:
        values=d['clinical_'+field].astype(str)
        if field=='AGE':
            a=np.array([float(v) if v.replace('.','',1).isdigit() else np.nan for v in values]);a=np.where(np.isfinite(a),a,np.nanmedian(a[fit]));parts.append(a[:,None])
        else:parts.append(np.column_stack([values==level for level in sorted(set(values[fit]))]))
    pieces=[]
    for x in [np.concatenate(parts,1),np.asarray(d['static'],float)]:
        mean=x[fit].mean(0);sd=x[fit].std(0);sd=np.where(sd<1e-8,1,sd);pieces.append((x-mean)/sd)
    return np.concatenate(pieces,1)

def literal_pool(bags,score,roi,cs,base):
    # Does not call the model forward or Bags.pool. Explicit WSI softmax and
    # patient equal-slide averaging, preserving float32 checkpoint arithmetic.
    pooled=[]
    for x,mask,quota,ids,order,k,quota_np in bags.buckets:
        s=torch.einsum('bnd,d->bn',x,score).masked_fill(~mask,-torch.inf)
        if bags.dynamic:
            v,ix=torch.topk(s,k,dim=1,sorted=True);keep=torch.arange(k,device=x.device)[None,:]<quota[:,None];v=v.masked_fill(~keep,-torch.inf);patch=x.gather(1,ix[:,:,None].expand(-1,-1,768))
        else:v=s;patch=x
        a=torch.softmax(v,dim=1);pooled.append(torch.bmm(a[:,None,:],patch)[:,0,:])
    p=bags.merge@torch.cat(pooled);return cs@base+cfg()['training']['residual_scale']*torch.tanh(p@roi)

def replay(d,z,arm,seed,fold):
    check_tag(z['tag'],arm,seed,fold);part=parent.splits(d,'DSS',fold)
    for k in part:
        if not np.array_equal(part[k],z['partitions'][k]):raise ValueError('partition drift')
    te=part['outer_test'];maximum=0
    if not z['actual_fit']:
        p=Path(z['reference_path'])
        if sha(p)!=z['reference_sha256']:raise ValueError('historical artifact changed')
        old=torch.load(p,map_location='cpu',weights_only=False)['test_risk'] if p.suffix=='.pt' else np.load(p)['risk']
        maximum=float(np.max(abs(old-z['risk'])))
        if arm=='CS_P15':
            a=torch.load(p,map_location='cpu',weights_only=False);actual=cs_independent(d,part['outer_train'])[te]@a['w'];maximum=max(maximum,float(np.max(abs(actual-z['risk']))))
        return maximum
    if arm in ['CS_CONCH_GLOBAL','CS_TITAN_NATIVE']:
        from sklearn.decomposition import PCA
        for role,st in z['states'].items():
            fit=part['inner_train' if role=='inner' else 'outer_train']
            if not np.array_equal(fit,st['fit']) or not np.array_equal(st['global_preprocess']['fit'],fit):raise ValueError('global preprocessing fit')
            raw=np.load(P/'conch_global.npy') if arm=='CS_CONCH_GLOBAL' else np.asarray(d['titan'],float);mean=raw[fit].mean(0);sd=raw[fit].std(0);sd[sd<1e-8]=1;v=(raw-mean)/sd;pca=PCA(32,svd_solver='full').fit(v[fit]);pc=pca.transform(v);pmean=pc[fit].mean(0);psd=pc[fit].std(0);psd[psd<1e-8]=1
            stp=st['global_preprocess']
            if np.max(abs(pca.components_-stp['components']))>1e-8:raise ValueError('PCA not training-only')
            design=np.c_[cs_independent(d,fit),(pc-pmean)/psd]
            if role=='outer':maximum=max(maximum,float(np.max(abs(design[te]@st['w']-z['risk']))))
            else:
                scores=[];val=part['inner_val'];tt=parent.endpoints(d)['DSS']['time'][val];ee=parent.endpoints(d)['DSS']['event'][val];ii,jj=pairs(tt,ee)
                for row in z['selection']:
                    candidate=z['inner_models'][str(row['lambda'])];pred=design[val]@candidate['w'];dd=pred[ii]-pred[jj];score=float(np.where(abs(dd)<=1e-8,.5,dd>0).mean())
                    if abs(score-row['score'])>1e-12 or np.max(abs(pred-candidate['risk']))>1e-8:raise ValueError('global inner selection replay')
                    scores.append(score)
                best=max(scores);ties=[r for r in z['selection'] if abs(r['score']-best)<1e-12];winner=next((r['lambda'] for r in ties if r['lambda']=='OFF'),ties[0]['lambda'])
                if winner!=z['chosen_lambda']:raise ValueError('global winner mismatch')
    else:
        selector,mm=arm.rsplit('_',1);bags=Bags(selector,int(mm),seed);owner=np.load(P/'row_owners.npy',mmap_mode='r');m=manifest()
        totals=np.zeros(512,int)
        for i,ids in z['selected_rows'].items():check_rows(ids,owner,m['slides'][i]['patient']);totals[m['slides'][i]['patient']]+=len(ids)
        alln=np.bincount(owner.astype(int),minlength=512)
        if not np.array_equal(totals,np.minimum(int(mm),alln)):raise ValueError('not patient-level exact budget')
        for role,st in z['states'].items():
            fit=part['inner_train' if role=='inner' else 'outer_train'];test=part['inner_val' if role=='inner' else 'outer_test']
            if not np.array_equal(fit,st['fit']) or not np.array_equal(st['preprocess']['fit_indices'],fit):raise ValueError('neural preprocessing fit')
            cs=torch.tensor(cs_independent(d,fit),dtype=torch.float32,device='cuda');state={k:v.to('cuda') for k,v in st['model'].items()}
            with torch.no_grad():r=literal_pool(bags,state['score'],state['roi'],cs,state['base']).cpu().numpy()
            if role=='outer':maximum=max(maximum,float(np.max(abs(r[test]-z['risk']))))
            else:
                values=z['training']['inner'];ema=None;trace=[]
                for row in values:ema=row['val_cox'] if ema is None else .3*row['val_cox']+.7*ema;trace.append(ema)
                if np.argmin(trace)+1!=z['selected_epoch'] or abs(test_loss(parent.endpoints(d)['DSS']['time'][test],parent.endpoints(d)['DSS']['event'][test],r[test])-values[z['selected_epoch']-1]['val_cox'])>2e-6:raise ValueError('inner selection/checkpoint inconsistent')
        del bags;gc.collect();torch.cuda.empty_cache()
    if maximum>3e-6:raise ValueError('checkpoint replay mismatch '+str(maximum))
    return maximum

def contrasts():
    core=[f'{s}_{m}' for s in cfg()['active_selectors'] for m in cfg()['budgets']]
    result=[(a,b) for a in core for b in ['CS_P15','C4_LEGACY']]+[(f'{s}_{m}',f'GRID_{m}') for s in ['DIVERSE','MIL'] for m in [128,512]]+[(f'{s}_512',f'{s}_128') for s in cfg()['active_selectors']]+[(a,'CS_P15') for a in ['CS_CONCH_GLOBAL','CS_TITAN_NATIVE']]
    if len(result)!=26:raise ValueError('secondary family changed')
    return result

def statistics(d,endpoints,weights,risks,seedrows):
    output=[];literal_error=0;hashes={}
    for ep in ['DSS','PFI']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];local=np.flatnonzero(valid);ii,jj=pairs(t[local],e[local]);ii,jj=local[ii],local[jj];keep=d['folds'][ii]==d['folds'][jj];ii,jj=ii[keep],jj[keep];credit={}
        for arm in arm_names():
            for seed in cfg()['seeds']:
                dd=risks[arm,seed][ii]-risks[arm,seed][jj];credit[arm,seed]=np.where(abs(dd)<=1e-8,.5,dd>0)
        comparisons=contrasts();delta=np.stack([np.mean([credit[a,s]-credit[b,s] for s in cfg()['seeds']],0) for a,b in comparisons],1);rng=np.random.default_rng(cfg()['bootstrap']['seed']+(ep=='PFI'));samples={v:[] for v in ['Harrell','Uno60']};count_hash=hashlib.sha256()
        for start in range(0,cfg()['bootstrap']['draws'],250):
            count=np.zeros((min(250,cfg()['bootstrap']['draws']-start),512),int);count[:,local]=rng.multinomial(len(local),np.ones(len(local))/len(local),len(count));count_hash.update(count.tobytes());q=(count[:,ii]*count[:,jj]).astype(float)
            for metric in samples:
                w=np.ones(len(ii)) if metric=='Harrell' else weights[ep][ii];qq=q*w;den=qq.sum(1);samples[metric].append(qq@delta/den[:,None])
                if start==0:
                    a,b=comparisons[0]
                    for n in range(3):
                        direct=0;denom=0
                        for seed in cfg()['seeds']:
                            num=0;dden=0
                            for fold in cfg()['folds']:
                                ids=np.flatnonzero(valid&(d['folds']==fold));expanded=np.repeat(ids,count[n,ids]);ww=None if metric=='Harrell' else weights[ep][expanded];an,ad=components(t[expanded],e[expanded],risks[a,seed][expanded],ww);bn,bd=components(t[expanded],e[expanded],risks[b,seed][expanded],ww);num+=an-bn;dden+=ad
                            direct+=num/dden/3
                        err=abs(direct-samples[metric][-1][n,0]);literal_error=max(literal_error,err)
                        if err>1e-10:raise ValueError('bootstrap statistic mismatch')
        hashes[ep]=count_hash.hexdigest()
        for metric in samples:
            sample=np.concatenate(samples[metric]);alpha=.05/52
            for j,(a,b) in enumerate(comparisons):
                values=[next(r[metric] for r in seedrows if (r['endpoint'],r['arm'],r['seed'])==(ep,a,s))-next(r[metric] for r in seedrows if (r['endpoint'],r['arm'],r['seed'])==(ep,b,s)) for s in cfg()['seeds']];lo,hi=np.quantile(sample[:,j],[.025,.975]);cl,ch=np.quantile(sample[:,j],[alpha/2,1-alpha/2]);output.append({'endpoint':ep,'comparison':a+'-'+b,'family':'prespecified_secondary','metric':metric,'delta':np.mean(values),'CI_low':lo,'CI_high':hi,'corrected_low':cl,'corrected_high':ch,'positive_seeds':sum(v>0 for v in values),'status':'COMPLETE',**{f'seed{s}':v for s,v in zip(cfg()['seeds'],values)}})
        for a,b in cfg()['primary_contrasts']:
            for metric in samples:output.append({'endpoint':ep,'comparison':a+'-'+b,'family':'reserved_primary_Bonferroni16','metric':metric,'status':'BLOCKED_CONCH_Z_ACCESS'})
    pd.DataFrame(output).to_csv(R/'PAIRED_COMPARISONS.csv',index=False);write(R/'BOOTSTRAP_AUDIT.json',{'status':'PASS','draws':cfg()['bootstrap']['draws'],'count_hashes':hashes,'literal_expanded_max_error':literal_error,'shared_patient_resampling':True,'within_fold_pairs_only':True,'averaged_seed_pair_credits_not_risks':True,'refit_variance_included':False,'secondary_family_tests':52,'primary_family_tests':16,'adjusted_CI_extreme_tail_expected_draws':cfg()['bootstrap']['draws']*.05/104})

def main():
    torch.set_num_threads(2);d=parent.load();endpoints=raw_endpoints(d,{});done=json.loads((P/'TRAINING_DONE.json').read_text())
    if done['status']!='PASS' or done['source_hash']!=source():raise ValueError('formal work incomplete/stale')
    cells=[];risks={};max_error=0;hashrows=[];trainrows=[]
    for arm in arm_names():
        for seed in cfg()['seeds']:
            risk=np.full(512,np.nan)
            for fold in cfg()['folds']:
                name=f'{arm}_seed{seed}_fold{fold}';c=json.loads((P/'cells'/f'{name}.json').read_text());check_tag(c['tag'],arm,seed,fold)
                if c['outer_test_selection'] or sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('formal integrity')
                z=torch.load(c['artifact'],map_location='cpu',weights_only=False);err=replay(d,z,arm,seed,fold);max_error=max(max_error,err);te=parent.splits(d,'DSS',fold)['outer_test']
                if not np.array_equal(z['test_time'],endpoints['DSS']['time'][te]) or not np.array_equal(z['test_event'],endpoints['DSS']['event'][te]) or not np.isfinite(z['risk']).all():raise ValueError('endpoint identity')
                risk[te]=z['risk'];cells.append({'arm':arm,'seed':seed,'fold':fold,**{k:c[k] for k in ['actual_fit','parameters','selected_epoch','seconds']},'peak_GPU_MiB':z.get('peak_GPU_MiB',0),'replay_error':err});hashrows.append({'cell':name,'sha256':c['artifact_sha256'],'bytes':Path(c['artifact']).stat().st_size})
                for role,values in z.get('training',{}).items():
                    for row in values:trainrows.append({'arm':arm,'seed':seed,'fold':fold,'role':role,**{k:v for k,v in row.items() if k!='gradient'},**{k+'_gradient':v for k,v in row['gradient'].items()}})
                print('REPLAY',name,flush=True)
            if not np.isfinite(risk).all():raise ValueError('OOF incomplete')
            risks[arm,seed]=risk
    weights={};rows=[]
    for ep in ['DSS','PFI']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];weights[ep]=np.zeros(512)
        for fold in cfg()['folds']:
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));weights[ep][te]=ipcw(t[tr],e[tr],t[te],e[te],60)
            for arm in arm_names():
                for seed in cfg()['seeds']:
                    risk=risks[arm,seed][te];hn,hd=components(t[te],e[te],risk);un,ud=components(t[te],e[te],risk,weights[ep][te]);hh=concordance_index_censored(e[te].astype(bool),t[te],risk)[0];uu=concordance_index_ipcw(Surv.from_arrays(e[tr].astype(bool),t[tr]),Surv.from_arrays(e[te].astype(bool),t[te]),risk,tau=60)[0]
                    if abs(hh-hn/hd)>1e-10 or abs(uu-un/ud)>1e-10:raise ValueError('independent sksurv mismatch')
                    info=next(r for r in cells if (r['arm'],r['seed'],r['fold'])==(arm,seed,fold));rows.append({'endpoint':ep,**info,'Harrell':hh,'Uno60':uu,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud,'n':len(te),'events':int(e[te].sum()),'training_scope':'DSS fit' if ep=='DSS' else 'DSS frozen transfer'})
    table=pd.DataFrame(rows);table.to_csv(R/'FOLD_METRICS.csv',index=False);seedrows=[]
    for (ep,arm,seed),g in table.groupby(['endpoint','arm','seed']):seedrows.append({'endpoint':ep,'arm':arm,'seed':int(seed),'Harrell':g.H_num.sum()/g.H_den.sum(),'Uno60':g.U_num.sum()/g.U_den.sum()})
    pd.DataFrame(seedrows).to_csv(R/'SEED_OOF_METRICS.csv',index=False);pd.DataFrame(cells).to_csv(R/'MODEL_SUMMARY.csv',index=False);pd.DataFrame(trainrows).to_csv(R/'TRAINING_DIAGNOSTICS.csv',index=False);pd.DataFrame(hashrows).to_csv(R/'PRIVATE_ARTIFACT_HASHES.csv',index=False);statistics(d,endpoints,weights,risks,seedrows)
    result={'status':'PASS','source_hash':source(),'verified_cells':len(cells),'actual_fits':sum(r['actual_fit'] for r in cells),'historical_references':sum(not r['actual_fit'] for r in cells),'metric_records':len(rows),'max_checkpoint_replay_error':max_error,'reference_replay_scope':'60 historical reference integrity/prediction checks;CS_P15 actual linear replay;old C/C4 checkpoints not newly replayed','new_model_checkpoint_replay':'all150 both inner/outer preprocessing and selected checkpoints','raw_endpoint_independently_verified':True,'sksurv_all_fold_metrics_verified':True}
    write(R/'INDEPENDENT_AUDIT.json',result);write(P/'EVALUATION_DONE.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
