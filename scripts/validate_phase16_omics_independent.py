"""Independent G5 feature arithmetic, checkpoint replay and paired statistics."""
import sys,json,gc,hashlib,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase11_independent import pairs,components,ipcw
from validate_phase13_independent import raw_endpoints,test_loss
from validate_phase16_independent import cs_independent,literal_pool
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
from sksurv.util import Surv
from src.phase16_semantic.omics import *
GR=R/'G5'

def independently_construct(d,z,fold):
    state=z['state'];fit=state['fit'];rr=np.asarray(d['rna_genes'],float);cc=np.asarray(d['cna_genes'],float)
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);median=np.nanmedian(rr[fit],0)
    median=np.where(np.isfinite(median),median,0);filled=np.where(np.isfinite(rr),rr,median);mu=filled[fit].mean(0);sd=filled[fit].std(0);usable=(np.isfinite(rr[fit]).mean(0)>=.9)&(sd>1e-8);rnames=d['rna_gene_symbols'].astype(str);cnames=d['cna_gene_symbols'].astype(str);loc={s:j for j,s in enumerate(cnames)};eligible=np.array([j for j,g in enumerate(rnames) if usable[j] and g in loc and np.isfinite(cc[fit,loc[g]]).mean()>=.9]);cindices=np.array([loc[rnames[j]] for j in eligible])
    if not np.array_equal(eligible,state['eligible_RNA_indices']) or not np.array_equal(cindices,state['eligible_CNA_indices']):raise ValueError('G5 common gene selection not train-only')
    permutation=np.arange(len(eligible))
    if state['random_mask']:
        rm=mu[eligible];rs=sd[eligible];bins=np.searchsorted(np.quantile(rm,[.2,.4,.6,.8]),rm)*5+np.searchsorted(np.quantile(rs,[.2,.4,.6,.8]),rs);rng=np.random.default_rng(169781+fold)
        for b in np.unique(bins):ix=np.flatnonzero(bins==b);permutation[ix]=rng.permutation(ix)
    if not np.array_equal(permutation,state['permutation']):raise ValueError('G5 matched random permutation mismatch')
    full_permutation=np.arange(len(rnames));full_permutation[eligible]=eligible[permutation];extra=np.setdiff1d(np.flatnonzero(usable),eligible)
    if state['random_mask'] and len(extra):
        mean=mu[extra];scale=sd[extra];bins=np.searchsorted(np.quantile(mean,[.2,.4,.6,.8]),mean)*5+np.searchsorted(np.quantile(scale,[.2,.4,.6,.8]),scale);rng=np.random.default_rng(169952+fold)
        for b in np.unique(bins):ix=extra[bins==b];full_permutation[ix]=rng.permutation(ix)
    if not np.array_equal(full_permutation,state['RNA_head_permutation']):raise ValueError('RNA-only/common matched permutation')
    full_z=(filled-mu)/np.where(sd>1e-8,sd,1);rlookup={s:j for j,s in enumerate(rnames)};lookup={s:j for j,s in enumerate(rnames[eligible])};groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text());RNA=[];GAIN=[];LOSS=[];RMISS=[];CMISS=[]
    for row,(name,genes) in zip(state['mapping'],groups.items()):
        original=np.array([lookup[g] for g in genes if g in lookup]);ix=permutation[original];ri=eligible[ix];ci=cindices[ix]
        if row['group']!=name or not np.array_equal(row['RNA_indices'],ri) or not np.array_equal(row['CNA_indices'],ci) or len(ix)!=row['size']:raise ValueError('G5 pathway gene identity mismatch')
        r=(rr[:,ri]-mu[ri])/np.where(sd[ri]>1e-8,sd[ri],1);c=cc[:,ci];gain=np.where(c>=0,c,0.);loss=np.where(c<=0,-c,0.);gain[~np.isfinite(c)]=np.nan;loss[~np.isfinite(c)]=np.nan
        if state['kind']=='RC_GATED':
            gate=1/(1+np.exp(-r));gain*=gate;loss*=gate
        full_indices=full_permutation[np.array([rlookup[g] for g in genes if g in rlookup and usable[rlookup[g]]])]
        if not np.array_equal(full_indices,row['RNA_head_indices']):raise ValueError('RNA head differs from locked query representation')
        RNA.append(full_z[:,full_indices].mean(1))
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);GAIN.append(np.nanmean(gain,1));LOSS.append(np.nanmean(loss,1))
        RMISS.append(1-np.isfinite(rr[:,full_indices]).mean(1));CMISS.append(1-np.isfinite(c).mean(1))
    RNA,GAIN,LOSS,RMISS,CMISS=[np.stack(v,1) for v in [RNA,GAIN,LOSS,RMISS,CMISS]]
    raw=np.c_[RNA,RMISS] if state['kind'] in ['RNA','RNA_ONLY'] else np.c_[GAIN,LOSS,CMISS] if state['kind']=='CNA_ONLY' else np.c_[RNA,GAIN,LOSS,RMISS,CMISS]
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(raw[fit],0)
    med=np.where(np.isfinite(med),med,0);v=np.where(np.isfinite(raw),raw,med);mean=v[fit].mean(0);scale=v[fit].std(0);scale=np.where(scale>1e-8,scale,1);expected=(v-mean)/scale;error=float(np.max(abs(expected-z['x'])))
    if error>1e-8 or np.max(abs(mean-state['mean']))>1e-8 or np.max(abs(scale-state['sd']))>1e-8:raise ValueError('G5 independent molecular arithmetic/scale mismatch')
    return expected,error

def load_prior_risks():
    risks={}
    for arm in base.arm_names()+arms():
        pp=base.P if arm in base.arm_names() else P
        for seed in cfg()['seeds']:
            v=np.full(512,np.nan)
            for fold in cfg()['folds']:
                c=json.loads((pp/'cells'/f'{label(arm,seed,fold)}.json').read_text())
                if sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('prior risk hash changed')
                z=torch.load(c['artifact'],map_location='cpu',weights_only=False);v[z['test_indices']]=z['risk']
            risks[arm,seed]=v
    return risks

def compare_defs():
    result=[]
    for s in mcfg()['selectors']:
        for m in mcfg()['budgets']:
            root=f'G5_{s}_{m}_';result +=[(root+'RNA',f'{s}_{m}'),(root+'RC_UNGATED',root+'RNA'),(root+'RC_GATED',root+'RC_UNGATED')]
        for kind in ['RC_GATED']:result.append((f'G5_{s}_512_{kind}',f'G5_{s}_512_RANDOM_{kind}'))
    if len(result)!=14:raise ValueError('G5 contrast family changed')
    return result

def statistics(d,endpoints,weights,risks,seedrows):
    result=[]
    for ep in ['DSS','PFI']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];local=np.flatnonzero(valid);i,j=pairs(t[local],e[local]);i,j=local[i],local[j];same=d['folds'][i]==d['folds'][j];i,j=i[same],j[same];credit={}
        for arm,seed in risks:
            v=risks[arm,seed];delta=v[i]-v[j];credit[arm,seed]=np.where(abs(delta)<=1e-8,.5,delta>0)
        pairs_to_compare=compare_defs();delta=np.stack([np.mean([credit[a,s]-credit[b,s] for s in cfg()['seeds']],0) for a,b in pairs_to_compare],1);rng=np.random.default_rng(1608531+(ep=='PFI'));samples={m:[] for m in ['Harrell','Uno60']}
        for start in range(0,30000,250):
            count=np.zeros((250,512),int);count[:,local]=rng.multinomial(len(local),np.ones(len(local))/len(local),250);q=(count[:,i]*count[:,j]).astype(float)
            for metric in samples:
                w=np.ones(len(i)) if metric=='Harrell' else weights[ep][i];ww=q*w;samples[metric].append(ww@delta/ww.sum(1)[:,None])
        for metric in samples:
            sample=np.concatenate(samples[metric])
            for k,(a,b) in enumerate(pairs_to_compare):
                pts=[next(r[metric] for r in seedrows if (r['endpoint'],r['arm'],r['seed'])==(ep,a,s))-next(r[metric] for r in seedrows if (r['endpoint'],r['arm'],r['seed'])==(ep,b,s)) for s in cfg()['seeds']];lo,hi=np.quantile(sample[:,k],[.025,.975]);cl,ch=np.quantile(sample[:,k],[.025/28,1-.025/28]);result.append({'endpoint':ep,'comparison':a+'-'+b,'metric':metric,'delta':np.mean(pts),'CI_low':lo,'CI_high':hi,'corrected_low':cl,'corrected_high':ch,'positive_seeds':sum(v>0 for v in pts),**{f'seed{s}':v for s,v in zip(cfg()['seeds'],pts)}})
    pd.DataFrame(result).to_csv(GR/'PAIRED_COMPARISONS.csv',index=False)

def main():
    torch.set_num_threads(2);GR.mkdir(parents=True,exist_ok=True);done=json.loads((MP/'TRAINING_DONE.json').read_text())
    if done['status']!='PASS' or done['source']!=msource():raise ValueError('G5 formal fits incomplete')
    d=parent.load();endpoints=raw_endpoints(d,{});risks=load_prior_risks();cache={};modelrows=[];traces=[];max_error=0;feature_error=0
    for arm,spec in armdefs().items():
        for seed in cfg()['seeds']:
            v=np.full(512,np.nan)
            for fold in cfg()['folds']:
                c=json.loads((MP/'cells'/f'{label(arm,seed,fold)}.json').read_text())
                if c['tag']['source']!=msource() or c['tag']['arm']!=arm or c['tag']['seed']!=seed or c['tag']['fold']!=fold or c['outer_test_selection'] or sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('G5 artifact identity/integrity')
                z=torch.load(c['artifact'],map_location='cpu',weights_only=False);part=parent.splits(d,'DSS',fold)
                for role,st in z['states'].items():
                    fit=part['inner_train' if role=='inner' else 'outer_train'];test=part['inner_val' if role=='inner' else 'outer_test'];fp=st['feature_path']
                    if not np.array_equal(fit,st['fit']) or not np.array_equal(test,st['test']) or sha(fp)!=st['feature_sha256']:raise ValueError('G5 preprocessing identity')
                    if fp not in cache:
                        saved=torch.load(fp,map_location='cpu',weights_only=False)
                        if saved['source']!=msource() or not np.array_equal(saved['state']['fit'],fit):raise ValueError('G5 feature source')
                        x,err=independently_construct(d,saved,fold);feature_error=max(feature_error,err);cache[fp]=x
                    x=cache[fp]
                    if spec['selector'] is None:
                        risk=x@st['w']
                        if role=='inner':
                            scores=[]
                            for candidate in z['selection']:
                                pred=x[test]@candidate['w'];hn,hd=components(endpoints['DSS']['time'][test],endpoints['DSS']['event'][test],pred);score=hn/hd
                                if abs(score-candidate['score'])>1e-12:raise ValueError('omics-only inner selection replay')
                                scores.append(score)
                            best=max(scores);tied=[r for r in z['selection'] if abs(r['score']-best)<1e-12];winner=next((r['lambda'] for r in tied if r['lambda']=='OFF'),tied[0]['lambda'])
                            if winner!=st['lambda_value']:raise ValueError('omics-only selected lambda mismatch')
                    else:
                        cs=torch.tensor(cs_independent(d,fit),dtype=torch.float32,device='cuda');mm=torch.tensor(x,dtype=torch.float32,device='cuda');state={k:v.cuda() for k,v in st['model'].items()};roi=Path(st['ROI_path'])
                        if sha(roi)!=st['ROI_sha256']:raise ValueError('G5 ROI changed')
                        bags=SemanticBags(roi,spec['budget'])
                        with torch.no_grad():risk=(literal_pool(bags,state['score'],state['roi'],cs,state['base'])+.25*torch.tanh(mm@state['mol'])).cpu().numpy()
                        del bags;gc.collect();torch.cuda.empty_cache()
                    if role=='outer':max_error=max(max_error,float(np.max(abs(risk[test]-z['risk']))))
                    elif spec['selector'] is not None:
                        history=z['history']['inner'];ema=None;values=[]
                        for row in history:ema=row['val_cox'] if ema is None else .3*row['val_cox']+.7*ema;values.append(ema)
                        epoch=np.argmin(values)+1
                        if epoch!=z['selected_epoch'] or abs(test_loss(endpoints['DSS']['time'][test],endpoints['DSS']['event'][test],risk[test])-history[epoch-1]['val_cox'])>2e-6:raise ValueError('G5 inner checkpoint replay')
                te=part['outer_test']
                if not np.array_equal(te,z['test_indices']) or not np.array_equal(z['time'],endpoints['DSS']['time'][te]) or not np.array_equal(z['event'],endpoints['DSS']['event'][te]):raise ValueError('G5 endpoint order')
                v[te]=z['risk'];modelrows.append({'arm':arm,'seed':seed,'fold':fold,'parameters':z['parameters'],'selected_epoch':z['selected_epoch'],'seconds':c['seconds'],'scope':'G5 new outer fit'})
                for role,history in z['history'].items():
                    for row in history:traces.append({'arm':arm,'seed':seed,'fold':fold,'role':role,**{k:v for k,v in row.items() if k!='gradient'},**{k+'_gradient':v for k,v in row['gradient'].items()}})
                print('G5_REPLAY',label(arm,seed,fold),flush=True)
            risks[arm,seed]=v
    if max_error>3e-6:raise ValueError('G5 checkpoint mismatch')
    rows=[];weights={}
    for ep in ['DSS','PFI']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];weights[ep]=np.zeros(512)
        for fold in cfg()['folds']:
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));iw=ipcw(t[tr],e[tr],t[te],e[te],60);weights[ep][te]=iw
            for (arm,seed),risk in risks.items():
                r=risk[te];hn,hd=components(t[te],e[te],r);un,ud=components(t[te],e[te],r,iw);h=concordance_index_censored(e[te].astype(bool),t[te],r)[0];u=concordance_index_ipcw(Surv.from_arrays(e[tr].astype(bool),t[tr]),Surv.from_arrays(e[te].astype(bool),t[te]),r,tau=60)[0]
                if abs(h-hn/hd)>1e-10 or abs(u-un/ud)>1e-10:raise ValueError('G5 independent metrics')
                rows.append({'endpoint':ep,'arm':arm,'seed':seed,'fold':fold,'n':len(te),'events':int(e[te].sum()),'Harrell':h,'Uno60':u,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud})
    df=pd.DataFrame(rows);df.to_csv(GR/'UNIFIED_FOLD_METRICS.csv',index=False);seedrows=[]
    for (ep,arm,seed),g in df.groupby(['endpoint','arm','seed']):seedrows.append({'endpoint':ep,'arm':arm,'seed':seed,'Harrell':g.H_num.sum()/g.H_den.sum(),'Uno60':g.U_num.sum()/g.U_den.sum()})
    pd.DataFrame(seedrows).to_csv(GR/'UNIFIED_SEED_OOF_METRICS.csv',index=False);pd.DataFrame(modelrows).to_csv(GR/'MODEL_SUMMARY.csv',index=False);pd.DataFrame(traces).to_csv(GR/'TRAINING_DIAGNOSTICS.csv',index=False);statistics(d,endpoints,weights,risks,seedrows)
    result={'status':'PASS','source':msource(),'G5_outer_replays':240,'combined_outer_fold_evaluations':720,'combined_actual_outer_fits':660,'historical_reference_records':60,'combined_metric_records':len(rows),'max_replay_error':max_error,'molecular_feature_max_error':feature_error,'feature_states_independently_checked':len(cache),'same_gene_RNA_CNA_alignment_verified':True,'separate_CNA_gain_loss_verified':True,'all_metrics_sksurv_verified':True};write(GR/'INDEPENDENT_AUDIT.json',result);write(MP/'EVALUATION_DONE.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
