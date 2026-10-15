"""Independent transforms, literal Cox gradients, heldout metrics and contrasts."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
from sksurv.util import Surv
from src.phase16r_light.core import *
from validate_phase11_independent import pairs,ipcw,components
from validate_phase13_independent import raw_endpoints
from validate_phase16_independent import cs_independent

def literal_gradient(x,t,e,w,penalty):
    r=x@w;g=-e.astype(float)
    for u in np.unique(t[e==1]):
        deaths=(t==u)&(e==1);risk=t>=u;number=int(deaths.sum());v=np.exp(r-r[risk].max())
        for k in range(number):a=risk.astype(float)-k/number*deaths;g+=a*v/(a@v)
    return x.T@g/e.sum()+np.asarray(penalty)*w

def replay_bank(d,z,visual,refit=False):
    fit=z['fit'];state=z['state'];answer={'CS':cs_independent(d,fit)};errors=[]
    def scaled(x,s):
        mean=x[fit].mean(0);sd=x[fit].std(0);sd=np.where(sd<1e-8,1,sd)
        if np.max(abs(mean-s['mean']))>1e-10 or np.max(abs(sd-s['sd']))>1e-10:raise ValueError('feature standardization includes nonfit patients')
        return (x-mean)/sd
    for name,st in state['visual'].items():
        raw=visual[name];mean=raw[fit].mean(0);sd=raw[fit].std(0);sd=np.where(sd<1e-8,1,sd)
        if np.max(abs(mean-st['scale']['mean']))>1e-12 or np.max(abs(sd-st['scale']['sd']))>1e-12:raise ValueError('visual statistics used nonfit patients')
        a=(raw-mean)/sd;pc=(a-st['mean'])@st['components'].T;answer[name]=np.c_[answer['CS'],scaled(pc,st['pc_scale'])]
        if refit:
            pp=PCA(32,svd_solver='full').fit(a[fit])
            if np.max(abs(pp.components_-st['components']))>1e-8:raise ValueError('PCA basis not fit-local')
    raw=np.asarray(d['rna_genes'],float);median=np.nanmedian(raw[fit],axis=0);median=np.where(np.isfinite(median),median,0);v=np.where(np.isfinite(raw),raw,median);mu=v[fit].mean(0);sd=v[fit].std(0);eligible=np.flatnonzero((np.isfinite(raw[fit]).mean(0)>=.9)&(sd>1e-8));genes=d['rna_gene_symbols'].astype(str);groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text());union=set().union(*[set(g) for g in groups.values()]);annotated=np.array([g in union for g in genes]);which=eligible[annotated[eligible]];ix=which[np.argsort(-sd[which],kind='stable')[:512]]
    if len(ix)<512:
        other=eligible[~annotated[eligible]];ix=np.r_[ix,other[np.argsort(-sd[other],kind='stable')[:512-len(ix)]]]
    st=state['RNA']
    for key,a in [('median',median),('mean',mu),('sd',sd)]:
        if np.max(abs(a-st['gene_state'][key]))>1e-10:raise ValueError('RNA imputation/gene scale includes nonfit patients')
    if not np.array_equal(ix,st['indices']):raise ValueError('inner RNA gene selection not fit-local')
    g=((v[:,ix]-mu[ix])/np.where(sd[ix]>1e-8,sd[ix],1)).astype(np.float32).astype(float);h=scaled(g,st['scale']);answer['HALL512']=h;pc=(h-st['pca_mean'])@st['components'].T
    for dim in [8,16,32,64,128]:
        answer['ORTHO'+str(dim)]=pc[:,:dim]
        if dim!=8:answer['PCA'+str(dim)]=scaled(pc[:,:dim],st['pc_scales']['PCA'+str(dim)])
    if np.max(abs(st['components']@st['components'].T-np.eye(128)))>1e-9:raise ValueError('nonorthogonal PCA')
    blocks=[]
    for (name,values),b in zip(groups.items(),st['blocks']):
        expected=np.flatnonzero(np.array([g in values for g in genes[ix]]))
        if not np.array_equal(expected,b['indices']):raise ValueError('biological block uses false gene memberships')
        blocks.append((h[:,expected]-b['mean'])@b['components'].T)
    for dim in [16,32,64,128]:
        alloc=st['block_allocations'][str(dim)]
        if sum(alloc)!=dim:raise ValueError('block dimension budget')
        a=np.concatenate([block[:,:q] for block,q in zip(blocks,alloc)],1);answer['BPCA'+str(dim)]=scaled(a,st['block_scales']['BPCA'+str(dim)])
    if refit:
        pp=PCA(128,svd_solver='full').fit(h[fit])
        if np.max(abs(pp.components_-st['components']))>1e-8:raise ValueError('RNA PCA basis not fit-local')
    for name,x in answer.items():
        err=float(abs(x-z['x'][name]).max());errors.append(err)
        if err>2e-9:raise ValueError('independent preprocessing replay '+name+' '+str(err))
    return max(errors),answer

def wrong(d,x,z,fit,test):
    perm=z['permutation']
    for part in [fit,test]:
        if set(perm[part])!=set(part) or np.any(perm[part]==part):raise ValueError('wrong patient cross-partition or identity')
    return np.c_[x['CS'],x['LOCAL'][perm,x['CS'].shape[1]:]]

def select(rows):
    best=max(v['score'] for v in rows);tied=[v for v in rows if abs(v['score']-best)<1e-12]
    return 'OFF' if any(v['lambda']=='OFF' for v in tied) else max(v['lambda'] for v in tied)

def check_global(visual):
    meta=base.manifest();raw=np.load(meta['features'],mmap_mode='r');owners=[];columns=[];values=[];number=np.zeros(512)
    for i,s in enumerate(meta['slides']):
        m=np.load(base.P/'coordinate_maps'/f'slide_{i+1:04d}.npz');idx=np.unique(m['conch_row'])
        if len(idx):number[s['patient']]+=1
    for i,s in enumerate(meta['slides']):
        m=np.load(base.P/'coordinate_maps'/f'slide_{i+1:04d}.npz');idx=np.unique(m['conch_row'])
        if len(idx):owners.extend([s['patient']]*len(idx));columns.extend((s['offset']+idx).tolist());values.extend([1/len(idx)/number[s['patient']]]*len(idx))
    matrix=csr_matrix((values,(owners,columns)),shape=(512,len(raw)));actual=matrix@raw;error=float(abs(actual-visual['GLOBAL_MATCHED']).max())
    if error>1e-10:raise ValueError('global matched pool mismatch')
    return error

def main():
    d=parent.load();endpoints=raw_endpoints(d,{});visual=dict(np.load(P/'VISUAL_SOURCES.npz'));gerr=check_global(visual);keys=cfg()['clinical_inputs']+cfg()['RNA_inputs'];risks={};rows=[];logs=[];cvrows=[];choices=[];components_out=[];preprocessed={};maxpre=0.;maxgrad=0.;maxrisk=0.;actual=0;reference=0;constant=0;innerfits=0;refit_count=0
    for seed in cfg()['seeds']:
        for fold in cfg()['folds']:
            name=label(seed,fold);artifact=P/'artifacts'/f'{name}.pt';c=json.loads((P/'cells'/f'{name}.json').read_text());z=torch.load(artifact,map_location='cpu',weights_only=False)
            if sha(artifact)!=c['sha256'] or z['source']!=source():raise ValueError('cell source/hash changed')
            part=parent.splits(d,'DSS',fold);train,test=part['outer_train'],part['outer_test'];t,e=endpoints['DSS']['time'],endpoints['DSS']['event'];innerden={key:{str(lam):[0.,0.] for lam in cfg()['lambda_grid']} for key in keys}
            expected_cv=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold*100+cfg()['split_seed_offset']).split(train,e[train]))
            for subfold,(record,(aa,bb)) in enumerate(zip(z['inner'],expected_cv)):
                fit,held=train[aa],train[bb]
                if not np.array_equal(fit,record['fit']) or not np.array_equal(held,record['test']) or set(fit)&set(held) or (set(fit)|set(held))!=set(train):raise ValueError('inner partition leakage')
                path=Path(record['feature_path'])
                if sha(path)!=record['feature_sha256']:raise ValueError('inner preprocessing changed')
                b=torch.load(path,map_location='cpu',weights_only=False)
                if not np.array_equal(b['fit'],fit):raise ValueError('inner feature fit includes validation patients')
                if str(path) not in preprocessed:
                    refit=seed==31 and fold in [1,3,5] and subfold==(fold//2)%3;err,x=replay_bank(d,b,visual,refit);maxpre=max(maxpre,err);preprocessed[str(path)]=x;refit_count+=int(refit)
                x=preprocessed[str(path)];design={**x,'WRONG_LOCAL':wrong(d,x,record['wrong'],fit,held)};ii,jj=pairs(t[held],e[held])
                for model in keys:
                    for candidate in record['models'][model]:
                        lam=candidate['lambda'];w=candidate['w'];pred=design[model][held]@w;err=float(abs(pred-candidate['risk']).max());maxrisk=max(maxrisk,err)
                        if err>1e-9:raise ValueError('inner prediction replay')
                        dd=pred[ii]-pred[jj];num=float(np.where(abs(dd)<=1e-8,.5,dd>0).sum());den=len(ii)
                        if num!=candidate['H_num'] or den!=candidate['H_den']:raise ValueError('inner selection metric mismatch')
                        innerden[model][str(lam)][0]+=num;innerden[model][str(lam)][1]+=den
                        if lam!='OFF':
                            gg=float(abs(literal_gradient(design[model][fit],t[fit],e[fit],w,float(lam))).max());maxgrad=max(maxgrad,gg);innerfits+=1
                            if gg>3e-5:raise ValueError('inner objective not stationary on training-only outcomes')
                        cvrows.append({'seed':seed,'fold':fold,'inner_fold':subfold,'model':model,'lambda':lam,'H_num':num,'H_den':den,'score':num/den})
            for model in keys:
                for row in z['CV'][model]:
                    num,den=innerden[model][str(row['lambda'])]
                    if abs(num/den-row['score'])>1e-12:raise ValueError('CV aggregation bias')
                if select(z['CV'][model])!=z['chosen'][model]:raise ValueError('CV winner/tie changed')
                choices.append({'seed':seed,'fold':fold,'model':model,'selected_lambda':z['chosen'][model]})
            op=Path(z['outer_feature_path'])
            if sha(op)!=z['outer_feature_sha256']:raise ValueError('outer preprocessing changed')
            outer=torch.load(op,map_location='cpu',weights_only=False)
            if not np.array_equal(outer['fit'],train):raise ValueError('outer preprocessing scope')
            if str(op) not in preprocessed:
                err,x=replay_bank(d,outer,visual,True);maxpre=max(maxpre,err);preprocessed[str(op)]=x;refit_count+=1
            x=preprocessed[str(op)];design={**x,'WRONG_LOCAL':wrong(d,x,z['outer_wrong'],train,test)}
            for model,m in z['models'].items():
                if m.get('reference_path'):
                    if sha(m['reference_path'])!=m['reference_sha256']:raise ValueError('historical comparator mutated')
                    if model=='C4_REF':
                        rr=np.load(m['reference_path'])
                        if not np.array_equal(rr['time'],t[test]) or not np.array_equal(rr['event'],e[test]):raise ValueError('C4 source endpoints do not match')
                    risk=m['risk'];reference+=1
                else:
                    xx=design[m['design']];risk=xx[test]@m['w'];err=float(abs(risk-m['risk']).max());maxrisk=max(maxrisk,err)
                    if err>1e-9:raise ValueError('outer risk replay')
                    if m.get('reuse_model'):
                        reference+=1
                        if model=='COMPACT_ADAPTIVE':
                            options=[{**row,'representation':family+str(dim),'dimension':dim,'family':family} for family in ['PCA','BPCA'] for dim in [16,32,64,128] for row in z['CV'][family+str(dim)]];best=max(v['score'] for v in options);tied=[v for v in options if abs(v['score']-best)<1e-12];tied.sort(key=lambda v:(v['lambda']!='OFF',v['dimension'],v['family']!='PCA',-v['lambda'] if v['lambda']!='OFF' else 0))
                            if tied[0]!=z['adaptive']:raise ValueError('compact dimension selected outside inner CV')
                    elif m.get('actual_optimizer_fit'):actual+=1
                    else:constant+=1
                    if m.get('actual_optimizer_fit'):
                        if model.endswith('_BLOCK'):
                            if m['base_lambda']=='OFF':
                                n=x['CS'].shape[1];gg=float(abs(literal_gradient(xx[train,n:],t[train],e[train],m['w'][n:],1.)).max())
                            else:gg=float(abs(literal_gradient(xx[train],t[train],e[train],m['w'],m['penalty'])).max())
                        else:gg=float(abs(literal_gradient(xx[train],t[train],e[train],m['w'],float(m['lambda']))).max())
                        maxgrad=max(maxgrad,gg)
                        if gg>3e-5:raise ValueError('outer objective not stationary')
                    if model in ['LOCAL__FIXED1','LOCAL__CV3','LOCAL_AT_CS_LAMBDA','LOCAL_BLOCK']:
                        n=x['CS'].shape[1];clinical=xx[test,:n]@m['w'][:n];pathrisk=xx[test,n:]@m['w'][n:];same=z['models']['CS_AT_LOCAL_LAMBDA' if model=='LOCAL__CV3' else 'CS__CV3' if model in ['LOCAL_BLOCK','LOCAL_AT_CS_LAMBDA'] else 'CS__FIXED1']['risk'];components_out.append({'seed':seed,'fold':fold,'model':model,'clinical_component_sd':clinical.std(),'path_component_sd':pathrisk.std(),'clinical_change_vs_control_sd':(clinical-same).std(),'path_mean':pathrisk.mean(),'coefficient_norm':np.linalg.norm(m['w']),'scope':'prediction accounting;not a causal attribution'})
                if (model,seed) not in risks:risks[model,seed]=np.full(512,np.nan)
                risks[model,seed][test]=risk
                for ep in ['DSS','PFI']:
                    tt,ee,valid=[endpoints[ep][k] for k in ['time','event','valid']];tr=train[valid[train]];te=test[valid[test]];rr=risk[valid[test]];weight=ipcw(tt[tr],ee[tr],tt[te],ee[te],60);hn,hd=components(tt[te],ee[te],rr);un,ud=components(tt[te],ee[te],rr,weight);hh=concordance_index_censored(ee[te].astype(bool),tt[te],rr)[0];uu=concordance_index_ipcw(Surv.from_arrays(ee[tr].astype(bool),tt[tr]),Surv.from_arrays(ee[te].astype(bool),tt[te]),rr,tau=60)[0]
                    if abs(hh-hn/hd)>1e-10 or abs(uu-un/ud)>1e-10:raise ValueError('independent metrics disagree')
                    rows.append({'endpoint':ep,'model':model,'seed':seed,'fold':fold,'n':len(te),'events':int(ee[te].sum()),'Harrell':hh,'Uno60':uu,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud})
            print('INDEPENDENT_CELL_PASS',name,flush=True)
    table=pd.DataFrame(rows);table.to_csv(R/'FOLD_METRICS.csv',index=False);pd.DataFrame(cvrows).to_csv(R/'INNER_CV_METRICS.csv',index=False);pd.DataFrame(choices).to_csv(R/'LAMBDA_SELECTION.csv',index=False);pd.DataFrame(components_out).to_csv(R/'CLINICAL_PATH_RISK_COMPONENTS.csv',index=False)
    seeds=[]
    for (ep,model,seed),g in table.groupby(['endpoint','model','seed']):seeds.append({'endpoint':ep,'model':model,'seed':seed,'Harrell':g.H_num.sum()/g.H_den.sum(),'Uno60':g.U_num.sum()/g.U_den.sum()})
    st=pd.DataFrame(seeds);st.to_csv(R/'SEED_OOF_METRICS.csv',index=False);summary=st.groupby(['endpoint','model'])[['Harrell','Uno60']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(R/'SUMMARY.csv',index=False)
    np.savez_compressed(P/'OOF_PRIVATE.npz',**{model+'__seed'+str(seed):r for (model,seed),r in risks.items()});statistics(d,endpoints,risks)
    result={'status':'PASS','outer_model_units':actual+reference+constant,'actual_outer_optimizer_fits':actual,'outer_OFF_constants':constant,'outer_references':reference,'inner_optimizer_fits_checked':innerfits,'metric_records':len(table),'preprocess_banks_independently_checked':len(preprocessed),'PCA_banks_refitted':refit_count,'preprocess_max_error':maxpre,'gradient_max':maxgrad,'risk_replay_error':maxrisk,'global_matched_sparse_pool_error':gerr,'all_metrics_crosschecked_sksurv':True,'source':source(),'C4_original_training_retained':True};write(R/'INDEPENDENT_AUDIT.json',result);print(json.dumps(result),flush=True)

def statistics(d,endpoints,risks):
    inventory=[]
    for a,b in cfg()['clinical_primary']:inventory.append((a,b,'clinical_primary',6))
    for a,b in cfg()['clinical_mechanism']:inventory.append((a,b,'clinical_mechanism',16))
    for schedule in ['FIXED1','CV3']:
        for family in ['PCA','BPCA']:
            for dim in [16,32,64,128]:inventory.append((family+str(dim)+'__'+schedule,'HALL512__'+schedule,'RNA_retention_'+schedule,16))
        for dim in [16,32,64,128]:inventory.append(('BPCA'+str(dim)+'__'+schedule,'PCA'+str(dim)+'__'+schedule,'RNA_grouping_'+schedule,8));inventory.append(('ORTHO'+str(dim)+'__'+schedule,'PCA'+str(dim)+'__'+schedule,'RNA_normalization_'+schedule,8))
    inventory += [('COMPACT_ADAPTIVE','HALL512__CV3','adaptive_compact',2),('LOCAL__CV3','LOCAL_SINGLE_INNER_REF','validation_schedule',2),('ORTHO8__FIXED1','PCA8_WHITE_REF','PCA8_normalization',2)]
    out=[];pairrows=[]
    for ep in ['DSS','PFI']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];local=np.flatnonzero(valid);i,j=pairs(t[local],e[local]);i,j=local[i],local[j];keep=d['folds'][i]==d['folds'][j];i,j=i[keep],j[keep];weights=np.zeros(512)
        for fold in cfg()['folds']:
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));weights[te]=ipcw(t[tr],e[tr],t[te],e[te],60)
        credit={}
        for key,r in risks.items():delta=r[i]-r[j];credit[key]=np.where(abs(delta)<=1e-8,.5,delta>0)
        differences=np.stack([np.mean([credit[a,s]-credit[b,s] for s in cfg()['seeds']],0) for a,b,_,_ in inventory],1);rng=np.random.default_rng(cfg()['bootstrap_seed']+(ep=='PFI'));draws={m:[] for m in ['Harrell','Uno60']}
        for start in range(0,cfg()['bootstrap_draws'],250):
            counts=np.zeros((250,512),int);counts[:,local]=rng.multinomial(len(local),np.ones(len(local))/len(local),250);mass=(counts[:,i]*counts[:,j]).astype(float)
            for metric in draws:w=np.ones(len(i)) if metric=='Harrell' else weights[i];q=mass*w;draws[metric].append(q@differences/q.sum(1)[:,None])
        for metric in draws:
            samples=np.concatenate(draws[metric]);w=np.ones(len(i)) if metric=='Harrell' else weights[i]
            for col,(a,b,family,n_tests) in enumerate(inventory):
                perseed=[float(w@(credit[a,s]-credit[b,s])/w.sum()) for s in cfg()['seeds']];lo,hi=np.quantile(samples[:,col],[.025,.975]);cl,ch=np.quantile(samples[:,col],[.025/n_tests,1-.025/n_tests]);out.append({'endpoint':ep,'a':a,'b':b,'family':family,'metric':metric,'delta':np.mean(perseed),'CI_low':lo,'CI_high':hi,'corrected_low':cl,'corrected_high':ch,'positive_seeds':sum(v>0 for v in perseed),**{'seed'+str(s):v for s,v in zip(cfg()['seeds'],perseed)}})
                pairrows.append({'endpoint':ep,'a':a,'b':b,'family':family,'metric':metric,'improved_pair_mass':float(w@np.mean([np.maximum(credit[a,s]-credit[b,s],0) for s in cfg()['seeds']],0)/w.sum()),'lost_pair_mass':float(w@np.mean([np.maximum(credit[b,s]-credit[a,s],0) for s in cfg()['seeds']],0)/w.sum())})
    pd.DataFrame(out).to_csv(R/'PAIRED_COMPARISONS.csv',index=False);pd.DataFrame(pairrows).to_csv(R/'PAIR_TRANSITIONS.csv',index=False)
    # The exact C-index difference telescopes across the common-lambda CS
    # control. This is descriptive decomposition, not biological causality.
    rows=[]
    for seed in cfg()['seeds']:
        for fold in cfg()['folds']:
            z=torch.load(P/'artifacts'/f'{label(seed,fold)}.pt',map_location='cpu',weights_only=False);ad=z['adaptive'];rows.append({'seed':seed,'fold':fold,'representation':ad['representation'],'dimension':ad['dimension'],'lambda':ad['lambda'],'inner_score':ad['score']})
    pd.DataFrame(rows).to_csv(R/'ADAPTIVE_COMPACT_SELECTION.csv',index=False)

if __name__=='__main__':
    torch.set_num_threads(2)
    while not (P/'TRAINING_DONE.json').exists():time.sleep(2)
    main()
