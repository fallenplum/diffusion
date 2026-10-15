"""Independent raw endpoint/feature/metric/stationarity/OOF mechanism audit."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from scipy.special import expit
from validate_phase11_independent import pairs,components,ipcw
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
from sksurv.util import Surv
from src.phase15_complement.core import PRIVATE,REPORT,CONFIG,MODELS,cfg,source,sha,write,name,parent


def independent_cs(d,fit):
    parts=[]
    for field in ['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE']:
        values=d['clinical_'+field].astype(str)
        if field=='AGE':
            a=np.asarray([float(v) if v.replace('.','',1).isdigit() else np.nan for v in values]);a=np.where(np.isfinite(a),a,np.nanmedian(a[fit]));parts.append(a[:,None])
        else:parts.append(np.column_stack([values==level for level in sorted(set(values[fit]))]))
    c=np.concatenate(parts,1);s=np.asarray(d['static'],float);pieces=[]
    for x in [c,s]:
        mu=x[fit].mean(0);sd=x[fit].std(0);sd=np.where(sd<1e-8,1,sd);pieces.append((x-mu)/sd)
    return np.concatenate(pieces,1)


def independent_rna(raw,state,fit):
    median=np.nanmedian(raw[fit],axis=0);median=np.where(np.isfinite(median),median,0);v=np.where(np.isfinite(raw),raw,median);mean=v[fit].mean(0);sd=v[fit].std(0);sd=np.where(sd>1e-8,sd,1)
    for k,value in [('median',median),('mean',mean),('sd',sd)]:
        if not np.array_equal(state[k],value):raise RuntimeError('RNA preprocessing not train-only')
    if not np.array_equal(state['fit_indices'],fit):raise RuntimeError('RNA preprocessing partition')
    return (v-mean)/sd


def error(actual,expected,label):
    err=float(np.max(abs(actual-expected)))
    if err>1e-8:raise RuntimeError(label+' '+str(err))
    return err


def literal_cox_gradient(x,t,e,w,penalty):
    r=x@w;g=-e.astype(float)
    for value in np.unique(t[e==1]):
        death=(t==value)&(e==1);risk=t>=value;number=int(death.sum());shift=r[risk].max();v=np.exp(r-shift)
        for k in range(number):
            a=risk.astype(float)-k/number*death;prob=a*v/(a@v);g+=prob
    return x.T@g/e.sum()+penalty*w


def replay_and_audit(d,ep,gene,seed,fold,cell,z):
    ev=parent.endpoints(d)[ep];t,e=ev['time'],ev['event'];replay={};maximum=0;gradient=0;counts=[]
    for role in ['inner','outer']:
        path=Path(cell['teacher_anchor_paths'][role]);a=torch.load(path,map_location='cpu',weights_only=False);fit,test=a['fit'],a['test'];part=z['partitions']
        expected=part['inner_train' if role=='inner' else 'outer_train']
        if not np.array_equal(fit,expected) or a['source_hash']!=source():raise RuntimeError('anchor source/partition')
        cs=independent_cs(d,fit);base=cs[test]@a['w'];maximum=max(maximum,error(base,a['test_risk'],'anchor replay'))
        if not np.array_equal(a['preprocessing']['fit_indices'],fit):raise RuntimeError('CS preprocessing')
        if not np.array_equal(a['preprocessing']['fit_hash'],hashlib.sha256(fit.astype(np.int64).tobytes()).hexdigest()):raise RuntimeError('CS fit hash')
        raw=np.asarray(d['rna_genes'][:,z['indices'][role]],float);v=independent_rna(raw,z['transforms'][role],fit);rr=z['RNA'][role]
        oofRNA=np.full(512,np.nan);oofCS=np.full(512,np.nan)
        for teacher,rteacher in zip(a['teachers'],rr['teachers']):
            train,held=teacher['train'],teacher['held']
            if set(train)&set(held) or set(train)|set(held)!=set(fit) or set(train)&set(test):raise RuntimeError('teacher held-out leak')
            if not np.array_equal(teacher['preprocessing']['fit_indices'],train) or not np.array_equal(rteacher['train'],train) or not np.array_equal(rteacher['held'],held):raise RuntimeError('teacher preprocessing/matched folds')
            xx=independent_cs(d,train);oofCS[held]=xx[held]@teacher['w'];rv=independent_rna(raw,rteacher['preprocessing'],train);oofRNA[held]=(rv[held]@rteacher['w']-rteacher['risk_mean'])/rteacher['risk_sd']
        maximum=max(maximum,error(oofCS[fit],a['oof'][fit],'CS crossfit replay'),error(oofRNA[fit],rr['oof'][fit],'RNA crossfit replay'))
        finalRNA=independent_rna(raw,rr['preprocessing'],fit);scalar=(finalRNA[test]@rr['w']-rr['risk_mean'])/rr['risk_sd'];maximum=max(maximum,error(scalar,rr['test'],'RNA scalar replay'))
        i,j=pairs(t[fit],e[fit]);keep=a['group'][fit[i]]==a['group'][fit[j]];i,j=i[keep],j[keep];saved=z['pairs'][role]
        if not np.array_equal(saved['i'],i) or not np.array_equal(saved['j'],j):raise RuntimeError('honest pair mask mismatch')
        margin=oofCS[fit[i]]-oofCS[fit[j]];maximum=max(maximum,error(margin,saved['margin'],'anchor pair margin'))
        for k,teacher in enumerate(a['teachers']):
            use=a['group'][fit[i]]==k;owners=set(fit[i[use]])|set(fit[j[use]])
            if owners&set(teacher['train']) or not owners<=set(teacher['held']):raise RuntimeError('both pair labels not excluded')
        lookup={int(x):n for n,x in enumerate(test)};wrong_test=z['permutations'][role+'_test'];lookupFit={int(x):n for n,x in enumerate(fit)};wrong_fit=z['permutations'][role+'_fit']
        strata=np.char.add(d['cancer_type'].astype(str),np.where(np.isnan(d['raw_rna']).any(1),'_missing','_complete'))
        for x,perm in [(fit,wrong_fit),(test,wrong_test)]:
            if set(x)!=set(perm) or np.any(x==perm) or not np.array_equal(strata[x],strata[perm]):raise RuntimeError('wrong patient crossed partition/stratum')
        counts.append({'endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'role':role,'fit_n':len(fit),'fit_events':int(e[fit].sum()),'pair_count':len(i),'pair_event_owners':len(np.unique(i)),'anchor_wrong_pairs':int((margin<=0).sum()),'anchor_lambda':a['lambda']})
        for key,w in z['coefficients'].items():
            if not key.startswith(role+'_'):continue
            label,lam_text=key[len(role)+1:].rsplit('_',1)
            if lam_text=='OFF':pred=base
            else:
                lam=float(lam_text)
                if label=='JOINT':
                    design=np.c_[cs,v];pred=design[test]@w;g=literal_cox_gradient(design[fit],t[fit],e[fit],w,np.r_[np.full(cs.shape[1],a['lambda']),np.full(512,lam)])
                else:
                    design=v.copy();weight=np.ones(len(i));offset=margin
                    if label=='WRONG':design[fit]=v[wrong_fit];design[test]=v[wrong_test]
                    if label=='SCALAR':design=np.zeros((512,1));design[fit,0]=oofRNA[fit];design[test,0]=scalar
                    if label not in ['ALL','SCALAR']:weight=expit(-margin)
                    if label=='HARD':weight=(margin<=0).astype(float)
                    if label=='NO_OFFSET':offset=np.zeros_like(margin)
                    if label=='RANDOM':
                        rng=np.random.default_rng(seed+fold*100+901)
                        for owner in np.unique(i):
                            positions=np.flatnonzero(i==owner);weight[positions]=rng.permutation(weight[positions])
                    D=design[fit[i]]-design[fit[j]]
                    g=D.T@(-expit(-(offset+D@w))*weight/max(weight.sum(),1e-30))+lam*w
                    pred=base+design[test]@w
                grad=float(abs(g).max());gradient=max(gradient,grad)
                if grad>2.5e-5:raise RuntimeError('independent nonstationary '+key+' '+str(grad))
            if role=='inner':
                row=next(row for row in z['selection'][label+'_rows'] if str(row['lambda'])==lam_text);ii,jj=pairs(t[test],e[test]);dd=pred[ii]-pred[jj];score=float(np.where(abs(dd)<=1e-8,.5,dd>0).mean())
                if abs(score-row['inner_Harrell'])>1e-12:raise RuntimeError('inner selection replay')
            else:
                if str(z['selection'][label])==lam_text:replay[label]=pred
                if lam_text=='1.0' and label in ['JOINT','ALL','SOFT','SCALAR','RANDOM','WRONG']:replay[label+'_FIXED1']=pred
        if role=='inner':
            for label in ['JOINT','ALL','SOFT','HARD','RANDOM','WRONG','NO_OFFSET','SCALAR']:
                candidates=z['selection'][label+'_rows'];best=max(r['inner_Harrell'] for r in candidates);ties=[r for r in candidates if abs(r['inner_Harrell']-best)<=1e-12];winner=next((r for r in ties if r['lambda']=='OFF'),ties[0])['lambda']
                if winner!=z['selection'][label]:raise RuntimeError('wrong inner winner')
        else:
            replay['CS']=base;replay['RNA_ONLY']=scalar
            oldcell=json.loads((parent.PRIVATE/'formal/cells'/f'{ep}_CS_seed31_fold{fold}.json').read_text());old=Path(oldcell['prediction_path'])
            if sha(old)!=oldcell['prediction_sha256']:raise RuntimeError('legacy anchor hash')
            replay['CS_LEGACY']=np.load(old)['risk']
    if set(replay)!=set(MODELS):raise RuntimeError('replay inventory')
    for model in MODELS:maximum=max(maximum,error(replay[model],z['risks'][model],model))
    return maximum,gradient,counts


def evaluate():
    configuration=cfg();d=parent.load();from validate_phase13_independent import raw_endpoints
    endpoints=raw_endpoints(d,configuration);rows=[];counts=[];selection=[];hashes=[];allrisks={};weights={};maximum=0;gradient=0
    for ep in configuration['endpoints']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];weights[ep]=np.zeros(512)
        for fold in configuration['folds']:
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));weights[ep][te]=ipcw(t[tr],e[tr],t[te],e[te],60)
        for gene in configuration['genes']:
            for seed in configuration['seeds']:
                for model in MODELS:allrisks[ep,gene,seed,model]=np.full(512,np.nan)
                for fold in configuration['folds']:
                    path=PRIVATE/'cells'/f'{name(ep,gene,seed,fold)}.json';c=json.loads(path.read_text());z=torch.load(c['artifact'],map_location='cpu',weights_only=False)
                    if c['source_hash']!=source() or z['source_hash']!=source() or c['protocol_sha256']!=sha(CONFIG) or c['outer_test_selection'] or sha(c['artifact'])!=c['artifact_sha256']:raise RuntimeError('cell identity/integrity')
                    for k,value in [('endpoint',ep),('gene',gene),('seed',seed),('fold',fold)]:
                        if c[k]!=value or z[k]!=value:raise RuntimeError('container identity')
                    te=np.flatnonzero(valid&(d['folds']==fold));tr=np.flatnonzero(valid&(d['folds']!=fold));part=z['partitions']
                    if not np.array_equal(part['outer_test'],te) or not np.array_equal(part['outer_train'],tr) or set(part['inner_train'])&set(part['inner_val']) or set(part['inner_train'])|set(part['inner_val'])!=set(tr):raise RuntimeError('patient partitions')
                    if not np.array_equal(z['test_time'],t[te]) or not np.array_equal(z['test_event'],e[te]):raise RuntimeError('endpoint alignment')
                    err,grad,ct=replay_and_audit(d,ep,gene,seed,fold,c,z);maximum=max(maximum,err);gradient=max(gradient,grad);counts+=ct
                    hashes.append({'pipeline':path.stem,'artifact_sha256':c['artifact_sha256'],'artifact_bytes':Path(c['artifact']).stat().st_size})
                    for label in ['JOINT','ALL','SOFT','HARD','RANDOM','WRONG','NO_OFFSET','SCALAR']:selection.append({'endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'model':label,'lambda':z['selection'][label]})
                    for model in MODELS:
                        risk=z['risks'][model];hn,hd=components(t[te],e[te],risk);un,ud=components(t[te],e[te],risk,weights[ep][te]);hh=concordance_index_censored(e[te].astype(bool),t[te],risk)[0];uu=concordance_index_ipcw(Surv.from_arrays(e[tr].astype(bool),t[tr]),Surv.from_arrays(e[te].astype(bool),t[te]),risk,tau=60)[0]
                        if abs(hh-hn/hd)>1e-10 or abs(uu-un/ud)>1e-10:raise RuntimeError('independent concordance mismatch')
                        allrisks[ep,gene,seed,model][te]=risk;rows.append({'endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'model':model,'Harrell':hh,'Uno60':uu,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud,'n':len(te),'events':int(e[te].sum())})
    table=pd.DataFrame(rows);table.to_csv(REPORT/'FOLD_METRICS.csv',index=False);pd.DataFrame(counts).to_csv(REPORT/'SUPERVISION_COUNTS.csv',index=False);pd.DataFrame(selection).to_csv(REPORT/'SELECTION.csv',index=False);pd.DataFrame(hashes).to_csv(REPORT/'PRIVATE_ARTIFACT_HASHES.csv',index=False)
    seedrows=[]
    for (ep,gene,seed,model),g in table.groupby(['endpoint','gene','seed','model']):seedrows.append({'endpoint':ep,'gene':gene,'seed':seed,'model':model,'Harrell':g.H_num.sum()/g.H_den.sum(),'Uno60':g.U_num.sum()/g.U_den.sum()})
    pd.DataFrame(seedrows).to_csv(REPORT/'SEED_OOF_METRICS.csv',index=False)
    statistics(configuration,d,endpoints,weights,allrisks,seedrows)
    result={'status':'PASS','pipelines':len(hashes),'expected_pipelines':90,'model_fold_records':len(rows),'expected_model_records':1530,'max_replay_error':maximum,'max_independent_gradient':gradient,'both_pair_labels_excluded':True,'all_inner_selection_replayed':True,'all_metrics_sksurv_crosschecked':True,'source_hash':source()}
    write(REPORT/'INDEPENDENT_AUDIT.json',result);write(PRIVATE/'EVALUATION_DONE.json',result);print(json.dumps(result))


def statistics(configuration,d,endpoints,weights,risks,seedrows):
    contrasts=[('SOFT','CS'),('SOFT','CS_LEGACY'),('SOFT','JOINT'),('SOFT','ALL'),('SOFT','RANDOM'),('SOFT','WRONG'),('HARD','SOFT'),('SOFT','NO_OFFSET'),('SCALAR','CS'),('SCALAR','SOFT'),('JOINT','CS'),('ALL','CS'),('SOFT_FIXED1','CS'),('SOFT_FIXED1','ALL_FIXED1'),('SOFT_FIXED1','RANDOM_FIXED1'),('SOFT_FIXED1','WRONG_FIXED1'),('SCALAR_FIXED1','CS')]
    output=[];errorrows=[];flips=[];diagnostic_ci=[];literal_max=0;literal_checks=0
    for ep in configuration['endpoints']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];local=np.flatnonzero(valid);ii,jj=pairs(t[local],e[local]);ii,jj=local[ii],local[jj];keep=d['folds'][ii]==d['folds'][jj];ii,jj=ii[keep],jj[keep];credits={};badmask={}
        for gene in configuration['genes']:
            for seed in configuration['seeds']:
                for model in MODELS:
                    dd=risks[ep,gene,seed,model][ii]-risks[ep,gene,seed,model][jj];credits[gene,seed,model]=np.where(abs(dd)<=1e-8,.5,dd>0)
                base=credits[gene,seed,'CS'];badmask[gene,seed]=base<1
                for metric in ['Harrell','Uno60']:
                    ww=np.ones(len(ii)) if metric=='Harrell' else weights[ep][ii];bad=badmask[gene,seed];good=~bad;rna=credits[gene,seed,'RNA_ONLY'];event=np.unique(ii[bad&(ww>0)])
                    masses=np.asarray([ww[(ii==v)&bad].sum() for v in event]);neff=float(masses.sum()**2/(masses@masses))
                    errorrows.append({'endpoint':ep,'gene':gene,'seed':seed,'metric':metric,'anchor_wrong_mass':float(ww[bad].sum()/ww.sum()),'RNA_accuracy_on_anchor_wrong':float(ww[bad]@rna[bad]/ww[bad].sum()),'RNA_accuracy_on_anchor_correct':float(ww[good]@rna[good]/ww[good].sum()),'wrong_pair_event_owners':len(event),'wrong_pair_event_Neff':neff})
                    for model in ['JOINT','ALL','SOFT','HARD','RANDOM','WRONG','SCALAR','SOFT_FIXED1']:
                        delta=credits[gene,seed,model]-base;up=float(ww@np.maximum(delta,0)/ww.sum());down=float(ww@np.maximum(-delta,0)/ww.sum());flips.append({'endpoint':ep,'gene':gene,'seed':seed,'metric':metric,'model':model,'rescued_mass':up,'damaged_mass':down,'net_delta':up-down})
        inventory=[(g,a,b) for g in configuration['genes'] for a,b in contrasts];delta=np.stack([np.mean([credits[g,s,a]-credits[g,s,b] for s in configuration['seeds']],axis=0) for g,a,b in inventory],1)
        rng=np.random.default_rng(configuration['bootstrap_seed']+(ep=='PFI'));draws={'Harrell':[],'Uno60':[]};conditional={'Harrell':[],'Uno60':[]}
        for start in range(0,configuration['bootstrap_draws'],250):
            n=min(250,configuration['bootstrap_draws']-start);cc=np.zeros((n,512),int);cc[:,local]=rng.multinomial(len(local),np.ones(len(local))/len(local),size=n);q=(cc[:,ii]*cc[:,jj]).astype(float)
            for metric in draws:
                ww=np.ones(len(ii)) if metric=='Harrell' else weights[ep][ii];qq=q*ww;draws[metric].append(qq@delta/qq.sum(1)[:,None]);sub=[]
                for g in configuration['genes']:
                    seedvalues=[]
                    for s in configuration['seeds']:
                        bad=badmask[g,s];num=qq@(bad*credits[g,s,'RNA_ONLY']);den=qq@bad
                        if np.any(den==0):raise RuntimeError('empty error-group bootstrap')
                        seedvalues.append(num/den)
                    sub.append(np.mean(seedvalues,axis=0))
                conditional[metric].append(np.stack(sub,1))
                # First ten literal expanded patient samples for two key contrasts.
                if start==0:
                    for row in range(10):
                        for a,b in [('SOFT','CS'),('SOFT','ALL')]:
                            a_score=risks[ep,'HALL',31,a];b_score=risks[ep,'HALL',31,b];direct=0.;denom=0.
                            for fold in configuration['folds']:
                                ix=np.flatnonzero(valid&(d['folds']==fold));expanded=np.repeat(ix,cc[row,ix]);w=None if metric=='Harrell' else weights[ep][expanded]
                                an,dd=components(t[expanded],e[expanded],a_score[expanded],w);bn,bd=components(t[expanded],e[expanded],b_score[expanded],w);direct+=an-bn;denom+=dd
                            matrix=float(qq[row]@(credits['HALL',31,a]-credits['HALL',31,b])/qq[row].sum());err=abs(matrix-direct/denom);literal_max=max(literal_max,err);literal_checks+=1
                            if err>1e-10:raise RuntimeError('literal bootstrap mismatch')
        for metric in draws:
            sample=np.concatenate(draws[metric])
            for j,(g,a,b) in enumerate(inventory):
                values=[next(r[metric] for r in seedrows if (r['endpoint'],r['gene'],r['seed'],r['model'])==(ep,g,s,a))-next(r[metric] for r in seedrows if (r['endpoint'],r['gene'],r['seed'],r['model'])==(ep,g,s,b)) for s in configuration['seeds']];primary=ep=='DSS' and g=='HALL' and (a,b) in contrasts[:6];alpha=.05/12 if primary else .05;lo,hi=np.quantile(sample[:,j],[.025,.975]);cl,ch=np.quantile(sample[:,j],[alpha/2,1-alpha/2]);output.append({'endpoint':ep,'gene':g,'comparison':a+'-'+b,'metric':metric,'delta':float(np.mean(values)),'CI_low':lo,'CI_high':hi,'corrected_low':cl,'corrected_high':ch,'primary':primary,'positive_seeds':sum(v>0 for v in values),**{f'seed{s}':v for s,v in zip(configuration['seeds'],values)}})
            csample=np.concatenate(conditional[metric])
            for j,g in enumerate(configuration['genes']):
                lo,hi=np.quantile(csample[:,j],[.025,.975]);point=np.mean([r['RNA_accuracy_on_anchor_wrong'] for r in errorrows if (r['endpoint'],r['gene'],r['metric'])==(ep,g,metric)]);diagnostic_ci.append({'endpoint':ep,'gene':g,'metric':metric,'RNA_accuracy_on_anchor_wrong':point,'CI_low':lo,'CI_high':hi,'scope':'diagnostic conditional-on-error;not used fortraining/selection;unadjusted'})
    pd.DataFrame(output).to_csv(REPORT/'PAIRED_COMPARISONS.csv',index=False);pd.DataFrame(errorrows).to_csv(REPORT/'RNA_ERROR_DIAGNOSTIC.csv',index=False);pd.DataFrame(flips).to_csv(REPORT/'REPAIR_DAMAGE.csv',index=False);pd.DataFrame(diagnostic_ci).to_csv(REPORT/'RNA_ERROR_DIAGNOSTIC_CI.csv',index=False)
    write(REPORT/'LITERAL_BOOTSTRAP_AUDIT.json',{'status':'PASS','checks':literal_checks,'max_error':literal_max,'shared_patient_draws':configuration['bootstrap_draws'],'risk_ensembles_used':False})


if __name__=='__main__':evaluate()
