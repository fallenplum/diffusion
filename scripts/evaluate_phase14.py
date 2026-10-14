"""Independent concordance/IPCW; checkpoint replay separate from metrics."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase11_independent import pairs,ipcw,components
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
from sksurv.util import Surv
from run_phase14 import PRIVATE,REPORT,MODELS,cfg,source,sha,unit,role_data,apply_rep,write,parent,index_map
from src.phase13_rna.models import RNAOnly,extract_rna
from src.phase14_rna.models import Conditional


def compare(risk,expected,label):
    error=float(abs(risk-expected).max())
    if error>3e-6:raise RuntimeError('replay mismatch '+label+' '+str(error))
    return error


def replay(z,ep,gene,seed,fold,configuration):
    f,m,a,tr,te=role_data(ep,gene,fold,'outer');representations={};errors=[]
    for name in ['RAW512','RAW64']:representations[name]=apply_rep(f[gene][te],z['transforms']['outer_'+name])
    initial=RNAOnly(m['clinical_mask'],m[gene],seed).cuda();_,b=extract_rna(initial,f[gene],te)
    for name,key in [('INIT64','processed_full'),('INITPOOL64','processed_mean')]:representations[name]=apply_rep(b[key],z['transforms']['outer_'+name])
    for producer in ['TOKEN','RANK']:
        initial.load_state_dict(z['states']['outer_'+producer]);_,b=extract_rna(initial,f[gene],te)
        for name,key in [(producer+'512','processed_full'),(producer+'64','processed_full'),('POOL64' if producer=='TOKEN' else 'RANKPOOL64','processed_mean')]:representations[name]=apply_rep(b[key],z['transforms']['outer_'+name])
    del initial;cs=np.c_[f['C'],f['S']][te]
    for model in MODELS[:10]:errors.append(compare(representations[model]@z['probes'][model]['w'],z['risks'][model],model))
    errors.append(compare(cs@z['probes']['CS']['w'],z['risks']['CS'],'CS'))
    wrong=index_map(te,z['permutations']['outer_test'])
    for model,rep,w in [('COXRAW','RAW64',False),('COX','RANK64',False),('COXPOOL','RANKPOOL64',False),('WRONG_COX','RANK64',True)]:
        r=representations[rep];r=r[wrong] if w else r;errors.append(compare(np.c_[cs,r]@z['probes']['outer_'+model]['w'],z['risks'][model],model))
    for arm,interaction,w in [('ADDITIVE',False,False),('INTERACT',True,False),('WRONG_ADDITIVE',False,True),('WRONG_INTERACT',True,True)]:
        q=Conditional(cs.shape[1],64,np.zeros(cs.shape[1]+64),interaction,configuration,seed).cuda();q.load_state_dict(z['states']['outer_'+arm]);q.eval();r=representations['RANK64'];r=r[wrong] if w else r
        with torch.no_grad():rr=q(torch.tensor(cs,device='cuda'),torch.tensor(r,device='cuda')).cpu().numpy()
        errors.append(compare(rr,z['risks'][arm],arm));del q
    return max(errors)


def evaluate():
    torch.set_num_threads(2);configuration=cfg();REPORT.mkdir(parents=True,exist_ok=True)
    d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));endpoint=dict(np.load(parent.PRIVATE/'endpoints.npz'))
    # Check raw clinical independently, including PFI validity, using previous independent reader.
    from validate_phase13_independent import raw_endpoints
    endpoints=raw_endpoints(d,configuration)
    folds=d['folds'];weights={};allrisks={};rows=[];diag=[];train=[];maximum=0;cellcount=0
    for ep in configuration['endpoints']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];weights[ep]=np.zeros(512)
        for fold in configuration['folds']:
            tr=np.flatnonzero(valid&(folds!=fold));te=np.flatnonzero(valid&(folds==fold));weights[ep][te]=ipcw(t[tr],e[tr],t[te],e[te],60)
        for gene in configuration['genes']:
            for seed in configuration['seeds']:
                for model in MODELS:allrisks[ep,gene,seed,model]=np.full(512,np.nan)
                for fold in configuration['folds']:
                    name=unit(ep,gene,seed,fold);cell=json.loads((PRIVATE/'cells'/f'{name}.json').read_text());z=torch.load(cell['artifact'],map_location='cpu',weights_only=False)
                    if cell['source_hash']!=source() or sha(cell['artifact'])!=cell['artifact_sha256'] or cell['outer_test_selection']:raise RuntimeError('artifact/source mismatch')
                    tr=np.flatnonzero(valid&(folds!=fold));te=np.flatnonzero(valid&(folds==fold));part=z['partitions']
                    if not np.array_equal(te,part['outer_test']) or not np.array_equal(tr,part['outer_train']) or set(part['inner_train'])&set(part['inner_val']) or set(part['inner_train'])|set(part['inner_val'])!=set(tr):raise RuntimeError('partition leakage')
                    if not np.array_equal(z['test_time'],t[te]) or not np.array_equal(z['test_event'],e[te]):raise RuntimeError('OOF outcomes')
                    strata=np.char.add(d['cancer_type'].astype(str),np.where(np.isnan(d['rna_genes']).any(1),'_missing','_complete'))
                    for role,fit,test in [('inner',part['inner_train'],part['inner_val']),('outer',tr,te)]:
                        for side,ix in [('fit',fit),('test',test)]:
                            perm=z['permutations'][role+'_'+side]
                            if set(perm)!=set(ix) or np.any(perm==ix) or not np.array_equal(strata[perm],strata[ix]):raise RuntimeError('wrong-patient identity/stratum')
                        covered=[]
                        for a in z['states'][role+'_teacher']['audit']:
                            if set(a['train'])&set(a['test']) or set(a['train'])|set(a['test'])!=set(range(len(fit))):raise RuntimeError('teacher leakage')
                            covered+=a['test']
                        if sorted(covered)!=list(range(len(fit))):raise RuntimeError('teacher OOF coverage')
                    maximum=max(maximum,replay(z,ep,gene,seed,fold,configuration));cellcount+=1
                    for model in MODELS:
                        risk=np.asarray(z['risks'][model],float)
                        if not np.isfinite(risk).all():raise RuntimeError('risk finite')
                        hn,hd=components(t[te],e[te],risk);un,ud=components(t[te],e[te],risk,weights[ep][te])
                        hh=concordance_index_censored(e[te].astype(bool),t[te],risk)[0];uu=concordance_index_ipcw(Surv.from_arrays(e[tr].astype(bool),t[tr]),Surv.from_arrays(e[te].astype(bool),t[te]),risk,tau=60)[0]
                        if abs(hh-hn/hd)>1e-10 or abs(uu-un/ud)>1e-10:raise RuntimeError('independent concordance mismatch')
                        allrisks[ep,gene,seed,model][te]=risk
                        rows.append({'endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'model':model,'Harrell':hh,'Uno60':uu,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud,'n':len(te),'events':int(e[te].sum())})
                    diag.extend([{**a,'endpoint':ep,'gene':gene,'seed':seed,'fold':fold} for a in cell['decodability']])
                    for kind,infos in [('producer',cell['producers']),('readout',cell['downstream'])]:
                        for label,info in infos.items():
                            logs=info['logs'];selected=info['selected_epoch'];picked=logs[selected-1]
                            train.append({'endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'kind':kind,'stage':label,'selected_epoch':selected,'actual_epochs':info['actual_epochs'],'parameters':info['parameters'],'connected_parameters':logs[-1]['connected_parameters'],'train_initial':logs[0]['train_cox'],'train_last':logs[-1]['train_cox'],'train_selected':picked['train_cox'],'validation_selected':picked['validation_cox'],'last_gradient':logs[-1]['gradient_norm'],'last_rank_KL':logs[-1]['rank_KL'],'test_correction_max':info.get('test_correction_max',0)})
    table=pd.DataFrame(rows);table.to_csv(REPORT/'FOLD_METRICS.csv',index=False);pd.DataFrame(diag).to_csv(REPORT/'TEACHER_DECODABILITY.csv',index=False);pd.DataFrame(train).to_csv(REPORT/'TRAINING_DIAGNOSTICS.csv',index=False)
    seedrows=[]
    for (ep,gene,seed,model),group in table.groupby(['endpoint','gene','seed','model']):seedrows.append({'endpoint':ep,'gene':gene,'seed':seed,'model':model,'Harrell':group.H_num.sum()/group.H_den.sum(),'Uno60':group.U_num.sum()/group.U_den.sum()})
    pd.DataFrame(seedrows).to_csv(REPORT/'SEED_OOF_METRICS.csv',index=False)
    statistics(configuration,d,endpoints,weights,allrisks,seedrows)
    audit={'status':'PASS','pipelines':cellcount,'model_fold_records':len(rows),'expected_records':1710,'max_checkpoint_replay_error':maximum,'all_metric_cells_crosschecked_sksurv':True,'teacher_and_wrong_pairing_audited':True,'parent_raw_endpoint_independent':True,'source_hash':source()}
    write(REPORT/'INDEPENDENT_AUDIT.json',audit);write(PRIVATE/'EVALUATION_DONE.json',audit);print(json.dumps(audit))


def statistics(configuration,d,endpoints,weights,risks,seedrows):
    comparisons=[('RANK64','TOKEN64'),('POOL64','TOKEN64'),('INTERACT','ADDITIVE'),('INTERACT','COX'),('INTERACT','WRONG_INTERACT'),('TOKEN64','RAW64'),('RANK64','RAW64'),('RANKPOOL64','RANK64'),('TOKEN512','RAW512'),('RANK512','TOKEN512'),('COX','CS'),('COX','COXRAW'),('COX','COXPOOL'),('COX','WRONG_COX'),('ADDITIVE','COX'),('INTERACT','CS'),('TOKEN64','INIT64'),('POOL64','INITPOOL64')]
    output=[];pairdiag=[]
    for ep in configuration['endpoints']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']]
        # Exclude two invalid PFI events before calling independent pair builder.
        local=np.flatnonzero(valid);ii,jj=pairs(t[local],e[local]);ii,jj=local[ii],local[jj];keep=d['folds'][ii]==d['folds'][jj];ii,jj=ii[keep],jj[keep]
        credits={}
        for gene in configuration['genes']:
            for model in MODELS:
                perseed=[]
                for seed in configuration['seeds']:
                    delta=risks[ep,gene,seed,model][ii]-risks[ep,gene,seed,model][jj];perseed.append(np.where(abs(delta)<=1e-8,.5,delta>0))
                credits[gene,model]=np.mean(perseed,axis=0)
        inventory=[(gene,a,b) for gene in configuration['genes'] for a,b in comparisons];differences=np.stack([credits[g,a]-credits[g,b] for g,a,b in inventory],1)
        rng=np.random.default_rng(configuration['bootstrap_seed']+(0 if ep=='DSS' else 1));draws={'Harrell':[],'Uno60':[]}
        for start in range(0,configuration['bootstrap_draws'],250):
            n=min(250,configuration['bootstrap_draws']-start);counts=np.zeros((n,512),int);counts[:,local]=rng.multinomial(len(local),np.ones(len(local))/len(local),size=n)
            q=(counts[:,ii]*counts[:,jj]).astype(float);qw=q*weights[ep][ii]
            draws['Harrell'].append(q@differences/q.sum(1)[:,None]);draws['Uno60'].append(qw@differences/qw.sum(1)[:,None])
        for metric in draws:
            sample=np.concatenate(draws[metric]);ww=np.ones(len(ii)) if metric=='Harrell' else weights[ep][ii]
            for j,(gene,a,b) in enumerate(inventory):
                values=[]
                for seed in configuration['seeds']:
                    aa=next(r[metric] for r in seedrows if (r['endpoint'],r['gene'],r['seed'],r['model'])==(ep,gene,seed,a));bb=next(r[metric] for r in seedrows if (r['endpoint'],r['gene'],r['seed'],r['model'])==(ep,gene,seed,b));values.append(aa-bb)
                primary=ep=='DSS' and gene=='HALL' and (a,b) in comparisons[:5];alpha=.05/10 if primary else .05
                lo,hi=np.quantile(sample[:,j],[.025,.975]);cl,ch=np.quantile(sample[:,j],[alpha/2,1-alpha/2]);point=float(ww@differences[:,j]/ww.sum())
                if abs(point-np.mean(values))>1e-10:raise RuntimeError('point/statistic mismatch')
                output.append({'endpoint':ep,'gene':gene,'comparison':a+'-'+b,'metric':metric,'delta':point,'CI_low':lo,'CI_high':hi,'corrected_low':cl,'corrected_high':ch,'primary':primary,'positive_seeds':sum(v>0 for v in values),**{f'seed{s}':v for s,v in zip(configuration['seeds'],values)}})
                pairdiag.append({'endpoint':ep,'gene':gene,'comparison':a+'-'+b,'metric':metric,'improved_pair_mass':float(ww[differences[:,j]>0].sum()/ww.sum()),'worsened_pair_mass':float(ww[differences[:,j]<0].sum()/ww.sum())})
    pd.DataFrame(output).to_csv(REPORT/'PAIRED_COMPARISONS.csv',index=False);pd.DataFrame(pairdiag).to_csv(REPORT/'PAIR_TRANSITIONS.csv',index=False)


if __name__=='__main__':evaluate()
