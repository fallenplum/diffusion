"""Independent raw OOF/clinical evaluation; imports NO training math/features."""
import hashlib,json,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd
from validate_phase11_independent import pairs,ipcw,components
R=ROOT/'reports/phase13_rna_information_audit';P=Path('/root/phase13_rna_private');F=P/'formal'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()


def expected_models(cfg):
    models=['C','S','CS']
    for gene in cfg['genes']:
        models += [gene+'_'+n for n in ['R_tuned','R_fixed1','PCA16_tuned','RESID_tuned','C_RAW','S_RAW','CS_RAW','CS_PCA16']]
        models += ['RO_'+gene+'_HEAD']+['RO_'+gene+'_'+state+'_'+tap for state in ['before','after'] for tap in cfg['RNA_only_taps']]
        models += ['CS_RO_'+gene+'_'+tap for tap in cfg['conditional_probe_taps']]
        models += ['F_'+gene+'_ORIGINAL']+['F_'+gene+'_'+state+'_'+tap for state in ['before','after'] for tap in cfg['fusion_taps']]
        models += ['CS_F_'+gene+'_'+tap for tap in cfg['fusion_conditional_probe_taps']]
    if len(models)!=111 or len(set(models))!=111:raise RuntimeError('locked model inventory')
    return models


def contrasts(cfg):
    output={};primary=set()
    for gene in cfg['genes']:
        name=gene+'_CS_RAW-CS';output[name]=[(gene+'_CS_RAW',1),('CS',-1)];primary.add(name)
        name=gene+'_FUSED_processed_full-RAW_fixed1';output[name]=[('F_'+gene+'_after_processed_full',1),(gene+'_R_fixed1',-1)];primary.add(name)
        terms={
            'RNAonly_native-RNAlinear':('RO_'+gene+'_HEAD',gene+'_R_tuned'),
            'RNAonly_full-RAW_fixed1':('RO_'+gene+'_after_processed_full',gene+'_R_fixed1'),
            'Fused_full-RNAonly_full':('F_'+gene+'_after_processed_full','RO_'+gene+'_after_processed_full'),
            'Fused_after-before':('F_'+gene+'_after_processed_full','F_'+gene+'_before_processed_full'),
            'RNAonly_after-before':('RO_'+gene+'_after_processed_full','RO_'+gene+'_before_processed_full'),
            'Fused_mean-full':('F_'+gene+'_after_processed_mean','F_'+gene+'_after_processed_full'),
            'Retrieval_full-RNA_full':('F_'+gene+'_after_RNApatch_full','F_'+gene+'_after_processed_full'),
            'Retrieval_mean-full':('F_'+gene+'_after_RNApatch_mean','F_'+gene+'_after_RNApatch_full'),
            'CS_Fused_full-CS':('CS_F_'+gene+'_after_processed_full','CS'),
            'CS_RNAonly_full-CS':('CS_RO_'+gene+'_after_processed_full','CS'),
            'CS_Retrieval-CS':('CS_F_'+gene+'_after_RNApatch_full','CS'),
            'CS_Retrieval-CS_RAW':('CS_F_'+gene+'_after_RNApatch_full',gene+'_CS_RAW'),
            'C_RAW-C':(gene+'_C_RAW','C'),'S_RAW-S':(gene+'_S_RAW','S'),'CS_PCA16-CS':(gene+'_CS_PCA16','CS'),
            'ResidualRNA-RNAlinear':(gene+'_RESID_tuned',gene+'_R_tuned')}
        for suffix,(a,b) in terms.items():output[gene+'_'+suffix]=[(a,1),(b,-1)]
    output['HALL_linear-TOP_linear']=[('HALL_R_tuned',1),('TOP_R_tuned',-1)]
    output['HALL_linear-MATCH_linear']=[('HALL_R_tuned',1),('MATCH_R_tuned',-1)]
    return output,primary


def raw_endpoints(d,cfg):
    file=Path('/root/autodl-tmp/diffusion_data/data/data_8/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt')
    raw=pd.read_csv(file,sep='\t',comment='#',dtype=str);raw['pid']=raw['PATIENT_ID'].str.extract(r'^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})')[0]
    if raw['pid'].duplicated().any():raise RuntimeError('ambiguous raw clinical patient')
    a=raw.set_index('pid').loc[d['patient_ids'].astype(str)];stored=dict(np.load(P/'endpoints.npz'));out={}
    registry=json.loads((R/'SOURCE_REGISTRY.json').read_text())
    if digest('/root/phase7a_foundation_headroom_v1/private_dataset.npz')!=registry['dataset_sha256'] or digest(P/'endpoints.npz')!=registry['endpoints_npz_sha256']:raise RuntimeError('data/endpoint integrity')
    for ep,alias in [('DSS','DSS'),('PFI','PFS')]:
        t=pd.to_numeric(a[alias+'_MONTHS'],errors='coerce').to_numpy();e=a[alias+'_STATUS'].str.extract(r'^\s*([01])')[0].astype(float).to_numpy();valid=np.isfinite(t)&(t>0)&np.isin(e,[0,1])
        if not np.array_equal(t,stored[ep+'_time']) or not np.array_equal(e,stored[ep+'_event']) or not np.array_equal(valid,stored[ep+'_valid']):raise RuntimeError('independent raw endpoint mismatch')
        out[ep]={'time':t,'event':e.astype(int),'valid':valid}
    return out


def test_loss(t,e,r):
    # Literal independent Efron; no training objective import.
    terms=[]
    for value in np.unique(t[e==1]):
        death=(t==value)&(e==1);at_risk=t>=value;rr=r[at_risk];mx=rr.max();total=np.exp(rr-mx).sum();dd=np.exp(r[death]-mx).sum();number=death.sum()
        terms.extend(mx+np.log(total-j/number*dd) for j in range(number))
    return float((sum(terms)-r[e==1].sum())/e.sum())


def validate():
    cfg=json.loads((ROOT/'configs/phase13_rna_information_audit.json').read_text());gate=json.loads((R/'HARD_GATES.json').read_text());d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));endpoints=raw_endpoints(d,cfg);models=expected_models(cfg)
    from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
    from sksurv.util import Surv
    rows=[];risks={};failed=[];weights={}
    for ep in cfg['endpoints']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];weights[ep]=np.zeros(512)
        for fold in cfg['folds']:
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));weights[ep][te]=ipcw(t[tr],e[tr],t[te],e[te],cfg['tau'])
        for model in models:
            for seed in cfg['seeds']:
                r=np.full(512,np.nan)
                for fold in cfg['folds']:
                    name=f'{ep}_{model}_seed{seed}_fold{fold}'
                    try:
                        c=json.loads((F/'cells'/f'{name}.json').read_text());z=dict(np.load(c['prediction_path']));m=json.loads(Path(c['manifest_path']).read_text());te=np.flatnonzero(valid&(d['folds']==fold));tr=np.flatnonzero(valid&(d['folds']!=fold))
                        if c['status']!='COMPLETE' or c['source_hash']!=gate['source_hash'] or c['protocol_sha256']!=digest(ROOT/'configs/phase13_rna_information_audit.json') or c['outer_test_selection']:raise RuntimeError('status/protocol/source')
                        if c['endpoint']!=ep or c['model']!=model or c['seed']!=seed or c['fold']!=fold or str(z['endpoint'])!=ep or str(z['model'])!=model or int(z['seed'])!=seed or int(z['fold'])!=fold:raise RuntimeError('container mismatch')
                        if digest(c['prediction_path'])!=c['prediction_sha256'] or digest(c['manifest_path'])!=c['manifest_sha256'] or digest(c['checkpoint_path'])!=c['checkpoint_sha256']:raise RuntimeError('artifact hash')
                        if not np.array_equal(z['test_indices'],te) or not np.array_equal(z['time'],t[te]) or not np.array_equal(z['event'],e[te]) or not np.isfinite(z['risk']).all():raise RuntimeError('wrong patient/endpoint/risk')
                        if m['outer_train']!=tr.tolist() or m['outer_test']!=te.tolist() or set(m['inner_train'])&set(m['inner_val']) or set(m['inner_train'])|set(m['inner_val'])!=set(tr):raise RuntimeError('partition leakage')
                        if m['patient_hash']!=ahash(d['patient_ids']) or m['fold_hash']!=ahash(d['folds']):raise RuntimeError('patient/fold integrity')
                        for role in ['inner','outer']:
                            a=m['preprocessing_'+role];fit=m['inner_train'] if role=='inner' else m['outer_train']
                            if a['fit_indices']!=fit or a['fit_hash']!=ahash(np.array(fit,dtype=np.int64)) or set(a['source_fields'])!=set(['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE']):raise RuntimeError('preprocessing fit/fields')
                        if model.startswith('F_') or model.startswith('CS_F_'):
                            if m['producer']['kind']!='FUSION_DSS_FROZEN':raise RuntimeError('PFI producer transfer mislabel')
                        r[te]=z['risk'];hn,hd=components(t[te],e[te],r[te]);un,ud=components(t[te],e[te],r[te],weights[ep][te])
                        hh=concordance_index_censored(e[te].astype(bool),t[te],r[te])[0];uu=concordance_index_ipcw(Surv.from_arrays(e[tr].astype(bool),t[tr]),Surv.from_arrays(e[te].astype(bool),t[te]),r[te],tau=cfg['tau'])[0]
                        if abs(hh-hn/hd)>1e-10 or abs(uu-un/ud)>1e-10:raise RuntimeError('independent metric mismatch')
                        rows.append({'endpoint':ep,'model':model,'seed':seed,'fold':fold,'test_n':len(te),'test_events':int(e[te].sum()),'Harrell':hn/hd,'Uno60':un/ud,'Harrell_num':hn,'Harrell_den':hd,'Uno_num':un,'Uno_den':ud,'test_Efron_loss':test_loss(t[te],e[te],r[te]),'parameters':c['parameters'],'selected_epoch':c.get('selected_epoch',0),'actual_fit':c.get('actual_fit'),'RNA_lambda':c.get('RNA_lambda'),'lambda':c.get('lambda')})
                    except Exception as error:failed.append({'cell':name,'error':str(error)})
                if np.isfinite(r[valid]).all():risks[ep,model,seed]=r
    status={'status':'FAIL' if failed else 'PASS','expected_records':3330,'verified_records':len(rows),'failures':failed,'independent_raw_clinical_verified':True,'metrics_crosschecked':'scikit-survival0.23.1'}
    pd.DataFrame(rows).to_csv(R/'FOLD_METRICS.csv',index=False);(R/'INDEPENDENT_METRICS.json').write_text(json.dumps(status,indent=2))
    if failed:print(json.dumps({'status':'FAIL','records':len(rows),'first_failures':failed[:5]}));return 2
    seedrows=[]
    for ep,model,seed in risks:
        cells=[r for r in rows if r['endpoint']==ep and r['model']==model and r['seed']==seed];n=sum(r['test_events'] for r in cells)
        seedrows.append({'endpoint':ep,'model':model,'seed':seed,'Harrell':sum(r['Harrell_num'] for r in cells)/sum(r['Harrell_den'] for r in cells),'Uno60':sum(r['Uno_num'] for r in cells)/sum(r['Uno_den'] for r in cells),'test_Efron_loss':sum(r['test_Efron_loss']*r['test_events'] for r in cells)/n})
    pd.DataFrame(seedrows).to_csv(R/'SEED_OOF_METRICS.csv',index=False)
    statistics(cfg,d,endpoints,weights,risks,seedrows)
    print(json.dumps(status));return 0


def statistics(cfg,d,endpoints,weights,risks,seedrows):
    contrast,primary=contrasts(cfg);rows=[];pair_rows=[];events=[];influence=[];audit=[]
    for ep in cfg['endpoints']:
        t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];pi,pj=pairs(t,e);keep=valid[pi]&valid[pj]&(d['folds'][pi]==d['folds'][pj]);pi,pj=pi[keep],pj[keep];w=weights[ep][pi]
        credits={}
        for model in expected_models(cfg):
            for seed in cfg['seeds']:
                delta=risks[ep,model,seed][pi]-risks[ep,model,seed][pj];credits[model,seed]=np.where(abs(delta)<=1e-8,.5,delta>0)
        contrasts_credit=np.stack([np.mean([sum(c*credits[a,seed] for a,c in terms) for seed in cfg['seeds']],axis=0) for terms in contrast.values()],1)
        rng=np.random.default_rng(cfg['bootstrap']['seed']+(0 if ep=='DSS' else 1));draws={metric:[] for metric in ['Harrell','Uno60']};hashcounts=hashlib.sha256()
        indices=np.flatnonzero(valid)
        for start in range(0,cfg['bootstrap']['draws'],250):
            n=min(250,cfg['bootstrap']['draws']-start);cc=rng.multinomial(len(indices),np.ones(len(indices))/len(indices),size=n);counts=np.zeros((n,512),np.int64);counts[:,indices]=cc;hashcounts.update(counts.tobytes());q=(counts[:,pi]*counts[:,pj]).astype(float);qw=q*w
            draws['Harrell'].append((q@contrasts_credit)/q.sum(1)[:,None]);draws['Uno60'].append((qw@contrasts_credit)/qw.sum(1)[:,None])
        for metric,blocks in draws.items():
            sample=np.concatenate(blocks)
            for j,(name,terms) in enumerate(contrast.items()):
                values=[sum(c*next(r[metric] for r in seedrows if r['endpoint']==ep and r['model']==a and r['seed']==seed) for a,c in terms) for seed in cfg['seeds']];is_primary=ep=='DSS' and name in primary;alpha=.05/cfg['primary_family_tests'] if is_primary else .05;lo,hi=np.quantile(sample[:,j],[.025,.975]);cl,ch=np.quantile(sample[:,j],[alpha/2,1-alpha/2])
                rows.append({'endpoint':ep,'comparison':name,'metric':metric,'family':'primary_DSS' if is_primary else 'secondary_unadjusted','mean_delta':np.mean(values),**{'seed'+str(s)+'_delta':v for s,v in zip(cfg['seeds'],values)},'positive_seeds':sum(v>0 for v in values),'negative_seeds':sum(v<0 for v in values),'CI95_low':lo,'CI95_high':hi,'corrected_low':cl,'corrected_high':ch,'draws':len(sample)})
        for metric,weight in [('Harrell',np.ones(len(pi))),('Uno60',w)]:
            mass=np.bincount(pi,weights=weight,minlength=512);active=mass>0;neff=float(mass.sum()**2/(mass@mass))
            events.append({'endpoint':ep,'metric':metric,'events_observed':int(e[valid].sum()),'events_with_comparable_credit':int(active.sum()),'effective_event_credit_Neff':neff,'largest_event_mass_fraction':float(mass.max()/mass.sum()),'comparable_mass':float(mass.sum()),'notice':'credit concentration,not independent-pair sample size'})
            for name,terms in contrast.items():
                if name not in primary:continue
                j=list(contrast).index(name);credit=contrasts_credit[:,j];point=weight@credit/weight.sum();values=[]
                for event in np.flatnonzero(active):
                    retain=(pi!=event)&(pj!=event);ww=weight[retain];values.append(float(ww@credit[retain]/ww.sum()))
                influence.append({'endpoint':ep,'comparison':name,'metric':metric,'point':point,'leave_one_event_min':min(values),'leave_one_event_max':max(values),'removed_event_count':len(values),'positive_removals':sum(v>0 for v in values),'scope':'fixed models,no refit,event omissions not used to alter primary'})
        for gene in cfg['genes']:
            for probe in ['RO_'+gene+'_HEAD','RO_'+gene+'_after_processed_full','F_'+gene+'_after_processed_full','F_'+gene+'_after_processed_mean','F_'+gene+'_after_RNApatch_full','F_'+gene+'_after_RNApatch_mean',gene+'_CS_RAW']:
                reference=gene+'_R_fixed1' if probe!=gene+'_CS_RAW' else 'CS'
                for metric,weight in [('Harrell',np.ones(len(pi))),('Uno60',w)]:
                    for seed in cfg['seeds']:
                        a=credits[reference,seed];b=credits[probe,seed];ties=(a==.5)|(b==.5);loss=(a==1)&(b==0);gain=(a==0)&(b==1);both=(a==1)&(b==1);neither=(a==0)&(b==0);base=credits['CS',seed];wrong=base==0
                        pair_rows.append({'endpoint':ep,'gene':gene,'probe':probe,'reference':reference,'seed':seed,'metric':metric,'both_correct_mass':float(weight[both].sum()),'reference_correct_lost_mass':float(weight[loss].sum()),'reference_wrong_rescued_mass':float(weight[gain].sum()),'both_wrong_mass':float(weight[neither].sum()),'any_tie_mass':float(weight[ties].sum()),'total_mass':float(weight.sum()),'gain_minus_loss_fraction':float((weight[gain].sum()-weight[loss].sum())/weight.sum()),'probe_credit_on_CS_wrong_pairs':float(weight[wrong]@b[wrong]/weight[wrong].sum()),'RNA_reference_credit_on_CS_wrong_pairs':float(weight[wrong]@credits[gene+'_R_fixed1',seed][wrong]/weight[wrong].sum()),'CS_wrong_pair_mass':float(weight[wrong].sum())})
        audit.append({'endpoint':ep,'draws':cfg['bootstrap']['draws'],'counts_sha256':hashcounts.hexdigest(),'patient_count':int(valid.sum()),'same_patientcounts_all_seeds_models':True,'risk_averaging':False,'IPCW_train_fixed':True,'retraining_variance_included':False})
    frame=pd.DataFrame(rows);frame.to_csv(R/'ALL_PAIRED.csv',index=False);frame[frame.family=='primary_DSS'].to_csv(R/'PRIMARY_PAIRED.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(R/'PAIR_RETENTION_TRANSITIONS.csv',index=False);pd.DataFrame(events).to_csv(R/'EVENT_INFORMATION_MASS.csv',index=False);pd.DataFrame(influence).to_csv(R/'LEAVE_ONE_EVENT_INFLUENCE.csv',index=False)
    (R/'BOOTSTRAP_AUDIT.json').write_text(json.dumps({'status':'PASS','primary_tests':12,'endpoint_audits':audit},indent=2))


if __name__=='__main__':raise SystemExit(validate())
