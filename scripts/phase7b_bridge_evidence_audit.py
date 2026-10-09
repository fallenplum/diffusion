"""Two locked exploratory diagnostics before any Bridge redesign."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from phase7_p0_biology_prompt_audit import REPO,BASE,digest,ahash,write_json,STATES,gene_z,group_scores,shuffled_mask
from phase7a_foundation_survival_headroom import locked as parent_locked,features,fit_cox,concordance,km_left

CONFIG=REPO/'configs/phase7b_bridge_evidence_audit.json'
SELF=REPO/'scripts/phase7b_bridge_evidence_audit.py'
DEFAULT=Path('/root/phase7b_bridge_evidence_v2')


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);assert not (out/'manifest.json').exists()
    p=json.loads(CONFIG.read_text());parent=Path(p['parent_run']);m=parent_locked(parent)
    assert json.loads((parent/'independent_validation.json').read_text())['status']=='PASS'
    assert (parent/'EXITCODE').read_text().strip()=='0'
    old=json.loads((Path(p['p0_run'])/'manifest.json').read_text())
    mask=np.asarray(old['concept_mask'])>0;assert mask.shape==(32,8) and mask.sum()==15
    sources=[CONFIG,SELF,REPO/'scripts/phase7a_foundation_survival_headroom.py',REPO/'scripts/phase7_p0_biology_prompt_audit.py',
        REPO/'configs/phase7_p0_molecular_morphology_prior.json',REPO/'configs/phase6a_titan_crc_morphology_concepts.json',
        Path(p['p0r_run'])/'sample_quality.tsv',BASE.parent/'coadread_tcga'/'data_clinical_patient.txt',BASE/'data_clinical_patient.txt']
    write_json(out/'protocol.json',p)
    write_json(out/'manifest.json',{'protocol_sha256':digest(out/'protocol.json'),'locked_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'parent_protocol_sha256':m['protocol_sha256'],'parent_dataset_sha256':m['private_dataset_sha256'],
        'sources':[{'path':str(f),'sha256':digest(f)} for f in sources], 'RNA_prior_mask':mask.astype(int).tolist(),
        'prior_edges':15,'covered_concepts':int(mask.any(1).sum()),'exploratory':True})
    for sub in ['oof','cells','coefficients','preprocessing']:(out/sub).mkdir(exist_ok=True)
    print('BRIDGE_EVIDENCE_PROTOCOL_LOCKED',flush=True)


def locked(out):
    m=json.loads((out/'manifest.json').read_text());p=json.loads((out/'protocol.json').read_text())
    assert digest(out/'protocol.json')==m['protocol_sha256']
    for f in m['sources']:assert digest(f['path'])==f['sha256']
    pm=parent_locked(Path(p['parent_run']));assert pm['private_dataset_sha256']==m['parent_dataset_sha256']
    return p,m,pm


def worker(out,worker_id):
    p,m,pm=locked(out);parent=Path(p['parent_run']);pp=json.loads((parent/'protocol.json').read_text())
    d=dict(np.load(parent/'private_dataset.npz'));t=d['DSS_time'];e=d['DSS_event'];seed=pp['seeds'][worker_id]
    for fold in range(1,6):
        tag=f'DSS_seed{seed}_fold{fold}';pr=np.load(parent/'preprocessing'/f'{tag}_outer.npz');tr=pr['tr'];te=pr['te'];it=pr['it'];iv=pr['iv']
        fi,ai,pi=features(d,it,pm['groups']);fo,ao,po=features(d,tr,pm['groups'])
        fi['CNA8']=fi['MC'][:,8:];fo['CNA8']=fo['MC'][:,8:]
        for scope,private in [('inner',pi),('outer',po)]:
            before=np.load(parent/'preprocessing'/f'{tag}_{scope}.npz')
            for k in private:
                if private[k].dtype.kind=='f':assert np.allclose(private[k],before[k],rtol=1e-12,atol=1e-12)
                else:assert np.array_equal(private[k],before[k])
            np.savez_compressed(out/'preprocessing'/f'{tag}_{scope}.npz',**private,tr=tr,te=te,it=it,iv=iv)
        for method,blocks in p['A']['new_methods'].items():
            name=tag+'_'+method;cp=out/'cells'/f'{name}.json';op=out/'oof'/f'{name}.npz'
            if cp.exists() and op.exists():continue
            xi=np.concatenate([fi[k] for k in blocks],1);xo=np.concatenate([fo[k] for k in blocks],1);trials=[]
            for lam in pp['ridge_grid']:
                w,detail=fit_cox(xi[it],t[it],e[it],lam)
                trials.append({'ridge':lam,'inner_harrell':concordance(t[iv],e[iv],xi[iv]@w),**detail})
            choice=max(trials,key=lambda z:(z['inner_harrell'],z['ridge']))
            w,detail=fit_cox(xo[tr],t[tr],e[tr],choice['ridge']);train=xo[tr]@w;raw=xo[te]@w;mean=train.mean();sd=train.std();assert sd>1e-10
            g=km_left(t[tr],e[tr],t[te]);active=(e[te]==1)&(t[te]<=60);assert (g[active]>0).all()
            weight=np.zeros(len(te));weight[active]=1/g[active]**2
            np.savez_compressed(op,test_indices=te,patient_ids=d['patient_ids'][te],time=t[te],event=e[te],risk=(raw-mean)/sd,
                raw_risk=raw,uno_weight=weight,censor_g=g,seed=seed,fold=fold,method=method,protocol_sha256=m['protocol_sha256'])
            np.savez_compressed(out/'coefficients'/f'{name}.npz',coefficients=w,train_mean=mean,train_sd=sd)
            write_json(cp,{'status':'PASS','seed':seed,'fold':fold,'method':method,'blocks':blocks,'feature_dim':xo.shape[1],
                'trials':trials,'selected_ridge':choice['ridge'],'outer_fit':detail,'inner_preprocessing':ai,'outer_preprocessing':ao,
                'oof_sha256':digest(op),'outer_test_used_for_selection':False,'same_parent_split_and_preprocessing':True})
        print('A_FOLD_DONE',seed,fold,flush=True)
    locked(out);write_json(out/f'worker_{worker_id}_DONE.json',{'status':'PASS'})


def load_oof(root,method,seed,d):
    arrays={k:np.full(512,np.nan) for k in ['risk','raw_risk','uno_weight']};count=np.zeros(512,int)
    for fold in range(1,6):
        tag=f'DSS_seed{seed}_fold{fold}_{method}';op=root/'oof'/f'{tag}.npz';a=np.load(op);c=json.loads((root/'cells'/f'{tag}.json').read_text())
        assert digest(op)==c['oof_sha256'];idx=a['test_indices'];assert np.array_equal(idx,np.flatnonzero(d['folds']==fold))
        assert np.array_equal(a['patient_ids'],d['patient_ids'][idx]) and np.array_equal(a['event'],d['DSS_event'][idx]) and np.array_equal(a['time'],d['DSS_time'][idx])
        for k in arrays:arrays[k][idx]=a[k]
        count[idx]+=1
    assert (count==1).all() and all(np.isfinite(a).all() for a in arrays.values())
    return arrays


def aggregate_a(out):
    p,m,pm=locked(out);parent=Path(p['parent_run']);pp=json.loads((parent/'protocol.json').read_text());d=dict(np.load(parent/'private_dataset.npz'))
    for w in range(3):assert json.loads((out/f'worker_{w}_DONE.json').read_text())['status']=='PASS'
    methods=p['A']['reuse_methods']+list(p['A']['new_methods']);t=d['DSS_time'];e=d['DSS_event'];fold=d['folds'];pred={};rows=[]
    for seed in pp['seeds']:
        for method in methods:
            root=parent if method in p['A']['reuse_methods'] else out
            a=load_oof(root,method,seed,d);pred[seed,method]=a
            rows.append({'seed':seed,'method':method,'reused_parent_oof':root==parent,'patients':512,'events':65,
                'oof_harrell':concordance(t,e,a['risk']),'oof_uno60':concordance(t,e,a['risk'],a['uno_weight']),
                'raw_oof_harrell':concordance(t,e,a['raw_risk']),
                'within_fold_pair_weighted_harrell':concordance(t,e,a['risk'],folds=fold),
                'within_fold_pair_weighted_uno60':concordance(t,e,a['risk'],a['uno_weight'],fold)})
    rng=np.random.default_rng(88211);boot=np.zeros((5000,512))
    for f in range(1,6):
        for ev in [0,1]:
            group=np.flatnonzero((fold==f)&(e==ev));boot[:,group]=rng.multinomial(len(group),np.ones(len(group))/len(group),size=5000)
    eligible=(e[:,None]==1)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&(e[None,:]==0)))
    distributions={}
    for method in methods:
        for metric in ['harrell','uno60']:
            vals=[]
            for seed in pp['seeds']:
                a=pred[seed,method];r=a['risk'];credit=(r[:,None]>r[None,:]).astype(float)+.5*(r[:,None]==r[None,:])
                den=eligible*(np.ones(512) if metric=='harrell' else a['uno_weight'])[:,None]
                vals.append(np.sum((boot@(den*credit))*boot,1)/np.sum((boot@den)*boot,1))
            distributions[method,metric]=np.mean(vals,0)
    table=pd.DataFrame(rows);pairs=[]
    for pair in p['A']['primary_pairs']+p['A']['secondary_anchor_pairs']:
        a,b=pair.split('-');primary=pair in p['A']['primary_pairs'];row={'pair':pair,'family':'primary_omics' if primary else 'secondary_anchor'}
        for metric in ['oof_harrell','oof_uno60','raw_oof_harrell','within_fold_pair_weighted_harrell','within_fold_pair_weighted_uno60']:
            ds=[float(table[(table.seed==s)&(table.method==a)].iloc[0][metric]-table[(table.seed==s)&(table.method==b)].iloc[0][metric]) for s in pp['seeds']]
            row['delta_'+metric]=float(np.mean(ds));row[metric+'_seed_deltas']=json.dumps(ds)
        for metric in ['harrell','uno60']:
            delta=distributions[a,metric]-distributions[b,metric];alpha=.05/(5 if primary else 2)
            row[metric+'_ci95_low'],row[metric+'_ci95_high']=np.quantile(delta,[.025,.975]).tolist()
            row[metric+'_family_ci_low'],row[metric+'_family_ci_high']=np.quantile(delta,[alpha/2,1-alpha/2]).tolist()
        row['locked_increment_gate']=row['harrell_family_ci_low']>0 and row['delta_oof_uno60']>0 and min(json.loads(row['oof_harrell_seed_deltas']))>0
        row['calibration_robust_gate']=row['locked_increment_gate'] and row['delta_raw_oof_harrell']>0 and row['delta_within_fold_pair_weighted_harrell']>0
        pairs.append(row)
    table.to_csv(out/'A_seed_oof_metrics.csv',index=False)
    table.groupby('method').mean(numeric_only=True).drop(columns=['seed','reused_parent_oof']).reset_index().to_csv(out/'A_three_seed_oof_means.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'A_paired_increments.csv',index=False)
    cells=[json.loads(f.read_text()) for f in (out/'cells').glob('*.json')];assert len(cells)==60
    write_json(out/'A_decision.json',{'status':'PASS','primary_DSS':True,'new_outer_fits':60,'new_inner_fits':240,'reused_parent_outer_oof':45,
        'primary_pairs_passed':[r['pair'] for r in pairs if r['family']=='primary_omics' and r['calibration_robust_gate']],
        'maximum_iterations':max([c['outer_fit']['iterations'] for c in cells]+[t['iterations'] for c in cells for t in c['trials']]),
        'maximum_gradient':max([c['outer_fit']['gradient_max'] for c in cells]+[t['gradient_max'] for c in cells for t in c['trials']]),
        'scope':'conditional fixed-OOF retrospective inference, not conditional mutual information estimation'})
    print('A_COMPLETE',flush=True)


def correlation(x,y):
    x=x-x.mean(0);y=y-y.mean(0)
    return (x.T@y)/np.sqrt((x*x).sum(0)[:,None]*(y*y).sum(0)[None,:])


def platform_names(ids,quality):
    q=pd.read_csv(quality,sep='\t',dtype=str).fillna('');q=q[q['platform'].isin(['IlluminaGA_RNASeqV2','IlluminaHiSeq_RNASeqV2'])&(q['aliquot_barcode'].str[13:15]=='01')]
    result=[]
    for pid in ids:
        choices=sorted(set(q.loc[q.patient_barcode==pid,'platform']));assert choices
        result.append('+'.join(choices))
    return np.asarray(result,dtype='U80')


def nuisance(d,fit_idx,platform):
    f,_,_=features(d,fit_idx,json.loads((Path(json.loads(CONFIG.read_text())['parent_run'])/'manifest.json').read_text())['groups'])
    cols=[np.ones((512,1)),f['C']]
    for values in [d['cancer_type'],platform]:
        for level in np.unique(values[fit_idx]):cols.append((values==level)[:,None].astype(float))
    return np.concatenate(cols,1),f


def derangement_groups(groups,rng):
    perm=np.arange(512)
    for g in groups:
        if len(g)>1:
            order=rng.permutation(g);perm[order]=np.roll(order,1)
    return perm


def degree_map(mask,rng):
    result=mask.copy();accepted=0;attempts=0
    while accepted<200 and attempts<20000:
        attempts+=1;edges=np.argwhere(result);(r1,c1),(r2,c2)=edges[rng.choice(len(edges),2,replace=False)]
        if r1!=r2 and c1!=c2 and not result[r1,c2] and not result[r2,c1]:
            result[r1,c1]=result[r2,c2]=False;result[r1,c2]=result[r2,c1]=True;accepted+=1
    assert accepted==200 and np.array_equal(result.sum(0),mask.sum(0)) and np.array_equal(result.sum(1),mask.sum(1))
    return result


def morphology_audit(out,d,platform):
    primary=pd.read_csv(BASE/'data_clinical_patient.txt',sep='\t',comment='#',dtype=str).set_index('PATIENT_ID').loc[d['patient_ids'].astype(str)]
    sup_path=BASE.parent/'coadread_tcga'/'data_clinical_patient.txt'
    sup=pd.read_csv(sup_path,sep='\t',comment='#',dtype=str).set_index('PATIENT_ID').loc[d['patient_ids'].astype(str)]
    text=sup['HISTOLOGICAL_DIAGNOSIS'].fillna('').str.lower();valid=text.str.contains('adenocarcinoma').to_numpy();y=text.str.contains('mucinous').to_numpy().astype(int)
    icd=primary['ICD_O_3_HISTOLOGY'].fillna('').to_numpy();verify=valid&(icd!='');agree=(y==1)==(icd=='8480/3');concordant=verify&agree
    score=d['static'][:,9];auc=roc_auc_score(y[valid],score[valid]);rng=np.random.default_rng(88311);draw=np.zeros((2000,512))
    for cancer in np.unique(d['cancer_type']):
        for label in [0,1]:
            group=np.flatnonzero(valid&(d['cancer_type']==cancer)&(y==label))
            if len(group):draw[:,group]=rng.multinomial(len(group),np.ones(len(group))/len(group),size=2000)
    positive=np.flatnonzero(valid&(y==1));negative=np.flatnonzero(valid&(y==0));credit=(score[positive,None]>score[negative]).astype(float)+.5*(score[positive,None]==score[negative])
    bauc=((draw[:,positive]@credit)*draw[:,negative]).sum(1)/(draw[:,positive].sum(1)*draw[:,negative].sum(1))
    predictions=np.full((512,2),np.nan);iterations=[]
    for fold in range(1,6):
        tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));z,f=nuisance(d,tr,platform)
        for k in range(2):
            x=z[:,1:];x=np.concatenate([x,f['S'][:,9,None]],1) if k else x
            model=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=5000,tol=1e-7).fit(x[tr],y[tr]);assert model.n_iter_.max()<5000
            predictions[te,k]=model.predict_proba(x[te])[:,1];iterations.append(int(model.n_iter_.max()))
    np.savez_compressed(out/'private_morphology_oof.npz',valid=valid,diagnosis=y,C10_score=score,probabilities=predictions)
    result={'status':'PASS','independent_from_TITAN_scores':True,'label':'supplementary histological diagnosis mucinous substring',
        'valid_patients':int(valid.sum()),'mucinous_patients':int(y[valid].sum()),'unknown_diagnoses_excluded':int((~valid).sum()),
        'primary_ICD_mucinous_compared_n':int(verify.sum()),'primary_ICD_mucinous_disagreements':int((verify&~agree).sum()),
        'concordant_subset_n':int(concordant.sum()),'concordant_subset_mucinous_n':int(y[concordant].sum()),
        'concordant_subset_raw_C10_auc':float(roc_auc_score(y[concordant],score[concordant])),
        'zero_shot_C10_auc':float(auc),'conditional_bootstrap_ci95':np.quantile(bauc,[.025,.975]).tolist(),
        'nuisance_oof_auc':float(roc_auc_score(y[valid],predictions[valid,0])),
        'nuisance_plus_C10_oof_auc':float(roc_auc_score(y[valid],predictions[valid,1])),
        'maximum_classifier_iterations':max(iterations),'source_sha256':digest(sup_path),
        'scope':'one independent patient diagnosis; not patch-verified labels; C10 is not an edge-covered concept, so this cannot validate the 15 RNA mappings; foundation pretraining overlap not excluded'}
    write_json(out/'B_morphology_label_audit.json',result)


def alignment(out):
    p,m,pm=locked(out);parent=Path(p['parent_run']);d=dict(np.load(parent/'private_dataset.npz'));platform=platform_names(d['patient_ids'].astype(str),Path(p['p0r_run'])/'sample_quality.tsv')
    mask=np.asarray(m['RNA_prior_mask'],bool).T;assert mask.shape==(8,32)
    molecular=np.empty((512,16));static=np.empty((512,32));raw_molecular=np.empty_like(molecular);raw_static=np.empty_like(static);fold_corr=[];fits=[];cancel=[]
    for fold in range(1,6):
        tr=np.flatnonzero(d['folds']!=fold);te=np.flatnonzero(d['folds']==fold);z,f=nuisance(d,tr,platform)
        x=f['MC'];y=f['S'];bx=np.linalg.lstsq(z[tr],x[tr],rcond=None)[0];by=np.linalg.lstsq(z[tr],y[tr],rcond=None)[0]
        xr=x-z@bx;yr=y-z@by;sx=xr[tr].std(0);sy=yr[tr].std(0);assert np.all(sx>1e-10) and np.all(sy>1e-10)
        molecular[te]=xr[te]/sx;static[te]=yr[te]/sy;raw_molecular[te]=x[te];raw_static[te]=y[te]
        fold_corr.append(correlation(molecular[te],static[te]))
        np.savez_compressed(out/'preprocessing'/f'alignment_fold{fold}.npz',tr=tr,te=te,nuisance_coefficients_molecular=bx,nuisance_coefficients_concepts=by,
            molecular_train_residual_sd=sx,concept_train_residual_sd=sy,nuisance_rank=np.linalg.matrix_rank(z[tr]))
        fits.append({'fold':fold,'n_train':len(tr),'n_test':len(te),'nuisance_columns':z.shape[1],'nuisance_rank':int(np.linalg.matrix_rank(z[tr])),
            'fit_patient_indices_sha256':ahash(tr),'test_patient_indices_sha256':ahash(te),'DSS_columns_accessed':False})
        _,st=gene_z(d['cna_genes'],tr);_,sets=group_scores(d['cna_genes'],d['cna_gene_symbols'],pm['groups'],st['usable'])
        for k,genes in enumerate(sets):
            a=d['cna_genes'][te][:,genes];med=np.nanmedian(d['cna_genes'][tr][:,genes],0);a=np.where(np.isfinite(a),a,med)
            signed=np.abs(a.mean(1));absolute=np.abs(a).mean(1);ratio=1-signed/np.maximum(absolute,1e-12)
            for j in range(len(te)):cancel.append({'state':STATES[k],'mean_absolute_dosage':absolute[j],'cancellation_fraction':ratio[j]})
        print('B_NUISANCE_FOLD_DONE',fold,flush=True)
    corr=correlation(molecular,static);uncorr=correlation(raw_molecular,raw_static);fold_corr=np.asarray(fold_corr)
    keys=np.asarray([f'{f}|{c}|{plat}' for f,c,plat in zip(d['folds'],d['cancer_type'],platform)]);groups=[np.flatnonzero(keys==key) for key in np.unique(keys)]
    rng=np.random.default_rng(88411);permcorr=np.empty((1999,16,32));global_stat=float(np.square(corr[:8][mask]).mean())
    for b in range(1999):
        ix=np.arange(512)
        for g in groups:ix[g]=rng.permutation(g)
        permcorr[b]=correlation(molecular[ix],static)
    global_null=np.square(permcorr[:,:8][:,mask]).mean(1);pair_p=(1+int((global_null>=global_stat).sum()))/2000
    wrong=[]
    for b in range(99):
        ix=derangement_groups(groups,rng);wrong.append(float(np.square(correlation(molecular[ix],static)[:8][mask]).mean()))
    maps=[];seen={ahash(mask)};mapping_stats=[]
    while len(maps)<999:
        sh=degree_map(mask,rng);sha=ahash(sh)
        if sha in seen:continue
        seen.add(sha);maps.append(sh);mapping_stats.append(float(np.square(corr[:8][sh]).mean()))
    map_p=(1+int((np.asarray(mapping_stats)>=global_stat).sum()))/1000
    primary_max=np.max(np.abs(permcorr[:,:8][:,mask]),axis=1);rna_max=np.max(np.abs(permcorr[:,:8]),axis=(1,2));cna_max=np.max(np.abs(permcorr[:,8:]),axis=(1,2))
    concept=json.loads((REPO/'configs/phase6a_titan_crc_morphology_concepts.json').read_text())['concepts'];rows=[]
    for j in range(16):
        for k in range(32):
            r=float(corr[j,k]);familymax=rna_max if j<8 else cna_max
            null=permcorr[:,j,k];stable=int(np.sum(np.sign(fold_corr[:,j,k])==np.sign(r)))
            row={'modality':'RNA' if j<8 else 'CNA','state':STATES[j%8],'concept_id':concept[k]['id'],'concept_name':concept[k]['name'],
                'prior_edge':bool(j<8 and mask[j,k]),'unadjusted_oof_correlation':float(uncorr[j,k]),'adjusted_oof_correlation':r,
                'fold_correlations':json.dumps(fold_corr[:,j,k].tolist()),'matching_fold_signs':stable,
                'blocked_permutation_p':(1+int((np.abs(null)>=abs(r)).sum()))/2000,
                'exploratory_256_maxT_p':(1+int((familymax>=abs(r)).sum()))/2000}
            row['primary_15_maxT_p']=(1+int((primary_max>=abs(r)).sum()))/2000 if row['prior_edge'] else np.nan
            row['stable_prior_edge']=row['prior_edge'] and row['primary_15_maxT_p']<=.05 and stable>=4
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/'B_alignment_correlations.csv',index=False)
    pd.DataFrame(fits).to_csv(out/'B_nuisance_fit_audit.csv',index=False)
    pd.DataFrame({'patient_pairing_null':global_null}).to_csv(out/'B_pairing_null_benchmark.csv',index=False)
    pd.DataFrame({'wrong_patient_prior_mean_r_squared':wrong}).to_csv(out/'B_wrong_patient_benchmark.csv',index=False)
    pd.DataFrame({'degree_matched_mapping_prior_mean_r_squared':mapping_stats}).to_csv(out/'B_mapping_benchmark.csv',index=False)
    c=pd.DataFrame(cancel);summaries=[]
    for state,a in c.groupby('state',sort=False):
        usable=a.mean_absolute_dosage>.1
        summaries.append({'state':state,'mean_absolute_dosage_median':float(a.mean_absolute_dosage.median()),
            'signed_cancellation_median':float(a.cancellation_fraction.median()),'patients_mean_abs_gt_point1':int(usable.sum()),
            'fraction_cancellation_ge50percent_among_nontrivial':float((a.loc[usable,'cancellation_fraction']>=.5).mean())})
    pd.DataFrame(summaries).to_csv(out/'B_CNA_cancellation_descriptive.csv',index=False)
    np.savez_compressed(out/'private_alignment_cache.npz',molecular_residual=molecular,concept_residual=static,raw_molecular=raw_molecular,raw_static=raw_static,
        fold_correlations=fold_corr,correlation=corr,permutation_correlations=permcorr,random_mapping_masks=np.asarray(maps),
        platform=platform,block_keys=keys,patient_ids=d['patient_ids'])
    stable_edges=[r for r in rows if r['stable_prior_edge']];pairgate=pair_p<=.025 and len(stable_edges)>0;mapgate=map_p<=.025
    singleton=sum(len(g)==1 for g in groups)
    decision={'status':'PASS','patients':512,'DSS_used':False,'prior_edges':15,'covered_concepts':12,'global_prior_mean_r_squared':global_stat,
        'patient_pairing_benchmark_p':pair_p,'degree_matched_mapping_benchmark_p':map_p,'wrong_patient_mean':float(np.mean(wrong)),
        'wrong_patient_quantile95':float(np.quantile(wrong,.95)),'mapping_null_quantile95':float(np.quantile(mapping_stats,.95)),
        'stable_primary_prior_edges':[{k:r[k] for k in ['state','concept_id','concept_name','adjusted_oof_correlation','primary_15_maxT_p','matching_fold_signs']} for r in stable_edges],
        'patient_pairing_gate':bool(pairgate),'mapping_specificity_gate':bool(mapgate),'exact_prior_supported':bool(pairgate and mapgate),
        'exchangeability_blocks':len(groups),'singleton_patients_not_mismatched':singleton,'mismatched_patients':512-singleton,
        'confidence':'conditional internal proxy alignment; no causal direction or independent slide annotation validation'}
    write_json(out/'B_decision.json',decision);morphology_audit(out,d,platform);locked(out)
    write_json(out/'B_DONE.json',{'status':'PASS','protocol_sha256':m['protocol_sha256']});print('B_COMPLETE',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['prepare','worker','aggregate_a','alignment'],required=True);a.add_argument('--out',type=Path,default=DEFAULT);a.add_argument('--worker-id',type=int,default=0);args=a.parse_args()
    globals()[args.mode](args.out,args.worker_id) if args.mode=='worker' else globals()[args.mode](args.out)
