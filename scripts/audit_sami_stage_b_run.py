"""Read-only independent NumPy replay of all locked SAMI Stage B models.

No optimizer or model fit is called. Model weights, identities and predictions
remain private; the output contains only aggregate discrepancies and counts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['MEAN','COMPOSITION','ADDITIVE','TISSUE_COND','RANDOM_GATE','WRONG_DONOR','SPATIAL_SHUFFLE']
DIMS = dict(COMPOSITION=6,ADDITIVE=70,TISSUE_COND=166,RANDOM_GATE=166,WRONG_DONOR=166,SPATIAL_SHUFFLE=166)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def index_sha(indices):
    return hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()


def arrays(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}


def weights(donor,section,tissue=None):
    groups={}
    for i,(d,s) in enumerate(zip(donor,section)):
        c=None if tissue is None else int(tissue[i])
        groups.setdefault(str(d),{}).setdefault(c,{}).setdefault(str(s),[]).append(i)
    result=np.zeros(len(donor),np.float64)
    for classes in groups.values():
        for slides in classes.values():
            for positions in slides.values():
                result[positions]=1./(len(groups)*len(classes)*len(slides)*len(positions))
    require(abs(result.sum()-1.)<1e-12,'Independent training weights do not sum to one')
    return result


def mean_scale(x,weight):
    x=np.asarray(x,np.float64)
    mean=np.sum(weight[:,None]*x,axis=0)
    scale=np.sqrt(np.sum(weight[:,None]*(x-mean)**2,axis=0))
    scale[scale<=1e-12]=1.
    return mean,scale


def error(a,b):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid comparison shape/value')
    return float(np.max(np.abs(a-b),initial=0.))


def close(a,b,message,tolerance=1e-9):
    value=error(a,b)
    require(value<=tolerance,message)
    return value


def compare_tree(actual,expected,path='metric',tolerance=1e-9):
    maximum=0.
    if isinstance(actual,dict):
        require(isinstance(expected,dict),path+': expected object')
        for key,value in actual.items():
            require(key in expected,path+': missing key '+key)
            maximum=max(maximum,compare_tree(value,expected[key],path+'.'+key,tolerance))
    elif isinstance(actual,(list,tuple)):
        require(isinstance(expected,(list,tuple)) and len(actual)==len(expected),path+': sequence mismatch')
        for i,(a,b) in enumerate(zip(actual,expected)):
            maximum=max(maximum,compare_tree(a,b,path+f'[{i}]',tolerance))
    elif actual is None or isinstance(actual,(str,bool)):
        require(actual==expected,path+': metadata mismatch')
    else:
        maximum=close(actual,expected,path+': numeric mismatch',tolerance)
    return maximum


def probabilities(x,model):
    logits=((np.asarray(x,np.float64)-model['mean'])/model['scale'])@model['W']+model['b']
    logits-=logits.max(axis=1,keepdims=True)
    p=np.exp(logits)
    return p/p.sum(axis=1,keepdims=True)


def teacher_objective_gradient(model,x,y,w):
    z=(np.asarray(x,np.float64)-model['mean'])/model['scale']
    logits=z@model['W']+model['b']
    logits-=logits.max(1,keepdims=True)
    ex=np.exp(logits)
    p=ex/ex.sum(1,keepdims=True)
    logp=logits-np.log(ex.sum(1,keepdims=True))
    loss=float(-np.dot(w,logp[np.arange(len(y)),y])+.5*np.square(model['W']).sum())
    residual=p.copy();residual[np.arange(len(y)),y]-=1.;residual*=w[:,None]
    g=np.r_[(z.T@residual+model['W']).ravel(),residual.sum(0)]
    return loss,float(np.max(np.abs(g)))


def huber_objective_gradient(model,design,true_y,w,target_map):
    z=(design-model['xmean'])/model['xscale']
    targets=((true_y-model['ymean'])/model['yscale'])[target_map]
    residual=z@model['W']+model['b']-targets
    absr=np.abs(residual)
    data=float(np.sum(w[:,None]*np.where(absr<=1.,.5*residual**2,absr-.5))/4.)
    reg=.05*float(np.square(model['W']).sum())
    derivative=np.clip(residual,-1.,1.)*w[:,None]/4.
    g=np.r_[(z.T@derivative+.1*model['W']).ravel(),derivative.sum(0)]
    return data+reg,data,reg,float(np.max(np.abs(g)))


def design_matrix(local,neighbor,p_local,p_neighbor,arm):
    if arm=='COMPOSITION':
        return np.c_[p_local,p_neighbor]
    base=np.c_[local,neighbor,p_local,p_neighbor]
    if arm=='ADDITIVE':
        return base
    if arm=='RANDOM_GATE':
        projection=np.random.default_rng(7032002).normal(size=(32,3))
        logits=neighbor@projection/np.sqrt(32)
        e=np.exp(logits-logits.max(1,keepdims=True));gate=e/e.sum(1,keepdims=True)
    else:
        gate=p_neighbor
    return np.c_[base,*[local*gate[:,c:c+1] for c in range(3)]]


def gate_diagnostics(gate,weight):
    """Descriptive probability geometry; no fit or outcome input."""
    gate=np.asarray(gate,float);weight=np.asarray(weight,float)
    require(gate.ndim==2 and gate.shape[1]==3 and len(gate)==len(weight)
            and np.isfinite(gate).all() and np.all(gate>=0),'Invalid gate diagnostic input')
    close(gate.sum(1),np.ones(len(gate)),'Gate rows do not sum to one')
    mean=weight@gate;centered=gate-mean
    covariance=(centered*weight[:,None]).T@centered
    eigenvalues=np.maximum(np.linalg.eigvalsh(covariance),0.)
    positive=eigenvalues[eigenvalues>1e-12]
    spectrum=positive/positive.sum() if len(positive) else positive
    rank=float(np.exp(-np.sum(spectrum*np.log(spectrum)))) if len(spectrum) else 0.
    return dict(mean_entropy_nats=float(weight@(-np.sum(gate*np.log(np.maximum(gate,1e-300)),axis=1))),
                mean_max_probability=float(weight@gate.max(1)),
                mean_probability=mean.tolist(),
                std_across_spots=np.sqrt(np.maximum(np.diag(covariance),0.)).tolist(),
                centered_covariance_effective_rank=rank,
                effective_rank_rule='exp_entropy_positive_covariance_eigenvalues_gt_1e-12;constant_zero',
                weighting='donor_section_spot_equal',calibrated_source_probability_claim=False)


def target_mapping(donor,section,tissue,arm):
    index=np.arange(len(donor),dtype=np.int64)
    if arm=='WRONG_DONOR':
        rng=np.random.default_rng(7032003);donors=np.unique(donor)
        for k,d in enumerate(donors):
            source=donors[(k+1)%len(donors)]
            for c in np.unique(tissue[donor==d]):
                recipient=np.flatnonzero((donor==d)&(tissue==c))
                candidate=np.flatnonzero((donor==source)&(tissue==c))
                require(len(candidate)>0,'Wrong-donor target source class absent')
                shuffled=rng.permutation(candidate)
                index[recipient]=shuffled[np.arange(len(recipient))%len(shuffled)]
        require(np.all(donor[index]!=donor) and np.array_equal(tissue[index],tissue),'Invalid wrong-donor control')
    elif arm=='SPATIAL_SHUFFLE':
        rng=np.random.default_rng(7032004)
        for s in np.unique(section):
            for c in np.unique(tissue[section==s]):
                order=rng.permutation(np.flatnonzero((section==s)&(tissue==c)))
                index[order]=np.r_[order[-1],order[:-1]]
        require(len(np.unique(index))==len(index) and np.array_equal(section[index],section)
                and np.array_equal(tissue[index],tissue),'Invalid spatial target control')
    return index


def expected_teacher_rows(donor,heldout_donor,recipient_donor):
    excluded={heldout_donor,recipient_donor}
    return np.flatnonzero(~np.isin(donor,list(excluded)))


def _available(v):
    v=[x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _complete(v):
    return float(np.mean(v)) if v and all(x is not None for x in v) else None


def _correlations(truth,prediction,donor,section,tissue,stratified):
    coverage=[dict(candidate_groups=0,eligible_groups=0,below_minimum_n_groups=0,
                   constant_truth_groups=0,constant_prediction_groups=0) for _ in range(4)]
    values=[]
    for d in np.unique(donor):
        drows=np.flatnonzero(donor==d);row=[]
        for p in range(4):
            cv=coverage[p];classes=np.unique(tissue[drows]) if stratified else [None]
            class_values=[]
            for c in classes:
                indices=drows if c is None else drows[tissue[drows]==c]
                section_values=[]
                for s in np.unique(section[indices]):
                    selected=indices[section[indices]==s];y=truth[selected,p];hat=prediction[selected,p]
                    cv['candidate_groups']+=1
                    if len(selected)<20:
                        cv['below_minimum_n_groups']+=1;section_values.append(None)
                    elif np.ptp(y)==0:
                        cv['constant_truth_groups']+=1;section_values.append(None)
                    else:
                        cv['eligible_groups']+=1
                        if np.ptp(hat)==0:
                            cv['constant_prediction_groups']+=1;section_values.append(0.)
                        else:
                            r1,r2=rankdata(y),rankdata(hat)
                            r1-=r1.mean();r2-=r2.mean()
                            section_values.append(float(np.dot(r1,r2)/np.sqrt(np.dot(r1,r1)*np.dot(r2,r2))))
                class_values.append(_available(section_values))
            row.append(_available(class_values))
        values.append(row)
    donor_values=[_complete(row) for row in values]
    return dict(aggregate=_complete(donor_values),available_donor_description=_available(donor_values),
                complete_donors=sum(v is not None for v in donor_values),
                per_program=[_complete([row[k] for row in values]) for k in range(4)],
                complete_donors_per_program=[sum(row[k] is not None for row in values) for k in range(4)],coverage=coverage)


def program_metrics(y,pred,donor,section,tissue,ymean,yscale,library_size=None,n_detected_genes=None):
    residual=(pred-ymean)/yscale-(y-ymean)/yscale
    w=weights(donor,section)
    hp=np.sum(w[:,None]*np.where(abs(residual)<=1.,.5*residual**2,abs(residual)-.5),axis=0)
    mp=np.sum(w[:,None]*residual**2,axis=0)
    wt=_correlations(y,pred,donor,section,tissue,True)
    ws=_correlations(y,pred,donor,section,tissue,False)
    result=dict(huber=float(hp.mean()),mse=float(mp.mean()),within_tissue_spearman=wt['aggregate'],
                within_section_spearman=ws['aggregate'],n_spots=len(y),n_donors=len(np.unique(donor)),
                n_sections=sum(len(np.unique(section[donor==d])) for d in np.unique(donor)),minimum_group_n=20,
                prediction_units='original_program_score',
                incomplete_primary_policy='null_if_any_donor_lacks_any_of_four_program_correlations',
                within_tissue_complete_donors=wt['complete_donors'],within_section_complete_donors=ws['complete_donors'],
                available_donor_description=dict(within_tissue_spearman=wt['available_donor_description'],
                                                 within_section_spearman=ws['available_donor_description']),per_program=[])
    for k in range(4):
        result['per_program'].append(dict(program_index=k,huber=float(hp[k]),mse=float(mp[k]),
            within_tissue_spearman=wt['per_program'][k],within_section_spearman=ws['per_program'][k],
            eligible_groups=wt['coverage'][k]['eligible_groups'],within_tissue_coverage=wt['coverage'][k],
            within_section_coverage=ws['coverage'][k],within_tissue_complete_donors=wt['complete_donors_per_program'][k],
            within_section_complete_donors=ws['complete_donors_per_program'][k]))
    qc={}
    for name,values in [('library_size',library_size),('n_detected_genes',n_detected_genes)]:
        if values is not None:
            nuisance=np.repeat(np.log1p(values)[:,None],4,axis=1)
            qc[name]=dict(transform='log1p',metric='within_section_Spearman_descriptive_only',
                          truth=_correlations(nuisance,y,donor,section,tissue,False),
                          prediction=_correlations(nuisance,pred,donor,section,tissue,False))
    result['qc_associations']=qc
    return result


def check_success(fit,objective,gradient):
    require(fit['success'] is True and np.isfinite(gradient) and gradient<=1e-5,'Model failed independent numerical convergence')
    require(fit['accepted_gradient_inf']==1e-5,'Optimizer acceptance threshold changed')
    close(gradient,fit['gradient_inf'],'Independent final gradient differs',1e-8)
    close(objective,fit['objective'],'Independent final objective differs',1e-8)


def checked_receipt(directory,lineage,artifacts):
    receipt=read(directory/'COMPLETE.json')
    require(receipt['status']=='COMPLETE' and receipt['lineage']==lineage,'Completed model lineage mismatch')
    for name in artifacts:
        require(sha(directory/f'{name}.npz')==receipt[name+'_sha256'],'Checkpoint artifact digest mismatch')
    return receipt


def audit(args):
    prepared,run=Path(args.prepared),Path(args.run)
    manifest=read(prepared/'manifest.json')
    config_path=ROOT/'configs/sami_surv_stage_b_v1.json'
    program_path=ROOT/'configs/sami_surv_stage_b_programs_v1.json'
    config=read(config_path);plan=read(run/'PLAN.json');lineage=plan['lineage']
    require(config['status']=='LOCKED_BEFORE_MODEL_FIT','Stage B configuration is not locked')
    require(manifest['status']=='PASS_PREPARED_SIGNED_PROGRAMS','Stage B input qualification failed')
    require(lineage['config_sha256']==manifest['config_sha256']==sha(config_path),'Config digest mismatch')
    require(lineage['program_sha256']==manifest['program_definition_sha256']==sha(program_path)
            ==config['data_contract']['program_definition_sha256'],'Program digest mismatch')
    require(lineage['prepared_sha256']==manifest['prepared_data_sha256']==sha(prepared/'prepared.npz'),'Prepared data digest mismatch')
    require(lineage['manifest_sha256']==sha(prepared/'manifest.json'),'Prepared manifest digest mismatch')
    require(manifest['original_ST_test_read'] is False and manifest['TCGA_DSS_read'] is False,'Forbidden prepared inputs declared')
    source_files=['scripts/run_sami_stage_b.py','src/sami_surv/morphology.py',
                  'src/sami_surv/spatial_programs.py','src/sami_surv/program_metrics.py']
    require(set(lineage['source_sha256'])==set(source_files),'Unexpected scientific source closure')
    require(all(lineage['source_sha256'][p]==sha(ROOT/p) for p in source_files),'Scientific source bytes changed')
    require(all(isinstance(lineage['runtime'][k],str) and lineage['runtime'][k]
                for k in ['python','numpy','scipy','sklearn']),'Missing runtime version evidence')
    data=arrays(prepared/'prepared.npz')
    expected_fields={'X','neighbor_X','tissue_y','donor','section','barcodes','coords_um',
                     'program_y','library_size','n_detected_genes'}
    require(set(data)==expected_fields,'Unexpected prepared arrays')
    n=len(data['X']);donor=data['donor'];section=data['section'];tissue=data['tissue_y'];y=data['program_y']
    require(n==13011 and data['X'].shape==(n,1536) and data['neighbor_X'].shape==(n,1536)
            and y.shape==(n,4),'Locked input dimensions changed')
    donors=np.unique(donor).tolist()
    require(len(donors)==6 and len(np.unique(section))==12,'Development donor/section counts changed')
    require(all(len(v)==n for v in data.values()),'Prepared row alignment differs')
    for key in ['X','neighbor_X','program_y','coords_um','library_size','n_detected_genes']:
        require(np.isfinite(data[key]).all(),'Nonfinite prepared input')
    require(plan['optimized_models']==57 and plan['hyperparameter_selection'] is False,'Model budget or selection contract changed')
    expected_subsets=[tuple(subset) for size in (4,5) for subset in itertools.combinations(donors,size)]
    require(len(plan['teachers'])==21 and len(plan['outer'])==6,'Incomplete teacher/outer plan')
    require(sorted(p.name for p in (run/'teachers').iterdir() if p.is_dir())==[f't_{i:02d}' for i in range(21)],
            'Teacher directories differ from exact locked set')
    models={};teacher_rows={};teacher_receipts=[];teacher_diagnostics=[]
    maxima=defaultdict(float)
    for ti,(spec,subset) in enumerate(zip(plan['teachers'],expected_subsets)):
        require(spec['teacher_index']==ti and spec['train_donors']==list(subset),'Teacher subset plan differs')
        tr=np.flatnonzero(np.isin(donor,subset))
        directory=run/'teachers'/f't_{ti:02d}'
        receipt=checked_receipt(directory,lineage,['model','indices'])
        require(receipt['teacher_index']==ti and receipt['train_index_sha256']==spec['train_index_sha256']
                ==index_sha(tr),'Teacher row identity digest differs')
        idx=arrays(directory/'indices.npz')
        require(set(idx)=={'train_index'} and np.array_equal(idx['train_index'],tr),'Teacher fit includes wrong rows')
        model=arrays(directory/'model.npz')
        require(set(model)=={'W','b','mean','scale'} and model['W'].shape==(1536,3)
                and model['b'].shape==(3,),'Teacher model schema differs')
        natural_w=weights(donor[tr],section[tr]);m,s=mean_scale(data['X'][tr],natural_w)
        maxima['teacher_train_scaler']=max(maxima['teacher_train_scaler'],
            close(m,model['mean'],'Teacher mean is not training-only'),close(s,model['scale'],'Teacher scale is not training-only'))
        balanced_w=weights(donor[tr],section[tr],tissue[tr])
        objective,gradient=teacher_objective_gradient(model,data['X'][tr],tissue[tr],balanced_w)
        check_success(receipt['fit_audit'],objective,gradient)
        require(receipt['fit_audit']['C']==1. and receipt['fit_audit']['n_train_donors']==len(subset),
                'Teacher C or training donor count differs')
        models[ti]=model;teacher_rows[ti]=tr;teacher_receipts.append(receipt)
        teacher_diagnostics.append(dict(teacher_index=ti,training_donors=len(subset),
                                       independently_recomputed_gradient_inf=gradient))
    require(sorted(p.name for p in (run/'outer').iterdir() if p.is_dir())==[f'o_{i:02d}' for i in range(6)],
            'Outer directories differ from exact locked set')
    predictions=np.full((7,n,4),np.nan);outer_assignment=np.full(n,-1,np.int64)
    fold_metrics=[];regressor_diagnostics=[];application_counts={4:0,5:0};pca_diagnostics=[];gate_geometry=[]
    for oi,held in enumerate(donors):
        tr=np.flatnonzero(donor!=held);te=np.flatnonzero(donor==held)
        spec=plan['outer'][oi]
        require(spec['outer_index']==oi and spec['heldout_donor']==held
                and spec['train_index_sha256']==index_sha(tr) and spec['test_index_sha256']==index_sha(te),'Outer split plan differs')
        require(not set(donor[tr])&set(donor[te]),'Outer donor leakage')
        directory=run/'outer'/f'o_{oi:02d}'
        outer_done=checked_receipt(directory,lineage,[])
        require(outer_done['outer_index']==oi and set(outer_done['arms'])==set(ARMS),'Outer completion schema differs')
        context_receipt=read(directory/'context_receipt.json')
        require(context_receipt['lineage']==lineage and context_receipt['outer_index']==oi,'Context lineage mismatch')
        for name in ['context','pca']:
            require(sha(directory/f'{name}.npz')==context_receipt[name+'_sha256'],'Context/PCA digest mismatch')
        context=arrays(directory/'context.npz');pca=arrays(directory/'pca.npz')
        require(set(context)=={'train_index','test_index','local_pc','neighbor_pc','p_local','p_neighbor','teacher_for_row'},
                'Unexpected context arrays')
        require(np.array_equal(context['train_index'],tr) and np.array_equal(context['test_index'],te),'Context training/test rows differ')
        require(context['teacher_for_row'].shape==(n,) and context['teacher_for_row'].dtype.kind in 'iu','Invalid teacher assignment vector')
        for recipient in donors:
            rows=np.flatnonzero(donor==recipient)
            ids=np.unique(context['teacher_for_row'][rows])
            require(len(ids)==1 and int(ids[0]) in models,'Rows assigned to invalid teacher')
            ti=int(ids[0]);expected=expected_teacher_rows(donor,held,recipient)
            require(np.array_equal(teacher_rows[ti],expected),'Teacher crossfit includes recipient or outer held-out donor')
            k=len(np.unique(donor[expected]))
            require(k==(5 if recipient==held else 4),'Teacher must train on exactly four inner or five outer donors')
            application_counts[k]+=1
            for xkey,pkey in [('X','p_local'),('neighbor_X','p_neighbor')]:
                prob=probabilities(data[xkey][rows],models[ti])
                maxima['teacher_probability_replay']=max(maxima['teacher_probability_replay'],
                    close(prob,context[pkey][rows],'Independent tissue probability replay differs'))
        require(set(pca)=={'mean','scale','components','singular_values','component_scale'},'Unexpected PCA schema')
        w=weights(donor[tr],section[tr]);m,s=mean_scale(data['X'][tr],w)
        maxima['PCA_train_scaler']=max(maxima['PCA_train_scaler'],
            close(m,pca['mean'],'PCA mean not training-only'),close(s,pca['scale'],'PCA scale not training-only'))
        require(pca['components'].shape==(32,1536) and pca['component_scale'].shape==(32,)
                and np.all(pca['component_scale']>1e-12),'PCA dimension or whitening scale invalid')
        close(pca['components']@pca['components'].T,np.eye(32),'PCA directions not orthogonal')
        pivot=np.argmax(abs(pca['components']),axis=1)
        require(np.all(pca['components'][np.arange(32),pivot]>0),'PCA direction sign contract differs')
        projected=((data['X'][tr].astype(float)-pca['mean'])/pca['scale'])@pca['components'].T
        score_mean=w@projected
        actual_scale=np.sqrt(np.sum(w[:,None]*(projected-score_mean)**2,axis=0))
        maxima['PCA_whitening']=max(maxima['PCA_whitening'],close(actual_scale,pca['component_scale'],'PCA scale uses non-training evidence'))
        for xkey,pckey in [('X','local_pc'),('neighbor_X','neighbor_pc')]:
            replay=((data[xkey].astype(float)-pca['mean'])/pca['scale'])@pca['components'].T/pca['component_scale']
            maxima['PCA_transform_replay']=max(maxima['PCA_transform_replay'],close(replay,context[pckey],'PCA transform replay differs'))
        transformed=context['local_pc'][tr]
        unit_variance=np.sum(w[:,None]*(transformed-w@transformed)**2,axis=0)
        close(unit_variance,np.ones(32),'Train PCA component variance is not one')
        pf=context_receipt['pca_fit_audit']
        require(pf['training_rows']==len(tr) and pf['training_donors']==5 and pf['components']==32
                and pf['seed']==7032001 and pf['n_iter']==7 and pf['neighbor_features_used_to_fit_PCA'] is False,
                'PCA training qualification receipt differs')
        pca_diagnostics.append(dict(outer_index=oi,maximum_unit_variance_error=error(unit_variance,np.ones(32)),
                                    direction_refit=False,raw_train_scaler_and_projected_train_variance_recomputed=True))
        random_projection=np.random.default_rng(7032002).normal(size=(32,3))
        random_logits=context['neighbor_pc']@random_projection/np.sqrt(32)
        random_exponent=np.exp(random_logits-random_logits.max(1,keepdims=True))
        random_probability=random_exponent/random_exponent.sum(1,keepdims=True)
        gate_geometry.append(dict(outer_index=oi,**{
            name:{'tissue':gate_diagnostics(context['p_neighbor'][rows],weights(donor[rows],section[rows])),
                  'random':gate_diagnostics(random_probability[rows],weights(donor[rows],section[rows]))}
            for name,rows in [('train',tr),('heldout',te)]}))
        ymean,yscale=mean_scale(y[tr],w)
        require(sorted(p.name for p in (directory/'arms').iterdir() if p.is_dir())==sorted(ARMS),'Arm directories differ from fixed matrix')
        one_fold={}
        for ai,arm in enumerate(ARMS):
            cell=directory/'arms'/arm
            artifacts=['predictions'] if arm=='MEAN' else ['model','predictions','target_map']
            receipt=checked_receipt(cell,lineage,artifacts)
            require(receipt['arm']==arm and receipt['outer_index']==oi,'Arm completion identity mismatch')
            saved=arrays(cell/'predictions.npz')
            require(set(saved)=={'row_index','prediction'} and np.array_equal(saved['row_index'],te),'Prediction row order mismatch')
            if arm=='MEAN':
                require(receipt['no_fit'] is True,'Mean baseline was optimized')
                maxima['target_train_scaler']=max(maxima['target_train_scaler'],
                    close(ymean,receipt['ymean'],'Mean baseline target mean differs'),
                    close(yscale,receipt['yscale'],'Mean baseline target scale differs'))
                ym,ys=np.asarray(receipt['ymean']),np.asarray(receipt['yscale'])
                pred=np.repeat(ym[None,:],len(te),axis=0)
            else:
                design=design_matrix(context['local_pc'],context['neighbor_pc'],context['p_local'],context['p_neighbor'],arm)
                require(design.shape==(n,DIMS[arm]),'Independent design dimension differs')
                model=arrays(cell/'model.npz')
                require(set(model)=={'W','b','xmean','xscale','ymean','yscale'} and model['W'].shape==(DIMS[arm],4)
                        and model['b'].shape==(4,),'Regressor schema or parameter budget differs')
                xm,xs=mean_scale(design[tr],w)
                maxima['design_train_scaler']=max(maxima['design_train_scaler'],
                    close(xm,model['xmean'],'Regressor design mean differs'),close(xs,model['xscale'],'Regressor design scale differs'))
                maxima['target_train_scaler']=max(maxima['target_train_scaler'],
                    close(ymean,model['ymean'],'Regressor target mean uses wrong rows'),close(yscale,model['yscale'],'Regressor target scale uses wrong rows'))
                mapping=target_mapping(donor[tr],section[tr],tissue[tr],arm)
                stored_map=arrays(cell/'target_map.npz')
                require(set(stored_map)=={'index'} and np.array_equal(mapping,stored_map['index']),
                        'Target remapping differs from fixed train-only rule')
                objective,data_loss,regularizer,gradient=huber_objective_gradient(model,design[tr],y[tr],w,mapping)
                check_success(receipt['fit_audit'],objective,gradient)
                close(data_loss,receipt['fit_audit']['data_loss'],'Huber data loss differs',1e-8)
                close(regularizer,receipt['fit_audit']['regularizer'],'Huber regularizer differs',1e-8)
                require(receipt['fit_audit']['ridge']==.1 and receipt['fit_audit']['delta']==1.
                        and receipt['fit_audit']['targets']==4 and receipt['fit_audit']['intercept_penalized'] is False,
                        'Regressor fixed objective differs')
                require(receipt['fit_audit']['allocated_parameters']==4*(DIMS[arm]+1)
                        and receipt['fit_audit']['regularized_parameters']==4*DIMS[arm], 'Parameter count differs')
                reuse=np.bincount(mapping,minlength=len(tr));values,counts=np.unique(reuse,return_counts=True)
                require(receipt['target_source_reuse_frequency_histogram']=={str(v):int(c) for v,c in zip(values,counts)},
                        'Target source reuse histogram differs')
                rank=int(np.linalg.matrix_rank((design[tr]-model['xmean'])/model['xscale']))
                require(rank==receipt['standardized_design_rank'],'Standardized design rank differs')
                ym,ys=model['ymean'],model['yscale']
                pred=(((design[te]-model['xmean'])/model['xscale'])@model['W']+model['b'])*ys+ym
                regressor_diagnostics.append(dict(outer_index=oi,arm=arm,
                    independently_recomputed_gradient_inf=gradient,standardized_design_rank=rank))
            maxima['program_prediction_replay']=max(maxima['program_prediction_replay'],close(pred,saved['prediction'],'Regressor prediction replay differs'))
            computed=program_metrics(y[te],pred,donor[te],section[te],tissue[te],ym,ys,
                                     data['library_size'][te],data['n_detected_genes'][te])
            maxima['metrics_recompute']=max(maxima['metrics_recompute'],compare_tree(computed,receipt['metrics']))
            maxima['outer_summary_metrics']=max(maxima['outer_summary_metrics'],compare_tree(computed,outer_done['arms'][arm]['metrics']))
            one_fold[arm]=computed;predictions[ai,te]=saved['prediction']
        fold_metrics.append(one_fold);outer_assignment[te]=oi
    require(len(teacher_diagnostics)==21 and len(regressor_diagnostics)==36 and application_counts=={4:30,5:6},
            'Incomplete optimized model count or crossfit applications')
    oof=arrays(run/'OOF_PRIVATE.npz');public=read(run/'RESULTS.json')
    require(set(oof)=={'predictions','row_index','outer_fold'} and np.array_equal(oof['row_index'],np.arange(n))
            and np.array_equal(oof['outer_fold'],outer_assignment),'OOF partition coverage differs')
    close(predictions,oof['predictions'],'OOF predictions differ from saved held-out cells',0.)
    require(public['lineage']==lineage and public['arm_order']==ARMS and public['optimized_teacher_models']==21
            and public['optimized_program_models']==36 and public['OOF_private_sha256']==sha(run/'OOF_PRIVATE.npz'),
            'Final public result identity or budget differs')
    require(public['original_ST_test_read'] is False and public['TCGA_DSS_read'] is False and public['source_truth_claim'] is False,
            'Public scope declarations differ')
    metric_names=['huber','mse','within_tissue_spearman','within_section_spearman']
    aggregate={}
    for arm in ARMS:
        rows=[f[arm] for f in fold_metrics]
        aggregate[arm]={key:_complete([row[key] for row in rows]) for key in metric_names}
        aggregate[arm]['per_program']=[dict(program_index=k,**{
            key:_complete([row['per_program'][k][key] for row in rows]) for key in metric_names}) for k in range(4)]
    maxima['aggregate_metrics']=compare_tree(aggregate,public['aggregate'])
    for oi,rows in enumerate(fold_metrics):
        require(public['per_fold'][oi]['outer_index']==oi,'Public fold ordering differs')
        for arm in ARMS:
            maxima['public_fold_metrics']=max(maxima['public_fold_metrics'],compare_tree(rows[arm],public['per_fold'][oi]['arms'][arm]['metrics']))
    for ti,receipt in enumerate(teacher_receipts):
        compare_tree(dict(teacher_index=ti,**receipt['fit_audit']),public['teacher_fit_audits'][ti])
    draw=np.random.default_rng(7032005).integers(0,6,size=(20000,6))
    for control in ['COMPOSITION','ADDITIVE','RANDOM_GATE','WRONG_DONOR','SPATIAL_SHUFFLE']:
        for metric_name in metric_names:
            candidate=[f['TISSUE_COND'][metric_name] for f in fold_metrics]
            comparator=[f[control][metric_name] for f in fold_metrics]
            if any(v is None for v in candidate+comparator):
                expected=dict(status='UNDEFINED_MISSING_PROGRAM_OR_DONOR',n_donors=6)
            else:
                delta=np.asarray(comparator)-np.asarray(candidate) if metric_name in ['huber','mse'] else np.asarray(candidate)-np.asarray(comparator)
                ci=np.quantile(delta[draw].mean(1),[.025,.975])
                expected=dict(status='COMPLETE',mean_improvement=float(delta.mean()),ci95=ci.tolist(),positive_donors=int((delta>0).sum()),
                              per_fold_improvement=delta.tolist(),bootstrap_repeats=20000,bootstrap_unit='donor',
                              scope='descriptive_fixed_OOF_not_full_pipeline_refit')
            maxima['paired_intervals']=max(maxima['paired_intervals'],compare_tree(expected,public['paired_contrasts'][control][metric_name]))
    return dict(schema_version='sami_stage_b_independent_run_audit_v1',status='PASS_INDEPENDENT_REPLAY',
                lineage=lineage,audit_script_sha256=sha(Path(__file__)),plan_sha256=sha(run/'PLAN.json'),
                results_sha256=sha(run/'RESULTS.json'),OOF_private_sha256=sha(run/'OOF_PRIVATE.npz'),
                verified_unique_teacher_models=21,verified_four_donor_teacher_models=15,verified_five_donor_teacher_models=6,
                verified_training_crossfit_applications=30,verified_outer_prediction_applications=6,
                verified_program_regressors=36,verified_nonfitted_mean_references=6,
                verified_outer_predictions=42,complete_spot_coverage=n,
                teacher_recipient_and_outer_donor_disjoint=True,true_target_scaler_shared_before_control_remapping=True,
                all_target_remaps_current_train_only=True,original_ST_test_arrays_read=False,TCGA_DSS_arrays_read=False,
                optimizer_steps=0,new_model_fits=0,PCA_directions_refitted=False,
                PCA_direction_provenance='saved_direction_source_hash_and_receipt_bound; train_scaler_orthogonality_sign_whitening_and_transform_replayed',
                maxima_absolute_discrepancy=dict(maxima),teacher_diagnostics=teacher_diagnostics,
                regressor_diagnostics=regressor_diagnostics,pca_diagnostics=pca_diagnostics,
                gate_distribution_diagnostics=gate_geometry,
                statistical_scope='descriptive_six_development_donor_OOF_not_external_confirmation_or_cell_source_truth')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.output.exists():
        require(args.output.read_text(encoding='utf-8')==text,'Refusing to replace a different completed independent audit')
    else:
        args.output.write_text(text,encoding='utf-8')
    print(json.dumps({'status':result['status'],'teachers':21,'regressors':36,'new_optimizer_steps':0}))


if __name__=='__main__':
    main()
