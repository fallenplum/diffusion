"""Fold-local M1--M3 inputs and exact, trainable R4 warm mapping.

No endpoint file is opened. Raw cohort values and all per-person returns are
private. Fixed GRID features are shared by all arms because the REGIONS mean
cannot exactly reconstruct the historical CP_SCALAR function.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
import numpy as np
import torch
from src.semantic_b0b7.data import ProgramTransform
from src.semantic_b0b7.p1r_data import require_no_outcomes
from src.semantic_b0b7.p2_anchor import array_sha

CLINICAL=('AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE')


def standardize(x,fit):
    x=np.asarray(x,np.float64)
    if not np.isfinite(x).all():raise ValueError('NONFINITE_COMMON_COVARIATE')
    mean=x[fit].mean(0);sd=x[fit].std(0)
    sd=np.where(sd>1e-8,sd,1.)
    return (x-mean)/sd,{'mean':mean,'sd':sd}


def historical_covariates(raw,rows,state):
    """Apply the existing current-training-partition maps without refitting."""
    cols=[]
    for field in CLINICAL:
        v=np.asarray(raw['clinical_raw'][field])[rows].astype(str)
        coding=state['clinical']['coding'][field]
        if field=='AGE':
            x=np.asarray([float(s) if s.replace('.','',1).isdigit() else np.nan for s in v])
            cols.append(np.where(np.isfinite(x),x,coding['median'])[:,None])
        else:cols.append(v[:,None]==np.asarray(coding['levels'])[None])
    c=np.concatenate(cols,axis=1).astype(np.float64)
    cs=state['clinical']['scale'];c=(c-cs['mean'])/cs['sd']
    ss=state['S32'];s=(np.asarray(raw['S32'])[rows]-ss['mean'])/ss['sd']
    vs=state['visual']['LOCAL'];x=np.asarray(raw['conch_local'],np.float64)[rows]
    z=(x-vs['before']['mean'])/vs['before']['sd']
    pc=(z-vs['mean'])@vs['components'].T
    pc=(pc-vs['after']['mean'])/vs['after']['sd']
    ms=state['molecular'];x=np.asarray(raw['rna'],np.float64)[rows];mask=np.isfinite(x)
    filled=np.where(mask,x,ms['median'])
    z=(filled-ms['scale']['mean'])/ms['scale']['sd']
    z=np.where(mask&(ms['measured'][None]>0),z,0.).astype(np.float32)
    r=z[:,ms['selected']].astype(np.float64)
    r=(r-ms['selected_scale']['mean'])/ms['selected_scale']['sd']
    return c,s,pc,r


def equal_slide_mean(patches,mask,slide):
    means=[]
    for p,m,s in zip(patches,mask,slide):
        groups=np.unique(s[m])
        if len(groups)==0:raise ValueError('EMPTY_WSI_PATIENT')
        means.append(np.mean([p[m&(s==g)].mean(0) for g in groups],axis=0))
    return np.stack(means)


@dataclass
class Prepared:
    batch:dict
    rows:np.ndarray
    train:np.ndarray
    evaluation:np.ndarray
    warm:dict
    baseline:np.ndarray
    flat:np.ndarray
    state:dict
    receipt:dict


def prepare(raw,frozen,anchor,fit,evaluation,memberships):
    require_no_outcomes(raw)
    fit,evaluation=np.asarray(fit,np.int64),np.asarray(evaluation,np.int64)
    if np.intersect1d(fit,evaluation).size or len(np.unique(fit))!=len(fit):
        raise ValueError('M123_TRAIN_EVALUATION_OVERLAP')
    if not np.array_equal(anchor.train_rows,fit) or not np.array_equal(anchor.evaluation_rows,evaluation):
        raise ValueError('WARM_TEACHER_CURRENT_PARTITION_MISMATCH')
    rows=np.r_[fit,evaluation];tr=np.arange(len(fit));ev=np.arange(len(fit),len(rows))
    expected_local=np.searchsorted(np.sort(rows),fit)
    full=anchor.state['full'];old=full['preprocessing']
    if not np.array_equal(old['fit'],expected_local):
        raise ValueError('OLD_PREPROCESSING_WAS_NOT_FITTED_ON_CURRENT_TRAINING_ROWS')
    c,s,grid,r=historical_covariates(raw,rows,old)
    registry=np.asarray(frozen['global_rows'],np.int64)
    if len(np.unique(registry))!=len(registry):raise ValueError('FM_REGISTRY_DUPLICATE')
    reverse={int(v):i for i,v in enumerate(registry)}
    take=torch.tensor([reverse[int(v)] for v in rows])
    image=frozen['image']
    patch=image['patch_values'][take].numpy();mask=image['patch_mask'][take].numpy()
    slide=image['patch_slide'][take].numpy()
    if not np.isfinite(patch[mask]).all():raise ValueError('NONFINITE_OBSERVED_PATCH')
    mean=equal_slide_mean(patch,mask,slide)
    mean_z,mean_st=standardize(mean,tr)
    native,native_st=standardize(image['titan_native'][take].numpy(),tr)
    aligned,aligned_st=standardize(image['titan_aligned'][take].numpy(),tr)
    # No information from another outer-test patient enters program moments.
    names,rm,cm=memberships
    rr=np.where(np.asarray(raw['rna_measured'])[rows],np.asarray(raw['rna'])[rows],np.nan)
    cc=np.where(np.asarray(raw['cna_measured'])[rows],np.asarray(raw['cna'])[rows],np.nan)
    program_state=ProgramTransform.fit(rr,cc,tr,rm,cm,min_coverage=.5,min_genes=5)
    program=program_state.transform(rr,cc)
    p=np.concatenate([s,grid,native,aligned],1)
    g=np.concatenate([r,program['values'].reshape(len(rows),-1),
        program['measured'].reshape(len(rows),-1),program['coverage'].reshape(len(rows),-1)],1)
    cp=full['full_models']['LOCAL'];rna=full['full_models']['R'];beta=full['beta']
    coef=np.asarray(cp['coef']);cwidth=c.shape[1]
    if len(coef)!=cwidth+64 or r.shape[1]!=512:
        raise ValueError('EXACT_CP_S32_GRID32_RNA512_REQUIRED')
    wp=np.zeros(p.shape[1]);wp[:64]=coef[cwidth:]/cp['scale']['sd']
    wg=np.zeros(g.shape[1]);wg[:512]=beta*np.asarray(rna['coef'])/rna['scale']['sd']
    wc=coef[:cwidth]/cp['scale']['sd']
    pb=-cp['scale']['mean']/cp['scale']['sd']
    gb=-beta*rna['scale']['mean']/rna['scale']['sd']
    baseline=p@wp+g@wg+c@wc+pb+gb
    reference=np.r_[anchor.full_fit_train_in_sample_risk_diagnostic,anchor.b0_exact_evaluation]
    error=float(np.max(np.abs(baseline-reference)))
    if error>1e-8:raise ValueError('WARM_LINEAR_RECIPE_REPLAY_FAILED:'+str(error))
    # This reference is the full current-fit R4 function, never its cross-fitted
    # training offsets; it is used only for initialization and diagnostics.
    flat0=np.c_[p,g,c,mean_z]
    flat,flat_state=standardize(flat0,tr)
    batch={'patch':torch.from_numpy(patch),'mask':torch.from_numpy(mask),'slide':torch.from_numpy(slide),
        'mean':torch.from_numpy(mean_z.astype(np.float32)), 'P':torch.from_numpy(p.astype(np.float32)),
        'G':torch.from_numpy(g.astype(np.float32)),'C':torch.from_numpy(c.astype(np.float32)),
        'program':torch.from_numpy(program['values']),'program_mask':torch.from_numpy(program['measured']),
        'coverage':torch.from_numpy(program['coverage'])}
    warm={'p_coef':wp,'g_coef':wg,'c_coef':wc,'p_bias':pb,'g_bias':gb}
    state={'native':native_st,'aligned':aligned_st,'mean':mean_st,'program':program_state,
           'historical_preprocessing':old,'flat':flat_state}
    receipt={'schema':'m123_fold_input_v1','patients':len(rows),'fit_patients':len(tr),'evaluation_patients':len(ev),
        'fit_rows_sha256':array_sha(fit),'evaluation_rows_sha256':array_sha(evaluation),
        'legacy_RNA512_selected_sha256':array_sha(np.asarray(old['molecular']['selected'])),
        'legacy_RNA512_columns':512,'Hallmark_programs':len(names),'clinical_columns':c.shape[1],
        'global_pathology_columns':p.shape[1],'global_omics_columns':g.shape[1],
        'M0plus_fixed_columns':flat.shape[1],'exact_warm_FP64_max_error':error,
        'DSS_access_in_preprocessing':False,'R4_GRID_in_all_arms_for_exact_warm_and_input_fairness':True,
        'train_reference_is_full_fit_not_OOF':True,'old_r0_is_fixed_offset':False,
        'patches_per_patient_limit':patch.shape[1],'slide_weighting':'within_slide_then_equal_slide',
        'input_sha256':hashlib.sha256(json.dumps({k:array_sha(v.numpy()) for k,v in batch.items()},sort_keys=True).encode()).hexdigest()}
    return Prepared(batch,rows,tr,ev,warm,baseline,flat,state,receipt)

