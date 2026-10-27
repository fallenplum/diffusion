"""Locked outer anchor/scalar confirmation; call only after inner selection lock.

This file does not authorize a neural outer test before its per-outer inner
selection finishes. Deterministic CP fits are reused, scalar teacher blocks vary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.bgb_en.data import load_raw,sha256,outer_partition
from src.bgb_en.omics import load_gmt_membership
from src.bgb_en.protocol import prepare_partition,array_hash
from src.bgb_en.baselines import prepare_anchors
from src.bgb_en.survival import fit_offset_ridge,standardized_risk,fit_scalar_same_teacher,evaluate,combine_metrics
from scripts.bgb_en_run import load_inputs

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def source_lineage_hash(lineage):
    return hashlib.sha256(json.dumps(lineage,sort_keys=True).encode()).hexdigest()

def validate_selection_lock(lock,inner_report,lineage,config,config_sha,inner_report_sha,fold):
    """Bind the inner decision to exact source, configuration and report bytes."""
    if lock.get('status')!='LOCKED_BEFORE_OUTER_EVALUATION' or lock.get('outer_test_evaluated') is not False:
        raise ValueError('A completed pre-outer selection lock is required')
    if fold not in (1,2,3,4,5) or lock.get('outer_folds')!=[1,2,3,4,5]:
        raise ValueError('Outer fold scope differs from the pre-outer lock')
    digest=source_lineage_hash(lineage)
    if lock.get('source_lineage_sha256')!=digest or inner_report.get('source_lineage_sha256')!=digest:
        raise ValueError('Selection lock or inner report has different scientific source/data lineage')
    if lock.get('config_sha256')!=config_sha:
        raise ValueError('Selection lock has a different configuration SHA256')
    canonical=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if inner_report.get('configuration_canonical_sha256')!=canonical:
        raise ValueError('Inner report has a different configuration')
    if lock.get('inner_report_sha256')!=inner_report_sha:
        raise ValueError('Inner report bytes changed after selection was locked')
    if inner_report.get('status')!='COMPLETE_CORE_INNER' or inner_report.get('no_outer_test_evaluation') is not True:
        raise ValueError('A complete, outer-blind inner matrix is required')
    expected={}
    for stage in ('R1','R2','R3'):
        arms=config[stage]['arms'][:]
        if stage=='R3':arms+=config[stage].get('additional_real_retraining',[])
        expected[stage]=sum(config['prototype']['inner_splits']*(1 if arm in ('P0','O0','O1','F0','F1') or arm.startswith('O2_') else len(config['prototype']['neural_seeds'])) for arm in arms)
        found=inner_report.get('stage_counts',{}).get(stage,{})
        if found.get('status')!='COMPLETE' or found.get('completed_unique_cells')!=expected[stage] or found.get('planned_unique_cells')!=expected[stage]:
            raise ValueError('Incomplete or incompatible stage counts in inner report')
    if inner_report.get('completed_unique_cells')!=sum(expected.values()):
        raise ValueError('Inner matrix completion count does not match protocol')
    finalists=lock.get('finalists')
    if (not isinstance(finalists,list) or len(finalists)!=len(set(finalists))
            or not {'CP_STRONG','CP_SCALAR'}.issubset(finalists)
            or not set(finalists).issubset({'CP_STRONG','CP_SCALAR','NESTED_INTERACTION'})):
        raise ValueError('Finalist lock must explicitly include both baseline references and no unknown model')
    return dict(executed_finalists=['CP_STRONG','CP_SCALAR'],
                neural_finalists_not_executed=[name for name in finalists if name not in ('CP_STRONG','CP_SCALAR')],
                this_tool_fits_neural_models=False)

def risk_distribution(base,cp,rna):
    train=base['train'];output={}
    for name,cf,key in [('CP',np.asarray(cp)[train],'LOCAL'),('RNA',np.asarray(rna)[train],'R')]:
        full=np.asarray(base['full_risks'][key])[train]
        a,b=float(cf.std()),float(full.std())
        output[name]=dict(crossfit_mean=float(cf.mean()),fullfit_mean=float(full.mean()),crossfit_sd=a,fullfit_sd=b,
            pearson=float(np.corrcoef(cf,full)[0,1]) if a>0 and b>0 else None,
            sd_ratio=a/b if b>0 else None,RMS_gap=float(np.sqrt(np.mean((cf-full)**2))),
            alignment='each_teacher_own_train_mean_SD_not_heldout_moments')
    return output

def lean_teacher(teacher):
    # Keep the transformations, member sets and Cox coefficients needed to
    # replay its held block. No feature design or raw RNA matrix is copied.
    return dict(train=teacher['train'],held=teacher['held'],
                preprocessing=teacher.get('preprocessing',teacher.get('state')),
                models={key:teacher['models'][key] for key in ('LOCAL','CS','R') if key in teacher['models']})

def calibration_risks(base,cp_cf,rna_cf,teacher_groups,beta):
    """Tiny n-patient arrays required to reconstruct frozen outer offsets."""
    train=base['train'];n=len(base['CP'])
    cp=np.asarray(base['full_risks']['LOCAL']).copy();rna=np.asarray(base['full_risks']['R']).copy()
    cp[train]=np.asarray(cp_cf)[train];rna[train]=np.asarray(rna_cf)[train]
    cp_only_cf=np.zeros(n);rna_only_cf=np.zeros(n)
    cp_only_cf[train]=cp[train];rna_only_cf[train]=rna[train]
    return dict(CP_cf=cp_only_cf,RNA_cf=rna_only_cf,teacher_groups=np.asarray(teacher_groups).copy(),
                CP=cp,RNA=rna,base=cp+beta*rna,
                crossfit_values_valid_only_where_teacher_groups_nonnegative=True)

def base_calibration(base):
    train=base['train']
    cs_cf=np.zeros(len(base['CS']));cs_cf[train]=base['CS'][train]
    return dict(beta=base['beta'],scalar_audit=base['scalar_audit'],
                teachers=[lean_teacher(t) for t in base['teachers']],
                **calibration_risks(base,base['CP'],base['RNA'],base['teacher_groups'],base['beta']),
                CS_cf=cs_cf,CS=np.asarray(base['CS']).copy(),CS_teacher_assignment='seed31_teacher_groups',
                risk_distribution=risk_distribution(base,base['CP'],base['RNA']),
                actual_new_Cox_fits=13,actual_new_scalar_fits=1)

def recalibrate(raw,base,membership,seed):
    train,valid=base['train'],base['valid'];n=len(raw['t']);cp=np.zeros(n);rna=np.zeros(n);assignment=np.full(n,-1,int);teachers=[]
    for group,held in enumerate(np.array_split(np.random.default_rng(seed).permutation(train),3)):
        fit=np.setdiff1d(train,held);feature=prepare_partition(raw,fit,membership)
        teacher=dict(train=fit,held=held,state=feature['state'],models={})
        for key,destination,lam in [('LOCAL',cp,10.),('R',rna,1.)]:
            coef,info=fit_offset_ridge(feature[key][fit],raw['t'][fit],raw['e'][fit],lam)
            prediction=feature[key]@coef;scaled,scale=standardized_risk(prediction[fit],prediction)
            destination[held]=scaled[held];teacher['models'][key]=dict(coef=coef,scale=scale,optimizer=info)
        assignment[held]=group;teachers.append(teacher)
    beta,info=fit_scalar_same_teacher(cp[train],rna[train],raw['t'][train],raw['e'][train],assignment[train])
    return dict(beta=beta,scalar_audit=info,teachers=[lean_teacher(t) for t in teachers],
        **calibration_risks(base,cp,rna,assignment,beta),
        risk_distribution=risk_distribution(base,cp,rna),actual_new_Cox_fits=6,actual_new_scalar_fits=1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True,type=Path);p.add_argument('--private',required=True,type=Path)
    p.add_argument('--selection-lock',required=True,type=Path);p.add_argument('--fold',required=True,type=int,choices=range(1,6));p.add_argument('--resume',action='store_true')
    p.add_argument('--inner-report',type=Path,default=ROOT/'reports/bgb_en/INNER_REPORT_AUDIT.json')
    p.add_argument('--config',type=Path,default=ROOT/'configs/bgb_en/prototype_v1.json')
    args=p.parse_args();torch.set_num_threads(2)
    lock=json.loads(args.selection_lock.read_text())
    inner_report=json.loads(args.inner_report.read_text())
    assets,config,raw,member,lineage=load_inputs(args)
    execution_scope=validate_selection_lock(lock,inner_report,lineage,config,sha256(args.config),sha256(args.inner_report),args.fold)
    cache_receipt=json.loads((Path(assets['patch_grid'])/'READY.json').read_text(encoding='utf-8-sig'))
    full_mean_sha=sha256(Path(assets['patch_grid'])/'full_mean.npy')
    if full_mean_sha!=cache_receipt['array_sha256']['full_mean']:
        raise ValueError('Full CONCH mean differs from its frozen native-cache receipt')
    train,test=outer_partition(raw['folds'],args.fold)
    directory=args.private/'outer'/f'fold{args.fold}'
    done=directory/'COMPLETE.json';artifact=directory/'models.pt'
    contract=dict(selection_lock_sha256=sha256(args.selection_lock),inner_report_sha256=sha256(args.inner_report),
        config_sha256=sha256(args.config),source_lineage=lineage,source_lineage_sha256=source_lineage_hash(lineage),
        assets_configuration_sha256=sha256(args.assets),
        visual_sources_sha256=raw['audit'].get('visual_sources_sha256'),full_CONCH_mean_sha256=full_mean_sha,
        cohort_sha256=raw['audit']['cohort_sha256'],endpoints_sha256=raw['audit']['endpoints_sha256'],
        train_sha256=array_hash(train),test_sha256=array_hash(test),source_sha256=sha256(Path(__file__)),
        module_sha256={name:sha256(ROOT/'src/bgb_en'/name) for name in ('protocol.py','baselines.py','survival.py','data.py','omics.py')})
    if done.exists():
        old=json.loads(done.read_text())
        if not args.resume or old['contract']!=contract or sha256(artifact)!=old['artifact_sha256']:raise ValueError('Outer reuse contract mismatch')
        return
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'START.json').open('x') as f:json.dump(contract,f)
    start=time.monotonic()
    try:
        base=prepare_anchors(raw,train,test,member.numpy(),seed=27011+args.fold*100+31)
        target=base['full_risks']['LOCAL'][test];rna=base['full_risks']['R'][test]
        results={};models={};actual_cox=13;actual_scalar=1
        for seed in (31,47,83):
            calibrated=base_calibration(base) if seed==31 else recalibrate(raw,base,member.numpy(),27011+args.fold*100+seed)
            if seed!=31:actual_cox+=6;actual_scalar+=1
            models[seed]=calibrated;risk=target+calibrated['beta']*rna
            metrics={}
            for name,pred in [('CP_STRONG',target),('CP_SCALAR',risk)]:
                metric=evaluate(raw['t'][train],raw['e'][train],raw['t'][test],raw['e'][test],pred,target)
                pvalid=raw['pfi_valid'][test];ptrain=raw['pfi_valid'][train]
                metric['PFI_frozen_DSS_risk']=evaluate(raw['pfi_t'][train][ptrain],raw['pfi_e'][train][ptrain],
                    raw['pfi_t'][test][pvalid],raw['pfi_e'][test][pvalid],pred[pvalid])
                metrics[name]=metric
            results[str(seed)]=dict(models=metrics,beta=calibrated['beta'],risk_distribution=calibrated['risk_distribution'])
            models[seed]['test_CP_risk']=target;models[seed]['test_CP_SCALAR_risk']=risk
        # Store transformations/coefficients and OOF privately, not a second
        # copy of the large molecular design arrays.
        torch.save(dict(contract=contract,train=train,test=test,full_models=base['full_models'],
            full_preprocessing=base['features']['state'],full_risks=base['full_risks'],
            seed31_CS_risk=base['CS'],teachers=models,results=results),artifact)
        write(done,dict(status='COMPLETE_OUTER_BASELINES',contract=contract,fold=args.fold,results=results,
            actual_Cox_fits=actual_cox,actual_scalar_calibration_fits=actual_scalar,
            deterministic_CP_test_vector_fits=1,CP_seed_references=3,artifact_sha256=sha256(artifact),
            elapsed_seconds=time.monotonic()-start,selection_used_outer_test=False,**execution_scope))
        print(json.dumps(dict(status='COMPLETE_OUTER_BASELINES',fold=args.fold,actual_Cox_fits=actual_cox,actual_scalar_fits=actual_scalar)))
    except Exception as exc:
        write(directory/'FAILED.json',dict(status='FAILED',type=type(exc).__name__,message=str(exc)));raise

if __name__=='__main__':main()
