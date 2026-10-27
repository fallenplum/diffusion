"""Numerical-precision repair of exactly two failed PCA geometry units.

The original source, lock, seven successful models and nine START records stay
unchanged. Only L-BFGS ftol/gtol change, under a separate pre-fit repair lock.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys
import threading

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import bgb_en_pca_geometry as original
from scripts.bgb_en_checkpoint_audit import require,sha256
from src.phase10_trimodal.math import efron_weights

THREAD=threading.local()
OPTIONS=dict(maxiter=2000,ftol=1e-15,gtol=1e-9,maxls=50)


def strict_fit(x,t,e,lam=1.,offset=None,maxiter=2000):
    require(lam==1. and offset is None and maxiter==2000,'REPAIR_STATISTICAL_OBJECTIVE_CHANGED')
    x=np.asarray(x,np.float64);t=np.asarray(t,float);e=np.asarray(e,int)
    a=efron_weights(t,e)
    with np.errstate(divide='ignore'):la=np.log(a)
    def fun(w):
        risk=x@w;scores=la+risk[None,:];den=logsumexp(scores,axis=1)
        dr=(np.exp(scores-den[:,None]).sum(0)-e)/e.sum()
        return float((den.sum()-risk[e==1].sum())/e.sum()+lam/2*(w@w)),x.T@dr+lam*w
    opt=minimize(fun,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options=OPTIONS)
    value,grad=fun(opt.x)
    info=dict(loss=value,gradient_max=float(np.max(np.abs(grad))),iterations=int(opt.nit),
              solver='L-BFGS-B',success=bool(opt.success),parameters=x.shape[1],precision_options=OPTIONS,
              external_gradient_guard=2e-5)
    if getattr(THREAD,'directory',None) is not None:
        # Kept private even on another failure; no invented last-state details.
        torch.save(dict(coefficient=opt.x,optimizer=info,status=int(opt.status),message=str(opt.message)),THREAD.directory/'SOLVER_LAST_STATE.pt')
    require(np.isfinite(value) and np.isfinite(opt.x).all() and np.max(np.abs(grad))<=2e-5,
            'STRICT_RETRY_DID_NOT_MEET_UNCHANGED_GRADIENT_CONTRACT')
    return opt.x,info


def repair_contract(args,contract):
    config=json.loads(args.repair_config.read_text(encoding='utf-8'))
    require(config['locked_units']==[dict(dimension=128,partition=2),dict(dimension=256,partition=0)]
        and config['lambda']==1. and {key:config[key] for key in OPTIONS}==OPTIONS
        and config['unchanged_external_gradient_guard']==2e-5,'REPAIR_CONFIG_OUTSIDE_PRECISION_ONLY_LOCK')
    require(config['PCA_refits']==config['teacher_refits']==0 and config['changes_R4_finalists'] is False,
            'REPAIR_SCOPE_CHANGED')
    starts={};successes={}
    for dim in [32,128,256]:
        for part in [0,1,2]:
            label=f'PCA{dim}_UNWHITENED_p{part}';directory=args.private/'cells'/label
            require((directory/'START.json').exists(),'ORIGINAL_ATTEMPT_ACCOUNTING_MISSING')
            starts[label]=sha256(directory/'START.json')
            if (directory/'COMPLETE.json').exists():successes[label]=sha256(directory/'COMPLETE.json')
    require(len(starts)==9 and len(successes)==7,'ORIGINAL_ATTEMPT_COUNTS_CHANGED')
    for unit in config['locked_units']:
        label=f"PCA{unit['dimension']}_UNWHITENED_p{unit['partition']}"
        require(label not in successes,'REPAIR_ATTEMPTS_TO_REFIT_VALID_ORIGINAL_UNIT')
    sources={name:sha256(ROOT/name) for name in ('scripts/bgb_en_pca_geometry_solver_repair.py',
        'scripts/bgb_en_pca_geometry.py','scripts/bgb_en_pca_geometry_audit.py','src/bgb_en/survival.py',
        'src/phase10_trimodal/reference.py','src/phase10_trimodal/math.py')}
    return dict(config=config,repair_config_sha256=sha256(args.repair_config),original_contract_sha256=original.canonical(contract),
        original_lock_sha256=sha256(args.private/'LOCK.json'),original_source_sha256=sha256(ROOT/'scripts/bgb_en_pca_geometry.py'),
        original_geometry_config_sha256=sha256(args.geometry_config),original_START_receipts=starts,
        original_successful_receipts=successes,source_sha256=sources,
        original_failure_detail='initial_run_reported_gradient_contract_exception;two_units_have_no_COMPLETE;per_unit_last_coefficients_and_failure_details_not_saved')


def ensure_repair_lock(args,contract,create=False):
    value=repair_contract(args,contract);path=args.repair_private/'LOCK.json'
    if path.exists():
        old=json.loads(path.read_text());require(old['contract']==value and old['status']=='LOCKED_BEFORE_TWO_PRECISION_RETRIES','REPAIR_SOURCE_LOCK_CHANGED')
        snapshot=args.repair_private/'original_source_snapshot'
        require(sha256(snapshot/'bgb_en_pca_geometry.py')==value['original_source_sha256']
            and sha256(snapshot/'pca_geometry_v1.json')==value['original_geometry_config_sha256'],'ORIGINAL_SOURCE_SNAPSHOT_CHANGED')
        return old
    require(create,'REPAIR_PREFIT_LOCK_MISSING')
    require(not (args.repair_private/'cells').exists(),'REPAIR_CELLS_EXIST_BEFORE_LOCK')
    args.repair_private.mkdir(parents=True,exist_ok=True)
    snapshot=args.repair_private/'original_source_snapshot';snapshot.mkdir(exist_ok=True)
    for source,target in [(ROOT/'scripts/bgb_en_pca_geometry.py',snapshot/'bgb_en_pca_geometry.py'),
                          (args.geometry_config,snapshot/'pca_geometry_v1.json')]:
        with target.open('xb') as handle:handle.write(source.read_bytes())
    lock=dict(status='LOCKED_BEFORE_TWO_PRECISION_RETRIES',created_utc=datetime.now(timezone.utc).isoformat(),contract=value,
              original_solver_attempts=9,original_successful_units=7,original_failed_units=2,new_solver_attempts_before_lock=0)
    with path.open('x',encoding='utf-8') as handle:json.dump(lock,handle,indent=2,allow_nan=False)
    original.write(args.repair_private/'INITIAL_FAILURE_ACCOUNTING.json',dict(
        original_solver_attempts=9,original_valid_units=7,original_failed_units=2,failed_units=value['config']['locked_units'],
        original_START_sha256=value['original_START_receipts'],original_complete_sha256=value['original_successful_receipts'],
        numerical_details='Unavailable: original fit raised before returning or saving last coefficient'))
    return lock


def merged_report(args,contract,lock):
    base=original.report(args,contract);rows=base['partition_rows'][:];sources={}
    for dim in [32,128,256]:
        for part in [0,1,2]:
            label=f'PCA{dim}_UNWHITENED_p{part}';directory=args.private/'cells'/label
            if not (directory/'COMPLETE.json').exists():directory=args.repair_private/'cells'/label
            if not (directory/'COMPLETE.json').exists():continue
            value=json.loads((directory/'COMPLETE.json').read_text());sources[dim,part]=value
            require(value['artifact_sha256']==sha256(directory/'model.pt'),'MERGED_GEOMETRY_ARTIFACT_CHANGED')
            if any(row['dimension']==dim and row['partition']==part for row in rows):continue
            row=dict(scope='PARTITION',dimension=dim,partition=part,status='COMPLETE',new_Harrell=value['metrics']['Harrell'],new_Uno60=value['metrics']['Uno60'],
                original_O2_Harrell=value['original_O2_metrics']['Harrell'],original_O2_Uno60=value['original_O2_metrics']['Uno60'],
                O0_Harrell=value['O0_metrics']['Harrell'],O0_Uno60=value['O0_metrics']['Uno60'],
                delta_H_vs_original_O2=value['metrics']['Harrell']-value['original_O2_metrics']['Harrell'],
                delta_U60_vs_original_O2=value['metrics']['Uno60']-value['original_O2_metrics']['Uno60'],
                delta_H_vs_O0=value['metrics']['Harrell']-value['O0_metrics']['Harrell'],delta_U60_vs_O0=value['metrics']['Uno60']-value['O0_metrics']['Uno60'],
                new_vs_O0_standardized_valid_SD_difference=value['diagnostics']['new_vs_O0_valid_SD_difference'],
                old_vs_O0_standardized_valid_SD_difference=value['diagnostics']['old_vs_O0_valid_SD_difference'],
                new_vs_O0_valid_Pearson=value['diagnostics']['new_vs_O0_valid_Pearson'],old_vs_O0_valid_Pearson=value['diagnostics']['old_vs_O0_valid_Pearson'],
                original_PC_gene_L2_directional_penalty_ratio=value['PCA_math_audit']['old_gene_L2_directional_penalty_ratio'],
                original_saved_RNA_decoder_Pearson=(value['original_RNA_retention'] or {}).get('risk_decoder_Pearson'),
                original_saved_RNA_decoder_R2=(value['original_RNA_retention'] or {}).get('risk_decoder_R2'),
                actual_new_Cox_fits=1,PCA_refits=0,teacher_refits=0)
            rows.append(row)
    for row in rows:row['numerical_solver']='strict_L-BFGS-B_retry' if dict(dimension=row['dimension'],partition=row['partition']) in lock['contract']['config']['locked_units'] else 'original_L-BFGS-B'
    summaries=[]
    for dim in [32,128,256]:
        cells=[sources[dim,p] for p in [0,1,2] if (dim,p) in sources]
        row=dict(scope='THREE_PARTITION_COUNTS',dimension=dim,partition='ALL',status='COMPLETE' if len(cells)==3 else 'RUNNING',completed_new_Cox_fits=len(cells))
        if len(cells)==3:
            new,old,o0=(original.aggregate([v[key] for v in cells]) for key in ('metrics','original_O2_metrics','O0_metrics'))
            row.update(new_Harrell=new['Harrell'],new_Uno60=new['Uno60'],original_O2_Harrell=old['Harrell'],original_O2_Uno60=old['Uno60'],O0_Harrell=o0['Harrell'],O0_Uno60=o0['Uno60'],
                delta_H_vs_original_O2=new['Harrell']-old['Harrell'],delta_U60_vs_original_O2=new['Uno60']-old['Uno60'],delta_H_vs_O0=new['Harrell']-o0['Harrell'],delta_U60_vs_O0=new['Uno60']-o0['Uno60'],
                new_counts=new,original_O2_counts=old,O0_counts=o0)
        summaries.append(row)
    new_attempts=sum((args.repair_private/'cells'/f"PCA{u['dimension']}_UNWHITENED_p{u['partition']}"/'START.json').exists() for u in lock['contract']['config']['locked_units'])
    base.update(status='COMPLETE_NINE_VALID_UNITS_ELEVEN_SOLVER_ATTEMPTS' if len(rows)==9 else 'PARTIAL_WITH_PRECISION_REPAIR',
        new_Cox_fits=9+new_attempts,unique_valid_Cox_models=len(rows),actual_Cox_solver_attempts=9+new_attempts,
        initial_solver_attempts=9,initial_valid_units=7,initial_failed_units=2,precision_retry_attempts=new_attempts,
        new_Cox_fits_definition='all actual numerical solver attempts including the two original failed attempts',
        summaries=summaries,partition_rows=sorted(rows,key=lambda r:(r['dimension'],r['partition'])),
        solver_repair=dict(lock_sha256=sha256(args.repair_private/'LOCK.json'),contract=lock['contract'],
                           original_source_and_seven_successful_models_unchanged=True))
    original.write(args.output/'R2_PCA_GEOMETRY_CONTROL.json',base)
    flat=base['partition_rows']+[{k:v for k,v in row.items() if not k.endswith('_counts')} for row in summaries]
    fields=list(dict.fromkeys(key for row in flat for key in row))
    with (args.output/'R2_PCA_GEOMETRY_CONTROL.csv').open('w',encoding='utf-8',newline='') as handle:
        out=csv.DictWriter(handle,fieldnames=fields);out.writeheader();out.writerows(flat)
    path=args.output/'R2_PCA_GEOMETRY_CONTROL.md';text=path.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(text):
        if line.startswith('状态：'):text[i]=f"状态：**{base['status']}**；有效唯一模型{len(rows)}/9，实际solver尝试{9+new_attempts}次（原9次含2次失败＋修复{new_attempts}次）；PCA/teacher重拟合0，GPU执行0。"
    start=next(i for i,line in enumerate(text) if line.startswith('| PC维度'))
    for i,row in enumerate(summaries):
        fmt=lambda value:'—' if value is None else f'{value:.6f}'
        text[start+2+i]='| '+str(row['dimension'])+' | '+' | '.join(fmt(row.get(key)) for key in ('new_Harrell','new_Uno60','original_O2_Harrell','original_O2_Uno60','O0_Harrell','O0_Uno60','delta_H_vs_original_O2','delta_H_vs_O0'))+' |'
    text+=['','## 数值修复溯源','',
        '原9次求解中7次通过，PCA128/p2与PCA256/p0无COMPLETE；原运行报告收敛守护异常。未保存两单元末次系数及各自失败细节，不能补写具体梯度或迭代原因。'
        '单独修复锁仅将L-BFGS ftol/gtol收紧为1e-15/1e-9；λ、输入、PCA、患者、终点及外部2e-5梯度守护保持一致。'
        '原源码/config快照、原锁与7份有效模型保留，新增两次尝试单独计数。独立Efron解析梯度和saved-coefficient回放另见R2_PCA_GEOMETRY_REPLAY_AUDIT.json。','']
    path.write_text('\n'.join(text),encoding='utf-8')
    return base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['lock','run','report'],required=True)
    for key in ('assets','core-private','private','repair-private'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/bgb_en');parser.add_argument('--config',type=Path,default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--geometry-config',type=Path,default=ROOT/'configs/bgb_en/pca_geometry_v1.json')
    parser.add_argument('--repair-config',type=Path,default=ROOT/'configs/bgb_en/pca_geometry_solver_repair_v1.json')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    contract,prepared,raw=original.source_contract(args);original.enforce_lock(args,contract)
    lock=ensure_repair_lock(args,contract,create=args.stage=='lock')
    if args.stage=='lock':print(json.dumps(dict(status=lock['status'],new_solver_attempts=0)));return
    if args.stage=='run':
        retry_args=copy.copy(args);retry_args.private=args.repair_private
        previous=original.fit_offset_ridge;original.fit_offset_ridge=strict_fit
        def unit(spec):
            THREAD.directory=args.repair_private/'cells'/f"PCA{spec['dimension']}_UNWHITENED_p{spec['partition']}"
            return original.fit_unit(retry_args,contract,prepared[spec['partition']],raw,spec['partition'],spec['dimension'])
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                for future in as_completed([pool.submit(unit,u) for u in lock['contract']['config']['locked_units']]):print(json.dumps(future.result()),flush=True)
        finally:original.fit_offset_ridge=previous
    result=merged_report(args,contract,lock)
    print(json.dumps({k:result[k] for k in ('status','unique_valid_Cox_models','actual_Cox_solver_attempts','PCA_refits','teacher_refits')}))


if __name__=='__main__':main()
