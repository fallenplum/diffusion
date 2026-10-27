"""One locked BG-BEN qualification, preparation, or formal inner fitting unit."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.decomposition import PCA
from src.bgb_en.data import load_raw,read_patch_cache,sha256
from src.bgb_en.protocol import fixed_inner_splits,patient_mismatch
from src.bgb_en.omics import load_gmt_membership
from src.bgb_en.baselines import prepare_anchors
from src.bgb_en.training import PatchBank,MechanismModel,train_neural,predict
from src.bgb_en.survival import fit_offset_ridge,evaluate

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def torch_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);temporary.replace(path)

def source(config):
    files=[ROOT/'src/bgb_en'/name for name in ('__init__.py','data.py','protocol.py','survival.py','baselines.py','omics.py','pathology.py','interaction.py','training.py')]
    files+=[Path(__file__),ROOT/'configs/bgb_en/prototype_v1.json']
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in sorted(files)}

def load_inputs(args):
    assets=json.loads(args.assets.read_text());config=json.loads(args.config.read_text())
    raw=load_raw(assets)
    # Fresh full-source pooling is independent of patient outcomes and sampling.
    cache=read_patch_cache(assets['patch_grid'],verify_hash=False)
    raw['conch_mean']=np.asarray(cache['full_mean'],np.float64)
    member,annotation=load_gmt_membership(assets['gmt'],raw['rna_symbols'],expected_sha256=assets.get('gmt_sha256'))
    groups=json.loads(Path(assets['legacy_groups']).read_text())
    universe=set().union(*(set(v) for v in groups.values()))
    raw['_direct_gene_universe']=np.array([s in universe for s in raw['rna_symbols']])
    lineage=dict(source=source(config),cohort_sha256=raw['audit']['cohort_sha256'],
        endpoints_sha256=raw['audit']['endpoints_sha256'],GMT_sha256=annotation['sha256'],
        legacy_group_sha256=sha256(assets['legacy_groups']),protocol_sha256=sha256(args.config))
    lineage['patch_cache_receipts']={k:sha256(Path(assets[k])/'READY.json') for k in ('patch_grid','patch_regions','patch_rand','patch_diverse')}
    return assets,config,raw,member,lineage

def prep_paths(args):
    key=f'outer{args.outer_fold}_inner{args.partition}'
    return args.private/'prepared'/(key+'.pt'),args.private/'prepared'/(key+'.json')

def prepare(args,raw,member,lineage):
    path,receipt=prep_paths(args)
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous['lineage']!=lineage or previous['sha256']!=sha256(path):raise ValueError('Prepared source mismatch')
        return torch.load(path,map_location='cpu',weights_only=False)
    if path.exists():raise ValueError('Investigate interrupted preparation before retry')
    split=fixed_inner_splits(raw['folds'],raw['e'],outer=args.outer_fold)
    train,valid=split[args.partition]
    start=time.monotonic();value=prepare_anchors(raw,train,valid,member.numpy(),seed=27011+args.outer_fold*100+args.partition)
    value['lineage']=lineage
    torch_write(path,value)
    write(receipt,dict(status='COMPLETE',lineage=lineage,sha256=sha256(path),
        train_n=len(train),valid_n=len(valid),train_events=int(raw['e'][train].sum()),
        valid_events=int(raw['e'][valid].sum()),upstream_optimizer_fits=value['upstream_optimizer_fits'],
        upstream_preprocessing_fits=value['upstream_preprocessing_fits'],
        risk_distribution=value['risk_distribution'],scalar_beta=value['beta'],
        elapsed_seconds=time.monotonic()-start))
    return value

def qualifier(args,assets,raw,member,config):
    cache=read_patch_cache(assets['patch_regions']);bank=PatchBank(cache,args.device)
    train,valid=fixed_inner_splits(raw['folds'],raw['e'],outer=args.outer_fold)[0]
    # No optimized anchor needed for a real-input gradient and throughput check.
    from src.bgb_en.protocol import prepare_partition
    features=prepare_partition(raw,train,member.numpy())
    fake=dict(features=features,train=train[:16],valid=valid[:8],CS=np.zeros(512),RNA=np.zeros(512),CP=np.zeros(512),base=np.zeros(512),beta=0.)
    output=[]
    for stage,arm in [('R1','P1'),('R1','P3'),('R2','O6'),('R3','F5')]:
        torch.manual_seed(31);model=MechanismModel(stage,arm,fake,member,bank,args.device)
        idx=torch.as_tensor(train[:8],device=args.device)
        torch.cuda.synchronize();start=time.monotonic()
        result=model(idx);result['risk'].square().mean().backward();torch.cuda.synchronize()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        out=model(idx);difference=float((result['risk'].detach()-out['risk'].detach()).abs().max())
        output.append(dict(stage=stage,arm=arm,patients=8,real_patch_nodes=int(bank.mask[idx].sum()),
            seconds_forward_backward=time.monotonic()-start,replay_max=difference,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            nonzero_gradient_tensors=sum(bool(torch.any(g!=0)) for g in grads),gradient_tensors=len(grads),
            geometry=result['diagnostics'] if stage=='R1' else None))
        del model
    write(args.private/'QUALIFICATION.json',dict(status='PASS_REAL_INPUT_FORWARD_BACKWARD',models=output,
        scientific_result=False,optimized_models=0,survival_updates=0,weights_saved=False))
    print(json.dumps(dict(status='PASS_REAL_INPUT_FORWARD_BACKWARD',models=output)),flush=True)

def convex_unit(args,prepared,raw):
    train,valid=prepared['train'],prepared['valid'];features=prepared['features'];arm=args.arm
    reference=prepared['full_risks']['LOCAL'][valid]
    if arm in ('P0','F0','F1','O0'):
        risk={'P0':prepared['full_risks']['GLOBAL'],'F0':prepared['CP'],
            'F1':prepared['base'],'O0':prepared['RNA']}[arm][valid]
        return dict(risk=risk,metrics=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],risk,reference),
            actual_optimizer_fits=0,reference_to_new_upstream_anchor=True,parameters=1 if arm=='F1' else len(prepared['full_models']['GLOBAL' if arm=='P0' else 'R' if arm=='O0' else 'LOCAL']['coef']))
    if arm=='O1':
        x=features['rna'].astype(float);lam=10.;state={}
    elif arm.startswith('O2_PCA'):
        dimension=int(arm.split('PCA')[1]);x=features['R'];dimension=min(dimension,len(train)-1)
        pca=PCA(dimension,svd_solver='full').fit(x[train]);x=pca.transform(x)
        mean,sd=x[train].mean(0),x[train].std(0);sd=np.where(sd>1e-8,sd,1.)
        x=(x-mean)/sd;lam=1.;state=dict(components=pca.components_,mean=pca.mean_,post_mean=mean,post_scale=sd)
    else:raise ValueError('Unknown convex arm')
    coefficient,optimizer=fit_offset_ridge(x[train],raw['t'][train],raw['e'][train],lam)
    risk=x[valid]@coefficient
    reference_risk=prepared['RNA'][valid]
    pearson=float(np.corrcoef(reference_risk,risk)[0,1]) if risk.std()>0 else None
    denominator=float(np.sum((reference_risk-reference_risk.mean())**2))
    # Train-only risk decoding from the representation, evaluated against a
    # teacher that excludes these patients; no held-out decoder fitting.
    target=prepared['full_risks']['R'];a=x[train];scale=target[train].std()
    decoder=np.linalg.solve(a.T@a+np.eye(a.shape[1]),a.T@target[train]) if a.shape[1]<=512 else None
    decoded=x[valid]@decoder if decoder is not None else None
    retention=dict(reference='current_partition_RNA_only_teacher_never_fitted_on_validation',
        prediction_risk_Pearson=pearson,decoder_available=decoder is not None,
        risk_decoder_Pearson=float(np.corrcoef(decoded,reference_risk)[0,1]) if decoded is not None and decoded.std()>0 else None,
        risk_decoder_R2=float(1.-np.sum((decoded-reference_risk)**2)/denominator) if decoded is not None and denominator>0 else None)
    return dict(risk=risk,coefficient=coefficient,feature_state=state,optimizer=optimizer,parameters=len(coefficient),
        actual_optimizer_fits=1,retention=retention,
        metrics=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],risk,reference))

def representation_retention(model,prepared):
    """Fit reference-risk decoders on fit rows; score only unseen validation rows."""
    train,valid=prepared['train'],prepared['valid'];active=np.r_[train,valid]
    full_tokens=[];pooled=[]
    model.eval()
    with torch.no_grad():
        for j in range(0,len(active),16):
            idx=torch.as_tensor(active[j:j+16],device=model.device)
            output=model.omics(**{k:v[idx] for k,v in model.inputs.items()})
            full_tokens.append(output['tokens'].flatten(1).cpu().numpy())
            pooled.append(output['embedding'].cpu().numpy())
    target=prepared['full_risks']['R'];y=target[train];truth=target[valid];out={}
    for name,blocks in [('complete_tokens',full_tokens),('pooled32',pooled)]:
        x=np.concatenate(blocks).astype(float);a,b=x[:len(train)],x[len(train):]
        mean=a.mean(0);sd=a.std(0);sd=np.where(sd>1e-8,sd,1.)
        a=(a-mean)/sd;b=(b-mean)/sd
        # Dual ridge avoids a features-by-features allocation for token matrices.
        coef=a.T@np.linalg.solve(a@a.T+np.eye(len(train)),y-y.mean())
        prediction=b@coef+y.mean();den=float(np.sum((truth-truth.mean())**2))
        out[name]=dict(dimension=a.shape[1],Pearson=float(np.corrcoef(prediction,truth)[0,1]) if prediction.std()>0 else None,
            R2=float(1.-np.sum((prediction-truth)**2)/den) if den>0 else None,
            target='current_train_fitted_RNA512_reference_unseen_validation',decoder_lambda=1.,decoder_fit_rows=len(train))
    return out

def run_unit(args,assets,config,raw,member,lineage):
    prepared=prepare(args,raw,member,lineage)
    label=f'{args.stage}_outer{args.outer_fold}_inner{args.partition}_{args.arm}_seed{args.seed}'
    cell=args.private/'cells'/label;done=cell/'COMPLETE.json';artifact=cell/'model.pt'
    if done.exists():
        previous=json.loads(done.read_text())
        if not args.resume:raise ValueError('Completed unit exists; use explicit --resume to verify')
        if previous['lineage']!=lineage or previous['artifact_sha256']!=sha256(artifact):raise ValueError('Completed unit changed')
        return
    cell.mkdir(parents=True,exist_ok=True)
    with (cell/'START.json').open('x') as handle:json.dump(dict(lineage=lineage,pid=os.getpid(),start=time.time()),handle)
    start=time.monotonic()
    try:
        convex=args.arm in ('P0','F0','F1','O0','O1') or args.arm.startswith('O2_')
        if convex:
            result=convex_unit(args,prepared,raw)
            metrics=result['metrics'];params=result['parameters'];cost=0;epochs=0
        else:
            torch.manual_seed(args.seed);np.random.seed(args.seed)
            cache_key='patch_rand' if args.arm=='P7_RAND' else 'patch_diverse' if args.arm=='P7_DIVERSE' else 'patch_grid' if args.arm=='P7_GRID' else 'patch_uni2' if args.arm.endswith('_UNI2') else 'patch_regions'
            cache=read_patch_cache(assets[cache_key]);bank=PatchBank(cache,args.device)
            model=MechanismModel(args.stage,args.arm,prepared,member,bank,args.device,
                gamma=config[args.stage].get('gamma',.1),intervention='wrong_query' if args.arm=='F5_WRONG_QUERY' else None)
            train,valid=prepared['train'],prepared['valid']
            strata=np.array([str(raw['cancer'][i])+'|'+str(raw['clinical_raw']['AJCC_PATHOLOGIC_TUMOR_STAGE'][i])+'|'+str(raw['platform'][i]) for i in range(512)])
            pairing={}
            for name,part in [('train',train),('validation',valid)]:
                mapping,counts=patient_mismatch(part,strata,270077+args.partition+(name=='validation'))
                model.query_map[part]=mapping[part];pairing[name]=counts
            result=train_neural(model,raw,prepared,epochs=config['training']['epochs'],batch_size=config['training']['batch_size'],
                lr=config['training']['learning_rate'],evaluate_epochs=tuple(config['training']['evaluation_epochs']))
            # All fixed checkpoints are retained. No outer scores select them.
            metrics=result['snapshots'][config['training']['epochs']]['metrics'];params=result['parameters'];cost=result['peak_GPU_bytes'];epochs=result['epochs']
            model.training_query_mean()
            controls={}
            if args.stage=='R3':
                for control in ('wrong_query','mean_query','whole_omics','attention_permutation','interaction_off','direct_off'):
                    risk,delta=predict(model,valid,config['training']['batch_size'],control)
                    controls[control]=dict(risk=risk,delta=delta,metrics=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],risk,prepared['base'][valid]))
            result['frozen_controls']=controls;result['pairing_audit']=pairing
            result['active_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
            if model.omics is not None:result['omics_parameter_audit']=model.omics.parameter_audit()
            if args.stage=='R2':result['retention']=representation_retention(model,prepared)
        result.update(lineage=lineage,train=prepared['train'],valid=prepared['valid'],seed=args.seed,arm=args.arm,stage=args.stage,
            prototype=True,outer_test_evaluated=False)
        torch_write(artifact,result)
        public=dict(status='COMPLETE',lineage=lineage,stage=args.stage,arm=args.arm,outer_fold=args.outer_fold,
            inner_partition=args.partition,seed=args.seed,metrics=metrics,parameters=params,peak_GPU_bytes=cost,
            epochs=epochs,actual_optimizer_fits=result.get('actual_optimizer_fits',1),
            elapsed_seconds=time.monotonic()-start,artifact_sha256=sha256(artifact),
            train_n=len(prepared['train']),valid_n=len(prepared['valid']),outer_test_evaluated=False)
        if 'curves' in result:public['curves']=result['curves']
        if 'retention' in result:public['retention']=result['retention']
        if 'frozen_controls' in result:public['frozen_controls']={k:v['metrics'] for k,v in result['frozen_controls'].items()}
        write(done,public);print(json.dumps({k:public[k] for k in ('status','stage','arm','inner_partition','seed','elapsed_seconds')}),flush=True)
    except Exception as exc:
        write(cell/'FAILED.json',dict(status='FAILED',lineage=lineage,type=type(exc).__name__,message=str(exc)));raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',required=True,type=Path);parser.add_argument('--private',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--stage',choices=['qualify','prepare','R1','R2','R3'],required=True)
    parser.add_argument('--partition',type=int,default=0);parser.add_argument('--outer-fold',type=int,default=1)
    parser.add_argument('--arm',default='P0');parser.add_argument('--seed',type=int,default=31)
    parser.add_argument('--device',default='cuda');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--max-workers',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(2)
    assets,config,raw,member,lineage=load_inputs(args)
    if args.dry_run:
        print(json.dumps(dict(status='PASS_DRY_RUN_NO_FIT',patients=len(raw['t']),events=int(raw['e'].sum()),pathways=member.shape[0])));return
    if args.stage=='qualify':qualifier(args,assets,raw,member,config)
    elif args.stage=='prepare':prepare(args,raw,member,lineage)
    else:run_unit(args,assets,config,raw,member,lineage)

if __name__=='__main__':main()
