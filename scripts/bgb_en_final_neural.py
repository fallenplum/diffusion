"""Fit one within-outer-selected interaction with saved common anchors."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.bgb_en_run import load_inputs,write,torch_write
from scripts.bgb_en_outer import validate_selection_lock,source_lineage_hash
from scripts.bgb_en_nested import select_receipts
from src.bgb_en.data import sha256,read_patch_cache,outer_partition
from src.bgb_en.protocol import array_hash,patient_mismatch
from src.bgb_en.replay import apply_saved_preprocessing,replay_full_risks
from src.bgb_en.training import PatchBank,MechanismModel,train_neural,predict
from src.bgb_en.survival import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    p.add_argument('--primary-private',type=Path,required=True)
    p.add_argument('--selection-lock',type=Path,required=True);p.add_argument('--fold-lock',type=Path,required=True)
    p.add_argument('--inner-report',type=Path,required=True);p.add_argument('--fold',type=int,choices=range(1,6),required=True)
    p.add_argument('--seed',type=int,choices=[31,47,83],required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--config',type=Path,default=ROOT/'configs/bgb_en/prototype_v1.json');p.add_argument('--resume',action='store_true')
    args=p.parse_args();torch.set_num_threads(2)
    assets,config,raw,member,lineage=load_inputs(args)
    lock=json.loads(args.selection_lock.read_text());inner=json.loads(args.inner_report.read_text());selected=json.loads(args.fold_lock.read_text())
    validate_selection_lock(lock,inner,lineage,config,sha256(args.config),sha256(args.inner_report),args.fold)
    if ('NESTED_INTERACTION' not in lock['finalists'] or selected['status']!='LOCKED_WITHIN_OUTER_TRAIN_BEFORE_TEST'
        or selected['outer_fold']!=args.fold or selected['outer_test_evaluated'] is not False
        or selected['selected_arm'] not in config['selection']['nested_interaction_candidates']
        or selected['source_lineage_sha256']!=source_lineage_hash(lineage)
        or selected['selection_lock_sha256']!=sha256(args.selection_lock)
        or selected['inner_report_sha256']!=sha256(args.inner_report)
        or selected['source_sha256']!=sha256(ROOT/'scripts/bgb_en_nested.py')
        or selected['selected_epochs']!=80 or selected['selected_gamma']!=.1):raise ValueError('Per-outer selection lock failed')
    inner_fits=args.primary_private if args.fold==1 else args.private/'nested_inner'
    chosen,candidates,receipts=select_receipts(inner_fits,args.fold,config,lineage)
    if chosen['arm']!=selected['selected_arm'] or candidates!=selected['candidates'] or receipts!=selected['receipts']:
        raise ValueError('Per-outer candidates or their saved source artifacts changed')
    baseline_dir=args.private/'outer'/f'fold{args.fold}';receipt=json.loads((baseline_dir/'COMPLETE.json').read_text())
    baseline_path=baseline_dir/'models.pt'
    if (receipt['contract']['selection_lock_sha256']!=sha256(args.selection_lock)
        or receipt['contract']['source_lineage']!=lineage or receipt['artifact_sha256']!=sha256(baseline_path)):
        raise ValueError('The shared final baseline must already be complete and unchanged')
    train,test=outer_partition(raw['folds'],args.fold)
    contract=dict(selection_lock_sha256=sha256(args.selection_lock),fold_lock_sha256=sha256(args.fold_lock),
        baseline_artifact_sha256=sha256(baseline_path),core_lineage=lineage,
        additional_sources={name:sha256(ROOT/name) for name in ('src/bgb_en/replay.py','scripts/bgb_en_final_neural.py')},
        train_hash=array_hash(train),test_hash=array_hash(test),seed=args.seed,arm=selected['selected_arm'])
    directory=args.private/'neural'/f'outer{args.fold}_seed{args.seed}';done=directory/'COMPLETE.json';artifact=directory/'model.pt'
    if done.exists():
        old=json.loads(done.read_text())
        if not args.resume or old['contract']!=contract or old['artifact_sha256']!=sha256(artifact):raise ValueError('Final neural reuse lineage changed')
        return
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'START.json').open('x') as handle:json.dump(contract,handle)
    start=time.monotonic()
    try:
        common=torch.load(baseline_path,map_location='cpu',weights_only=False)
        if not np.array_equal(common['train'],train) or not np.array_equal(common['test'],test):raise ValueError('Saved outer split changed')
        features=apply_saved_preprocessing(raw,common['full_preprocessing'])
        full=replay_full_risks(features,common['full_models'])
        if max(float(np.max(np.abs(full[key]-common['full_risks'][key]))) for key in full)>2e-12:
            raise ValueError('No-refit common anchor replay failed')
        calibrated=common['teachers'][args.seed]
        prepared=dict(features=features,train=train,valid=test,CP=calibrated['CP'],RNA=calibrated['RNA'],
            base=calibrated['base'],CS=common['seed31_CS_risk'],beta=calibrated['beta'],full_risks=full)
        bank=PatchBank(read_patch_cache(assets['patch_regions']),args.device)
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        model=MechanismModel('R3',selected['selected_arm'],prepared,member,bank,args.device,gamma=.1)
        strata=np.array([str(raw['cancer'][i])+'|'+str(raw['clinical_raw']['AJCC_PATHOLOGIC_TUMOR_STAGE'][i])+'|'+str(raw['platform'][i]) for i in range(512)])
        for name,part in [('train',train),('validation',test)]:
            mapping,_=patient_mismatch(part,strata,270077+args.fold*100+(name=='validation'));model.query_map[part]=mapping[part]
        # Final training has no test-set evaluation or epoch selection while
        # weights are updated. Exactly the locked epoch budget is run first.
        from src.bgb_en.survival import cached_full_risk_step
        batches=[torch.as_tensor(train[j:j+8],device=args.device) for j in range(0,len(train),8)]
        optimizer=torch.optim.Adam([value for value in model.parameters() if value.requires_grad],lr=.001)
        torch.cuda.reset_peak_memory_stats() if str(args.device).startswith('cuda') else None
        curves=[]
        for epoch in range(1,81):
            step=cached_full_risk_step(model,batches,raw['t'][train],raw['e'][train],optimizer,args.device)
            if epoch in (20,40,80):curves.append(dict(epoch=epoch,**step,outer_test_read=False))
        risk,delta=predict(model,test,8);model.training_query_mean();metrics={}
        references={'CP_STRONG':common['full_risks']['LOCAL'][test],'CP_SCALAR':calibrated['base'][test]}
        for name,ref in references.items():metrics[name]=evaluate(raw['t'][train],raw['e'][train],raw['t'][test],raw['e'][test],risk,ref)
        ptrain=raw['pfi_valid'][train];ptest=raw['pfi_valid'][test]
        pfi=evaluate(raw['pfi_t'][train][ptrain],raw['pfi_e'][train][ptrain],raw['pfi_t'][test][ptest],raw['pfi_e'][test][ptest],risk[ptest])
        controls={}
        for control in ('wrong_query','mean_query','whole_omics','attention_permutation','interaction_off','direct_off'):
            cr,cd=predict(model,test,8,control)
            controls[control]=dict(risk=cr,delta=cd,metrics=evaluate(raw['t'][train],raw['e'][train],raw['t'][test],raw['e'][test],risk,cr))
        result=dict(contract=contract,train=train,test=test,final_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            risk=risk,delta=delta,metrics=metrics,PFI_frozen_DSS_risk=pfi,frozen_controls=controls,curves=curves,
            epochs=80,optimizer_updates=80,outer_evaluations_during_training=0)
        torch_write(artifact,result)
        write(done,dict(status='COMPLETE_OUTER_NEURAL',contract=contract,fold=args.fold,seed=args.seed,selected_arm=selected['selected_arm'],
            epochs=80,actual_new_optimizer_fits=1,actual_new_preprocessing_fits=0,metrics=metrics,PFI_frozen_DSS_risk=pfi,
            frozen_controls={key:value['metrics'] for key,value in controls.items()},
            parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            peak_GPU_bytes=torch.cuda.max_memory_allocated() if str(args.device).startswith('cuda') else 0,
            elapsed_seconds=time.monotonic()-start,artifact_sha256=sha256(artifact),outer_test_read_for_selection=False,
            delta_SD=float(delta.std()),curves=curves))
        print(json.dumps(dict(status='COMPLETE_OUTER_NEURAL',fold=args.fold,seed=args.seed,selected_arm=selected['selected_arm'])))
    except Exception as exc:
        write(directory/'FAILED.json',dict(status='FAILED',error_type=type(exc).__name__));raise


if __name__=='__main__':main()
