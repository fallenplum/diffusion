"""Exact risk-pair repair/damage from saved final-checkpoint interventions."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.bgb_en.data import load_raw,sha256
from src.bgb_en.survival import evaluate
from scripts.bgb_en_recovery_audit import verify_recovery_receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',required=True,type=Path);p.add_argument('--assets',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/bgb_en/prototype_v1.json')
    p.add_argument('--recovery-receipt',type=Path,help='independent F7 recovery receipt; old partial schedule remains unchanged')
    args=p.parse_args()
    recovery=None
    if args.recovery_receipt is not None:
        recovery=verify_recovery_receipt(args.recovery_receipt,args.private,config_path=args.config)
    raw=load_raw(json.loads(args.assets.read_text()));rows=[];audited=0
    for receipt_path in sorted((args.private/'cells').glob('R3_*/COMPLETE.json')):
        receipt=json.loads(receipt_path.read_text());artifact=receipt_path.parent/'model.pt'
        if sha256(artifact)!=receipt['artifact_sha256']:raise ValueError('Private trained artifact changed')
        model=torch.load(artifact,map_location='cpu',weights_only=False)
        if 'frozen_controls' not in model:continue
        train,valid=model['train'],model['valid'];full=model['snapshots'][80]['risk']
        for control,value in model['frozen_controls'].items():
            counterfactual=np.asarray(value['risk'])
            full_metric=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],full,counterfactual)
            control_metric=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],counterfactual)
            dh=full_metric['Harrell']-control_metric['Harrell'];du=full_metric['Uno60']-control_metric['Uno60']
            if abs(full_metric['repair']-full_metric['damage']-dh)>1e-12 or abs(full_metric['Uno_repair']-full_metric['Uno_damage']-du)>1e-12:
                raise ValueError('Repair minus damage does not equal concordance difference')
            rows.append(dict(arm=receipt['arm'],control=control,seed=receipt['seed'],inner_partition=receipt['inner_partition'],
                Harrell_full=full_metric['Harrell'],Harrell_control=control_metric['Harrell'],delta_Harrell=dh,
                Uno60_full=full_metric['Uno60'],Uno60_control=control_metric['Uno60'],delta_Uno60=du,
                repair=full_metric['repair'],damage=full_metric['damage'],Uno_repair=full_metric['Uno_repair'],Uno_damage=full_metric['Uno_damage'],
                repair_num=full_metric['repair']*full_metric['Harrell_den'],damage_num=full_metric['damage']*full_metric['Harrell_den'],
                Uno_repair_num=full_metric['Uno_repair']*full_metric['Uno_den'],Uno_damage_num=full_metric['Uno_damage']*full_metric['Uno_den'],
                Harrell_den=full_metric['Harrell_den'],Uno_den=full_metric['Uno_den'],
                risk_shift_sd_ratio=full_metric['risk_shift_sd_ratio'],pair_reference='full_trained_checkpoint_vs_frozen_control',
                intervention_not_real_retraining=True))
        audited+=1
    schedule_path=args.private/'R3_SCHEDULE.json'
    schedule=json.loads(schedule_path.read_text()) if schedule_path.exists() else {}
    result=dict(schema='bgb_en_mechanism_pair_replay_v1',status='COMPLETE_AVAILABLE_CHECKPOINTS',
        audited_final_neural_checkpoints=audited,comparisons=rows,optimizer_steps=0,
        whole_matrix_complete=recovery['core_matrix_complete'] if recovery is not None else schedule.get('status')=='COMPLETE',
        original_R3_schedule_status=schedule.get('status'),original_schedule_rewritten=False,
        recovery_receipt_sha256=sha256(args.recovery_receipt) if recovery is not None else None,
        recovery_audit_source_sha256=sha256(ROOT/'scripts/bgb_en_recovery_audit.py'),
        recovered_core_unique_cells=recovery['core_unique_cells'] if recovery is not None else None,
        repaired_F7_complete_receipts_verified=len(recovery['repaired_cells']) if recovery is not None else 0,
        scientific_scope='saved_risk_dependence_not_retraining_necessity_or_independent_confirmation')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],checkpoints=audited,comparisons=len(rows))))

if __name__=='__main__':main()
