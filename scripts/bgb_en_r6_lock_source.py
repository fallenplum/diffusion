"""Create the finite source-only R6 lock after the complete R5 decision gate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r6_pretrain import FIXED_DEFAULTS, current_code_hashes, validate_lock


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_cells():
    cells=[]
    groups=[('S0','MATCHED','PRIMARY'),('S1','MATCHED','PRIMARY'),
        ('S2','MATCHED','PRIMARY'),('S3','MATCHED','PRIMARY'),
        ('S4','MATCHED','PRIMARY'),('S5','WRONG_DONOR','PRIMARY'),
        ('S4','WITHIN_SECTION_TISSUE_DEPTH','PRIMARY'),('S4','RANDOM_GATE','PRIMARY'),
        ('S6','MATCHED','SPATIAL'),('S6','WRONG_NEIGHBOR','SPATIAL'),
        ('S7','MATCHED','SPOT_ALIGNMENT'),('S7','WRONG_SPOT_ALIGNMENT','SPOT_ALIGNMENT')]
    for arm,control,phase in groups:
        for heldout in range(6):
            for seed in (31,47):
                cells.append(dict(arm=arm,control=control,heldout_donor=heldout,
                                  seed=seed,schedule_phase=phase))
    for control in ('MATCHED','WRONG_DONOR','WITHIN_SECTION_TISSUE_DEPTH'):
        cells.append(dict(arm='S7',control=control,heldout_donor=-1,seed=31,
                          schedule_phase='FULL_TRANSFER'))
    assert len(cells)==147 and sum(c['arm']!='S0' for c in cells)==135
    return cells


def exclusive_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('PREEXISTING_R6_LOCK_OR_GATE_PRESERVED')
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--r5-report-root',type=Path,required=True)
    p.add_argument('--source-cache',type=Path,required=True)
    p.add_argument('--fresh-certificate',type=Path,required=True)
    p.add_argument('--private-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'configs/bgb_en/r6_source_v1.json')
    args=p.parse_args()
    r5=args.r5_report_root
    names=('decision.json','R5_REPORT_AUDIT.json','R5_CHECKPOINT_REPLAY_AUDIT.json','PROTOCOL_LOCK.json')
    values={n:json.loads((r5/n).read_text()) for n in names}
    decision,audit,replay,lock=(values[n] for n in names)
    if decision.get('decision') not in ('R5_READOUT_STOP_R6_ELIGIBLE','R5_PERFORMANCE_ONLY_NOT_MECHANISM'):
        raise ValueError('R5_GATE_DOES_NOT_AUTHORIZE_AUTOMATIC_R6_FORMAL_START')
    if not (decision.get('R6_formal_start_eligible') is True
        and audit.get('status')=='COMPLETE_REQUIRED_R5_TRAINING_AND_REPLAY'
        and audit.get('qualification_passed') and audit.get('replay_passed')
        and replay.get('status')=='PASS_COMPLETE_R5_CHECKPOINT_REPLAY'
        and replay.get('passed_units')==108 and replay.get('failed_units')==0
        and replay.get('main_passed')==63 and replay.get('mechanism_passed')==45
        and replay.get('contract_sha256')==audit.get('contract_sha256')==lock.get('contract_sha256')):
        raise ValueError('COMPLETE_63_PLUS_45_ACTUAL_R5_REPLAY_REQUIRED')
    cache_manifest=args.source_cache/'manifest.private.json'
    manifest=json.loads(cache_manifest.read_text())
    fresh=sha(args.fresh_certificate)
    if manifest['source_audit']['fresh_certificate_sha256']!=fresh:
        raise ValueError('SOURCE_CACHE_NOT_BOUND_TO_FRESH_REAL_ALIGNMENT')
    config=dict(schema='bgb_en_r6_source_pretrain_lock_v1',status='LOCKED_BEFORE_SOURCE_FITS',
        upstream_commit='ecd273d2036d98e40dd9e997dd45e8de9eba7947',
        source_code_sha256=current_code_hashes(),fixed_defaults=dict(FIXED_DEFAULTS),
        source_cache_manifest_sha256=sha(cache_manifest),fresh_certificate_sha256=fresh,
        cells=build_cells(),maximum_source_workers=8,logical_LODO_units=144,optimized_LODO_units=132,
        full_source_optimized_units=3,actual_optimized_source_units=135,
        source_ST_test_sections_used=False,source_DSS_or_PFI_loaded=False,
        transfer_family='S7_fixed_a_priori_no_source_or_DSS_score_selection',
        full_source_seed=31,source_gate_affects_claims_not_data_qualified_matrix_execution=True,
        source_development_donors_reused_not_new_external_confirmation=True)
    validate_lock(config)
    gate=dict(schema='bgb_en_r6_source_launch_gate_v1',r6_formal_training_allowed=True,
        R5_decision=decision['decision'],R5_contract_sha256=lock['contract_sha256'],
        R5_public_evidence_sha256={n:sha(r5/n) for n in names},
        source_data_alignment_qualified=True,TCGA_labels_loaded_by_gate_builder=False,
        source_code_and_schedule_locked_before_formal_fits=True)
    exclusive_json(args.output,config)
    exclusive_json(args.private_root/'SOURCE_LAUNCH_GATE.private.json',gate)
    print(json.dumps(dict(status='LOCKED_BEFORE_SOURCE_FITS',
        config_sha256=sha(args.output),gate_sha256=sha(args.private_root/'SOURCE_LAUNCH_GATE.private.json'),
        source_logical_units=147,source_actual_planned_fits=135)))


if __name__=='__main__':main()
