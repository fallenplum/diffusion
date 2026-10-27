"""Create a separate dependency gate after independently certified F7 recovery."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bgb_en_recovery_audit import verify_recovery_receipt,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    certificate=verify_recovery_receipt(args.receipt,args.private,config_path=ROOT/'configs/bgb_en/prototype_v1.json')
    destination=args.output_dir.resolve()
    if destination.name!='recovered_core_gate' or destination.parent!=args.private.parent.resolve():
        raise ValueError('Gate must stay in the explicitly named BG-BEN private parent')
    destination.mkdir(exist_ok=False)
    for stage in ('R1','R2'):
        original=args.private/(stage+'_SCHEDULE.json')
        if json.loads(original.read_text())['status']!='COMPLETE':raise ValueError('Earlier stage is incomplete')
        (destination/original.name).write_bytes(original.read_bytes())
    value=dict(status='COMPLETE',unique_units=72,recovery_receipt_sha256=sha256(args.receipt),
               original_R3_schedule_sha256=sha256(args.private/'R3_SCHEDULE.json'),
               original_schedule_preserved=True,scope='dependency gate after actual certified F7 replacement fits')
    (destination/'R3_SCHEDULE.json').write_text(json.dumps(value,indent=2)+'\n')
    value=dict(status='COMPLETE_SEPARATE_RECOVERED_DEPENDENCY_GATE',recovery_receipt_sha256=sha256(args.receipt),
               core_unique_cells=certificate['core_unique_cells'],original_schedules_modified=False,
               script_sha256=sha256(__file__),new_fits=0)
    (destination/'SCHEDULER_RECEIPT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
