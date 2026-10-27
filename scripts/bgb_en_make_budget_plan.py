"""Create the finite O3/O4 160/320 plan; no optimizer or model selection."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_run import write
from src.bgb_en.data import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True,type=Path);p.add_argument('--core-private',required=True,type=Path)
    p.add_argument('--private',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--public-output',required=True,type=Path)
    args=p.parse_args();tasks=[]
    for epochs in (320,160):
        for arm in ('O4','O3'):
            for part in range(3):
                for seed in (31,47):
                    tasks.append(dict(label=f'budget_{arm}_e{epochs}_p{part}_s{seed}',script='scripts/bgb_en_budget_retrain.py',
                        args=[str(x) for x in ['--stage','train','--assets',args.assets,'--primary-private',args.core_private,
                            '--private',args.private,'--arm',arm,'--partition',part,'--seed',seed,'--epochs',epochs,'--device','cuda']]))
    value=dict(status='LOCKED_BEFORE_ADDITIONAL_FITS',outer_test_evaluated=False,tasks=tasks,
        config_sha256=sha256(ROOT/'configs/bgb_en/budget_sensitivity_v1.json'),
        driver_sha256=sha256(ROOT/'scripts/bgb_en_budget_retrain.py'),R4_epochs=80,new_neural_fits=24,new_optimizer_updates=5760)
    if args.output.exists() and json.loads(args.output.read_text())!=value:raise ValueError('Budget plan changed')
    if not args.output.exists():write(args.output,value)
    public={key:v for key,v in value.items() if key!='tasks'}
    public.update(plan_sha256=sha256(args.output),planned_units=[task['label'] for task in tasks],
        interpretation='new same-seed fits; report both endpoints; no best-budget choice; source teachers reused',
        initialization='fresh_model_and_fresh_Adam_no_optimizer_resume')
    if args.public_output.exists() and json.loads(args.public_output.read_text())!=public:raise ValueError('Budget public lock changed')
    if not args.public_output.exists():write(args.public_output,public)
    print(json.dumps(dict(status=value['status'],planned_units=len(tasks))))


if __name__=='__main__':main()
