"""Lock the previously specified factor/order/budget/source sensitivity cells."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_run import write
from src.bgb_en.data import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True,type=Path);p.add_argument('--core-private',required=True,type=Path)
    p.add_argument('--private-root',required=True,type=Path);p.add_argument('--path-private',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--public-output',required=True,type=Path);args=p.parse_args()
    task=[]
    def add(label,script,argv,priority):task.append(dict(label=label,script='scripts/'+script,args=[str(x) for x in argv],priority=priority))
    for variant in ('O8_F2','O8_F8','O5_RANDOM_ORDERS','O6_RANDOM_ORDERS'):
        for part in range(3):
            for seed in (31,47):
                add(f'omics_{variant}_p{part}_s{seed}','bgb_en_omics_sensitivity.py',
                    ['--stage','train','--assets',args.assets,'--core-private',args.core_private,'--private',args.private_root/'omics_sensitivity_v1',
                     '--variant',variant,'--partition',part,'--seed',seed,'--device','cuda'],300 if 'ORDERS' in variant else 80)
    for encoder,budget in [('CONCH',128),('CONCH',1024),('UNI2',128)]:
        for arm in ('P1','P3'):
            for part in range(3):
                for seed in (31,47):
                    add(f'path_{encoder}_{budget}_{arm}_p{part}_s{seed}','bgb_en_path_sensitivity.py',
                        ['--stage','train','--assets',args.assets,'--private',args.path_private,'--primary-private',args.core_private,
                         '--encoder',encoder,'--budget',budget,'--arm',arm,'--partition',part,'--seed',seed,'--device','cuda'],
                        budget*(4 if arm=='P3' else 1))
    projects={'TRUE':args.private_root/'external_v2_unitnorm/fit_06/UNI2_PROJECT_TRANSFER.pt',
              'WRONG_DONOR':args.private_root/'external_v2_WRONG_DONOR/fit_06/UNI2_PROJECT_TRANSFER.pt',
              'SPATIAL_SHUFFLE':args.private_root/'external_v2_SPATIAL_SHUFFLE/fit_06/UNI2_PROJECT_TRANSFER.pt'}
    for control,project in projects.items():
        if not project.is_file():raise ValueError('Qualified source projection is not complete')
        arms=['F5_UNI2','F11_UNI2'] if control=='TRUE' else ['F11_UNI2']
        for arm in arms:
            for part in range(3):
                for seed in (31,47):
                    command=['--assets',args.assets,'--primary-private',args.core_private,'--private',args.private_root/f'auxiliary_{control}_v2',
                        '--arm',arm,'--partition',part,'--seed',seed,'--device','cuda']
                    if arm=='F11_UNI2':command+=['--project',project]
                    add(f'aux_{control}_{arm}_p{part}_s{seed}','bgb_en_aux_run.py',command,64)
    task.sort(key=lambda row:(-row['priority'],row['label']))
    if len(task)!=84:raise ValueError('Additional matrix count drift')
    value=dict(status='LOCKED_BEFORE_ADDITIONAL_FITS',outer_test_evaluated=False,planned_unique_neural_fits=84,
        source_projections={key:sha256(path) for key,path in projects.items()},
        path_sensitivity_lock_sha256=sha256(args.path_private/'PATH_SENSITIVITY_LOCK.json'),
        original_core_source_modified=False,teacher_refits=0,schedule='largest estimated work first; no scientific parameter changes',tasks=task)
    if args.output.exists() and json.loads(args.output.read_text())!=value:raise ValueError('Existing additional plan differs')
    write(args.output,value)
    public=dict(status=value['status'],outer_test_evaluated=False,plan_sha256=sha256(args.output),
        planned_neural_fits=84,groups=dict(omics_factor_and_order=24,pathology_budget_and_encoder=36,external_source_transfer=24),
        external_source_actual_fits_separately=16,teacher_refits=0,path_sensitivity_lock_sha256=value['path_sensitivity_lock_sha256'],
        source_projections=value['source_projections'],tasks=[dict(unit=x['label'],script=x['script']) for x in task],
        plan_generator_sha256=sha256(__file__),explicitly_not_hyperparameter_search=True)
    if args.public_output.exists() and json.loads(args.public_output.read_text())!=public:raise ValueError('Existing anonymous additional lock differs')
    write(args.public_output,public);print(json.dumps(dict(status=value['status'],tasks=len(task))))


if __name__=='__main__':main()
