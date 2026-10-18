"""Final fit/resource accounting from immutable receipts; no endpoint access."""
from pathlib import Path
import sys
import json
import platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import scipy
import sklearn
import torch
from src.phase26_multiquery import data,runtime


def main():
    identity=runtime.source();target=runtime.REPORT/'RUN_AUDIT.json'
    if target.exists():raise ValueError('append-only accounting')
    scheduler=json.loads((runtime.PRIVATE/'SCHEDULER_PRIVATE.json').read_text())
    if scheduler['status']!='COMPLETE' or scheduler['source']!=identity or scheduler['failures']:
        raise ValueError('scheduler incomplete')
    cells=[];steps=0;epochs=0
    for fold in (1,2,3):
        for seed in (31,47):
            folder=runtime.cell_dir(seed,fold);done=runtime.verify_done(folder,identity,seed,fold)
            record=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
            timing={}
            for arm in runtime.NEURAL_ARMS:
                saved=record['arms'][arm];history=saved['history'];audit=saved['training_audit']
                assert len(history)==audit['actual_epochs']==audit['optimizer_steps']==80
                steps+=audit['optimizer_steps'];epochs+=len(history)
                timing[arm]=sum(r['epoch_seconds'] for r in history)
            cells.append({'seed':seed,'fold':fold,'wall_seconds':done['wall_seconds'],'training_seconds':timing,
                          'checkpoint_sha256':done['checkpoint_sha256']})
    main_rows=[];gradients=[]
    for fold in (1,2,3):
        saved,receipt=data.read_prepared(fold);state=saved['mainstate']
        optimizers=[state['full_RNA_optimizer']]
        for teacher in state['teachers']:optimizers.extend((teacher['CP_optimizer'],teacher['RNA_optimizer']))
        assert len(optimizers)==7
        gradients.extend(float(v['gradient_max']) for v in optimizers)
        main_rows.append({'fold':fold,'new_convex_fits':7,'reused_full_CP':1,
            'CP_parameters':saved['audit']['CP_parameters'],'RNA_parameters':512,
            'main_distribution':saved['audit']['main_distribution'],'prepared_sha256':receipt['sha256'],
            'maximum_saved_optimizer_gradient':max(v['gradient_max'] for v in optimizers)})
    assert steps==epochs==1440 and len(gradients)==21
    qualify=json.loads((runtime.REPORT/'REAL_QUALIFICATION.json').read_text())
    assert qualify['status']=='PASS' and qualify['source']==identity and qualify['optimizer_steps']==12
    value={'status':'COMPLETE_FORMAL_PROTOTYPE_ACCOUNTING','source':identity,'parent_commit':runtime.PARENT,
        'cells':6,'model_slots':24,'formal_neural_fits':18,'formal_optimizer_steps':steps,'formal_epochs':epochs,
        'new_convex_fits':21,'reused_full_CP_models':3,'neural_seeds_share_prepared_main':True,
        'formal_failed_cells':0,'short_qualification_models':6,'short_optimizer_steps':12,
        'short_weights_or_arrays_saved':False,'cell_rows':cells,'fold_main_distribution':main_rows,
        'maximum_saved_convex_optimizer_gradient':max(gradients),'convex_gradients_independently_recomputed_here':False,
        'numerical_module_test_counts':{'model':31,'data':11,'initial_evaluation':27,'training':4},
        'evaluation_guard_hardening_targeted_tests':16,'targeted_tests_partly_overlap_initial_tests':True,
        'training_population_SD_diagnostic_targeted_recheck':1,
        'versions':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
                    'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'gpu_name':torch.cuda.get_device_name(),'GPU_total_MiB':32760,
        'training_point_GPU_used_MiB':11176,'training_point_GPU_utilization_percent':99,
        'continuous_GPU_average_or_peak_claimed':False,
        'MAIN_replay_sha256':runtime.sha(runtime.REPORT/'MAIN_REPLAY_AUDIT.json'),
        'neural_replay_sha256':runtime.sha(runtime.REPORT/'REPLAY_AUDIT.json'),
        'new_endpoint_API_calls_here':0,'new_fits_or_neural_forward_calls_here':0,
        'existing_private_fit_label_bytes_deserialized':True,'outer_started':False,
        'patient_arrays_weights_or_identifiers_public':False}
    runtime.atomic_json(target,value)
    print(json.dumps({'status':value['status'],'formal_neural_fits':18,'formal_optimizer_steps':1440,'new_convex_fits':21}))


if __name__=='__main__':main()
