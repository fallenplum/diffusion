"""Anonymous fixed-head fit diagnostics; no fresh predictors or endpoint APIs."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.phase27_readout import runtime


if __name__=='__main__':
    target=runtime.REPORT/'TRAIN_DIAGNOSTIC.json'
    if target.exists():raise ValueError('append-only diagnostics')
    identity=runtime.source();rows=[];cells=[];feature_rows=[]
    for fold in (1,2,3):
        for seed in (31,47):
            path=runtime.cell_dir(seed,fold);done=runtime.verify_done(path,identity,seed,fold)
            record=torch.load(path/'head.pt',map_location='cpu',weights_only=False)
            for arm,value in record['heads'].items():
                optimizer=value['optimizer']
                rows.append({'seed':seed,'fold':fold,'arm':arm,
                    'parameter_count':16,'optimizer':{key:optimizer[key] for key in
                        ('status','penalty','parameters','objective','cox','gradient_max','iterations','newton_polish_steps')},
                    'optimization_history':optimizer['history'],**value['fit_diagnostics']})
            cells.append({'seed':seed,'fold':fold,'new_convex_fits':3,'head_sha256':done['checkpoint_sha256'],
                          'wall_seconds':done['wall_seconds']})
            feature=json.loads(runtime.feature_path(seed,fold).with_suffix('.json').read_text())
            feature_rows.append({'seed':seed,'fold':fold,'wall_seconds':feature['wall_seconds'],
                'feature_sha256':feature['sha256'],'feature_audit':feature['audit']})
    if runtime.source()!=identity:raise ValueError('source drift')
    result={'status':'COMPLETE_FROZEN_READOUT_FIT_DIAGNOSTICS','source':identity,
        'new_convex_heads':18,'new_neural_steps':0,'main_effect_refits':0,'rank':16,'lambda':10,
        'head_rows':rows,'cell_rows':cells,'feature_rows':feature_rows,'new_fits_or_forward_calls_here':0,
        'new_valid_endpoint_reads_here':0,'training_fit_representation_is_supervised_in_sample':True,
        'optimization_history_is_fit_only_not_validation_selection':True,
        'pair_logistic_loss_is_not_additive_Efron_decomposition':True,'patient_arrays_or_weights_public':False}
    runtime.atomic_json(target,result)
    print(json.dumps({'status':result['status'],'new_heads_referenced':18,'new_fits_here':0}))
