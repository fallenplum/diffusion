"""Six selected B3 models: fixed-weight query dependence, never new fitting."""
import argparse,hashlib,json,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.qpr.gate2_worker import Evidence
from src.qpr.survival_model import FullRelation
from src.qpr.protocol import require_stage
from src.bgb_en.survival import evaluate


def run(a):
    require_stage('gate2',a.reports);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=[]
    for inner in (1,2,3):
        with (Path(a.ram_root)/f'qpr_G2_inner{inner}_20261102.private.pkl').open('rb') as f:data=pickle.load(f)
        store=Evidence(data,'B3');nt=len(data['fit']);va=np.arange(nt,len(store.rows))
        for seed in (31,47):
            cell=Path(a.source)/f'inner{inner}_B3_seed{seed}';meta=json.loads((cell/'COMPLETE.private.json').read_text());epoch=meta['selected_epoch']
            cp=cell/f'epoch{epoch:03d}.private.pt'
            if hashlib.sha256(cp.read_bytes()).hexdigest()!=meta['selected']['checkpoint_sha256']:raise ValueError('SELECTED_CHECKPOINT_CHANGED')
            model=FullRelation(data['clinical'].shape[1],'B3').cuda();model.load_state_dict(torch.load(cp,weights_only=True)['state']);model.stage(epoch);model.eval()
            with torch.no_grad():
                real=store.predict(model,va);store.constant_query(model);constant=store.constant.detach()
                hook=model.query.register_forward_hook(lambda module,inputs,output:constant[None].expand(len(output),-1,-1))
                mean=store.predict(model,va);hook.remove()
                store.arm='B5';wrong=store.predict(model,va);store.arm='B3'
                # Exact decomposition: mean-of-edge value head plus all remaining
                # query readout terms. The latter can still contain main effects.
                vm=(store.v.double()*store.edge_mask[...,None]).sum(1)/store.edge_mask.sum(1)[:,None]
                slide_u=model.unconditional(vm).squeeze(-1)
                uncond=torch.stack([slide_u[store.owner_to_slides[int(o)]].mean() for o in data['evaluation']])
            saved=np.load(cell/f'epoch{epoch:03d}_predictions.private.npz')['evaluation_delta'];error=float(np.max(np.abs(saved-real.cpu().numpy())))
            if error>1e-10:raise ValueError('SELECTED_CHECKPOINT_FROZEN_REPLAY')
            r0=data['r0_evaluation'];args=(data['train_time'],data['train_event'],data['evaluation_time'],data['evaluation_event'])
            own=evaluate(*args,r0+real.cpu().numpy());comparisons={}
            for label,x in [('fit_mean_query',mean),('wrong_RNA_query',wrong)]:
                dr=(x-real).cpu().numpy();metric=evaluate(*args,r0+x.cpu().numpy())
                comparisons[label]={'risk_difference_SD':float(dr.std()),'risk_difference_abs_max':float(np.abs(dr).max()),
                                    'Harrell_change_from_original':metric['Harrell']-own['Harrell'],'Uno60_change_from_original':metric['Uno60']-own['Uno60']}
            rows.append({'inner':inner,'seed':seed,'selected_epoch':epoch,'replay_max_abs':error,
                         'total_new_delta_SD':float(real.std(unbiased=False)),
                         'unconditional_edge_mean_head_SD':float(uncond.std(unbiased=False)),
                         'query_readout_terms_SD':float((real-uncond).std(unbiased=False)),
                         'query_readout_can_contain_additional_main_effects':True,'interventions':comparisons})
            del model;torch.cuda.empty_cache()
        del store;torch.cuda.empty_cache()
    out={'role':'POST_TRAINING_FIXED_WEIGHT_DEPENDENCE_NOT_NEW_HELDOUT_PERFORMANCE_EVIDENCE',
         'new_optimizer_steps':0,'selected_models':6,'outer_test_labels_decoded':0,
         'mean_query_reference':'current_model_train_partition_mean_query_preserving_original_valid_program_mask',
         'wrong_query_reference':'same_B5_training_only_donor_map_original_Clinical_S32_values_and_main_risk_preserved',
         'interpretation':'Equal ordering metrics do not imply identical continuous risks. These are dependence probes, not retraining ablations.',
         'rows':rows}
    Path(a.output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('source','output','reports','ram-root'):p.add_argument('--'+x,required=True)
    run(p.parse_args())
