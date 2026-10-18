"""Two-step real GPU qualification; discard short model arrays and weights."""
from pathlib import Path
import sys
import json
from unittest.mock import patch
import contextlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase26_multiquery import data, runtime, train


def blocked(*args,**kwargs):raise RuntimeError('qualification forbids endpoint/preprocess/main fits')


def main():
    identity=runtime.source();target=runtime.PRIVATE/'REAL_QUALIFICATION_PRIVATE.json'
    if target.exists():raise ValueError('qualification already exists; no silent rerun')
    with contextlib.ExitStack() as stack:
        from src.phase25_local import data as old
        for obj,attr in ((old.parent,'endpoints'),(old.parent,'splits'),(old,'prepare'),(old,'fit_ridge'),
                         (old,'fit_preprocessor'),(old.PCA,'fit'),(data,'prepare')):
            stack.enter_context(patch.object(obj,attr,blocked))
        ds=data.Data(1,31,device='cuda');rows=[]
        for arm in runtime.NEURAL_ARMS:
            fingerprints=[]
            for repeat in range(2):
                model,initial,history,audit,counts=train.train_arm(ds,arm,31,1,epochs=2)
                with torch.no_grad():
                    risk,unit,_=train._forward(model,ds,ds.valid)
                    off,_,_=train._forward(model,ds,ds.valid,intervention='OFF')
                if not np.array_equal(off.cpu().numpy(),ds.R0_valid):raise ValueError('OFF failed exact main')
                if unit.abs().max()>1+1e-7:raise ValueError('hard bound failed')
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                fingerprint={'state':state,'risk':risk.detach().cpu().clone(),
                             'history':[{k:v for k,v in x.items() if k!='epoch_seconds'} for x in history]}
                fingerprints.append(fingerprint)
                del model,initial,history,audit
            a,b=fingerprints
            if not all(torch.equal(a['state'][k],b['state'][k]) for k in a['state']) or not torch.equal(a['risk'],b['risk']) or a['history']!=b['history']:
                raise ValueError('same source/seed real GPU nondeterministic')
            rows.append({'arm':arm,'repeat_models':2,'epochs_each':2,'exact_state_risk_history_repeat':True,
                         'allocated_parameters':counts['allocated_trainable']})
            del fingerprints;torch.cuda.empty_cache()
    if identity!=runtime.source():raise ValueError('qualification source drift')
    result={'status':'PASS','source':identity,'real_GPU':torch.cuda.get_device_name(),'short_models':6,
            'optimizer_steps':12,'short_arrays_or_weights_saved':False,'formal_training_started':False,
            'heldout_endpoint_API_calls':0,'new_convex_fits':0,'rows':rows}
    runtime.atomic_json(target,result)
    runtime.atomic_json(runtime.REPORT/'REAL_QUALIFICATION.json',result)
    print(json.dumps({'status':'PASS','source':identity,'short_models':6,'optimizer_steps':12}),flush=True)


if __name__=='__main__':main()
