"""Two deterministic in-memory short fits; never save short-model arrays."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def fingerprint(model,history,record):
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
    for row in history:
        h.update(json.dumps({k:v for k,v in row.items() if k!='epoch_seconds'},sort_keys=True).encode())
    for name in ('fit_risk','risk'):
        h.update(record[name].tobytes())
    for name,v in sorted(record['sensitivities'].items()):h.update(name.encode());h.update(v.tobytes())
    return h.hexdigest()


def main():
    import torch
    from src.phase25_local import data,train
    from src.phase25_local.runtime import source,PRIVATE,REPORT,atomic_json,NEURAL_ARMS
    from src.phase13_rna import data as parent
    identity=source();path=PRIVATE/'REAL_QUALIFICATION_PRIVATE.json'
    if path.exists():raise ValueError('qualification already recorded; review receipt instead of rerunning')
    started=time.monotonic();ds=data.Data(1,'prototype',31,device='cuda')
    before=ds.base.clone();raw=data.read_prepared(1,'prototype')[0]
    forbidden={'valid_time','valid_event','test_time','test_event','DSS_time','DSS_event','endpoint','endpoints'}
    if forbidden.intersection(vars(ds)) or forbidden.intersection(raw):raise ValueError('heldout endpoint-bearing field escaped')
    torch.cuda.reset_peak_memory_stats();runs=[]
    def blocked(*a,**k):raise RuntimeError('endpoint reader/new convex fit called during neural fitting')
    # Raw preparation is independently verified. The real training interface
    # keeps only fit labels; this is not falsely called a raw-file label poison.
    with patch.object(parent,'endpoints',side_effect=blocked),patch.object(parent,'load',side_effect=blocked),patch.object(data,'fit_ridge',side_effect=blocked):
        for repeat in range(2):
            hashes={}
            for arm in NEURAL_ARMS:
                model,_,history,audit=train.train_arm(ds,arm,31,1,epochs=2)
                result=train.predict_arm(model,ds)
                if audit['heldout_endpoint_reads']!=0 or not torch.equal(before,ds.base):raise ValueError('isolation/main drift')
                hashes[arm]=fingerprint(model,history,result)
                del model,result;gc.collect()
            runs.append(hashes)
    if runs[0]!=runs[1] or source()!=identity:raise ValueError('deterministic short-fit/source parity failed')
    record={'status':'PASS','source':identity,'short_models':6,'epochs_each':2,'optimizer_steps':12,
        'repeat_hashes_identical':True,'all_three_arms_qualified':True,
        'heldout_endpoints_absent_from_training_view':True,'endpoint_readers_blocked_during_training':True,
        'new_convex_fit_blocked_during_training':True,
        'raw_file_label_poison_experiment_claimed':False,
        'scope':'actual_prepared_inputs_fit_only_training_interface_and_repeatability;raw_teacher_isolation_covered_by_data_tests_and_cache_receipts',
        'main_offset_unchanged':True,'new_validation_selection':False,
        'short_weights_or_arrays_saved':False,'raw_legacy_endpoint_bytes_never_loaded':False,
        'seconds':time.monotonic()-started,'peak_GPU_reserved_bytes':int(torch.cuda.max_memory_reserved()),
        'versions':{'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda},
        'patient_arrays_or_weights_public':False}
    atomic_json(path,record);atomic_json(REPORT/'REAL_QUALIFICATION.json',record)
    print(json.dumps({'status':'PASS','source':identity,'short_models':6,'updates':12,'seconds':record['seconds']}),flush=True)


if __name__=='__main__':main()
