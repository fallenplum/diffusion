"""Audit the historical-only evaluator repair; no model fitting or replay.

The snapshot is taken before replacing the erroneous evaluator. Qualification
uses the immutable snapshot, actual tests and saved historical/reference data.
Do not repeat in an already qualified release directory.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase20_selective.data import P,R,source,source_files,write
from src.phase20_selective.provenance import require_record

BEFORE='05d081ee25c1bd6fac58781eacf4c7b0f4005e24682d234c84940a362086ded3'
TRAINING='e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548'
ARCHIVE=P/'recovery/history_reference_evaluation'
ENV=dict(os.environ,PYTHONPATH='.:/root/phase10_trimodal_eval_deps',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)

def snapshot():
    require(source()==TRAINING,'training source changed')
    require(not (R/'HISTORICAL_REFERENCE_REPAIR_AUDIT.json').exists(),'historical repair already qualified; do not repeat')
    require(sha(ROOT/'scripts/validate_phase20.py')==BEFORE,'original erroneous evaluator bytes differ')
    ARCHIVE.mkdir(parents=True,exist_ok=True);records={}
    originals={'evaluator_before.py':ROOT/'scripts/validate_phase20.py',
        'math_audit_before.json':R/'EVALUATION_MATH_AUDIT.json',
        'independent_failure_before.log':P/'logs/independent_d2_evaluation.log',
        'pipeline_failure_before.log':P/'logs/evaluation_d2_pipeline.log',
        'pipeline_failed_before.json':P/'EVALUATION_D2_FAILED.json'}
    for name,path in originals.items():
        target=ARCHIVE/name
        require(path.is_file(),'missing original failure evidence: '+name)
        if target.exists():require(sha(target)==sha(path),'existing failure archive differs: '+name)
        else:shutil.copyfile(path,target)
        records[name]={'sha256':sha(path),'original_path':str(path)}
    write(ARCHIVE/'SNAPSHOT.json',{'source':TRAINING,'original_evaluator_sha256':BEFORE,'files':records,
        'original_math_test_sha256':sha(ROOT/'tests/test_phase20_evaluation.py'),
        'training_source_files':source_files(),'repair_lock_sha256':sha(R/'AUDIT_REPAIR_LOCK.json'),
        'training_qualification_sha256':sha(R/'AUDIT_REPAIR_TESTS.json')})
    print('HISTORICAL_REFERENCE_FAILURE_SNAPSHOT_SAVED',flush=True)

def ast_audit():
    before=ast.parse((ARCHIVE/'evaluator_before.py').read_text());after=ast.parse((ROOT/'scripts/validate_phase20.py').read_text())
    functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    a,b=functions(before),functions(after)
    require(set(b)-set(a)=={'historical_reference_identity'} and not set(a)-set(b),'unexpected function changes')
    require(all(ast.dump(a[name])==ast.dump(b[name]) for name in a if name!='collect'),'non-historical evaluator function changed')
    def without_history(node):
        node=copy.deepcopy(node)
        node.body=[q for q in node.body if not (isinstance(q,ast.For) and any(isinstance(n,ast.Name) and n.id=='historic' for n in ast.walk(q)))]
        return ast.dump(node)
    require(without_history(a['collect'])==without_history(b['collect']),'collect changed outside historical reference loop')
    other=lambda tree:[ast.dump(n) for n in tree.body if not isinstance(n,ast.FunctionDef)]
    require(other(before)==other(after),'module constants/imports changed')
    return {'all_preexisting_functions_except_collect_unchanged':True,'collect_outside_historical_loop_unchanged':True,
        'replay_AST_identical':True,'primary_metrics_bootstrap_and_contrasts_AST_identical':True,
        'module_constants_imports_unchanged':True,'only_added_function':'historical_reference_identity'}

def actual_tests():
    result={}
    for label,path,count in [('math','tests/test_phase20_evaluation.py',7),('history','tests/test_phase20_history_reference.py',5)]:
        junit=ARCHIVE/(label+'_actual_tests.xml')
        run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--junitxml='+str(junit),path],cwd=ROOT,env=ENV,capture_output=True,text=True)
        (ARCHIVE/(label+'_actual_tests.log')).write_text(run.stdout+run.stderr)
        require(run.returncode==0,'actual '+label+' tests failed: '+run.stdout[-2000:])
        cases=list(ET.parse(junit).iter('testcase'))
        require(len(cases)==count and not any(list(q) for q in cases),'wrong actual test coverage')
        result[label]={'actual_tests':len(cases),'failures_errors_skips':0,'test_source_sha256':sha(ROOT/path),'junit_sha256':sha(junit)}
    return result

def preserved_replay():
    records={};rows=[]
    for i in range(4):
        marker=read(P/f'REPLAY_SHARD_{i}.json')
        require(marker['status']=='PASS' and marker['source']==TRAINING and marker['shards']==4,'saved replay shard not PASS')
        frame=pd.read_csv(R/f'CHECKPOINT_REPLAY_{i}.csv');require(len(frame)==marker['rows'],'saved replay count differs')
        rows+=frame.to_dict('records')
        for path in [P/f'REPLAY_SHARD_{i}.json',*[R/f'{name}_{i}.csv' for name in ['CHECKPOINT_REPLAY','FEATURE_RECONSTRUCTION','CONVEX_REFERENCE_AUDIT']]]:
            records[path.name]=sha(path)
    expected={(a,s,f,r) for a in ['L0','P1','P16','S1','S16','D16','F16','WC16','WO16'] for s in [31,47,83] for f in range(1,6) for r in ['inner','outer']}
    keys=[(q['arm'],q['seed'],q['fold'],q['role']) for q in rows]
    require(len(keys)==270 and set(keys)==expected,'saved 270 checkpoint coverage differs')
    require(all(q['source']==TRAINING and q['max_replay_error']==0 and q['anchor_replay_max_error']==0 for q in rows),'saved replay is not exact')
    require(sum(q['extra_representation_modes'] for q in rows)==120,'extra representation replays differ')
    for q in rows:
        path=P/q['role']/f"{q['arm']}_seed{q['seed']}_fold{q['fold']}.pt"
        receipt=require_record(path.with_suffix('.json'),checkpoint_path=path)
        require(receipt['checkpoint_sha256']==q['checkpoint_sha256'] and receipt['source']==q['actual_fit_source'],'saved replay checkpoint changed')
    return {'checkpoint_count':270,'prediction_vectors':1740,'maximum_error':0.,'files':records,'new_forward_calls':0,'new_training_fits':0}

def qualify():
    from scripts.validate_phase20 import historical_reference_identity
    from src.phase19_attention.data import P as parentP,R as parentR
    require(not (R/'HISTORICAL_REFERENCE_REPAIR_AUDIT.json').exists(),'already qualified; preserve original audit')
    saved=read(ARCHIVE/'SNAPSHOT.json');require(saved['original_evaluator_sha256']==BEFORE,'wrong archived evaluator')
    require(sha(ARCHIVE/'evaluator_before.py')==BEFORE,'original evaluator archive changed')
    require(source()==TRAINING and source_files()==saved['training_source_files'],'training source changed')
    require(sha(R/'AUDIT_REPAIR_LOCK.json')==saved['repair_lock_sha256'] and sha(R/'AUDIT_REPAIR_TESTS.json')==saved['training_qualification_sha256'],'frozen training qualification changed')
    require(sha(ROOT/'tests/test_phase20_evaluation.py')==saved['original_math_test_sha256'],'original seven mathematical tests changed')
    ast_result=ast_audit();tests=actual_tests();replay=preserved_replay()
    base=Path('/root/phase16r_light_rna_private');npz=np.load(base/'OOF_PRIVATE.npz');rows=[]
    published=pd.read_csv(parentR/'HISTORICAL_LOCAL_REFERENCE_AUDIT.csv',float_precision='round_trip').set_index(['seed','fold'])
    for seed in [31,47,83]:
        vector=npz['LOCAL__CV3__seed'+str(seed)]
        require(vector.shape==(512,) and vector.dtype==np.dtype('float64') and np.isfinite(vector).all(),'invalid complete original historical vector')
        for fold in range(1,6):
            path=base/'artifacts'/f'seed{seed}_fold{fold}.pt';receipt=read(base/'cells'/f'seed{seed}_fold{fold}.json')
            ck=torch.load(path,map_location='cpu',weights_only=False)
            bank=torch.load(parentP/'preprocessing'/f'fold{fold}_outer.pt',map_location='cpu',weights_only=False);ix=bank['valid']
            item=historical_reference_identity(vector[ix],ck['models']['LOCAL__CV3']['risk'],ix,ck['part']['outer_test'],sha(path),receipt['sha256'])
            prior=published.loc[seed,fold]
            require(item['artifact_sha256']==prior.artifact_sha256 and item['maximum_error']==prior.maximum_error,'published Phase19 reference audit differs')
            rows.append({'seed':seed,'fold':fold,**item,'matches_published_Phase19_audit':True})
    for name,record in saved['files'].items():require(sha(ARCHIVE/name)==record['sha256'],'original failure evidence archive changed')
    evaluator=sha(ROOT/'scripts/validate_phase20.py')
    write(R/'EVALUATION_MATH_AUDIT.json',{'status':'PASS','source':TRAINING,'server_actual_tests':7,'failures_errors_skips':0,
        'evaluator_sha256':evaluator,'test_source_sha256':tests['math']['test_source_sha256'],'junit_sha256':tests['math']['junit_sha256'],
        'historical_reference_regression_tests':5,'historical_reference_test_sha256':tests['history']['test_source_sha256'],
        'total_actual_evaluation_tests':12,'stdout_summary':'Actual server runs: original seven math tests plus five separate historical-reference regression tests; no skips/errors/failures.'})
    write(R/'HISTORICAL_REFERENCE_REPAIR_AUDIT.json',{'status':'PASS','source':TRAINING,'evaluator_before_sha256':BEFORE,
        'evaluator_after_sha256':evaluator,'original_math_test_sha256':saved['original_math_test_sha256'],
        'history_regression_test_sha256':tests['history']['test_source_sha256'],'actual_math_tests':7,'actual_history_regression_tests':5,
        'actual_total_evaluation_tests':12,'failures_errors_skips':0,'actual_test_runs':tests,
        'math_audit_sha256':sha(R/'EVALUATION_MATH_AUDIT.json'),'audit_program_sha256':sha(Path(__file__)),
        'AST':ast_result,'historical_cells':rows,'historical_cell_count':len(rows),'maximum_historical_reference_error':max(q['maximum_error'] for q in rows),
        'original_failure_evidence':{name:record['sha256'] for name,record in saved['files'].items()},
        'original_historical_OOF_sha256':sha(base/'OOF_PRIVATE.npz'),'preserved_replay':replay,
        'published_reference_rule':{'absolute_tolerance':1e-9,'Phase19_validator_sha256':sha(ROOT/'scripts/validate_phase19.py'),
            'Phase16R_independent_validator_sha256':sha(ROOT/'scripts/validate_phase16r_light_rna.py'),
            'scope':'existing independently recomputed float64 historical OOF only; not current Phase20 predictions or anchor'},
        'training_source_files_unchanged':True,'original_math_test_bytes_unchanged':True,'current_risks_and_historical_OOF_modified':False,
        'new_training_fits':0,'new_prediction_replays':0,'recovery_scope':'statistics-only, reusing all 270 qualified checkpoints and 1740 exact prediction replays',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print('HISTORICAL_REFERENCE_REPAIR_AUDIT_PASS',evaluator,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',action='store_true');args=parser.parse_args()
    snapshot() if args.snapshot else qualify()
