"""Qualify the anchor-only audit repair; never refit a retained checkpoint.

The original D1 snapshot was saved before any source changes. Qualification
replays all stored formal prediction modes without labels, verifies the exact
unchanged optimization code, and preserves original checkpoint source fields.
"""
import argparse
import ast
import collections
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase20_selective.data import P,R,source,source_files,write
from src.phase10_trimodal.data import digest
from src.phase20_selective.provenance import LEGACY_SOURCE

ARCHIVE=P/'recovery/D1'
ENV=dict(os.environ,PYTHONPATH='.:/root/phase10_trimodal_eval_deps',OPENBLAS_NUM_THREADS='2',
         OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')


def snapshot():
    value=json.loads((ARCHIVE/'SNAPSHOT.json').read_text())
    if value['legacy_source']!=LEGACY_SOURCE or len(value['legacy_checkpoints'])!=189:
        raise ValueError('the immutable initial D1 snapshot is incomplete')
    return value


def tests():
    all_files=['tests/test_phase20_'+name+'.py' for name in ['operators','model','data','training','diagnostics']]
    results={}
    for kind,files,expected in [('training',all_files,64),('math',['tests/test_phase20_evaluation.py'],7)]:
        junit=P/'logs'/('repair_'+kind+'_tests.xml')
        cmd=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--junitxml='+str(junit),*files]
        run=subprocess.run(cmd,cwd=ROOT,env=ENV,capture_output=True,text=True)
        (P/'logs'/('repair_'+kind+'_tests.log')).write_text(run.stdout+run.stderr)
        if run.returncode: raise ValueError(kind+' repair tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        cases=list(ET.parse(junit).getroot().iter('testcase'))
        if len(cases)!=expected or any(list(c) for c in cases):
            raise ValueError('wrong test count, skip, failure or error in '+kind)
        results[kind]={'actual_tests':len(cases),'junit_sha256':digest(junit)}
    files=all_files+['tests/test_phase20_evaluation.py']
    write(R/'AUDIT_REPAIR_TESTS.json',{'status':'PASS','source':source(),'actual_training_tests':64,
        'actual_math_tests':7,'failures_errors_skips':0,'test_files':{f:digest(ROOT/f) for f in files},
        'training_source_files':source_files(),'actual_test_runs':results})
    write(R/'EVALUATION_MATH_AUDIT.json',{'status':'PASS','source':source(),'server_actual_tests':7,
        'failures_errors_skips':0,'evaluator_sha256':digest(ROOT/'scripts/validate_phase20.py'),
        'test_source_sha256':digest(ROOT/'tests/test_phase20_evaluation.py'),
        'junit_sha256':results['math']['junit_sha256'],'stdout_summary':'Actual repair qualification: 7 passed, 0 skips/failures/errors'})
    print('PHASE20_REPAIR_TESTS_PASS',source(),flush=True)


def static_audit(old):
    current=source_files()
    changed=sorted(k for k in set(current)|set(old['legacy_source_files']) if current.get(k)!=old['legacy_source_files'].get(k))
    expected=sorted(['src/phase20_selective/train.py','tests/test_phase20_training.py',
                     'scripts/run_phase20.py','src/phase20_selective/provenance.py'])
    if changed!=expected: raise ValueError(('unexpected training-source change',changed))
    before=ast.parse((ARCHIVE/'source/src/phase20_selective/train.py').read_text())
    after=ast.parse((ROOT/'src/phase20_selective/train.py').read_text())
    a={n.name:n for n in before.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    b={n.name:n for n in after.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    for name in set(a)-{'forward'}:
        if ast.dump(a[name])!=ast.dump(b[name]): raise ValueError('changed optimization/model call: '+name)
    forward=copy.deepcopy(b['forward']);repaired=0
    for node in ast.walk(forward):
        for field,value in ast.iter_fields(node):
            if not isinstance(value,list): continue
            expanded=[]
            for child in value:
                if (isinstance(child,ast.If) and isinstance(child.test,ast.Compare) and
                    ast.dump(child.test)==ast.dump(ast.parse("mode == 'anchor'",mode='eval').body)):
                    expanded.extend(child.orelse);repaired+=1
                else: expanded.append(child)
            setattr(node,field,expanded)
    if repaired!=1 or ast.dump(forward)!=ast.dump(a['forward']):
        raise ValueError('non-anchor forward path is not exactly the original AST')
    return {'changed_source_files':changed,'all_other_train_functions_AST_identical':True,
            'nonanchor_forward_AST_identical_after_removing_anchor_only_branch':True,
            'model_operators_data_diagnostics_configuration_unchanged':True}


def legacy_module():
    name='src.phase20_selective._initial_train_D1'
    spec=importlib.util.spec_from_file_location(name,ARCHIVE/'source/src/phase20_selective/train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def replay(shard,shards):
    import numpy as np
    import torch
    from src.phase20_selective import data,train
    torch.set_num_threads(2);old=snapshot();static_audit(old);initial=legacy_module()
    jobs=sorted(old['legacy_checkpoints']);rows=[]
    modes=[('matched','matched_risk'),('wrongC','wrongC_risk'),('wrongO','wrongO_risk'),
           ('constantC','constantC_risk'),('off','interaction_off_risk')]
    for filename in jobs[shard::shards]:
        path=Path(filename)
        if digest(path)!=old['legacy_checkpoints'][filename] or digest(path.with_suffix('.json'))!=old['legacy_records'][str(path.with_suffix('.json'))]:
            raise ValueError('original D1 bytes changed')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['source']!=LEGACY_SOURCE: raise ValueError('original source rewritten')
        train.seed_all(ck['seed']+100*ck['fold']);ds=data.Data(ck['fold'],ck['role'],ck['seed'])
        model=train.make_model(ds,ck['arm']);model.load_state_dict(ck['state']);model.eval()
        chosen=modes+([('fullOmicsWrong','full_omics_wrong_risk'),('uniformPair','uniform_pair_risk')] if ck['arm'] in ['P1','P16'] else [])
        with torch.no_grad():
            for mode,key in chosen:
                previous=initial.forward(model,ds,ds.valid,mode).cpu().numpy()
                repaired=train.forward(model,ds,ds.valid,mode).cpu().numpy()
                if not np.array_equal(previous,repaired) or not np.array_equal(repaired,ck[key]):
                    raise ValueError(('formal prediction changed',path.name,mode))
            anchor=train.forward(model,ds,ds.valid,'anchor').cpu().numpy()
            if not np.array_equal(anchor,ck['anchor_risk']): raise ValueError('stored original anchor changed')
        rows.append({'checkpoint_sha256':digest(path),'formal_nonanchor_vectors':len(chosen),'anchor_bitwise_equal':True})
        print('REPAIR_COMPATIBLE',ck['role'],path.stem,len(chosen),flush=True)
        del model,ds;torch.cuda.empty_cache()
    write(P/f'REPAIR_COMPATIBILITY_SHARD_{shard}.json',{'status':'PASS','source':source(),'shard':shard,'shards':shards,'rows':rows})


def finalize(shards):
    import numpy as np
    import torch
    from src.phase20_selective import data,train
    torch.set_num_threads(2);old=snapshot();static=static_audit(old);identity=source()
    parts=[json.loads((P/f'REPAIR_COMPATIBILITY_SHARD_{i}.json').read_text()) for i in range(shards)]
    if any(q['status']!='PASS' or q['source']!=identity or q['shards']!=shards for q in parts): raise ValueError('compatibility shard stale')
    rows=[r for q in parts for r in q['rows']]
    if len(rows)!=189 or {q['checkpoint_sha256'] for q in rows}!=set(old['legacy_checkpoints'].values()) or sum(q['formal_nonanchor_vectors'] for q in rows)!=1029:
        raise ValueError('all189 original checkpoints/1029 vectors must be checked')
    initial=legacy_module();anchors=[]
    for role in ['inner','outer']:
        for fold in range(1,6):
            ds=data.Data(fold,role,31);model=train.make_model(ds,'L0').eval()
            with torch.no_grad():
                original=initial.forward(model,ds,ds.valid,'anchor').cpu().numpy()
                actual=train.forward(model,ds,ds.valid,'anchor').cpu().numpy()
            reference=ds.inherited_simple['models']['J0']['risk']
            if not np.array_equal(actual,reference): raise ValueError('repaired real-partition anchor is not exact')
            anchors.append({'role':role,'fold':fold,'original_mismatched_patients':int((original!=reference).sum()),
                'original_max_error':float(abs(original-reference).max()),'repaired_max_error':0.,'repaired_bitwise_identical':True})
            del model,ds;torch.cuda.empty_cache()
    for filename,expected in old['legacy_checkpoints'].items():
        if digest(filename)!=expected: raise ValueError('original checkpoint changed during qualification')
    qualification=json.loads((R/'AUDIT_REPAIR_TESTS.json').read_text())
    if qualification['source']!=identity or qualification['status']!='PASS' or qualification['training_source_files']!=source_files(): raise ValueError('repair tests stale')
    inventory={**old,'status':'PASS','source':identity,'legacy_source':LEGACY_SOURCE,
        'initial_snapshot_sha256':digest(ARCHIVE/'SNAPSHOT.json'),'qualification_sha256':digest(R/'AUDIT_REPAIR_TESTS.json')}
    write(P/'ANCHOR_AUDIT_REPAIR_COMPATIBILITY.json',inventory)
    write(R/'AUDIT_REPAIR_COMPATIBILITY.json',{'status':'PASS','source':identity,'legacy_source':LEGACY_SOURCE,
        'inventory_sha256':digest(P/'ANCHOR_AUDIT_REPAIR_COMPATIBILITY.json'),
        'repair_program_sha256':digest(Path(__file__)),'same_phase_retained_checkpoint_count':189,
        'retained_inner':135,'retained_outer':54,'remaining_fresh_outer_fits':81,
        'formal_nonanchor_prediction_vectors_checked':1029,'formal_predictions_bitwise_equal':True,'bitwise_equal':True,
        'additional_legacy_anchor_replays':189,'survival_outcomes_used_for_repair_selection':False,
        'weights_epochs_histories_or_original_sources_rewritten':False,'static_equivalence':static,
        'ten_partition_anchor_audits':anchors,'original_failed_attempts':1,'scheduler_interrupted_attempts':7,
        'scope':'anchor-only audit aggregation; all optimization, selection and formal prediction arithmetic unchanged'})
    write(R/'AUDIT_REPAIR_LOCK.json',{'status':'LOCKED_BEFORE_RESTART_AFTER_AUDIT_ONLY_REPAIR','source':identity,
        'training_files':source_files(),'compatibility_audit_sha256':digest(R/'AUDIT_REPAIR_COMPATIBILITY.json'),
        'qualification_sha256':digest(R/'AUDIT_REPAIR_TESTS.json'),'no_outer_endpoint_selection':True})
    print('PHASE20_ANCHOR_REPAIR_QUALIFIED',identity,flush=True)


def resume():
    from src.phase20_selective.provenance import repair_inventory,require_record,verify_qualified_tests
    repair_inventory();verify_qualified_tests(require_record(R/'OPERATOR_AND_PIPELINE_TESTS.json'))
    for name in ['FAILED.json','EVALUATION_FAILED.json']:
        path=P/name
        if path.exists(): path.rename(ARCHIVE/('closed_active_'+str(time.time_ns())+'_'+name))
    started=json.loads((P/'STARTED.json').read_text());stat=Path('/proc')/str(started['pid'])/'stat'
    if stat.exists() and stat.read_text().split()[2]!='Z': raise ValueError('old scheduler still active')
    stream=(P/'logs/scheduler.log').open('a');stream.write('\nAUDIT_ONLY_REPAIR_RESTART '+source()+'\n');stream.flush()
    child=subprocess.Popen([sys.executable,'scripts/schedule_phase20.py'],cwd=ROOT,env=ENV,
        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(P/'STARTED.json',{'pid':child.pid,'source':source(),'workers':json.loads((P/'WORKERS.json').read_text())['workers'],
        'legacy_source':LEGACY_SOURCE,'retained_unchanged_checkpoints':189,'fresh_outer_fits_remaining':81})
    print('PHASE20_REPAIR_RESUMED',child.pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['tests','replay','finalize','resume'])
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=4);args=parser.parse_args()
    if args.action=='tests':tests()
    elif args.action=='replay':replay(args.shard,args.shards)
    elif args.action=='finalize':finalize(args.shards)
    else:resume()
