"""Explicit anonymous Phase26 publication; no Git writes or model fitting."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase26_multiquery import runtime

NAMES={'PROTOCOL.md','RESULTS.md','QUALIFICATION.json','REAL_QUALIFICATION.json','MAIN_REPLAY_AUDIT.json',
       'REPLAY_AUDIT.json','INNER_RESULTS.json','INNER_GATE.json','decision.json','RUN_AUDIT.json',
       'BOUNDARY_AUDIT.json','ARTIFACT_MANIFEST.json'}


def build():
    spec=importlib.util.spec_from_file_location('_phase26_public_guard',ROOT/'scripts/publish_phase23_cdei.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    guard.PARENT=runtime.PARENT;guard.PARENT_FILES=2678
    guard.AGGREGATE_RECORD_LISTS|={'two_seed_means','primary_tests','anchor_guards','pairing_guards','mechanism_rows',
        'cell_replay','cell_rows','fullfit_rows','teacher_rows','parameter_rows','fold_main_distribution',
        'fit_distribution_rows','main_effect_guards','frozen_query_guards','arm_equal_cell_means'}
    inherited=guard.audit_aggregate_json
    def audit(value,location='root'):
        if location in ('prepared_sha256_by_fold','legacy_prepared_sha256_by_fold'):
            guard.require(isinstance(value,dict) and set(value)=={'1','2','3'} and all(
                isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in value.values()),
                'prepared receipt must be exactly three fold SHA256 strings')
            return
        if location=='raw_state_sha256_by_fold':
            is_sha=lambda v:isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v)
            guard.require(isinstance(value,dict) and set(value)=={'1','2','3'} and all(
                isinstance(v,dict) and set(v)=={'full','teachers'} and is_sha(v['full']) and
                isinstance(v['teachers'],list) and len(v['teachers'])==3 and all(is_sha(s) for s in v['teachers'])
                for v in value.values()),'raw state receipt must be three folds of four state SHA256 strings')
            return
        inherited(value,location)
    guard.audit_aggregate_json=audit
    parent=guard.verify_parent(runtime.PRIVATE/'PARENT_TRACKED_BYTES_PRIVATE.json')
    parent.pop('parent_all2556verified',None);parent['parent_all2678verified']=True
    paths=list((ROOT/'src/phase26_multiquery').glob('*.py'))
    paths+=list((ROOT/'scripts').glob('*phase26*.py'))+list((ROOT/'tests').glob('test_phase26_*.py'))
    paths+=[runtime.CONFIG]
    present={p.name for p in runtime.REPORT.iterdir() if p.is_file()}
    guard.require(present<=NAMES and NAMES-{'ARTIFACT_MANIFEST.json'}<=present,'missing/unreviewed Phase26 report')
    paths+=list(runtime.REPORT.iterdir())
    allowed={p.relative_to(ROOT).as_posix() for p in paths};guard.allowed_path=lambda p:p in allowed
    guard.REPORT=runtime.REPORT.relative_to(ROOT).as_posix()
    rows=[guard.audit_public_file(p.relative_to(ROOT).as_posix(),root=ROOT) for p in sorted(paths)
          if p!=runtime.REPORT/'ARTIFACT_MANIFEST.json']
    return {'status':'PASS','schema':'phase26_explicit_anonymous_manifest_v1',
        'training_source':runtime.source(),'training_files':runtime.source_files(),
        'evaluator_files':{p:runtime.sha(ROOT/p) for p in ('src/phase26_multiquery/evaluation.py','scripts/evaluate_phase26.py')},
        'parent_commit':runtime.PARENT,'parent_verification':parent,'public_files':rows,
        'total_files':len(rows),'total_bytes':sum(r['bytes'] for r in rows),'manifest_self_excluded':True,
        'scope':'new_Phase26_text_only;not_retrospective_whole_repo_privacy_audit',
        'patient_arrays_weights_or_credentials_public':False,'scientific_claim':'see_RESULTS_and_gate_not_technical_manifest_PASS'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    value=build();target=runtime.REPORT/'ARTIFACT_MANIFEST.json'
    if a.write:runtime.atomic_json(target,value)
    elif json.loads(target.read_text())!=value:raise ValueError('manifest drift')
    print(json.dumps({'status':'PASS','files_excluding_manifest':value['total_files'],'historical_files':2678}))


if __name__=='__main__':main()
