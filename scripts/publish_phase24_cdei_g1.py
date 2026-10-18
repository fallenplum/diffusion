"""Audit only new anonymous G1 text artifacts; never stage or push itself."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase24_cdei_g1.runtime import PRIVATE,REPORT,PARENT,source,source_files,atomic_json,sha


def build():
    # Reuse the already reviewed text/aggregate and Git-blob guard implementation
    # in a separate module instance. No historical source file is changed.
    spec=importlib.util.spec_from_file_location('_g1_public_guard',ROOT/'scripts/publish_phase23_cdei.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    guard.PARENT=PARENT;guard.PARENT_FILES=2595
    guard.AGGREGATE_RECORD_LISTS |= {'primary_tests','same_head_intervention_screen'}
    parent=guard.verify_parent(PRIVATE/'PARENT_TRACKED_BYTES_PRIVATE.json')
    parent.pop('parent_all2556verified',None)
    parent['parent_all2595verified']=True
    md={'PROTOCOL.md','RESULTS.md','MATH_AND_PRIOR_AUDIT.md'}
    js={'QUALIFICATION.json','REAL_DATA_QUALIFICATION_PUBLIC.json','decision.json',
        'INNER_RESULTS.json','INNER_GATE.json','MECHANISM_AUDIT.json','RUN_AUDIT.json',
        'OUTER_RESULTS.json','ARTIFACT_MANIFEST.json'}
    paths=list((ROOT/'src/phase24_cdei_g1').glob('*.py'))
    paths+=list((ROOT/'tests').glob('test_phase24_cdei_g1_*.py'))
    paths+=list((ROOT/'scripts').glob('*phase24_cdei_g1*.py'))
    paths+=[ROOT/'configs/phase24_cdei_g1.json']
    for p in REPORT.iterdir():
        if not p.is_file() or p.name not in md|js:raise ValueError('unreviewed report')
        paths.append(p)
    allowed={p.relative_to(ROOT).as_posix() for p in paths}
    guard.allowed_path=lambda name:name in allowed
    guard.REPORT=REPORT.relative_to(ROOT).as_posix()
    rows=[]
    for p in sorted(set(paths)):
        if p.name=='ARTIFACT_MANIFEST.json':continue
        if not p.resolve().is_relative_to(ROOT):raise ValueError('unsafe resolved path')
        rows.append(guard.audit_public_file(p.relative_to(ROOT).as_posix(),root=ROOT))
    identity=source()
    evaluations={p:sha(ROOT/p) for p in ['src/phase24_cdei_g1/evaluation.py','scripts/evaluate_phase24_cdei_g1.py',
                    'src/phase10_trimodal/reference.py','src/phase21_relative/evaluation.py']}
    return {'status':'PASS','schema':'phase24_g1_anonymous_publication_v1','training_source':identity,
        'training_files':source_files(),'evaluator_files':evaluations,'parent_commit':PARENT,
        'parent_verification':parent,'public_files':rows,'total_files':len(rows),
        'total_bytes':sum(r['bytes'] for r in rows),'public_patient_arrays_or_weights':False,
        'privacy_scope':'explicit_new_Phase24_files_only_not_a_retrospective_whole_repository_audit',
        'manifest_self_excluded':True,
        'scientific_claim':'see_RESULTS_and_decision_not_inferred_from_technical_PASS'}


def main():
    a=argparse.ArgumentParser();a.add_argument('--write',action='store_true');args=a.parse_args()
    manifest=build();p=REPORT/'ARTIFACT_MANIFEST.json'
    if args.write:atomic_json(p,manifest)
    elif json.loads(p.read_text())!=manifest:raise ValueError('manifest drift')
    print(json.dumps({'status':'PASS','mode':'write' if args.write else 'audit','files':manifest['total_files'],
        'parent_files_verified':2595,'source':manifest['training_source']}))


if __name__=='__main__':main()
