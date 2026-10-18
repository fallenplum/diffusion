"""Fail-closed anonymous artifact manifest for the appended Phase27 audit."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase27_readout import runtime

NAMES={'PROTOCOL.md','RESULTS.md','QUALIFICATION.json','FEATURE_AUDIT.json','REPLAY_AUDIT.json',
       'TRAIN_DIAGNOSTIC.json','INNER_RESULTS.json','INNER_GATE.json','decision.json','ARTIFACT_MANIFEST.json'}


def build():
    spec=importlib.util.spec_from_file_location('_phase27_guard',ROOT/'scripts/publish_phase23_cdei.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    guard.PARENT=runtime.PARENT;guard.PARENT_FILES=2711
    guard.AGGREGATE_RECORD_LISTS|={'head_rows','feature_rows','cell_rows','optimization_history',
        'two_seed_means','primary_tests','main_effect_guards','frozen_query_guards','comparison_rows',
        'cell_bindings','frozen_pairing_guards','provenance_audit'}
    parent=guard.verify_parent(runtime.PRIVATE/'PARENT_TRACKED_BYTES_PRIVATE.json')
    parent.pop('parent_all2556verified',None);parent['parent_all2711verified']=True
    paths=list((ROOT/'src/phase27_readout').glob('*.py'))+list((ROOT/'tests').glob('test_phase27_*.py'))
    paths+=list((ROOT/'scripts').glob('*phase27*.py'))+[runtime.CONFIG]
    present={p.name for p in runtime.REPORT.iterdir() if p.is_file()}
    guard.require(present<=NAMES and NAMES-{'ARTIFACT_MANIFEST.json'}<=present,'missing/unreviewed Phase27 report')
    paths+=list(runtime.REPORT.iterdir());guard.allowed_path=lambda name:name in {p.relative_to(ROOT).as_posix() for p in paths}
    guard.REPORT=runtime.REPORT.relative_to(ROOT).as_posix()
    rows=[guard.audit_public_file(p.relative_to(ROOT).as_posix(),root=ROOT) for p in sorted(paths)
          if p.name!='ARTIFACT_MANIFEST.json']
    return {'status':'PASS','schema':'phase27_anonymous_readout_manifest_v1','training_source':runtime.source(),
        'source_files':runtime.source_files(),'parent_commit':runtime.PARENT,'parent_verification':parent,
        'public_files':rows,'total_files':len(rows),'total_bytes':sum(r['bytes'] for r in rows),
        'manifest_self_excluded':True,'scope':'explicit_new_Phase27_files_only',
        'private_patient_arrays_weights_and_credentials_public':False,'scientific_conclusion':'see_RESULTS_and_gate'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    value=build();target=runtime.REPORT/'ARTIFACT_MANIFEST.json'
    if a.write:runtime.atomic_json(target,value)
    elif json.loads(target.read_text())!=value:raise ValueError('manifest drift')
    print(json.dumps({'status':'PASS','new_files_excluding_manifest':value['total_files'],'historical_files_verified':2711}))
