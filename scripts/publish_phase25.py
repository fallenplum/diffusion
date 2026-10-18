"""Publish only explicit anonymous Phase25 text; no Git writes in this tool."""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase25_local.runtime import PRIVATE,REPORT,PARENT,source,source_files,atomic_json,sha

A={'PROTOCOL.md','RESULTS.md','decision.json','CONCH_PC_ATTRIBUTION.json',*(name+ext for name in
    ('VISUAL_ATTRIBUTION','C4_ROUTING_ATTRIBUTION','SCALAR_ATTRIBUTION','LEGACY_ATTRIBUTION')
    for ext in ('.md','.json'))}
B={'PROTOCOL.md','RESULTS.md','QUALIFICATION.json','REAL_QUALIFICATION.json','MAIN_REUSE_AUDIT.json',
   'MAIN_REPLAY_AUDIT.json','REPLAY_AUDIT.json','INNER_RESULTS.json','INNER_GATE.json','decision.json',
   'RUN_AUDIT.json','ARTIFACT_MANIFEST.json'}


def build():
    spec=importlib.util.spec_from_file_location('_phase25_public_guard',ROOT/'scripts/publish_phase23_cdei.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    guard.PARENT=PARENT;guard.PARENT_FILES=2621
    guard.AGGREGATE_RECORD_LISTS |= {
        'primary_tests','same_head_intervention_screen','seed_metrics','model_summary','paired_component_credit_summary',
        'fold_local_basis_audit','concentration_by_cell','PC_contributions_by_cell','modulation','five_seed_means',
        'prediction_files','historical_checks','mechanism_rows','training_rows','model_rows','seed_summary',
        'parameter_rows','risk_component_rows','C4_query_interventions','cell_summary','summaries',
        'DSS_HALL_coefficient_diagnostics','DSS_HALL_fold_decomposition',
        'actual_teacher_held_label_perturbations','historical_C4_fixed_RNA_sensitivity',
        'historical_CONCH_summary','historical_DSS_HALL_paired_intervals_reused',
        'historical_confirmatory_and_mechanism_intervals','historical_repair_damage_credit_mass',
        'paired_credit_summary','repair_damage_summary','fullfit_rows','teacher_rows',
        'cell_replay','cell_rows','two_seed_means','R3_anchor_guards','R3_query_pairing_guards',
        'R1_effective_tests','R2_effective_tests'}
    inherited_json_audit=guard.audit_aggregate_json
    def audit_aggregate(value,location='root'):
        # The independent replay receipt indexes exactly three prepared-file
        # checksums by fold. This is not a patient-indexed record collection.
        if location=='prepared_sha256_by_fold':
            guard.require(isinstance(value,dict) and set(value)=={'1','2','3'} and
                all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in value.values()),
                'prepared fold receipt must contain exactly three SHA256 strings')
            return
        inherited_json_audit(value,location)
    guard.audit_aggregate_json=audit_aggregate
    parent=guard.verify_parent(PRIVATE.parent/'PARENT_TRACKED_BYTES_PRIVATE.json')
    parent.pop('parent_all2556verified',None);parent['parent_all2621verified']=True
    paths=list((ROOT/'src/phase25_local').glob('*.py'))
    paths+=list((ROOT/'tests').glob('test_phase25_*.py'))
    paths+=list((ROOT/'scripts').glob('*phase25*.py'))
    paths+=[ROOT/'configs/phase25_local_routing.json']
    for directory,names in [('reports/phase25_a',A),('reports/phase25_b',B)]:
        present={p.name for p in (ROOT/directory).iterdir() if p.is_file()}
        guard.require(not ((names-{'ARTIFACT_MANIFEST.json'})-present),
            'required Phase25 report is missing')
        for p in (ROOT/directory).iterdir():
            if not p.is_file() or p.name not in names:raise ValueError('unreviewed Phase25 report')
            paths.append(p)
    allowed={p.relative_to(ROOT).as_posix() for p in paths}
    guard.allowed_path=lambda name:name in allowed
    rows=[]
    for p in sorted(set(paths)):
        if p==REPORT/'ARTIFACT_MANIFEST.json':continue
        if not p.resolve().is_relative_to(ROOT) or p.is_symlink():raise ValueError('unsafe public path')
        guard.REPORT=p.parent.relative_to(ROOT).as_posix() if p.parent in (ROOT/'reports/phase25_a',REPORT) else '__not_a_report__'
        rows.append(guard.audit_public_file(p.relative_to(ROOT).as_posix(),root=ROOT))
    return {'status':'PASS','schema':'phase25_explicit_anonymous_artifact_manifest_v1','training_source':source(),
        'training_files':source_files(),'evaluator_files':{p:sha(ROOT/p) for p in
            ('src/phase25_local/evaluation.py','scripts/evaluate_phase25.py','src/phase10_trimodal/reference.py','src/phase21_relative/evaluation.py')},
        'parent_commit':PARENT,'parent_verification':parent,'public_files':rows,'total_files':len(rows),
        'total_bytes':sum(r['bytes'] for r in rows),'manifest_self_excluded':True,
        'scope':'explicit_new_Phase25_files_only_not_retrospective_whole_repository_privacy_audit',
        'public_patient_arrays_or_model_weights':False,
        'scientific_claim':'see_A_and_B_RESULTS_and_decisions_not_inferred_from_technical_PASS'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    manifest=build();target=REPORT/'ARTIFACT_MANIFEST.json'
    if a.write:atomic_json(target,manifest)
    elif json.loads(target.read_text())!=manifest:raise ValueError('manifest drift')
    print(json.dumps({'status':'PASS','mode':'write' if a.write else 'audit','files':manifest['total_files'],
                      'parent_files_verified':2621,'source':manifest['training_source']}),flush=True)


if __name__=='__main__':main()
