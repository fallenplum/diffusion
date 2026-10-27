"""Freeze the predeclared prototype gate; never read outer-test predictions."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_run import load_inputs,write
from scripts.bgb_en_outer import validate_selection_lock,source_lineage_hash
from src.bgb_en.data import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/bgb_en/prototype_v1.json')
    p.add_argument('--reports',type=Path,default=ROOT/'reports/bgb_en');args=p.parse_args()
    _,config,_,_,lineage=load_inputs(args)
    inner=args.reports/'INNER_REPORT_AUDIT.json';matrix=args.reports/'R3_MODEL_MATRIX.csv'
    report=json.loads(inner.read_text())
    if report['status']!='COMPLETE_CORE_INNER':raise ValueError('Core inner matrix must finish before finalist gate')
    if report['source_lineage_sha256']!=source_lineage_hash(lineage):raise ValueError('Source lineage differs')
    with matrix.open(newline='',encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
    expected=[arm for arm in config['R3']['arms'] if arm not in ('F0','F1')]
    selected=[row for row in rows if row['arm'] in expected]
    if len(selected)!=len(expected) or {row['arm'] for row in selected}!=set(expected):raise ValueError('Incomplete neural gate pool')
    if any(row['status']!='COMPLETE' or row['paired_reference']!='F1' for row in selected):raise ValueError('Invalid paired neural gate')
    qualified=[row for row in selected if float(row['paired_delta_H'])>0 and float(row['paired_delta_U60'])>0]
    finalists=['CP_STRONG','CP_SCALAR']+(['NESTED_INTERACTION'] if qualified else [])
    frozen=args.reports/'INNER_REPORT_FOR_SELECTION.json'
    if frozen.exists() and frozen.read_bytes()!=inner.read_bytes():raise ValueError('A previous immutable selection snapshot exists')
    if not frozen.exists():frozen.write_bytes(inner.read_bytes())
    value=dict(status='LOCKED_BEFORE_OUTER_EVALUATION',outer_test_evaluated=False,outer_folds=[1,2,3,4,5],
        finalists=finalists,source_lineage_sha256=source_lineage_hash(lineage),config_sha256=sha256(args.config),
        inner_report_sha256=sha256(frozen),inner_model_matrix_sha256=sha256(matrix),
        selector_source_sha256=sha256(__file__),gate=config['selection']['neural_finalist_gate'],
        qualified_prototype_arms=[dict(arm=row['arm'],delta_H=float(row['paired_delta_H']),delta_U60=float(row['paired_delta_U60'])) for row in qualified],
        nested_interaction_pool=config['selection']['nested_interaction_candidates'] if qualified else [],
        selection_is_not_independent_validation=True,
        scientific_scope='prototype gate and each-outer-train inner selection; repeatedly developed cohort is exploratory')
    validate_selection_lock(value,report,lineage,config,sha256(args.config),sha256(frozen),1)
    output=args.reports/'OUTER_SELECTION_LOCK.json'
    if output.exists() and json.loads(output.read_text())!=value:raise ValueError('Existing finalist lock differs')
    if not output.exists():write(output,value)
    print(json.dumps(dict(status=value['status'],finalists=finalists,qualified_arms=[x['arm'] for x in qualified])))


if __name__=='__main__':main()
