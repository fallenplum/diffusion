"""Publish an anonymous pre-outer protocol and scientific source closure."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.bgb_en.data import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    receipt=json.loads((args.private/'prepared/outer1_inner0.json').read_text())
    config=json.loads((ROOT/'configs/bgb_en/prototype_v1.json').read_text())
    report=dict(status='LOCKED_CORE_MATRIX_OUTER_TEST_NOT_EVALUATED',lineage=receipt['lineage'],
        config_sha256=sha256(ROOT/'configs/bgb_en/prototype_v1.json'),
        planned_unique_inner_units=dict(R1=57,R2=57,R3=72),
        planned_neural_fits=162,planned_additional_convex_fits=12,
        planned_upstream_anchor_and_teacher_fits=39,deterministic_baseline_reference_cells=12,
        seeds=config['prototype']['neural_seeds'],inner_partitions=3,
        epochs_per_neural_fit=80,outer_test_evaluated=False,
        full_riskset_Efron=True,IPCW_fit_partition_only=True,
        bootstrap=False,confidence_intervals=False,p_values=False,
        shared_patch_selector='REGIONS512_qualified_from_coordinates_before_survival_fits',
        source_auxiliary_tracked_separately=True,
        interpretation='exploratory_repeated_development_cohort_not_independent_confirmation')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text())!=report:raise ValueError('Existing protocol lock differs')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],core_cells=186)))

if __name__=='__main__':main()
