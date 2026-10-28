"""Model-independent heldout program/tissue scoring eligibility; no fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.bgb_en.r6_source import SourceContract,load_private_cache,sha256,PROGRAM_ORDER


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source-cache',type=Path,required=True)
    p.add_argument('--source-cache-manifest-sha256',required=True)
    p.add_argument('--fresh-certificate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('EXISTING_ELIGIBILITY_PRESERVED')
    certificate_sha=sha256(args.fresh_certificate)
    dataset=load_private_cache(args.source_cache,SourceContract(fresh_certificate_sha256=certificate_sha),
                               expected_manifest_sha256=args.source_cache_manifest_sha256)
    groups=[]
    for donor in range(6):
        for k,name in enumerate(PROGRAM_ORDER):
            for tissue in range(3):
                rows=(dataset.donor_codes==donor)&(dataset.tissue_y==tissue)&np.isfinite(dataset.program_targets[:,k])
                values=dataset.program_targets[rows,k]
                variable=bool(len(values)>0 and np.max(values)>np.min(values))
                groups.append(dict(heldout_donor=donor,program=name,tissue_class=tissue,
                                   support=int(rows.sum()),target_nonconstant=variable,
                                   eligible=bool(len(values)>=3 and variable)))
    result=dict(schema='bgb_en_r6_source_target_eligibility_v1',
        status='LOCKED_MODEL_INDEPENDENT_SCORING_SUPPORT',groups=groups,
        source_cache_manifest_sha256=args.source_cache_manifest_sha256,
        fresh_certificate_sha256=certificate_sha,
        eligibility_rule='valid_target_support_ge_3_and_raw_target_nonconstant',
        source_train_positive_affine_scaling_preserves_eligibility=True,
        model_predictions_or_source_performance_used=False,
        TCGA_labels_loaded=False,new_optimizer_updates=0,source_test_sections_read=False,
        patient_or_spot_identifiers_values_or_coordinates_exported=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],groups=len(groups),
        eligible_groups=sum(g['eligible'] for g in groups),sha256=sha256(args.output))))


if __name__=='__main__':main()
