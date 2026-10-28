"""Export only anonymous numeric proof from completed retention receipts."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import bgb_en_r6_retention as r


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('assets','private','primary-private','source-private','config','source-config','launch-gate'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--launch-gate-sha256',required=True)
    args=p.parse_args()
    lock=r.verify_lock(args)  # Metadata and SHA only, no target endpoint loader.
    rows=[]
    for entry in lock['representations']:
        directory=args.private/'cells'/f"{entry['representation']}_p{entry['partition']}"
        done=r.read(directory/'COMPLETE.json');replay=r.read(directory/'REPLAY.json')
        r.require(replay['status']=='PASS_RETENTION_COEFFICIENT_AND_INDEPENDENT_EFRON_REPLAY'
            and replay['contract']==done['contract']==r._contract(lock,entry)
            and replay['complete_sha256']==r.sha256(directory/'COMPLETE.json'),
            'RETENTION_EXPORTED_REPLAY_ORIGIN_CHANGED')
        r.require(replay['risk_max_abs_error']<=1e-12 and replay['independent_gradient_max']<=1e-6
            and replay['independent_objective_error']<=1e-10 and replay['new_optimizer_fits']==0,
            'RETENTION_EXPORTED_NUMERICAL_CHECK_FAILED')
        rows.append(dict(representation=entry['representation'],partition=entry['partition'],
            optimizer_representation=entry['optimizer_representation'],status=replay['status'],
            actual_optimizer_fits=done['actual_optimizer_fits'],solver_iterations=done['solver_iterations'],
            completion_sha256=r.sha256(directory/'COMPLETE.json'),replay_sha256=r.sha256(directory/'REPLAY.json'),
            risk_max_abs_error=replay['risk_max_abs_error'],independent_gradient_max=replay['independent_gradient_max'],
            independent_objective_error=replay['independent_objective_error']))
    result=dict(schema='bgb_en_r6_retention_numeric_replay_export_v1',status='PASS_18_REAL_RECEIPT_NUMERIC_SUMMARIES',
        exporter_sha256=r.sha256(__file__),retention_lock_sha256=r.sha256(args.private/'retentionLOCK.json'),
        retention_contract_sha256=lock['contract_sha256'],retention_lock_schema=lock['schema'],
        retention_lock_status=lock['status'],source_file_sha256=lock['sources'],
        planned_logical_cells=lock['planned_logical_cells'],planned_actual_optimizer_fits=lock['planned_actual_optimizer_fits'],
        actual_optimizer_fits=sum(x['actual_optimizer_fits'] for x in rows),rows=rows,
        new_optimizer_fits=0,new_forward_calls=0,target_endpoints_or_arrays_loaded=False,
        original_private_complete_and_replay_receipts_preserved=True)
    out=ROOT/'reports/bgb_en_r6/R6_RETENTION_INDEPENDENT_REPLAY_AUDIT.json'
    r.exclusive_json(out,result)
    print(json.dumps(dict(status=result['status'],actual_optimizer_fits=result['actual_optimizer_fits'],
        max_risk_error=max(x['risk_max_abs_error'] for x in rows),
        max_independent_gradient=max(x['independent_gradient_max'] for x in rows),
        max_independent_objective_error=max(x['independent_objective_error'] for x in rows))))


if __name__=='__main__':
    main()
