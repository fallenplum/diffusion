"""Frozen checkpoint role audit: actual update stages, no survival scoring."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def describe_state(state,epoch):
    epoch=int(epoch)
    if epoch<0:raise ValueError('NEGATIVE_SELECTED_EPOCH')
    local=state['local_out.weight'].detach().double()
    row={'selected_epoch':epoch,'actual_head_only_updates':min(epoch,20),'actual_joint_encoder_updates':max(epoch-20,0),
        'local_output_matrix_norm':float(local.norm()),'local_output_matrix_exactly_zero':bool(torch.count_nonzero(local)==0)}
    if epoch<=20 and not row['local_output_matrix_exactly_zero']:raise ValueError('HEAD_ONLY_SCHEDULE_LOCAL_WEIGHT_CHANGED')
    key='interaction_head.weight'
    row['explicit_interaction_head_present']=key in state
    row['explicit_interaction_head_norm']=float(state[key].detach().double().norm()) if key in state else None
    return row


def main():
    from scripts.semantic_m123_worker import check_lock,sha,read,write
    from scripts.semantic_m123_schedule import final_jobs,verify_complete,cell_path
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','role-lock','expected-role-sha256','private','output-newdir'):p.add_argument('--'+k,required=True)
    a=p.parse_args();check_lock(a.lock,a.expected_lock_sha256)
    role=read(a.role_lock)
    if sha(a.role_lock)!=a.expected_role_sha256 or role['master_lock_sha256']!=a.expected_lock_sha256:raise ValueError('ROLE_LOCK_CHANGED')
    for n,h in role['sources'].items():
        if sha(ROOT/n)!=h:raise ValueError('ROLE_SOURCE_CHANGED')
    private=Path(a.private)
    if (private/'OWNER.private.json').exists() or not (private/'STUDY_COMPLETE.private.json').exists():raise ValueError('ALL_FITS_MUST_COMPLETE')
    barrier=private/'ALL_OUTER_SELECTIONS_FROZEN.private.json'
    jobs=final_jobs(a.expected_lock_sha256,read(barrier),sha(barrier));rows=[]
    for j in jobs:
        if j['arm']=='M0PLUS':continue
        receipt=verify_complete(private,j)
        cp=torch.load(cell_path(private,j)/receipt['checkpoint_file'],map_location='cpu',weights_only=False)
        if cp['job']!=j:raise ValueError('CHECKPOINT_JOB_CHANGED')
        rows.append({'outer':int(j['split'][5]),'arm':j['arm'],'seed':j['seed'],
            **describe_state(cp['state_dict'],j['epochs'])})
    if len(rows)!=150:raise ValueError('EXPECTED150_CHECKPOINTS')
    out=Path(a.output_newdir);out.mkdir(parents=True,exist_ok=False)
    table=out/'FROZEN_CHECKPOINT_TRAINING_ROLES.csv'
    with table.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    audit={'schema':'m123_frozen_checkpoint_training_roles_v1','status':'COMPLETE','actual_checkpoints_loaded':150,
        'zero_optimizer_final_initializations':sum(r['selected_epoch']==0 for r in rows),
        'positive_head_only_final_fits':sum(0<r['selected_epoch']<=20 for r in rows),
        'positive_joint_encoder_final_fits':sum(r['actual_joint_encoder_updates']>0 for r in rows),
        'new_optimizer_fits':0,'new_model_forwards':0,'survival_endpoint_arrays_decoded':False,
        'archive_hashing_may_read_endpoint_containers_as_opaque_bytes':True,
        'master_lock_sha256':a.expected_lock_sha256,'role_lock_sha256':a.expected_role_sha256,
        'source_sha256':role['sources'],'table_sha256':sha(table),
        'limits':['Zero local_out disables local contributions inside the pathology main branch.',
          'M3 has a separate explicit interaction head, so zero local_out does not disable all M3 cross-modal dependence.',
          'Nonzero frozen path derivatives do not establish that encoder parameters received optimizer updates.']}
    write(out/'FROZEN_CHECKPOINT_TRAINING_ROLES_AUDIT.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('limits','source_sha256')}))


if __name__=='__main__':main()
