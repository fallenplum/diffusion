"""Prelocked dual-track selection and reversible scientific-data retention."""
from pathlib import Path
import json
from scripts.semantic_m123r_common import folder,verify_cell,read,write,sha
from src.semantic_m123r.protocol import configurations,choose
from src.semantic_m123r.training import SNAPSHOTS


def block_jobs(outer,arm,seed,lock_sha):
    return [{'phase':'qualification' if outer==1 else 'formal','role':'inner','outer':outer,'inner':i,
        'arm':arm,'seed':seed,'config':c,'epochs':140,'lock_sha256':lock_sha}
        for c in configurations(arm) for i in range(3)]


def train_eligible(arm,node):
    u=node['actual_update_counts']
    needed=['base'] if arm in ('R2','R3') else ['patch'] if arm=='R4' else ['patch','mil_query','mil_attention']
    if arm in ('R6','R7','R8','R9'):needed+=['molecular_query','cross_attention']
    return node['epoch']>=30 and all(u.get(g,0)>=20 for g in needed)


def select_block(private,outer,arm,seed,lock_sha):
    path=Path(private)/'selections'/f'outer{outer}_{arm}_seed{seed}.private.json'
    if path.exists():return read(path)
    jobs=block_jobs(outer,arm,seed,lock_sha);receipts={}
    for j in jobs:
        r=verify_cell(private,j)
        if r is None:raise ValueError('ALL_THREE_INNERS_ALL_CONFIGURATIONS_REQUIRED')
        receipts[(json.dumps(j['config'],sort_keys=True),j['inner'])]=(j,r)
    candidates=[];trained=[]
    for c in configurations(arm):
        rr=[receipts[(json.dumps(c,sort_keys=True),i)] for i in range(3)]
        for epoch in SNAPSHOTS:
            nodes=[next(s for s in r['snapshots'] if s['epoch']==epoch) for _,r in rr]
            row={'config':c,'epoch':epoch,'metrics':[s['metrics']['validation']['risk'] for s in nodes]}
            candidates.append(row)
            if all(train_eligible(arm,s) for s in nodes):trained.append(row)
    choices={'EARLY_ANY':choose(candidates,trained=False),'ENCODER_TRAINED':choose(trained,trained=True)}
    result={'schema':'m123r_three_inner_global_selection_v1','outer':outer,'arm':arm,'seed':seed,
        'lock_sha256':lock_sha,'choices':choices,'outer_results_used':False,
        'all_fixed_candidates':candidates,'early_and_trained_never_collapsed':True}
    write(path,result)
    # Retain only the two globally selected three-inner states. Every removed
    # state was first saved, hashed and loaded for a real prediction replay.
    for j,r in receipts.values():
        keep={str(v['epoch']) for v in choices.values() if v['config']==j['config']}
        f=folder(private,j);removed={cp['file']:cp['sha256'] for epoch,cp in r['checkpoints'].items() if epoch not in keep}
        manifest=f/'WEIGHT_RETENTION.private.json'
        write(manifest,{'schema':'m123r_prelocked_weight_retention_v1','selection_sha256':sha(path),
            'verified_removed':removed,'kept_epochs':sorted(map(int,keep)),
            'all_predictions_metrics_histories_preserved':True,'saved_replays_before_removal':len(r['checkpoints'])})
        for name,h in removed.items():
            target=f/name
            if not target.resolve().is_relative_to((Path(private)/'fits').resolve()) or target.is_symlink() or not name.startswith('checkpoint_'):
                raise ValueError('UNSAFE_CHECKPOINT_REMOVAL')
            if sha(target)!=h:raise ValueError('CHECKPOINT_CHANGED_BEFORE_RETENTION')
            target.unlink()
    return result
