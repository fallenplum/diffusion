"""Exact per-anchor retrieval references and saved-mask audits; no forwards.

All tensors stay on the private execution server. Preparing support reads the
qualified outcome-free partition caches. Final auditing deserializes saved
checkpoint arrays solely to compare pair-valid masks, not to rerun the model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.semantic_b0b7.pretext import PairMetadata,negative_support


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False,sort_keys=True);f.write('\n')


def exact_direction(valid,negatives):
    valid,negatives=np.asarray(valid),np.asarray(negatives)
    if valid.dtype!=bool or negatives.dtype!=bool or valid.shape!=negatives.shape or valid.ndim!=2 or valid.shape[0]!=valid.shape[1]:
        raise ValueError('PAIR_SUPPORT_MUST_BE_SQUARE_BOOLEAN')
    support=valid & negatives & ~np.eye(len(valid),dtype=bool)
    counts=support.sum(1);covered=np.diag(valid)&(counts>0)
    values,freq=np.unique(counts[covered],return_counts=True)
    return {'anchors':len(valid),'anchors_covered':int(covered.sum()),
        'exact_chance_mean':float(np.mean(1./(counts[covered]+1))) if covered.any() else None,
        'negative_count_mean':float(counts[covered].mean()) if covered.any() else None,
        'negative_count_histogram':[{'negative_count':int(a),'anchors':int(b)} for a,b in zip(values,freq)]}


def compute_scopes(metadata,valid):
    n=len(valid);scopes=[('global',None,np.ones(n,bool))]
    for name in ('stage','missing_pattern','cohort'):
        scopes.extend((name,str(group),metadata[name]==group) for group in np.unique(metadata[name]))
    rows=[]
    for scope,group,use in scopes:
        meta=PairMetadata(**{k:v[use] for k,v in metadata.items()});mask=valid[np.ix_(use,use)]
        for mode in ('matched','random'):
            negatives=(np.zeros_like(mask,dtype=bool) if len(mask)<2 else negative_support(meta,mode=mode)[0])
            for direction,v,neg in (('image_to_gene',mask,negatives),('gene_to_image',mask.T,negatives.T)):
                rows.append({'scope':scope,'group':group,'mode':'hard' if mode=='matched' else mode,
                             'direction':direction,**exact_direction(v,neg)})
    return rows


def append_reference(metrics,rows):
    output=copy.deepcopy(metrics)
    for row in rows:
        parent=output if row['scope']=='global' else output['strata'][row['scope']][row['group']]
        direction=parent['retrieval_'+row['mode']][row['direction']]
        if direction['anchors_covered']!=row['anchors_covered']:
            raise ValueError('RETRIEVAL_AND_CHANCE_ELIGIBLE_SUPPORT_DIFFER')
        old=direction['mean_negative_count'];new=row['negative_count_mean']
        if (old is None)!=(new is None) or old is not None and abs(old-new)>1e-12:
            raise ValueError('RETRIEVAL_AND_CHANCE_NEGATIVE_COUNT_DIFFER')
        chance=row['exact_chance_mean'];top1=direction['top1_tie_averaged']
        direction['exact_chance_mean']=chance
        direction['top1_minus_exact_chance']=top1-chance if top1 is not None and chance is not None else None
    return output


def prepare(args):
    from scripts.semantic_b0b7_p1r import validate_scope,lock_binding
    manifest=read(args.inputs_manifest);master=read(args.lock);supports={};reports=[]
    args.output.mkdir(parents=True,exist_ok=False)
    for row in manifest['partitions']:
        if sha(row['file'])!=row['sha256']:raise ValueError('PARTITION_FILE_CHANGED')
        data=torch.load(row['file'],map_location='cpu',weights_only=False)
        lock_binding(args.lock,data,row['sha256'])
        _,indices=validate_scope(data,data['train_indices'],data['valid_indices'],evaluate_valid=True)
        indices=np.asarray(indices,int)
        metadata={k:np.asarray(v)[indices] for k,v in data['metadata'].items()}
        gene_valid=np.asarray(data['omics']['measured'])[indices].any(axis=(1,2))
        valid=np.broadcast_to(gene_valid[None,:],(len(indices),len(indices))).copy()
        partition=int(row['partition']);rows=compute_scopes(metadata,valid)
        supports[partition]={'metadata':metadata,'pair_valid':valid,'input_sha256':row['sha256'],'rows':rows}
        reports.append({'partition':partition,'input_sha256':row['sha256'],'rows':rows,
                        'derived_pair_valid_sha256':hashlib.sha256(valid.tobytes()).hexdigest()})
        del data
    artifact=args.output/'CHANCE_SUPPORT.private.pt';torch.save(supports,artifact)
    receipt={'schema':'semantic_b0b7_p1r_chance_support_v1','status':'SUPPORT_PREPARED_PENDING_SAVED_MASK_AUDIT',
        'master_lock_sha256':sha(args.lock),'input_registry_sha256':sha(args.inputs_manifest),'source_sha256':sha(__file__),
        'support_artifact_sha256':sha(artifact),'partitions':reports,'neural_model_forwards':0,'fits':0,
        'outcomes_loaded':False,'checkpoint_selection_changed':False,'CI_computed':False}
    write(args.output/'CHANCE_SUPPORT_BINDING.json',receipt)
    return receipt


def audit(args):
    binding=read(args.support_root/'CHANCE_SUPPORT_BINDING.json')
    artifact=args.support_root/'CHANCE_SUPPORT.private.pt'
    if sha(artifact)!=binding['support_artifact_sha256'] or sha(__file__)!=binding['source_sha256']:
        raise ValueError('CHANCE_SUPPORT_OR_SOURCE_CHANGED')
    support=torch.load(artifact,map_location='cpu',weights_only=False)
    report=read(args.matrix_root/'MATRIX_REPORT.private.json')
    if report['status']!='COMPLETE_ACTUAL_REQUESTED_MATRIX':raise ValueError('MATRIX_NOT_COMPLETE')
    checked=[];metrics=[];new_state_count=0
    for cell in report['cells']:
        directory=args.matrix_root/'cells'/cell['key'];receipt=read(directory/'cell.private.json')
        if sha(directory/'cell.private.json')!=cell['receipt_sha256']:raise ValueError('CELL_RECEIPT_CHANGED')
        s=support[cell['partition']]
        for info in receipt['states']:
            path=directory/info['file']
            if path.resolve().parent!=directory.resolve() or sha(path)!=info['sha256']:
                raise ValueError('SAVED_STATE_SHA_CHANGED')
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            actual=np.asarray(checkpoint['expected_replay']['pair_valid'])
            if actual.dtype!=bool or not np.array_equal(actual,s['pair_valid']):
                raise ValueError('ACTUAL_SAVED_MASK_DIFFERS_FROM_METADATA_CHANCE_SUPPORT')
            if checkpoint['input_sha256']!=s['input_sha256'] or checkpoint['master_lock_sha256']!=report['master_lock_sha256']:
                raise ValueError('SAVED_STATE_SCOPE_CHANGED')
            checked.append({'key':cell['key'],'identity':'NEW_FIT','epoch':info['epoch'],'sha256':info['sha256'],
                            'pair_valid_bytes_equal_to_metadata_support':True})
            new_state_count+=1
            del checkpoint
        metrics.append({'key':cell['key'],'partition':cell['partition'],'variant':cell['variant'],'seed':cell['seed'],
                        'metrics':append_reference(cell['selected_validation'],s['rows'])})
    if new_state_count!=4*len(report['cells']):raise ValueError('INCOMPLETE_SAVED_MASK_AUDIT')
    if args.original_fits_root:
        for ref in report['historical_references']['references']:
            arm=ref['variant'].removeprefix('ORIGINAL_')
            directory=args.original_fits_root/'cells'/(arm+'_seed'+str(ref['seed']))
            if sha(directory/'cell.private.json')!=ref['source_receipt_sha256']:
                raise ValueError('HISTORICAL_REFERENCE_RECEIPT_CHANGED')
            receipt=read(directory/'cell.private.json');s=support[0]
            for info in receipt['states']:
                path=directory/info['file']
                if path.resolve().parent!=directory.resolve() or sha(path)!=info['sha256']:
                    raise ValueError('HISTORICAL_REFERENCE_STATE_CHANGED')
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                actual=np.asarray(checkpoint['expected_validation']['pair_valid'])
                if actual.dtype!=bool or not np.array_equal(actual,s['pair_valid']):
                    raise ValueError('HISTORICAL_REFERENCE_SAVED_MASK_CHANGED')
                checked.append({'key':ref['variant']+'_seed'+str(ref['seed']),'identity':'REPLAY_ONLY_REFERENCE',
                    'epoch':info['epoch'],'sha256':info['sha256'],'pair_valid_bytes_equal_to_metadata_support':True})
                del checkpoint
            metrics.append({'key':'reference_p0_'+ref['variant']+'_seed'+str(ref['seed']),
                'partition':0,'variant':ref['variant'],'seed':ref['seed'],'identity':'REPLAY_ONLY_REFERENCE',
                'metrics':append_reference(ref['selected_validation'],s['rows'])})
    result={'schema':'semantic_b0b7_p1r_exact_chance_saved_mask_audit_v1','status':'COMPLETE_ZERO_FORWARD_SAVED_MASK_AUDIT',
        'source_sha256':sha(__file__),'master_lock_sha256':report['master_lock_sha256'],
        'matrix_report_sha256':sha(args.matrix_root/'MATRIX_REPORT.private.json'),
        'support_binding_sha256':sha(args.support_root/'CHANCE_SUPPORT_BINDING.json'),
        'partitions':binding['partitions'],'checkpoint_masks_checked':checked,'selected_metric_projections':metrics,
        'new_fit_state_masks_checked':new_state_count,'historical_reference_state_masks_checked':len(checked)-new_state_count,
        'checkpoint_arrays_deserialized':True,'checkpoint_weights_not_used_for_forward':True,
        'neural_model_forwards':0,'new_fits':0,'checkpoint_selection_changed':False,
        'DSS_OS_PFI_or_time_events_accessed':False,'CI_computed':False,
        'definition':'eligible-anchor mean of reciprocal actual negative count plus one; never reciprocal of mean count'}
    args.output.mkdir(parents=True,exist_ok=False);write(args.output/'EXACT_CHANCE_AND_SAVED_MASK_AUDIT.private.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','audit'))
    p.add_argument('--inputs-manifest',type=Path);p.add_argument('--lock',type=Path)
    p.add_argument('--support-root',type=Path);p.add_argument('--matrix-root',type=Path)
    p.add_argument('--original-fits-root',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=prepare(a) if a.mode=='prepare' else audit(a)
    print(json.dumps({'status':r['status'],'neural_model_forwards':0,'new_fits':0}))
