"""Shared R5 private I/O, unchanged teacher qualification, and source guards."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bgb_en_run import load_inputs,write,torch_write
from src.bgb_en.protocol import fixed_inner_splits,array_hash
from src.bgb_en.r5_protocol import (sha256,require,make_protocol_lock,write_protocol_lock,
    verify_protocol_lock,rna_pattern_labels,hungarian_query_mismatch)
from src.bgb_en.training import PatchBank
from src.bgb_en.data import read_patch_cache

MAIN_CONFIG='configs/bgb_en/r5_main_v1.json'
MECH_CONFIG='configs/bgb_en/r5_mechanism_v1.json'
NEW_SOURCES=('src/bgb_en/r5_readout.py','src/bgb_en/r5_protocol.py','src/bgb_en/r5_training.py',
    'scripts/bgb_en_r5_common.py','scripts/bgb_en_r5_run.py','scripts/bgb_en_r5_qualify.py',
    'scripts/bgb_en_r5_report.py','scripts/bgb_en_r5_replay.py','scripts/bgb_en_r5_schedule.py',
    'src/bgb_en/replay.py','src/phase10_trimodal/math.py','src/phase10_trimodal/reference.py')


def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--private',type=Path,required=True)
    p.add_argument('--primary-private',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true')
    return p


def context(args,*,locked=True):
    original=argparse.Namespace(assets=args.assets,config=ROOT/'configs/bgb_en/prototype_v1.json')
    assets,legacy_config,raw,member,lineage=load_inputs(original)
    config=json.loads((ROOT/MAIN_CONFIG).read_text())
    training=config['training'];readout=config['readout']
    require(training['epochs']==80 and training['evaluation_epochs']==[20,40,80]
            and training['batch_size']==8 and training['learning_rate']==.001
            and training['parameter_L2']==1e-4 and training['directional_prior_L2']==1e-4,
            'R5_FIXED_TRAINING_SCHEDULE_CHANGED')
    require(readout['gamma']==.1 and readout['temperature']==4 and readout['rank']==4
            and readout['sham_rank']==75,'R5_FIXED_READOUT_CONTRACT_CHANGED')
    require(len(raw['t'])==512 and int(raw['e'].sum())==65,'CANONICAL_DSS_COHORT_CHANGED')
    require(int(raw['pfi_valid'].sum())==510 and int(raw['pfi_e'][raw['pfi_valid']].sum())==129,
            'ALIGNED_PFI_NOT_CANONICAL')
    patterns,pattern_audit=rna_pattern_labels(np.isfinite(raw['rna']))
    require(pattern_audit['complete_patients']==309 and pattern_audit['not_fully_measured_patients']==203,
            'RNA_MISSING_GROUPS_CHANGED')
    if locked:
        lock=json.loads((args.private/'LOCK.json').read_text());verify_protocol_lock(lock,ROOT)
        fingerprints={key:raw['audit'][key] for key in ('cohort_sha256','endpoints_sha256')}
        require(all(lock['data_fingerprints'][key]==value for key,value in fingerprints.items()),
                'DATA_LINEAGE_CHANGED_AFTER_R5_LOCK')
    else:lock=None
    return dict(assets=assets,legacy_config=legacy_config,raw=raw,member=member,lineage=lineage,
                config=config,lock=lock,patterns=patterns,pattern_audit=pattern_audit)


def qualified_prepared(args,ctx,partition):
    stem=f'outer1_inner{partition}';receipt=args.primary_private/'prepared'/(stem+'.json')
    payload=receipt.with_suffix('.pt');metadata=json.loads(receipt.read_text())
    require(metadata['status']=='COMPLETE' and metadata['lineage']==ctx['lineage'],
            'UPSTREAM_PREPARED_OR_SOURCE_CHANGED')
    require(sha256(payload)==metadata['sha256'],'UPSTREAM_PREPARED_PAYLOAD_CHANGED')
    if ctx['lock'] is not None:
        require(ctx['lock']['prepared_receipt_sha256'][stem]==sha256(receipt),'LOCKED_TEACHER_RECEIPT_CHANGED')
    prepared=torch.load(payload,map_location='cpu',weights_only=False)
    train,valid=fixed_inner_splits(ctx['raw']['folds'],ctx['raw']['e'],outer=1)[partition]
    require(np.array_equal(prepared['train'],train) and np.array_equal(prepared['valid'],valid),
            'R5_INNER_SPLIT_CHANGED')
    require(np.all(ctx['raw']['folds'][np.r_[train,valid]]!=1),'OUTER_TEST_ENTERED_R5_TRAINING')
    require(prepared['fit_hash']==array_hash(train) and prepared['valid_hash']==array_hash(valid),
            'PREPROCESSING_SCOPE_CHANGED')
    teacher_held=[]
    for teacher in prepared['teachers']:
        fit,held=teacher['train'],teacher['held'];teacher_held.extend(held.tolist())
        require(not np.intersect1d(fit,held).size and set(fit)|set(held)==set(train),
                'TEACHER_PATIENT_SCOPE_INVALID')
        require(not np.intersect1d(np.r_[fit,held],valid).size,'VALIDATION_ENTERED_TEACHER')
        require(teacher['preprocessing']['fit_hash']==array_hash(fit),'TEACHER_PREPROCESSING_NOT_LOCAL')
    require(sorted(teacher_held)==sorted(train.tolist()) and prepared['pair_teacher_rule']=='both_pair_members_in_one_excluded_teacher_block',
            'SAME_TEACHER_CALIBRATION_CONTRACT_CHANGED')
    require(prepared['scalar_audit']['gradient_max']<=1e-6,'SCALAR_CALIBRATION_NOT_CONVERGED')
    return prepared,dict(prepared_receipt_sha256=sha256(receipt),prepared_artifact_sha256=sha256(payload),
        train_hash=array_hash(train),valid_hash=array_hash(valid),
        train_n=len(train),valid_n=len(valid),teachers_reused=3,teacher_refits=0,
        upstream_supervised_fits_reused_not_new=14,risk_distribution=metadata['risk_distribution'],
        scalar_beta=prepared['beta'])


def create_lock(args,ctx):
    prepared={};receipts={}
    for part in (0,1,2):
        _,audit=qualified_prepared(args,ctx,part);prepared[str(part)]=audit
        receipts[f'outer1_inner{part}']=audit['prepared_receipt_sha256']
    fingerprints={key:ctx['raw']['audit'][key] for key in ('cohort_sha256','endpoints_sha256')}
    fingerprints.update(GMT_sha256=ctx['lineage']['GMT_sha256'],
        legacy_group_sha256=ctx['lineage']['legacy_group_sha256'],
        patch_regions_receipt_sha256=sha256(Path(ctx['assets']['patch_regions'])/'READY.json'),
        RNA_finite_mask_sha256=ctx['pattern_audit']['mask_shape_and_bytes_sha256'])
    source_files=list(ctx['lineage']['source'])+list(NEW_SOURCES)
    value=make_protocol_lock(ROOT,MAIN_CONFIG,MECH_CONFIG,source_files,
                            data_fingerprints=fingerprints,prepared_receipt_sha256=receipts)
    locked=write_protocol_lock(args.private/'LOCK.json',value)
    write(ROOT/'reports/bgb_en_r5/PROTOCOL_LOCK.json',locked)
    write(args.private/'PREPARED_REUSE_AUDIT.json',dict(status='PASS_THREE_UNCHANGED_PARTITION_LOCAL_TEACHER_BUNDLES',
        bundles=prepared,new_teacher_fits=0,outer_test_accessed=False,pattern_audit=ctx['pattern_audit']))
    return locked


def pairing(ctx,prepared,partition):
    raw=ctx['raw'];options=ctx['config']['pairing'];mapping=np.arange(512);audits={}
    for name,indices in [('train',prepared['train']),('validation',prepared['valid'])]:
        seed=options['base_seed']+partition*options['partition_seed_stride']+options['valid_seed_offset']*(name=='validation')
        current,audit=hungarian_query_mismatch(indices,missing_pattern=ctx['patterns'],cancer=raw['cancer'],
            stage=raw['clinical_raw']['AJCC_PATHOLOGIC_TUMOR_STAGE'],platform=raw['platform'],seed=seed)
        mapping[indices]=current[indices];audits[name]=audit
    return mapping,audits


def bank(ctx,device):return PatchBank(read_patch_cache(ctx['assets']['patch_regions']),device)


def public_snapshot_guard():
    """Only metadata: the old formal reports must not be rewritten by R5."""
    parent=ROOT/'reports/bgb_en'
    return {str(p.relative_to(parent)):sha256(p) for p in parent.rglob('*') if p.is_file()}


def validate_qualification(args,ctx):
    path=args.private/'QUALIFICATION.json';value=json.loads(path.read_text())
    require(value['status']=='PASS_R5_REAL_DATA_AND_LEGACY_PARITY'
            and value['contract_sha256']==ctx['lock']['contract_sha256'],
            'R5_REAL_QUALIFICATION_REQUIRED_BEFORE_FORMAL_FITS')
    return sha256(path)


def cell_name(arm,partition,seed,control='MATCHED'):
    return f'{arm}_{control}_p{partition}_seed{seed}'


def cell_contract(ctx,origin,arm,partition,seed,control,qualification_sha256):
    return dict(contract_sha256=ctx['lock']['contract_sha256'],prepared_origin=origin,
        arm=arm,partition=partition,seed=seed,training_control=control,
        qualification_sha256=qualification_sha256,epochs=80,
        main_config_sha256=sha256(ROOT/MAIN_CONFIG),mechanism_config_sha256=sha256(ROOT/MECH_CONFIG),
        outer_test_used=False)


def subgroup_diagnostics(raw,prepared,risk):
    from src.bgb_en.survival import evaluate
    from src.phase10_trimodal.reference import pairs,uno_weights
    train,valid=prepared['train'],prepared['valid'];base=prepared['base'][valid]
    groups={'RNA_COMPLETE':np.isfinite(raw['rna'][valid]).all(1),
            'RNA_BLOCK_MISSING':~np.isfinite(raw['rna'][valid]).all(1),
            'COAD':np.char.find(np.asarray(raw['cancer'][valid]).astype(str),'COAD')>=0,
            'READ':np.char.find(np.asarray(raw['cancer'][valid]).astype(str),'READ')>=0}
    result=[]
    for name,mask in groups.items():
        indices=valid[mask];n=len(indices)
        metrics=evaluate(raw['t'][train],raw['e'][train],raw['t'][indices],raw['e'][indices],risk[mask],base[mask]) if n else {}
        result.append(dict(group=name,patients=n,events=int(raw['e'][indices].sum()),metrics=metrics))
    i,j=pairs(raw['t'][valid],raw['e'][valid]);weight=uno_weights(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],tau=60.)
    margin=np.abs(base[i]-base[j]);edges=(0.,.25,.5,1.,float('inf'))
    def credit(x):
        d=x[i]-x[j];return np.where(np.abs(d)<=1e-8,.5,(d>0).astype(float))
    b,c=credit(base),credit(risk)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(margin>=lo)&(margin<hi);den=int(m.sum());u=float(weight[i[m]].sum())
        difference=c[m]-b[m]
        result.append(dict(group='CP_SCALAR_PAIR_MARGIN',lower=lo,upper=hi if np.isfinite(hi) else None,
            comparable_pairs=den,Uno_pair_weight=u,events_at_pair_left=int(len(np.unique(i[m]))),
            repair=float(np.maximum(difference,0).sum()/den) if den else None,
            damage=float(np.maximum(-difference,0).sum()/den) if den else None,
            Uno_repair=float(np.dot(weight[i[m]],np.maximum(difference,0))/u) if u else None,
            Uno_damage=float(np.dot(weight[i[m]],np.maximum(-difference,0))/u) if u else None))
    return result
