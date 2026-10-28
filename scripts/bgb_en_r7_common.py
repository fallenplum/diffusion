"""R7 independent target contracts; published R6 assets stay read-only."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import bgb_en_r6_common as old
from src.bgb_en.protocol import array_hash
from src.bgb_en.r6_pathology import source_model_state_sha256
from src.bgb_en.r6_omics import GeneValueBank
from src.bgb_en.omics import SparsePathwayEncoder
from src.bgb_en.data import sha256

require,read,write,torch_write=old.require,old.read,old.write,old.torch_write
canonical_hash=old.canonical_hash
configure_deterministic_runtime=old.configure_deterministic_runtime
validate_deterministic_runtime=old.validate_deterministic_runtime
PARTITIONS,SEEDS=(0,1,2),(31,47,83)
TARGET_ARMS=('ADD','RAW','NORM','INDEP','SLOT','GATE1','GATE2','SLOT_WRONG_RNA',
             'GATE1_WRONG_RNA','GATE1_WRONG_PATH','GATE2_WRONG_RNA')
EVALUATION_EPOCHS=(20,40,80)
DIAGNOSTIC_EPOCHS=(20,40,60,80)
FROZEN_CONTROLS=('matched_query','wrong_query','wrong_path','mean_query','interaction_off','main_off')
TARGET_SOURCE_FILES=('scripts/bgb_en_r7_common.py','scripts/bgb_en_r7_run.py','scripts/bgb_en_r7_replay.py',
    'scripts/bgb_en_r7_schedule.py','scripts/bgb_en_r7_qualify.py','scripts/bgb_en_r7_report.py',
    'scripts/bgb_en_r7_interface_audit.py',
    'reports/bgb_en_r7/PROTOCOL.md',
    'src/bgb_en/r7_survival.py','src/bgb_en/r7_training.py')


def parser(description):
    p=argparse.ArgumentParser(description=description)
    for name in ('assets','private','primary-private','source-private','config','source-config','r6-private','r6-config'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    return p


def frozen_controls_for_arm(arm):
    return tuple(c for c in FROZEN_CONTROLS if c!='wrong_path' or arm not in ('RAW','ADD'))


def legacy_args(args):
    """Never mix the new config/output namespace with the R6 asset namespace."""
    return argparse.Namespace(assets=args.assets,private=args.r6_private,primary_private=args.primary_private,
        source_private=args.source_private,config=args.r6_config,source_config=args.source_config,
        device=args.device,resume=False,launch_gate=None,launch_gate_sha256=None)


def readonly_r6_gate(args):
    """Metadata gate before context can load target arrays/source weights."""
    require(args.private.resolve()!=args.r6_private.resolve()
            and not args.private.resolve().is_relative_to(ROOT),'R7_NEW_PRIVATE_NAMESPACE_REQUIRED')
    qpath=args.r6_private/'QUALIFICATION.json';rpath=args.r6_private/'REPLAY_AUDIT.json'
    lpath=args.r6_private/'targetLOCK.json'
    lock,q,replay=read(lpath),read(qpath),read(rpath)
    lock_body={k:v for k,v in lock.items() if k not in ('contract_sha256','created_UTC')}
    require(lock.get('schema')=='bgb_en_r6_target_protocol_lock_v1'
        and canonical_hash(lock_body)==lock.get('contract_sha256') and lock['config_sha256']==sha256(args.r6_config)
        and all(sha256(ROOT/name)==digest for name,digest in lock['sources'].items()),
        'R7_READONLY_R6_LOCK_CODE_OR_CONFIG_CHANGED')
    require(q.get('status')=='PASS_R6_TARGET_REAL_DATA_AND_SOURCE_TRANSFER'
            and replay.get('status')=='PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY'
            and q['contract_sha256']==replay['contract_sha256']==lock['contract_sha256']
            and replay.get('qualification_sha256')==sha256(qpath)
            and replay.get('expected_units')==replay.get('available_units')==replay.get('passed_units')==90
            and replay.get('failed_units')==0 and not replay.get('missing_or_unreplayed_units'),
            'R7_COMPLETE_R6_ACTUAL_REPLAY_REQUIRED_BEFORE_TARGET_IO')
    expected={(a,p,s) for a in old.TARGET_ARMS for p in PARTITIONS for s in SEEDS}
    rows=replay.get('units',[])
    require(len(rows)==90 and {(r['arm'],r['partition'],r['seed']) for r in rows}==expected,
            'R7_R6_ACTUAL_MATRIX_CHANGED')
    for row in rows:
        require(row.get('unit')==f"{row['arm']}_p{row['partition']}_seed{row['seed']}",'R7_R6_UNIT_PATH_CHANGED')
        require(row.get('status')=='PASS' and row.get('actual_state_dict_forward_replayed') is True
                and row.get('source_frozen_weight_hash_replayed') is True
                and row.get('saved_query_mapping_recomputed') is True and row.get('changed_pair_credits')==0
                and np.isfinite(row.get('maximum_risk_error',np.inf)) and row['maximum_risk_error']<=2e-6
                and row.get('new_optimizer_updates')==0,'R7_R6_UNIT_NOT_ACTUALLY_REPLAYED')
        complete=args.r6_private/'cells'/row['unit']/'COMPLETE.json'
        require(sha256(complete)==row['completion_sha256'],'R7_R6_COMPLETE_RECEIPT_CHANGED')
    refs=replay.get('zero_fit_baseline_reference_replays',[])
    require(len(refs)==18 and {(r['arm'],r['partition'],r['seed']) for r in refs}
            =={(a,p,s) for a in ('B0','B1') for p in PARTITIONS for s in SEEDS}
            and all(r.get('status')=='PASS' and r.get('independent_fixed_anchor_metrics_recomputed') is True
                    and r.get('new_optimizer_fits')==0 for r in refs),'R7_R6_18_ZERO_FIT_REFERENCES_REQUIRED')
    for row in refs:
        reference=args.r6_private/'references'/f"{row['arm']}_p{row['partition']}_seed{row['seed']}.json"
        require(sha256(reference)==row['receipt_sha256'],'R7_R6_REFERENCE_RECEIPT_CHANGED')
    for control in old.FULL_SOURCE_CONTROLS:
        origin=lock['source_full_states'][control]
        directory=args.source_private/'source_cells'/f'S7_{control}_h-1_seed31'
        for name,field in (('state_epoch080.private.pt','checkpoint_sha256'),('COMPLETE.json','completion_sha256'),('REPLAY.json','replay_sha256')):
            require(sha256(directory/name)==origin[field],'R7_READONLY_SOURCE_FILE_CHANGED')
    return dict(target_lock_sha256=sha256(lpath),qualification_sha256=sha256(qpath),replay_sha256=sha256(rpath),
        contract_sha256=lock['contract_sha256'],config_sha256=sha256(args.r6_config),
        reused_actual_target_fit_count=90,new_source_fits=0,new_teacher_fits=0)


def validate_target_config(config):
    require(config.get('schema')=='bgb_en_r7_inner_v1' and tuple(config.get('trainable_arms',()))==TARGET_ARMS
            and config.get('formal_target_neural_fits')==99,'R7_LOCKED_99_ARM_MATRIX_REQUIRED')
    t=config['training']
    for name,value in dict(epochs=80,evaluation_epochs=[20,40,80],diagnostic_epochs=[20,40,60,80],
        primary_checkpoint_epoch=80,optimizer='Adam',learning_rate=.001,batch_size=8,dropout=0.,
        parameter_L2=.0001,total_delta_L2=.1,interaction_delta_L2=.1,gradient_clip_norm=5.,
        gradient_cache_replay_max_error=2e-5,saved_risk_replay_max_error=2e-6).items():
        require(t.get(name)==value,'R7_FIXED_TRAINING_CONTRACT_CHANGED_'+name.upper())
    proto=config['prototype']
    require(proto.get('outer_fold')==1 and proto.get('inner_partitions')==list(PARTITIONS)
        and proto.get('neural_seeds')==list(SEEDS) and proto.get('outer_test_access') is False,
        'R7_INNER_SPLIT_OR_SEEDS_CHANGED')
    require(config.get('primary_candidate')=='GATE1' and config.get('primary_endpoint')=='DSS'
        and config.get('secondary_endpoint')=='PFI_frozen_DSS_risk_only','R7_CANDIDATE_OR_ENDPOINT_CHANGED')
    return config


def qualified_prepared(args,ctx,partition):
    prepared,origin=old.qualified_prepared(ctx['r6_args'],ctx['r6_ctx'],partition)
    if ctx.get('lock') is not None:
        require(ctx['lock']['prepared_receipt_sha256'][str(partition)]==origin['prepared_receipt_sha256'],
                'R7_PREPARED_RECEIPT_CHANGED')
    return prepared,origin


def context(args,*,locked=True):
    gate=readonly_r6_gate(args)
    previous=legacy_args(args);r6=old.context(previous)
    config=validate_target_config(read(args.config))
    ctx=dict(r6_args=previous,r6_ctx=r6,upstream_r6=gate,config=config,raw=r6['raw'],member=r6['member'],
        sources=r6['sources'],source_audits=r6['source_audits'],lineage=r6['lineage'],lock=None)
    if locked:
        ctx['lock']=read(args.private/'targetLOCK.json');verify_target_lock(args,ctx)
    return ctx


def source_names(args,ctx):
    names=set(ctx['r6_ctx']['lock']['sources'])|set(TARGET_SOURCE_FILES)
    names.add(str(args.config.resolve().relative_to(ROOT)).replace('\\','/'))
    for candidate in ('src/bgb_en/r7_protocol.py','scripts/bgb_en_r7_interface_audit.py'):
        if (ROOT/candidate).exists():names.add(candidate)
    return sorted(names)


def create_lock(args,ctx):
    require(ctx.get('lock') is None,'R7_LOCK_CREATION_REQUIRES_UNLOCKED_CONTEXT')
    origins={str(p):qualified_prepared(args,ctx,p)[1]['prepared_receipt_sha256'] for p in PARTITIONS}
    names=source_names(args,ctx)
    require(all((ROOT/name).is_file() for name in names),'R7_ALL_FINAL_SOURCE_FILES_REQUIRED_BEFORE_LOCK')
    value=dict(schema='bgb_en_r7_target_protocol_lock_v1',status='LOCKED_BEFORE_R7_FITS',
        sources={name:sha256(ROOT/name) for name in names},config_sha256=sha256(args.config),
        source_config_sha256=sha256(args.source_config),upstream_r6=ctx['upstream_r6'],
        prepared_receipt_sha256=origins,source_full_states=ctx['source_audits'],
        calibration_receipt_sha256={f'p{p}_seed{s}':sha256(args.private/'calibration'/f'p{p}_seed{s}.json')
            for p in PARTITIONS for s in SEEDS},
        fixed_deterministic_runtime=validate_deterministic_runtime(),planned_neural_fits=99,
        planned_target_optimizer_updates=7920,teacher_refits=0,source_refits=0,outer_test_access=False)
    value['contract_sha256']=canonical_hash(value)
    path=args.private/'targetLOCK.json'
    if path.exists():require(read(path)==value,'R7_EXISTING_LOCK_PRESERVED_NO_OVERWRITE')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    return value


def verify_target_lock(args,ctx):
    lock=ctx['lock'];body={k:v for k,v in lock.items() if k!='contract_sha256'}
    require(lock.get('schema')=='bgb_en_r7_target_protocol_lock_v1' and lock.get('status')=='LOCKED_BEFORE_R7_FITS'
        and canonical_hash(body)==lock['contract_sha256'],'R7_IMMUTABLE_PROTOCOL_LOCK_CHANGED')
    require(lock['upstream_r6']==ctx['upstream_r6'] and lock['source_full_states']==ctx['source_audits']
        and lock['config_sha256']==sha256(args.config) and lock['source_config_sha256']==sha256(args.source_config)
        and lock['fixed_deterministic_runtime']==validate_deterministic_runtime(),'R7_UPSTREAM_CONFIG_OR_RUNTIME_CHANGED')
    require(set(source_names(args,ctx))==set(lock['sources'])
        and all(sha256(ROOT/name)==digest for name,digest in lock['sources'].items()),'R7_LOCKED_SCIENTIFIC_SOURCE_CHANGED')


def public_snapshot_guard():
    return {str(path.relative_to(ROOT)):sha256(path) for name in ('bgb_en','bgb_en_r5','bgb_en_r6')
        for path in (ROOT/'reports'/name).rglob('*') if path.is_file()}


def validate_qualification(args,ctx):
    path=args.private/'QUALIFICATION.json';value=read(path)
    require(value.get('status')=='PASS_R7_REAL_DATA_AND_READONLY_R6_TRANSFER'
        and value.get('contract_sha256')==ctx['lock']['contract_sha256'],'R7_REAL_DATA_QUALIFICATION_REQUIRED')
    return sha256(path)


class R7PatientModel(nn.Module):
    def __init__(self,arm,prepared,membership,raw_symbols,source,encoded_bank,device,*,seed=31):
        super().__init__()
        from src.bgb_en.r7_survival import R7SurvivalModel
        self.arm,self.device,self.patch_bank=arm,device,encoded_bank
        self.fit_indices,self.valid_indices=np.asarray(prepared['train']),np.asarray(prepared['valid'])
        self.inputs={k:torch.as_tensor(prepared['features'][k],device=device) for k in ('rna','rna_mask')}
        self.offset=torch.as_tensor(prepared['base'],device=device,dtype=torch.float32)
        self.query_map=np.arange(len(self.offset));self.mean_context=None
        self.intervention='wrong_query' if arm.endswith('_WRONG_RNA') else 'wrong_path' if arm.endswith('_WRONG_PATH') else 'matched'
        self.gene_bank=GeneValueBank(raw_symbols,source.gene_reader.mapping.gene_symbols).to(device)
        self.core=R7SurvivalModel(arm,deepcopy(source),source_kind='MATCHED_ST',init_seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.hallmark=SparsePathwayEncoder(membership,mode='pool',width=32,patterns=4)
        self.hallmark.pool_query.requires_grad_(False);self.hallmark.risk_head.requires_grad_(False)
        self.to(device)

    def genes(self,indices):return self.gene_bank(self.inputs['rna'][indices],self.inputs['rna_mask'][indices])

    def refresh_training_query_mean(self,batch_size=16):
        from src.bgb_en.r6_survival import MeanGeneQueryContext
        total=counts=None
        with torch.no_grad():
            for start in range(0,len(self.fit_indices),batch_size):
                idx=torch.as_tensor(self.fit_indices[start:start+batch_size],device=self.device)
                row=self.core.foundation.encode_gene_programs(self.genes(idx));mask=row['program_mask']
                num=torch.where(mask[...,None],row['tokens'],torch.zeros_like(row['tokens'])).sum(0);n=mask.sum(0)
                total=num if total is None else total+num;counts=n if counts is None else counts+n
        self.mean_context=MeanGeneQueryContext((total/counts.clamp_min(1)[:,None]).detach(),(counts>0).detach(),len(self.fit_indices))
        return self.mean_context

    def forward(self,indices,control=None):
        indices=torch.as_tensor(indices,device=self.device,dtype=torch.long)
        other=torch.as_tensor(self.query_map[indices.detach().cpu().numpy()],device=self.device)
        mode='matched' if control=='matched_query' else self.intervention if control is None else control
        conditioning=self.intervention if control in ('interaction_off','main_off') else mode
        patch,neighborhood=self.patch_bank.select(indices);genes=self.genes(indices)
        aux=self.hallmark(**{k:v[indices] for k,v in self.inputs.items()})
        kwargs=dict(neighborhood=neighborhood,hallmark_tokens=aux['tokens'],hallmark_mask=aux['token_mask'])
        if conditioning=='wrong_query':
            foreign=self.core.foundation.encode_gene_programs(self.genes(other))
            kwargs.update(query_context=foreign['tokens'],query_mask=foreign['program_mask'])
        elif conditioning=='wrong_path':
            foreign_patch,foreign_neighbors=self.patch_bank.select(other)
            foreign=self.core.unconditional_patch_base(foreign_patch,self.genes(other),neighborhood=foreign_neighbors)
            kwargs.update(path_query_context=foreign['states'],path_query_mask=foreign['program_mask'])
        elif conditioning=='mean_query':
            require(self.mean_context is not None,'R7_TRAIN_ONLY_MEAN_QUERY_REQUIRED');kwargs['mean_context']=self.mean_context
        output=self.core(patch,genes,self.offset[indices],intervention='off' if mode=='interaction_off' else
            conditioning if mode=='main_off' else mode,**kwargs)
        if mode=='main_off':
            output['main_delta']=torch.zeros_like(output['main_delta'])
            for key in ('path_main_delta','gene_main_delta','hallmark_main_delta'):output[key]=torch.zeros_like(output[key])
            output['delta']=output['total_delta']=output['cross_delta'];output['risk']=output['offset']+output['delta']
        output['hallmark_auxiliary']=aux
        if output.get('gates'):
            patient={}
            for direction,label in (('path','path'),('gene','gene')):
                if direction=='path':mask=output['patch_base_read']['program_mask']&output['patch_cond_read']['program_mask']
                else:
                    first,last=output['reverse_base_read'],output['reverse_cond_read']
                    mask=first['program_mask']&last['program_mask']
                value=output['gates'][direction].squeeze(-1);count=mask.sum(-1)
                mean=torch.where(mask,value,torch.zeros_like(value)).sum(-1)/count.clamp_min(1)
                variance=torch.where(mask,(value-mean[:,None]).square(),torch.zeros_like(value)).sum(-1)/count.clamp_min(1)
                patient['gate_'+label+'_mean']=mean;patient['gate_'+label+'_sd']=variance.sqrt()
                patient['gate_'+label+'_valid_programs']=count.to(mean.dtype)
                if len(output.get('state_trace',[]))==2:patient['gate_'+label+'_round2_mean']=mean
            output['diagnostic_patient_scalars']=patient
        return output

    def regularization(self):
        return self.core.regularization()+1e-4*sum(p.square().sum() for p in self.hallmark.parameters() if p.requires_grad)

    def parameter_budget(self):
        return dict(registered=sum(p.numel() for p in self.parameters()),requires_grad=sum(p.numel() for p in self.parameters() if p.requires_grad),
            core=self.core.parameter_budget(),frozen_source_in_optimizer=False,same_effective_capacity_or_FLOPs_claimed=False)


def construct(args,ctx,prepared,arm,seed):
    require(arm in TARGET_ARMS and seed in SEEDS,'R7_MODEL_OUTSIDE_LOCKED_MATRIX')
    bank,receipt=old.patch_bank(ctx['r6_args'],ctx['r6_ctx'],'MATCHED',create=False)
    model=R7PatientModel(arm,prepared,ctx['member'],ctx['raw']['rna_symbols'],ctx['sources']['MATCHED'],bank,args.device,seed=seed)
    model.query_map,pairing=old.pairing(ctx['r6_ctx'],prepared,args.partition)
    path=args.private/'calibration'/f'p{args.partition}_seed{seed}.json';calibration=read(path)
    require(calibration['status']=='PASS_R7_TRAIN_RMS_CALIBRATION'
        and calibration['origin']==calibration_origin(args,ctx,prepared,seed,receipt,shared_state_hash(model)),
        'R7_SHARED_NORM_CALIBRATION_ORIGIN_CHANGED')
    if ctx.get('lock') is not None:
        require(sha256(path)==ctx['lock']['calibration_receipt_sha256'][f'p{args.partition}_seed{seed}'],
            'R7_LOCKED_CALIBRATION_RECEIPT_CHANGED')
    model.core.set_calibration_scales(**calibration['calibration']['scales'])
    model.calibration=calibration['calibration']
    model.refresh_training_query_mean()
    return model,dict(source_key='MATCHED',source_origin=ctx['source_audits']['MATCHED'],
        patch_cache_receipt_sha256=receipt,pairing=pairing,calibration=calibration['calibration'],calibration_receipt_sha256=sha256(path))


def shared_state_hash(model):
    import hashlib
    excluded=('core.slot_path_weight','core.slot_gene_weight','core.path_state_gate.','core.gene_state_gate.')
    scale_names={'core.'+k for k in ('context_rms','patch_rms','query_rms','gene_key_rms','evidence_rms','reverse_query_rms')}
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name in scale_names or any(name.startswith(prefix) for prefix in excluded):continue
        value=value.detach().cpu().contiguous();digest.update(name.encode()+str(value.dtype).encode()+str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def calibration_origin(args,ctx,prepared,seed,patch_receipt,initial_shared_sha):
    return dict(partition=args.partition,seed=seed,train_hash=array_hash(prepared['train']),
        prepared_receipt_sha256=qualified_prepared(args,ctx,args.partition)[1]['prepared_receipt_sha256'],
        source_state_sha256=ctx['source_audits']['MATCHED']['complete_source_state_sha256'],
        patch_cache_receipt_sha256=patch_receipt,initial_shared_state_sha256=initial_shared_sha,
        core_source_sha256=sha256(ROOT/'src/bgb_en/r7_survival.py'),common_source_sha256=sha256(__file__),
        config_sha256=sha256(args.config),calibration_model='untrained_NORM',validation_used=False,outcomes_used=False)


def create_calibration_cache(args,ctx,prepared,seed):
    require(ctx.get('lock') is None,'R7_CALIBRATION_CACHE_CREATION_BEFORE_IMMUTABLE_LOCK_ONLY')
    bank,receipt=old.patch_bank(ctx['r6_args'],ctx['r6_ctx'],'MATCHED',create=False)
    model=R7PatientModel('NORM',prepared,ctx['member'],ctx['raw']['rna_symbols'],ctx['sources']['MATCHED'],bank,args.device,seed=seed)
    origin=calibration_origin(args,ctx,prepared,seed,receipt,shared_state_hash(model))
    path=args.private/'calibration'/f'p{args.partition}_seed{seed}.json'
    if path.exists():
        prior=read(path);require(prior['status']=='PASS_R7_TRAIN_RMS_CALIBRATION' and prior['origin']==origin,
            'R7_EXISTING_CALIBRATION_PRESERVED_ORIGIN_CHANGED');return prior
    calibration=calibrate_training_rms(model,prepared['train'])
    result=dict(status='PASS_R7_TRAIN_RMS_CALIBRATION',origin=origin,calibration=calibration,new_optimizer_updates=0)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


def calibrate_training_rms(model,indices,batch_size=8):
    """Core supplies raw float32 square sums; accumulate full train in float64."""
    require(np.array_equal(np.asarray(indices),model.fit_indices),'R7_RMS_CALIBRATION_NOT_CURRENT_TRAIN_ONLY')
    # Final statistic names are fixed with the core, not chosen from performance.
    stages=(('raw',{'raw_G0':'context_rms','raw_adapted_patch':'patch_rms','raw_base_query':'query_rms','raw_gene_key':'gene_key_rms'}),
            ('base',{'scaled_P0':'evidence_rms'}),('reverse',{'projected_reverse_query':'reverse_query_rms'}))
    scales={};audit={}
    previous=model.training;model.eval()
    with torch.no_grad():
        for stage,mapping in stages:
            sums={};counts={}
            for start in range(0,len(indices),batch_size):
                idx=torch.as_tensor(indices[start:start+batch_size],device=model.device)
                patch,neighbors=model.patch_bank.select(idx)
                row=model.core.calibration_batch(patch,model.genes(idx),neighborhood=neighbors,stage=stage)['statistics']
                for name,value in row.items():
                    ss=float(value['sum_squares']);count=int(value['count'])
                    require(np.isfinite(ss) and ss>=0 and count>=0,'R7_RMS_NONFINITE_OR_NEGATIVE_STATISTIC')
                    sums[name]=sums.get(name,0.)+ss;counts[name]=counts.get(name,0)+count
            for statistic,buffer in mapping.items():
                require(counts.get(statistic,0)>0,'R7_RMS_EMPTY_CALIBRATION_INTERFACE_'+statistic)
                scales[buffer]=max(float(np.sqrt(sums[statistic]/counts[statistic])),1e-6)
            model.core.set_calibration_scales(**scales)
            audit[stage]={name:dict(sum_squares=sums[name],count=counts[name]) for name in sums}
    model.train(previous)
    return dict(train_hash=array_hash(np.asarray(indices)),training_patients=len(indices),scales=scales,statistics=audit,
        train_only=True,outcomes_accessed=False,new_optimizer_updates=0,fixed_after_initialization=True)
