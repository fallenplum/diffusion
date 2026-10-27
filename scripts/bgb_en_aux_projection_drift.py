"""CPU-only parameter drift from verified source projection to final80 F11.

This audit does not construct a model, evaluate a patient or update weights.
Parameter movement is not a measurement of semantic retention or forgetting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.bgb_en_report_aux import (GROUPS,SOURCE_NAMES,canonical_hash,formatted,
                                      mean,require,sha256,verify_source_receipt)

NORMALIZATION='float32_per_row_L2_norm_division; reject_norm_le_1e-12'
PROJECT_SHAPES={'0.weight':(64,1536),'0.bias':(64,),'1.weight':(64,),'1.bias':(64,)}


def tensor_hash(value):
    array=np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def drift_statistics(source,final):
    a,b=np.asarray(source),np.asarray(final)
    require(a.shape==b.shape and a.size>0 and np.isfinite(a).all() and np.isfinite(b).all(),
            'Projection tensor shape or finite-value mismatch')
    aa,bb=a.astype(np.float64).ravel(),b.astype(np.float64).ravel()
    an,bn=np.linalg.norm(aa),np.linalg.norm(bb);change=bb-aa
    return dict(parameters=int(a.size),shape=list(a.shape),source_RMS=float(np.sqrt(np.mean(aa**2))),
        final_RMS=float(np.sqrt(np.mean(bb**2))),delta_RMS=float(np.sqrt(np.mean(change**2))),
        relative_L2=float(np.linalg.norm(change)/an) if an>0 else None,
        cosine=float(np.clip(np.dot(aa,bb)/(an*bn),-1.,1.)) if an>0 and bn>0 else None,
        relative_L2_status='DEFINED' if an>0 else 'NA_ZERO_SOURCE_NORM',
        cosine_status='DEFINED' if an>0 and bn>0 else 'NA_ZERO_SOURCE_OR_FINAL_NORM',
        source_tensor_sha256=tensor_hash(a),final_tensor_sha256=tensor_hash(b))


def source_projection(private_root,group):
    import torch
    folder=Path(private_root)/SOURCE_NAMES[group]/'fit_06'
    receipt=verify_source_receipt(folder/'COMPLETE.json',group,6)
    path=folder/'UNI2_PROJECT_TRANSFER.pt'
    require('UNI2_PROJECT_TRANSFER.pt' in receipt['artifacts_sha256']
            and sha256(path)==receipt['artifacts_sha256']['UNI2_PROJECT_TRANSFER.pt'],
            'Source transfer projection does not match completed source receipt')
    transfer=torch.load(path,map_location='cpu',weights_only=True)
    require(transfer.get('schema')=='bgb_en_external_projection_transfer_v2_unitnorm'
            and transfer.get('role')=='full_source_transfer' and transfer.get('epochs')==20
            and transfer.get('source_donors')==6 and transfer.get('input_normalization')==NORMALIZATION
            and transfer.get('TCGA_DSS_read') is False and transfer.get('original_ST_test_read') is False,
            'Source transfer normalization, role or outcome-exclusion contract differs')
    require(transfer.get('source_control','TRUE')==group,'Source transfer control identity differs')
    require(transfer.get('source_data_sha256')==receipt['source_data_sha256']
            and transfer.get('source_manifest_sha256')==receipt['source_manifest_sha256'],
            'Source transfer and source fit annotation identity differ')
    state=transfer.get('project_state_dict',{})
    require(set(state)==set(PROJECT_SHAPES),'Unexpected source projection tensor set')
    arrays={}
    for key,value in state.items():
        require(isinstance(value,torch.Tensor) and value.device.type=='cpu'
                and tuple(value.shape)==PROJECT_SHAPES[key],'Source projection dimensions/device differ')
        arrays[key]=value.detach().numpy().copy()
        require(np.isfinite(arrays[key]).all(),'Source projection contains nonfinite weights')
    return arrays,dict(source_group=group,source_transfer_sha256=sha256(path),
        source_fit_receipt_sha256=receipt['receipt_sha256'],source_data_sha256=receipt['source_data_sha256'],
        source_manifest_sha256=receipt['source_manifest_sha256'],source_epochs=20,source_donors=6,
        input_normalization=NORMALIZATION,transferred_parameters=sum(a.size for a in arrays.values()))


def audit_final_cell(path,group,part,seed,source,source_audit):
    import torch
    receipt=json.loads(path.read_text())
    require(receipt.get('status')=='COMPLETE' and receipt.get('stage')=='R3'
            and receipt.get('arm')=='F11_UNI2' and receipt.get('inner_partition')==part
            and receipt.get('seed')==seed and receipt.get('outer_fold')==1
            and receipt.get('outer_test_evaluated') is False and receipt.get('epochs')==80
            and receipt.get('actual_optimizer_fits')==1,'Expected completed inner-only F11 final80 cell')
    lineage=receipt['lineage'];origin=lineage['shared_teacher_origin_lineage']
    require(all(lineage.get(k)==v for k,v in origin.items()),'Source projection drift cell relabelled its teacher origin')
    require(lineage.get('external_projection_sha256')==source_audit['source_transfer_sha256'],
            'Final cell was not initialized by the named source projection')
    artifact=path.parent/'model.pt'
    require(artifact.is_file() and sha256(artifact)==receipt['artifact_sha256'],'Final80 model artifact byte SHA mismatch')
    checkpoint=torch.load(artifact,map_location='cpu',weights_only=False)
    require(checkpoint.get('lineage')==lineage and checkpoint.get('arm')=='F11_UNI2'
            and checkpoint.get('stage')=='R3' and checkpoint.get('seed')==seed
            and checkpoint.get('epochs')==80 and checkpoint.get('optimizer_updates')==80,
            'Final80 checkpoint payload identity or training budget differs')
    snapshot=checkpoint.get('snapshots',{}).get(80)
    require(snapshot is not None and snapshot.get('metrics')==receipt['metrics'],
            'Final projection is not from the completed final80 snapshot')
    complete_state=snapshot['state_dict'];prefix='path.project.'
    state={key[len(prefix):]:value for key,value in complete_state.items() if key.startswith(prefix)}
    require(set(state)==set(source),'Target projection tensor set differs from source initialization')
    tensors=[];flat_source=[];flat_final=[]
    for key in sorted(source):
        value=state[key]
        require(isinstance(value,torch.Tensor) and value.device.type=='cpu','Final projection was not mapped to CPU')
        final=value.detach().numpy()
        tensors.append(dict(tensor=key,**drift_statistics(source[key],final)))
        flat_source.append(source[key].ravel());flat_final.append(final.ravel())
    total=drift_statistics(np.concatenate(flat_source),np.concatenate(flat_final))
    return dict(group=group,arm='F11_UNI2',partition=part,seed=seed,final_epoch=80,
        source_transfer_sha256=source_audit['source_transfer_sha256'],
        final_model_artifact_sha256=receipt['artifact_sha256'],complete_receipt_sha256=sha256(path),
        teacher_origin_sha256=canonical_hash(origin),UNI2_cache_READY_sha256=lineage['UNI2_cache_receipt_sha256'],
        source_projection_loaded_by_recorded_lineage=True,per_tensor=tensors,all_projection_parameters=total)


def generate(private_root,output):
    sources={};cells=[];origins=set();caches=set()
    for group in GROUPS:
        initial,source_audit=source_projection(private_root,group);sources[group]=source_audit
        for part in (0,1,2):
            for seed in (31,47):
                path=Path(private_root)/f'auxiliary_{group}_v2'/'cells'/f'R3_outer1_inner{part}_F11_UNI2_seed{seed}'/'COMPLETE.json'
                if not path.is_file():continue
                row=audit_final_cell(path,group,part,seed,initial,source_audit)
                cells.append(row);origins.add(row['teacher_origin_sha256']);caches.add(row['UNI2_cache_READY_sha256'])
    require(len(origins)<=1 and len(caches)<=1,'Projection-drift cells mix target teachers or UNI2 caches')
    summary=[]
    for group in GROUPS:
        selected=[row for row in cells if row['group']==group];complete=len(selected)==6
        summary.append(dict(group=group,status='COMPLETE' if complete else 'RUNNING',completed_cells=len(selected),planned_cells=6,
            mean_delta_RMS=mean([r['all_projection_parameters']['delta_RMS'] for r in selected]) if complete else None,
            mean_relative_L2=mean([r['all_projection_parameters']['relative_L2'] for r in selected]) if complete else None,
            mean_cosine=mean([r['all_projection_parameters']['cosine'] for r in selected]) if complete else None))
    result=dict(schema='bgb_en_aux_projection_parameter_drift_v1',
        status='COMPLETE_PRELOCKED_F11_PROJECTION_DRIFT' if len(cells)==18 else 'RUNNING',
        completed_final80_F11_cells=len(cells),planned_final80_F11_cells=18,
        source_projections=sources,summary=summary,cells=cells,
        F5_initialization_drift_audited=False,
        F5_boundary='not constructed: exact F5 RNG initialization requires complete MechanismModel creation including preceding omics modules',
        measurement='parameter RMS difference, source-relative L2 and cosine; concatenated statistics are parameter-count weighted',
        scientific_boundary='parameter movement does not establish source semantic forgetting, preservation, medical mechanism or prognostic benefit',
        CPU_tensor_operations_only=True,model_construction_calls=0,model_forward_calls=0,
        optimizer_updates=0,GPU_used=False,patient_arrays_exported=False,
        report_script_sha256=sha256(__file__))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# 外部初始化投影：最终权重漂移审计','',
           f"**状态：{result['status']}；已核验 {len(cells)}/18 个完整 F11 最终80轮投影。**",'',
           '| 源初始化 | 完成 | 平均 ΔRMS | 平均 relative L2 | 平均 cosine |',
           '|---|---:|---:|---:|---:|']
    for row in summary:lines.append(f"| {row['group']} | {row['completed_cells']}/6 | {formatted(row['mean_delta_RMS'])} | {formatted(row['mean_relative_L2'])} | {formatted(row['mean_cosine'])} |")
    lines+=['','逐tensor统计覆盖Linear权重/偏置及LayerNorm权重/偏置，记录真实来源投影哈希、来源COMPLETE与最终model.pt哈希。仅在完整六个目标单元后汇总该组平均值；concat指标按参数数量加权，不能理解为各层同权。源范数为零时relative L2为NA，任一范数为零时cosine为NA。',
            '', '本脚本只在CPU读取已保存权重，不构建模型、不运行患者前向、不读结局值、不做优化。未完成单元不读取其checkpoint。',
            '', '## 解释边界','',
            '**参数漂移不等于语义遗忘。** 大幅移动不证明来源语义消失，小幅移动也不证明语义保留；需要独立来源任务回放或行为验证才可讨论这些问题。这里不作疾病机制或DSS增益结论。',
            '', '优势：逐tensor、真实初始化、真实最终权重和字节谱系直接对应。风险：参数空间具有尺度与重参数化依赖；LayerNorm与后续模块可能改变相同权重的行为，concat统计受大矩阵主导。',
            '', '未提供F5随机初始化漂移。完整MechanismModel先构建omics再构建path；只设置相同seed后单独创建PathEncoder会得到错误的RNG起点。当前审计仅比较三组F11各自实际加载的来源投影。']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/bgb_en/AUX_PROJECTION_DRIFT.json')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    result=generate(args.private_root,args.output)
    print(json.dumps({key:result[key] for key in ('status','completed_final80_F11_cells','optimizer_updates','GPU_used')}))


if __name__=='__main__':main()
