"""Zero-fit anonymous audit of direct-RNA versus pathway-token gene support.

Only prepared input/preprocessing metadata are consumed. No risk, endpoint,
patient identifier, gene symbol or selected-gene index is exported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts import bgb_en_run as driver
from scripts.bgb_en_checkpoint_audit import (
    require, sha256, verify_current_source, verify_preprocessing_state,
)
from src.bgb_en.omics import shuffled_membership


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def support_counts(selected, membership, legacy_union):
    """Count support only; return no symbols, indices or individual data."""
    member = np.asarray(membership)
    old = np.asarray(legacy_union)
    chosen = np.asarray(selected)
    require(member.ndim == 2 and member.dtype == bool, 'INVALID_MEMBERSHIP')
    require(old.dtype == bool and old.shape == (member.shape[1],), 'INVALID_LEGACY_UNION')
    require(chosen.ndim == 1 and np.issubdtype(chosen.dtype, np.integer), 'INVALID_SELECTED_INDICES')
    require(len(chosen) > 0 and len(np.unique(chosen)) == len(chosen), 'DUPLICATED_SELECTED_INDICES')
    require((chosen >= 0).all() and (chosen < member.shape[1]).all(), 'SELECTED_INDEX_OUT_OF_RANGE')
    degree = member.sum(axis=0, dtype=np.int64)[chosen]
    count = int(np.count_nonzero(degree))
    values, frequencies = np.unique(degree, return_counts=True)
    return {
        'direct_selected_gene_count': int(len(chosen)),
        'token_supported_selected_gene_count': count,
        'token_unsupported_selected_gene_count': int(len(chosen) - count),
        'token_supported_selected_gene_fraction': count / len(chosen),
        'selected_membership_count_histogram': {str(int(k)): int(v) for k, v in zip(values, frequencies)},
        'selected_membership_count_mean': float(degree.mean()),
        'selected_membership_count_min': int(degree.min()),
        'selected_membership_count_max': int(degree.max()),
        'selected_in_legacy_union_count': int(old[chosen].sum()),
        'selected_outside_legacy_union_count': int((~old[chosen]).sum()),
        'selected_legacy_but_not_token_supported_count': int(np.sum(old[chosen] & (degree == 0))),
    }


def union_counts(member, old):
    current = np.asarray(member, bool).any(axis=0)
    old = np.asarray(old, bool)
    require(old.shape == current.shape, 'UNION_GENE_ORDER_MISMATCH')
    common = int(np.sum(current & old))
    return {
        'canonical_measured_gene_universe_count': int(len(old)),
        'legacy_union_measured_gene_count': int(old.sum()),
        'Hallmark50_union_measured_gene_count': int(current.sum()),
        'legacy_Hallmark50_intersection_measured_gene_count': common,
        'legacy_union_not_in_Hallmark50_measured_gene_count': int(np.sum(old & ~current)),
        'Hallmark50_union_not_in_legacy_measured_gene_count': int(np.sum(current & ~old)),
        'legacy_union_covered_by_Hallmark50_fraction': common / int(old.sum()) if old.any() else None,
        'Hallmark50_union_covered_by_legacy_fraction': common / int(current.sum()) if current.any() else None,
    }


def render(report):
    lines = [
        '# R2 输入集合覆盖审计（0 拟合）', '',
        '本审计只核查输入支持集，不读取保存风险、不评估生存终点，也不修改训练结果或选择锁。', '',
        '## 数据来源与覆盖', '',
        f"状态：**{report['status']}**。核验 {len(report['partitions'])} 份主准备状态及 "
        f"{len(report['teachers'])} 份已有教师准备状态；新增优化器/PCA/教师拟合/GPU调用均为 0。", '',
        '基因符号按既有 `load_inputs` 的规范顺序及 GMT exact-symbol 映射核查；下表只给匿名计数。', '',
        '| 量 | 数值 |', '|---|---:|',
    ]
    lines += [f'| {key} | {value} |' for key, value in report['union_coverage'].items()]
    lines += ['', '## RNA512 与通路 Token 的实际输入重合', '',
              '真实 Hallmark50 用于 O3–O8；O9 使用预锁 seed 270099 的列置换，其实际支持集另列。', '',
              '| 准备状态 | 真实通路覆盖 / 512 | 比例 | O9 置换覆盖 / 512 | 原旧并集外回退基因 |',
              '|---|---:|---:|---:|---:|']
    for row in report['partitions'] + report['teachers']:
        true, null = row['true_membership'], row['O9_shuffled_membership']
        lines.append(f"| {row['unit']} | {true['token_supported_selected_gene_count']} / {true['direct_selected_gene_count']} | "
                     f"{true['token_supported_selected_gene_fraction']:.6f} | "
                     f"{null['token_supported_selected_gene_count']} / {null['direct_selected_gene_count']} | "
                     f"{true['selected_outside_legacy_union_count']} |")
    lines += ['', '各所选基因的通路隶属次数分布、准备文件和来源 SHA256 见同名 JSON；不公开基因名称、选择索引或患者数据。', '',
              '## 解释边界', '',
              '- 未覆盖基因确实不进入该通路分支；这与压缩/池化造成的表示损失是不同环节。',
              '- 即使全部 512 基因被覆盖，也只证明输入集合包含关系，不能证明聚合保留排序信息。',
              '- 线性 decoder 的低可解码性不能统一归因于池化；需同时考虑输入支持、有限解码器、正则几何与训练。',
              '- O3–O9 的直接 RNA512 风险路径仍被保留；此审计不改变既有整体模型性能结论。',
              '- 本统计基于规范 RNA 基因列表的静态成员关系，不是逐患者测量覆盖率或细胞来源真值。', '',
              '## 优势与局限', '',
              '**优势：**复用既有准备状态并校验哈希，零拟合，直接排查两条分支输入集合是否一致。', '',
              '**局限：**不测表示可逆性、不提供预后增量证据；分支中的实际数值缺失与 CNA 来源还需要各自审计。', '',
              '**置信度：高（限上述代码与文件哈希匹配后的计数结论）。**', '']
    return '\n'.join(lines)


def run(args):
    assets, _, raw, membership, lineage = driver.load_inputs(args)
    member = membership.cpu().numpy().astype(bool, copy=False)
    require(member.shape[0] == 50, 'EXPECTED_50_HALLMARK_SETS')
    old = np.asarray(raw['_direct_gene_universe'], bool)
    require(member.shape[1] == len(raw['rna_symbols']), 'RNA_GENE_ORDER_SIZE_MISMATCH')
    null = shuffled_membership(membership, seed=270099)[0].numpy()
    legacy = json.loads(Path(assets['legacy_groups']).read_text())
    rows, teachers = [], []
    for partition in range(3):
        stem = args.private / 'prepared' / f'outer1_inner{partition}'
        path, receipt_path = stem.with_suffix('.pt'), stem.with_suffix('.json')
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get('status') == 'COMPLETE', 'PREPARED_NOT_COMPLETE')
        require(receipt.get('sha256') == sha256(path), 'PREPARED_SHA256_MISMATCH')
        verify_current_source(receipt['lineage'], config_path=args.config, expected_lineage=lineage)
        prepared = torch.load(path, map_location='cpu', weights_only=False)
        require(prepared.get('lineage') == lineage, 'PREPARED_LINEAGE_MISMATCH')
        state = prepared['features']['state']
        verify_preprocessing_state(state, np.asarray(prepared['train']))

        def row(unit, state):
            selected = state['molecular']['selected']
            require(len(selected) == 512, 'EXPECTED_512_DIRECT_GENES')
            return {'unit': unit, 'prepared_sha256': receipt['sha256'],
                    'preprocessing_fit_sha256': state['fit_hash'],
                    'true_membership': support_counts(selected, member, old),
                    'O9_shuffled_membership': support_counts(selected, null, old)}

        rows.append(row(f'partition_{partition}', state))
        require(len(prepared['teachers']) == 3, 'EXPECTED_THREE_TEACHER_STATES')
        for teacher, value in enumerate(prepared['teachers']):
            verify_preprocessing_state(value['preprocessing'], np.asarray(value['train']))
            teachers.append(row(f'partition_{partition}_teacher_{teacher}', value['preprocessing']))
        del prepared
    return {
        'schema': 'bgb_en_input_coverage_audit_v1', 'status': 'COMPLETE_INPUT_METADATA_AUDIT',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'audit_source_sha256': sha256(__file__), 'protocol_sha256': lineage['protocol_sha256'],
        'core_source_lineage_sha256': canonical_hash(lineage['source']),
        'input_lineage_sha256': canonical_hash(lineage), 'GMT_sha256': lineage['GMT_sha256'],
        'legacy_group_sha256': lineage['legacy_group_sha256'],
        'canonical_gene_order_sha256': hashlib.sha256('\n'.join(map(str, raw['rna_symbols'])).encode()).hexdigest(),
        'pathway_count': int(member.shape[0]), 'legacy_program_count': len(legacy),
        'union_coverage': union_counts(member, old), 'partitions': rows, 'teachers': teachers,
        'actual_optimizer_fits': 0, 'teacher_refits': 0, 'PCA_refits': 0, 'GPU_calls': 0,
        'saved_risk_fields_consumed': False, 'outer_risk_fields_consumed': False,
        'individual_data_or_gene_indices_exported': False,
        'true_membership_arms': ['O3', 'O4', 'O5', 'O6', 'O7', 'O8'],
        'shuffled_membership_arm': 'O9', 'shuffled_membership_seed': 270099,
        'scope': 'Static input-support overlap only; not a decoder or survival-performance test.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/bgb_en/R2_INPUT_COVERAGE_AUDIT.json')
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    args.output.with_suffix('.md').write_text(render(report), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'partitions': len(report['partitions']),
                      'teachers': len(report['teachers']), 'optimizer_fits': 0, 'GPU_calls': 0}))


if __name__ == '__main__':
    main()
