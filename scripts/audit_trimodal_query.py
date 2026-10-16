"""Post-training, outcome-blind query/PCA audit. Never changes training files.

Only fixed first-32 inner-training patients are probed. The audit does not use
survival times/events, validation scores, checkpoint risk arrays, or outer-test
patients to choose examples or calculate its statistics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F

from src.trimodal_interaction.data import (
    Data, OLD_LOCAL, OLD_VISUAL, P, R, ROOT, ahash, cfg, digest, source, write,
)
from src.trimodal_interaction.model import Model, mixed_difference


def relative_rows(before, after):
    a = before.flatten(1)
    b = after.flatten(1)
    return (b - a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12)


def cosine_rows(before, after):
    return F.cosine_similarity(before.flatten(1), after.flatten(1), dim=1, eps=1e-12)


def _pre_normalization_tokens(interaction, c, r, rm):
    qr = interaction.gene[None] + interaction.rna(torch.stack([r, rm], -1))
    cc = interaction.c(c)
    return qr * (1 + .1 * torch.tanh(cc[:, None])) + .1 * cc[:, None]


def inspect_preprocessing(data, config):
    """Reconstruct actual PCA training rows from private source patch locations."""
    states = data.state
    fit_rows = np.asarray(states['patch_PCA_fit_rows'], np.int64)
    if fit_rows.ndim != 2 or fit_rows.shape[1] != 2:
        raise ValueError('patch PCA row manifest has wrong shape')
    fit_set = set(data.fit.tolist())
    if set(fit_rows[:, 0]) != fit_set:
        raise ValueError('patch PCA fitting patients differ from inner-train')
    if np.intersect1d(fit_rows[:, 0], data.part['inner_val']).size:
        raise ValueError('patch PCA includes inner-validation patient')
    if np.intersect1d(fit_rows[:, 0], data.part['outer_test']).size:
        raise ValueError('patch PCA includes outer-test patient')
    unique, counts = np.unique(fit_rows[:, 0], return_counts=True)
    if counts.max() > 16 or (counts < 1).any():
        raise ValueError('patch PCA sampling quota violated')
    if not data.f['PM'][fit_rows[:, 0], fit_rows[:, 1]].all():
        raise ValueError('PCA fitting row is masked/padded')
    locations = states['selected_patches']['slide_and_parent_row']
    ss, rr = locations[fit_rows[:, 0], fit_rows[:, 1]].T
    if (ss < 0).any() or (rr < 0).any():
        raise ValueError('PCA fitting row has no source location')
    raw = np.load(OLD_VISUAL / 'conch.npy', mmap_mode='r')
    owner = np.load(OLD_VISUAL / 'owner.npy')
    if not np.array_equal(owner[ss], fit_rows[:, 0]):
        raise ValueError('PCA source-patch ownership mismatch')
    sample = np.asarray(raw[ss, rr], np.float32)
    original = states['patch_pca']
    rebuilt = PCA(64, whiten=False, svd_solver='randomized',
                  random_state=int(config.get('preprocessing_seed', 17321))).fit(sample)
    component_error = float(np.max(abs(original.components_ - rebuilt.components_)))
    mean_error = float(np.max(abs(original.mean_ - rebuilt.mean_)))
    if component_error > 2e-5 or mean_error > 1e-7:
        raise ValueError('actual source-row PCA replay differs')
    max_reference_error = max(float(np.max(abs(data.references[k] - data.f[k][data.fit].mean(0))))
                              for k in ['C', 'R', 'RM'])
    if max_reference_error > 1e-7:
        raise ValueError('reference is not exact inner-training mean')
    inactive = np.setdiff1d(np.arange(512), data.active)
    if any(np.any(v[inactive] != 0) for v in data.f.values()):
        raise ValueError('cache contains features outside active inner partition')
    for key, values in data.f.items():
        if ahash(values[data.fit]) != data.audit['fit_feature_hashes'][key]:
            raise ValueError('actual training feature hash differs: ' + key)
        if ahash(values[data.valid]) != data.audit['valid_feature_hashes'][key]:
            raise ValueError('actual validation feature hash differs: ' + key)
    # Independently replay the historical LOCAL visual transform on fit rows.
    with np.load(OLD_LOCAL / 'VISUAL_SOURCES.npz') as saved:
        local = np.asarray(saved['LOCAL'], np.float64)
    fit = data.fit
    mean = local[fit].mean(0)
    sd = local[fit].std(0)
    sd[sd < 1e-8] = 1
    lpc = PCA(32, whiten=False, svd_solver='full').fit((local[fit] - mean) / sd)
    local_error = float(np.max(abs(lpc.components_ - states['LOCAL_pca'].components_)))
    if local_error > 2e-8:
        raise ValueError('LOCAL PCA replay on actual inner-train differs')
    return {'fold': data.fold, 'status': 'PASS', 'fit_patient_count': len(unique),
            'valid_patient_count': len(data.valid), 'PCA_fit_sample_count': len(sample),
            'PCA_max_samples_per_patient': int(counts.max()),
            'PCA_fit_partition_hash': ahash(data.fit),
            'PCA_source_samples_hash': ahash(sample),
            'PCA_fit_row_manifest_hash': ahash(fit_rows),
            'PCA_replayed_components_hash': ahash(rebuilt.components_),
            'PCA_component_replay_absmax': component_error,
            'PCA_mean_replay_absmax': mean_error,
            'LOCAL_PCA_replay_absmax': local_error,
            'reference_replay_absmax': max_reference_error,
            'cache_sha256': data.audit['cache_sha256'],
            'heldout_patients_in_PCA_fit': 0,
            'inactive_features_zero': True}


@torch.no_grad()
def probe_chunk(model, data, indices, device):
    inter = model.interaction
    ids = np.asarray(indices, np.int64)
    wrong = data.wrong[ids]
    if np.any(ids == wrong) or not np.isin(wrong, data.fit).all():
        raise ValueError('query wrong control not an inner-training derangement')
    tensor = lambda key, ix=ids: data.ft[key][ix].to(device)
    x, c, r, rm, p, mask = [tensor(key) for key in ['X', 'C', 'R', 'RM', 'P', 'PM']]
    rw, rmw = tensor('R', wrong), tensor('RM', wrong)
    q, _, _ = inter.queries(c, r, rm)
    qw, _, _ = inter.queries(c, rw, rmw)
    qs, _, _ = inter.queries(c, r * 1.1, rm)
    pre = _pre_normalization_tokens(inter, c, r, rm)
    pre_scaled = _pre_normalization_tokens(inter, c, r * 1.1, rm)
    saved_gene = inter.gene.detach().clone()
    try:
        inter.gene.zero_()
        q_no_gene, _, _ = inter.queries(c, r, rm)
    finally:
        inter.gene.copy_(saved_gene)
    c0, r0, rm0 = inter.c0.expand_as(c), inter.r0.expand_as(r), inter.rm0.expand_as(rm)
    pp = inter.pnorm(p)
    k, v = inter.k(pp), inter.v(pp)
    a11 = inter.attention(q, k, mask)
    a10 = inter.attention(inter.queries(c, r0, rm0)[0], k, mask)
    a01 = inter.attention(inter.queries(c0, r, rm)[0], k, mask)
    a00 = inter.attention(inter.queries(c0, r0, rm0)[0], k, mask)
    delta = mixed_difference(a11, a10, a01, a00)
    mixed = inter.message(delta, v)
    ordinary = inter.message(a11, v)
    uniform = mask[:, None, None, :] / mask.sum(-1)[:, None, None, None]
    centered = inter.message(a11 - uniform, v)
    target, den = mixed.detach().norm(dim=-1), centered.detach().norm(dim=-1)
    if bool(((den < 1e-15) & (target > 1e-12)).any()):
        raise ValueError('A7 norm control would be undefined on this query probe')
    ratio = torch.where(den > 1e-15, target / den, torch.zeros_like(den))
    centered_matched = centered * ratio[:, None]
    norm_error = (centered_matched.norm(dim=-1) - target).abs()
    if model.arm == 'A7' and not torch.allclose(centered_matched.norm(dim=-1), target,
                                              atol=1e-8, rtol=2e-5):
        raise ValueError('A7 pre-gate matching norm fails on own parameters')
    matched_residual, diagnostics = inter(c, r, rm, p, mask, diagnostic=True)
    wrong_residual = inter(c, rw, rmw, p, mask)
    c_null = inter(c0, r, rm, p, mask)
    r_null = inter(c, r0, rm0, p, mask)
    if model.arm in ['A5', 'A7'] and (torch.count_nonzero(c_null) or torch.count_nonzero(r_null)):
        raise ValueError('mixed operator reference risk is not exactly zero')
    own_main = x @ model.main
    matched_risk = own_main + matched_residual
    wrong_risk = own_main + wrong_residual
    result = {
        'query_matched_wrong_cosine': cosine_rows(q, qw),
        'query_matched_wrong_relative_L2': relative_rows(q, qw),
        'query_RNA_scale1p1_relative_L2': relative_rows(q, qs),
        'pre_LayerNorm_RNA_scale1p1_relative_L2': relative_rows(pre, pre_scaled),
        'query_gene_identity_ablation_relative_L2': relative_rows(q, q_no_gene),
        'query_gene_identity_ablation_cosine': cosine_rows(q, q_no_gene),
        'query_norm': q.flatten(1).norm(dim=1),
        'mixed_message_norm_before_gate': mixed.norm(dim=-1),
        'ordinary_message_norm': ordinary.norm(dim=-1),
        'mixed_to_ordinary_message_norm': mixed.norm(dim=-1) / ordinary.norm(dim=-1).clamp_min(1e-12),
        'mixed_row_sum_absmax': delta.sum(-1).abs().flatten(1).amax(-1),
        'A7_pregate_norm_match_abs_error': norm_error,
        'A7_norm_control_scale': ratio,
        'clinical_reference_interaction_abs': c_null.abs(),
        'RNA_and_mask_reference_interaction_abs': r_null.abs(),
        'matched_wrong_total_risk_abs': (matched_risk - wrong_risk).abs(),
        'matched_wrong_interaction_abs': (matched_residual - wrong_residual).abs(),
        'interaction_abs': matched_residual.abs(),
        'forward_message_norm_after_gate': diagnostics['message_norm'],
    }
    # Risk arrays exist only in memory for this fixed label-blind probe.
    return {key: value.cpu().numpy() for key, value in result.items()}, {
        'main': own_main.cpu().numpy(), 'residual': matched_residual.cpu().numpy(),
        'wrong_residual': wrong_residual.cpu().numpy(), 'total': matched_risk.cpu().numpy()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--budget160', action='store_true',
                        help='audit verified budget160 receipts; preserve original cap40 audit')
    args = parser.parse_args()
    if args.batch < 1 or args.batch > 16:
        raise ValueError('probe batch must be 1..16')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = cfg()
    training_source = source()
    done = json.loads((P / 'DONE.json').read_text())
    if done.get('status') != 'ALL_LOCKED_INNER_CELLS_FINISHED' or done['source'] != training_source:
        raise ValueError('audit requires all locked training jobs to finish')
    lock = json.loads((P / 'LOCK.json').read_text())
    if lock['source'] != training_source:
        raise ValueError('training lock changed')
    private_cells = P / 'cells'
    output = R
    budget_identity = None
    reusable_preprocessing = {}
    original_audit_hash = None
    if args.budget160:
        from scripts.extend_trimodal_budget import budget_source
        budget_identity = budget_source()
        budget_done = json.loads((P / 'budget160/DONE.json').read_text())
        if (budget_done.get('status') != 'ALL_BUDGET_CELLS_VERIFIED'
                or budget_done['source'] != training_source
                or budget_done['budget_source'] != budget_identity):
            raise ValueError('budget audit requires all budget160 jobs to finish')
        budget_lock = json.loads((P / 'budget160/LOCK.json').read_text())
        if (budget_lock['original_source'] != training_source
                or budget_lock['budget_source'] != budget_identity):
            raise ValueError('budget lock/source changed')
        private_cells = P / 'budget160/cells'
        output = R / 'budget160'
        previous_path = R / 'QUERY_REPRESENTATION_AUDIT.json'
        if previous_path.exists():
            previous = json.loads(previous_path.read_text())
            if previous['status'] != 'PASS' or previous['training_source'] != training_source:
                raise ValueError('original preprocessing audit has wrong status/source')
            original_audit_hash = digest(previous_path)
            reusable_preprocessing = {q['fold']: q for q in previous['preprocessing_actual_source_replay']}
    records, preprocess, receipts = [], [], []
    for fold in config['folds']:
        checked_preprocess = False
        for seed in config['seeds']:
            data = Data(fold, 'inner', seed)
            if not checked_preprocess:
                if fold in reusable_preprocessing:
                    saved = reusable_preprocessing[fold]
                    if (saved['status'] != 'PASS'
                            or saved['cache_sha256'] != data.audit['cache_sha256']
                            or saved['PCA_fit_partition_hash'] != ahash(data.fit)):
                        raise ValueError('PCA cache/partition changed after cap40 source replay')
                    # Data has rechecked the complete cache SHA, source, source
                    # visual SHA, upstream feature SHA and exact partitions.
                    preprocess.append({**saved, 'reused_original_audit_sha256': original_audit_hash})
                else:
                    preprocess.append(inspect_preprocessing(data, config))
                checked_preprocess = True
            patients = data.fit[:min(32, len(data.fit))]
            for arm in ['A2', 'A5', 'A7']:
                label = f'{arm}_seed{seed}_fold{fold}'
                cell = json.loads((private_cells / (label + '.json')).read_text())
                path = Path(cell['checkpoint_path']) if args.budget160 else P / 'checkpoints' / (label + '.pt')
                checkpoint_hash = digest(path)
                if cell['source'] != training_source or cell['checkpoint_sha256'] != checkpoint_hash:
                    raise ValueError('checkpoint/source receipt mismatch: ' + label)
                if args.budget160 and cell['budget_source'] != budget_identity:
                    raise ValueError('budget checkpoint receipt source mismatch: ' + label)
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                if checkpoint['source'] != training_source or not np.array_equal(checkpoint['fit'], data.fit):
                    raise ValueError('checkpoint fit/source drift: ' + label)
                if not np.array_equal(checkpoint['valid'], data.valid):
                    raise ValueError('checkpoint validation partition drift: ' + label)
                if checkpoint['preprocessing']['cache_sha256'] != data.audit['cache_sha256']:
                    raise ValueError('checkpoint preprocessing cache differs: ' + label)
                if args.budget160 and not cell['reused_reference'] and checkpoint['budget_source'] != budget_identity:
                    raise ValueError('retrained checkpoint budget source differs: ' + label)
                model = Model(data, arm, np.zeros(data.f['X'].shape[1]), config).to(args.device)
                model.load_state_dict(checkpoint['state'], strict=True)
                model.eval()
                values, risks = {}, {}
                for j in range(0, len(patients), args.batch):
                    result, risk = probe_chunk(model, data, patients[j:j + args.batch], args.device)
                    for key, value in result.items():
                        values.setdefault(key, []).append(value)
                    for key, value in risk.items():
                        risks.setdefault(key, []).append(value)
                row = {'arm': arm, 'seed': seed, 'fold': fold, 'probe_patients': len(patients),
                       'probe_partition': 'first32_inner_train_in_frozen_manifest_order',
                       'null_exact_zero_expected': arm in ['A5', 'A7']}
                for key, chunks in values.items():
                    value = np.concatenate(chunks)
                    row[key + '_mean'] = float(value.mean())
                    row[key + '_median'] = float(np.median(value))
                    row[key + '_max'] = float(value.max())
                risks = {key: np.concatenate(chunks) for key, chunks in risks.items()}
                for key, value in risks.items():
                    row[key + '_SD'] = float(value.std())
                main_sd = max(row['main_SD'], 1e-12)
                row['interaction_to_main_SD'] = row['residual_SD'] / main_sd
                row['total_to_main_SD'] = row['total_SD'] / main_sd
                row['matched_wrong_residual_SD'] = float((risks['residual'] - risks['wrong_residual']).std())
                records.append(row)
                receipts.append({'arm': arm, 'seed': seed, 'fold': fold,
                                 'checkpoint_sha256': checkpoint_hash,
                                 'reused_cap40_checkpoint': bool(cell.get('reused_reference', False))})
                print('QUERY_AUDIT', label, 'q_wrong_cos', row['query_matched_wrong_cosine_mean'],
                      'q_scale_rel', row['query_RNA_scale1p1_relative_L2_mean'],
                      'mixed/ordinary', row['mixed_to_ordinary_message_norm_mean'], flush=True)
                del model
    if source() != training_source:
        raise ValueError('training files changed during post-training audit')
    if args.budget160 and budget_source() != budget_identity:
        raise ValueError('budget training source changed during audit')
    table = pd.DataFrame(records)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / 'QUERY_REPRESENTATION_AUDIT.csv', index=False)
    arm_summary = {}
    summary_keys = ['query_matched_wrong_cosine_mean', 'query_matched_wrong_relative_L2_mean',
                    'query_RNA_scale1p1_relative_L2_mean',
                    'pre_LayerNorm_RNA_scale1p1_relative_L2_mean',
                    'query_gene_identity_ablation_relative_L2_mean',
                    'mixed_to_ordinary_message_norm_mean', 'interaction_to_main_SD',
                    'matched_wrong_total_risk_abs_mean']
    for arm in ['A2', 'A5', 'A7']:
        sub = table[table.arm == arm]
        arm_summary[arm] = {key: {'median_across_cells': float(sub[key].median()),
                                  'min_across_cells': float(sub[key].min()),
                                  'max_across_cells': float(sub[key].max())}
                            for key in summary_keys}
    summary = {'status': 'PASS', 'scope': 'label-blind mechanism diagnostics; no predictive efficacy inference',
               'training_source': training_source, 'audit_script_sha256': digest(__file__),
               'training_schedule': 'budget160_sensitivity' if args.budget160 else 'original_cap40',
               'budget_source': budget_identity,
               'preprocessing_original_audit_reuse_sha256': original_audit_hash,
               'cells': len(records), 'arms': ['A2', 'A5', 'A7'], 'probe_patients_maximum_per_cell': 32,
               'probe_selection': 'first inner-training rows; no outcome/score-based selection',
               'own_main_effect_unchanged_under_wrong_context': True,
               'risk_arrays_persisted': False, 'outer_test_probes': 0,
               'all_mixed_reference_risks_exact_zero': bool((table[table.arm.isin(['A5', 'A7'])]
                   [['clinical_reference_interaction_abs_max', 'RNA_and_mask_reference_interaction_abs_max']] == 0).all().all()),
               'mixed_row_sum_error_global_max': float(table.mixed_row_sum_absmax_max.max()),
               'A7_pregate_norm_match_error_global_max': float(table[table.arm == 'A7'].A7_pregate_norm_match_abs_error_max.max()),
               'arm_summary': arm_summary, 'preprocessing_actual_source_replay': preprocess,
               'checkpoint_receipts': receipts,
               'interpretation': [
                   'Raw RNA512 linear main effect preserves a direct information path; it does not prove that the query retains the same prognostic ranking information.',
                   'Per-token LayerNorm can reduce amplitude information. The 1.1 RNA scaling probe measures representation sensitivity, not RNA prognostic sufficiency.',
                   'Wrong-context query changes and nonzero mixed messages are operator diagnostics, not evidence of patient-specific predictive benefit.',
                   'A7 norm matching is checked before its gate using A7 own learned parameters; post-gate norms need not match A5 or the ordinary attention message.',
                   'All statistics use frozen first32 inner-training rows without survival labels; no independent morphology or raw-WSI semantic validation is supplied.'
               ]}
    write(output / 'QUERY_REPRESENTATION_AUDIT.json', summary)
    (output / 'QUERY_AUDIT_INTERPRETATION.md').write_text(
        '# 查询表征与机制审计\n\n'
        '本审计仅探测各内层训练分区按既定顺序的前32名患者，未按事件、风险或模型得分挑选患者，'
        '不读取外折测试患者做探测；所有患者级风险仅在内存中用于汇总。\n\n'
        '**保留 RNA512 主效应旁路，只能证明存在直接信息通路，不能证明 RNA→Query 编码无损。** '
        '逐 Token LayerNorm 可能削弱幅度信息；RNA 数值乘1.1、基因身份置零和正确/错误患者替换'
        '分别探测数值敏感性、身份依赖与患者条件变化，不能单凭这些统计量宣称预后排序保真。\n\n'
        '**A7 的范数匹配在门控之前、使用其自身参数检查。** 后续 Gate 可改变范数；'
        '不同模型独立学习后的消息范数也不要求逐元素相等。四路径参考退化、零行和以及非零消息'
        '只能验证算子属性；是否有预测收益应由独立锁定的比较回答。\n\n'
        'PCA 审计从实际私有选区位置重建训练样本，重新拟合 PCA64 与 LOCAL PCA32，'
        '核查训练患者归属、每患者采样上限、组件、参考值和实际缓存特征哈希。'
        'CSV 与 JSON 仅发布匿名聚合，不包含患者标识、所选基因列表或风险数组。\n', encoding='utf8')
    print('QUERY_AUDIT_DONE', len(records), flush=True)


if __name__ == '__main__':
    main()
