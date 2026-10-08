"""Publish only informative, aggregate artifacts after both independent gates."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from phase7_p0_biology_prompt_audit import REPO,digest,write_json

P0R=Path('/root/phase7_p0r_robustness_v1')
SURV=Path('/root/phase7a_foundation_headroom_v1')
LABEL={'C':'Clinical','R':'RNA-linear Top512','U':'UNI2-h full mean','T':'TITAN native','S':'Static TITAN concepts','M':'True RNA states','MC':'True RNA+CNA states',
       'CR':'Clinical + RNA','CU':'Clinical + UNI2-h','CT':'Clinical + TITAN','CS':'Clinical + concepts','CM':'Clinical + RNA states',
       'CMC':'Clinical + RNA/CNA states','CRU':'Clinical + RNA + UNI2-h','CRT':'Clinical + RNA + TITAN',
       'CRS':'Clinical + RNA + concepts','CRMC':'Clinical + RNA + RNA/CNA states'}


def main():
    for root in [P0R,SURV]:
        assert json.loads((root/'DONE.json').read_text())['status']=='PASS'
        assert json.loads((root/'independent_validation.json').read_text())['status']=='PASS'
        assert (root/'EXITCODE').read_text().strip()=='0'
    r=json.loads((P0R/'decision.json').read_text());platform=json.loads((P0R/'platform_source_audit.json').read_text())
    assert platform['missing_block_exactly_exclusive_GAII'] and platform['missing_patients_with_ambiguous_platform']==0
    ep=json.loads((P0R/'endpoint_provenance_audit.json').read_text())
    assert all(v['status_all_equal'] and v['times_all_equal_to_converted_CDR'] for v in ep['endpoints'].values())
    p0lines=['# Phase7-P0R：数据来源与 Stage 基因集鲁棒性', '',
        '**已完成 99 次独立匹配随机基因集 × 3 seeds × 5 folds；1485 份外折 OOF、297 份真正 seed-level OOF。分类模型、8 组通路及评分规则全部沿用 Phase7-P0。**', '',
        '## 决策一：Stage 的发现保留，但效应幅度修正', '',
        f'- 真基因集三 seed OOF BA 均值：**{r["true_mean_oof_ba"]:.4f}**。',
        f'- 99 个匹配随机集合均值：**{r["random_mean_oof_ba"]:.4f}**，随机分布 95% 分位：**{r["random_mean_quantiles"][3]:.4f}**，最大值：**{r["random_mean_quantiles"][-1]:.4f}**。',
        f'- 真−平均随机：**{r["delta_true_minus_average_random"]:+.4f} BA**，固定 OOF 下配对患者 bootstrap 95% 区间 **[{r["paired_conditional_ci95"][0]:+.4f}, {r["paired_conditional_ci95"][1]:+.4f}]**。',
        '- 三个 seed 均超过各自随机分布 95% 分位；99 次没有随机集合达到真实均值，有限 benchmark exceedance=(1+0)/(99+1)=0.01。',
        '**预先锁定鲁棒性判据 PASS。**上一轮相对单个随机集合的 +0.1009 应更新为相对 99 次随机均值的 +0.0654；不能把 0.01 解释为独立外部验证或分子机制因果 p 值。', '',
        '## 203×3024 缺失区块从哪里来', '',
        '1. 本地 raw RNA 与 pinned cBioPortal 官方 RNA 的所有数值和缺失掩码完全相同。官方新版本仅有 29 个 Hugo 注释与 1 个 Entrez 注释更新；本轮保留旧注释、所有原值及基因行。',
        '2. 原文件就是字面 NA；512 人中 203 人缺失同一 3024 个基因，共 613872 个值。没有整人 RNA 缺失，不能把这 203 人当成没有 RNA 模态。',
        '3. GDC 原始肿瘤 RNA 平台元数据：仅 GAII 的 203 人全部有区块；仅 HiSeq 的 295 人和双平台记录的 14 人全部完整。缺失人群没有平台歧义；14 个双平台病例的最终输入 aliquot 选择仍不能仅凭该元数据断言。',
        '4. PanCancer 方法说明 GAII→HiSeq 调整后会将低 reads / 残余批次效应基因替换为 NA。结合官方矩阵与平台对应，支持这是上游平台处理遗留，而不是项目解析错误或生物学“基因不存在”。[来源说明](https://groups.google.com/g/cbioportal/c/6MZsoLeomIo)。',
        '本轮没有插回这些值、换数据版本、做新批次校正或增加 missingness 特征；沿用 train coverage≥90% 的有效基因规则。', '',
        '## 证据边界与置信度', '',
        '- **高：**官方数值/NA 一致、203 人平台对应、计算及 OOF 验证完整。',
        '- **中：**指定匹配零模型下 Stage 优势稳健。基因均值/SD 匹配不等于所有生物与技术混杂完全匹配。',
        '- **低：**外部队列可迁移性、因果通路机制、DSS 增益；Stage 是上一轮已发现的 endpoint，同一队列重复随机集不是独立确认。', '',
        '优势：99 个匹配基因集、保留集合大小与重叠、同模型同内外折规则、外部源文件与平台证据。风险：发现队列复用、近似 gene exchangeability、未重训练 bootstrap。', '',
        '## 执行与复现', '',
        '固定随机 seeds 70001–70099；三个 split seeds 20261007/08/09。先拼每 seed 的五折 OOF，再算 BA。全部分类器最大 51/5000 次迭代，独立指标误差 <1e-16。',
        '运行：`phase7_p0r_robustness.py --mode prepare`，随后 worker-id 0/1/2，aggregate，再 `validate_phase7_audits.py --phase p0r --out /root/phase7_p0r_robustness_v1`。',
        '官方 pinned commit 和哈希见 source audit；GDC 平台原数据保留服务器，不进入仓库。', '',
        '## 拓展思考', '', '真实状态能解释 Stage，与其能提供 Clinical 以外的 DSS 信息是不同问题；因此按计划进入 Phase7A，不修改任何通路。', '',
        '## 你可能想知道', '', '不会把 203×3024 的 NA 当成可用标签或模态缺失指示，也不根据这轮结果追选基因集。']
    (P0R/'RESULTS.md').write_text('\n'.join(p0lines)+'\n',encoding='utf-8')
    (P0R/'CONCLUSIONS_ZH.md').write_text('\n'.join(p0lines)+'\n',encoding='utf-8')

    mean=pd.read_csv(SURV/'three_seed_oof_means.csv');seed=pd.read_csv(SURV/'seed_oof_metrics.csv');pair=pd.read_csv(SURV/'paired_conditional_increments.csv')
    conv=json.loads((SURV/'convergence_audit.json').read_text());validation=json.loads((SURV/'independent_validation.json').read_text())
    lines=['# Phase7A：Foundation Representation Survival Headroom Audit', '',
        '**正式完成：17 个预先指定表示组合 × 2 个独立终点 × 3 inner-split seeds × 5 固定患者外折；510 outer fits、2040 inner fits、102 个完整 seed OOF 模型。P0R 完成且独立验证通过后才开始本轮拟合。**', '',
        '## 决策二：Clinical 以外的稳定 DSS 增量尚未确立', '',
        '**六项预注册 Clinical 条件比较均未通过稳健正增量判据。静态概念是值得保留的小容量候选，但当前结果不足以扩大 molecular prompt 或宣称已解决 DSS。**',
        '有一个预先指定的敏感性比较通过：Clinical+RNA+concepts 相对 Clinical+RNA 的 Harrell 增量。但 Clinical+RNA 本身略弱于 Clinical，不能据此宣布概念在最强 Clinical anchor 上已得到稳定增量。', '',
        '## 1. 数据和终点', '',
        '- DSS 主终点：原始 DSS_MONTHS/DSS_STATUS，512 人、65 事件。',
        '- PFS 次要终点：原始 PFS_MONTHS/PFS_STATUS，510 人、129 事件；排除两名时间无效病例，保留已有患者外折，不插补结局。',
        '- 外部 TCGA-CDR 核验：DSS 对 DSS，源文件 PFS 对 CDR PFI，512 人的状态一致，月数与天数/30.417 在 5.5e-8 月内一致。**此处的 PFS 实际源于 TCGA-CDR PFI，不等同于临床试验判定的 PFS。**',
        '- DSS/PFS 独立拟合、独立汇总；不根据 PFS 替换主终点。[GDC 临床来源](https://gdc.cancer.gov/about-data/publications/pancanatlas)。', '',
        '## 2. 模型和泄露防护', '',
        '所有固定表示使用相同单阶段 ridge Cox（Efron ties，float64），候选 λ=0.01/0.1/1/10；inner-train 拟合预处理与 Cox，inner-val 只选 λ；整个 outer-train 同算法重拟合，最后评一次 outer-test。',
        'Clinical 仅 Age/Sex/Stage/T/N/M；Age 中位数、类别词表、标准化、RNA Top512、有效基因筛选与状态评分标准化全部 fit-local。仅使用 train variance 排 Top512，再做 z-score。',
        '联合模型直接拼接冻结表征后拟合一个 Cox，没有 in-sample prediction stacking，也没有拼接不同模型的 latent coordinates。TITAN/UNI2-h 不训练，基因集/概念词表不改。',
        'UNI2-h 全量读 520 张 slide、3295129 个 patches：每 slide 全 patch 均值，再等 slide 权重聚合到 512 个患者。**这是 mean-pool 的线性 headroom，不是 ABMIL，也不是 UNI2-h 所有可能下游模型的上限。**',
        'RNA 8 状态和 RNA+CNA 16 状态保持原定义；signed CNA dosage 不被解释为 pathway activation。', '',
        '## 3. 六类基础表示：真正三 seed OOF', '',
        '| 表示 | DSS Harrell | DSS Uno60 | PFS/PFI Harrell | PFS/PFI Uno60 |', '|---|---:|---:|---:|---:|']
    for method in ['C','U','R','T','S','M','MC']:
        a=mean[(mean.endpoint=='DSS')&(mean.method==method)].iloc[0];b=mean[(mean.endpoint=='PFS')&(mean.method==method)].iloc[0]
        lines.append(f'| {LABEL[method]} | {a.oof_harrell:.4f} | {a.oof_uno60:.4f} | {b.oof_harrell:.4f} | {b.oof_uno60:.4f} |')
    lines += ['', '数值为每个 seed 先拼五外折 OOF、算指标，再平均三个 seed；不是 15 个 fold-cell 指标的平均。所有 seed 使用同一患者队列，不能当成三个独立队列。', '',
        'OOF risk 仅按所属模型 outer-train 的 risk 均值/SD 标定；没有用测试分布。Harrell 保留完整时间；Uno 预锁 60 月，outer-train reverse censor KM，事件时点左极限权重，无 G clipping。',
        '相同时间的 event/censor 可比较，双 event tie 不比较；实现与验证独立逐事件核算。OOF Uno 是各患者所属训练折的删失权重拼接版本；另提供不比较跨 fold 患者的 pair-weighted 指标。', '',
        '## 4. Clinical 条件增量（主要比较族）', '',
        '| 终点 | 联合−Clinical | ΔHarrell | 条件 95% CI | Bonferroni 族区间 | ΔUno60 | 判据 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r0 in pair[pair.family=='clinical_primary'].itertuples(index=False):
        lines.append(f'| {r0.endpoint} | {r0.pair} | {r0.delta_harrell:+.4f} | [{r0.harrell_ci95_low:+.4f}, {r0.harrell_ci95_high:+.4f}] | [{r0.harrell_family_ci_low:+.4f}, {r0.harrell_family_ci_high:+.4f}] | {r0.delta_uno60:+.4f} | {"PASS" if r0.robust_positive_increment else "证据不足"} |')
    lines += ['', '代码：R=RNA-linear；U=UNI2-h；T=TITAN native；S=static concepts；M=true RNA states；MC=true RNA+CNA states。每个终点六个主要对比，使用 99.1667% bootstrap 分位区间作为 Bonferroni 族区间。',
        '通过条件为：Harrell 族区间下界>0、三个 seed Harrell 差值均>0、平均 Uno 差值>0。PFS 使用同规则单独描述，不升级为确认性主终点。', '',
        '## 5. Clinical+RNA anchor 的预注册敏感性比较', '',
        '| 终点 | 对比 | ΔHarrell | 95% CI | 四项族区间 | ΔUno60 | 判据 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r0 in pair[pair.family=='clinical_rna_sensitivity'].itertuples(index=False):
        lines.append(f'| {r0.endpoint} | {r0.pair} | {r0.delta_harrell:+.4f} | [{r0.harrell_ci95_low:+.4f}, {r0.harrell_ci95_high:+.4f}] | [{r0.harrell_family_ci_low:+.4f}, {r0.harrell_family_ci_high:+.4f}] | {r0.delta_uno60:+.4f} | {"PASS" if r0.robust_positive_increment else "证据不足"} |')
    lines += ['', '**解释限制：**DSS CR=0.7699、C=0.7767；CRS=0.7979。CRS−CR=+0.0281 的 Harrell 四项族区间 [+0.0031,+0.0544]，三个 seed 同向；Uno60 +0.0239 的 95% 区间 [+0.0025,+0.0476]，但族区间 [−0.0031,+0.0535] 跨零。',
        'CS−C=+0.0201 的 95% 区间 [−0.0124,+0.0556] 跨零。因此“静态概念为当前 RNA 联合模型补充信息”有条件内部证据，“稳定改善 Clinical 最强 anchor”尚未确立；这更不能支持 RNA→prompt、真实 mapping 或 patient-specific conditioning 机制。', '',
        '## 6. 训练预算和审计', '',
        f'- 全部 2550 个 inner/outer Cox 均达到梯度判据；最大 outer iterations **{conv["maximum_outer_iterations"]}/5000**，inner **{conv["maximum_inner_iterations"]}/5000**，最大绝对梯度 **{conv["maximum_gradient"]:.3g}**，上限 1e-5。',
        '- 内外层相同收敛训练，不使用 early stopping。本轮不存在需增加 epoch 才收敛的证据；收敛充分不证明正则网格或表示已达到最优。',
        f'- 独立验证 510 份 OOF/coefficients、60 个 train-only Top512/分割检查、102 份完整 seed OOF；指标误差 **{validation["independent_metrics_error"]:.3g}**，预测重构误差 **{validation["prediction_reconstruction_error"]:.3g}**。',
        '- 缓存和 OOF 持久化在服务器，可复算，不需要重新训练已有 raw cells。', '',
        '## 7. 风险与结论置信度', '',
        '**优势：**统一凸线性 head；严格 inner/outer 预处理；真实 OOF；训练删失权重；同患者配对和两个终点分开；P0R 独立完成后才进生存拟合。',
        '**劣势：**单队列且仅 65 DSS 事件；一次 inner hold-out；三个 seed 共用患者/外折；bootstrap 仅条件于固定 OOF，没有重训 pipeline。Bonferroni 处理固定比较族的区间，不消除历次研究探索偏差；2000 次 bootstrap 的尾部分位还有 Monte Carlo 误差。',
        'Cox 只估计线性风险；UNI2-h 均值池化可能丢局部信息；分子状态均值压缩可能丢预后信息。联合全局 ridge 也会改变 Clinical 系数，因此负点差不能证明新增模态没有条件信息。正则网格有边界选择，不能据收敛证明搜索范围已完备。',
        '风险标定会改变跨折配对排序；raw-risk 与 within-fold 指标全部公开。部分模型 raw pooled 与标定后 pooled 点值相差约 0.02–0.03，不能把本轮数字直接与旧未标定指标混作同一协议排名。',
        '- **高：**数据来源、泄露防护、收敛和 OOF 数值验证。',
        '- **中：**当前固定表示/共同线性头下的点差与条件区间；CS 是容量小、值得保留的候选。',
        '- **低：**外部稳定收益、分子—形态因果机制、复杂 prompt 的可行性，以及“事件少是唯一原因”。', '',
        '## 拓展思考', '',
        'Stage 的真实基因集优势与分子状态 DSS 条件增量可以同时前者成立、后者证据不足。当前结果将下一步问题收窄到：静态病理语义能否稳定补充最强 Clinical anchor，而非先扩大分子 prompt 容量。', '',
        '## 你可能想知道', '',
        '这轮不能区分“65 事件导致方差”与“表示丢失相关信号”的全部贡献，也不能因 PFS/PFI 有 129 事件仍无增量就排除监督预算限制：两个终点机制不同。仅凭这轮结果不应宣布医学知识 idea 成立或无效。', '',
        '## 复现', '',
        '顺序：cache → prepare（校验 P0R 两个完成门）→ worker-id 0–5（六进程）→ aggregate → 独立 validate → report。CLI 见 `phase7a_foundation_survival_headroom.py --help`。源哈希、协议、收敛、每 seed OOF 指标与配对对比均随报告公开；原始数据、患者预测、weights 不公开。', '',
        '- [Efron Cox ties 官方说明](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html)。',
        '- [IPCW/训练删失支持说明](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)。',
        '- [reverse KM ties 原始实现](https://github.com/sebp/scikit-survival/blob/master/sksurv/nonparametric.py)。']
    (SURV/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (SURV/'CONCLUSIONS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cells=[json.loads(f.read_text()) for f in (SURV/'cells').glob('*.json')]
    pd.DataFrame([{'endpoint':c['endpoint'],'seed':c['seed'],'fold':c['fold'],'method':c['method'],'dimension':c['dimension'],
        'parameter_count':c['parameter_count'],'selected_ridge':c['selected_ridge'],'outer_iterations':c['outer_fit']['iterations'],
        'gradient_max':c['outer_fit']['gradient_max'],'train_n':c['n_train'],'train_events':c['events_train'],
        'test_n':c['n_test'],'test_events':c['events_test']} for c in cells]).sort_values(['endpoint','seed','fold','method']).to_csv(SURV/'fit_summary.csv',index=False)
    sd=seed.groupby(['endpoint','method'])[['oof_harrell','oof_uno60']].std().rename(columns=lambda s:s+'_seed_sd').reset_index()
    mean.merge(sd,on=['endpoint','method']).to_csv(SURV/'oof_means_and_seed_sd.csv',index=False)
    for source,name,files in [
        (P0R,'phase7_p0r_robustness',['RESULTS.md','protocol.json','manifest.json','decision.json','validation.json','independent_validation.json',
         'missing_block_source_audit.json','platform_source_audit.json','missingness_source_groups.csv','seed_oof_benchmark.csv','permutation_benchmark.csv','DONE.json']),
        (SURV,'phase7a_foundation_headroom',['RESULTS.md','protocol.json','manifest.json','decision.json','independent_validation.json','convergence_audit.json',
         'endpoint_source_audit.json','mathematical_checks.json','uni2_cache_audit.json','seed_oof_metrics.csv','three_seed_oof_means.csv','oof_means_and_seed_sd.csv',
         'paired_conditional_increments.csv','fit_summary.csv','DONE.json'])]:
        target=REPO/'reports'/name;target.mkdir(parents=True,exist_ok=True)
        for file in files:shutil.copy2(source/file,target/file)
        if source==SURV:shutil.copy2(P0R/'endpoint_provenance_audit.json',target/'endpoint_provenance_audit.json')
        write_json(target/'artifact_checksums.json',{f.name:digest(f) for f in target.iterdir() if f.is_file() and f.name!='artifact_checksums.json'})
    print('BOTH_VALIDATED_REPORTS_PUBLISHED_LOCALLY',flush=True)

if __name__=='__main__':main()
