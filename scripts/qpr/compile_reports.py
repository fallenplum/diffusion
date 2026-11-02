"""Compile descriptive aggregate reports; no fitting or gate-threshold changes."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np


def csvwrite(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def gate1(reports):
    p=reports/'gate1';decision=json.loads((p/'decision.json').read_text());cells=json.loads((p/'ALL_SEED_RESULTS.json').read_text())
    cost=[];capacity=[];synthetic=[];summary=[];training=[]
    for arm in ('A0','A1','A2','A3','A4','A5'):
        values=[next(r for r in c['results'] if r['arm']==arm) for c in cells]
        summary.append({'arm':arm,'mean_probe_RMSE':np.mean([r['real']['RMSE_probe'] for r in values]),
                        'SD_probe_RMSE':np.std([r['real']['RMSE_probe'] for r in values],ddof=1),
                        'mean_true_query_RMSE':np.mean([r['real']['RMSE_true'] for r in values]),
                        'mean_probe_logZ_MAE':np.mean([r['real']['logZ_MAE_probe'] for r in values])})
    for c in cells:
        for r in c['results']:
            if r['neural_fits']:
                cost.append({'seed':c['seed'],'arm':r['arm'],'epochs':r['trained_epochs'],'selected_epoch':r['selected_epoch'],
                             'optimizer_updates':r['optimizer_updates'],'elapsed_seconds':r['elapsed_seconds']})
                capacity.append({'seed':c['seed'],'arm':r['arm'],'parameters':r['parameter_count'],'active_slots_min':r['active_slots_min'],'maximum_slot_mass':r['max_slot_mass']})
            for kind,m in r['synthetic'].items():synthetic.append({'seed':c['seed'],'arm':r['arm'],'kind':kind,**{k:v for k,v in m.items() if not isinstance(v,list)}})
        for curve in c['curves']:
            for row in curve['history']:training.append({'seed':c['seed'],'arm':curve['arm'],**row})
    csvwrite(p/'arm_summary.csv',summary);csvwrite(p/'training_cost.csv',cost);csvwrite(p/'capacity_profile.csv',capacity)
    csvwrite(p/'synthetic_negative_controls.csv',synthetic);csvwrite(p/'training_effectiveness.csv',training)
    ledger={'qualified_neural_fits':len(cost),'qualified_optimizer_updates':sum(r['optimizer_updates'] for r in cost),
            'qualified_fit_elapsed_seconds_sum':sum(r['elapsed_seconds'] for r in cost),'KMeans_fit_calls':10,
            'checkpoint_save_load_replays':sum(len(c['checkpoint_replays']) for c in cells),
            'maximum_replay_metric_error':max(r['metric_replay_max_abs'] for c in cells for r in c['checkpoint_replays']),
            'formal_lock_sha256':sorted({c['formal_lock_sha256'] for c in cells}),
            'no_survival_labels_used':all(not c['DSS_labels_read'] for c in cells),
            'validation_used_for_epoch_selection':True,'independent_clinical_test':False,
            'G1_temporary_weights_deleted_after_actual_replay':True,'G2_selected_weights_retained':True,
            'invalid_software_attempt_cost':'reported_separately_in_maintenance_G1_INVALID_SOFTWARE_ATTEMPTS'}
    (p/'RUN_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
    table='\n'.join(f"| {r['arm']} | {r['mean_probe_RMSE']:.8g} | {r['mean_true_query_RMSE']:.8g} | {r['mean_probe_logZ_MAE']:.8g} |" for r in summary)
    text=f"""# G1：无 DSS 查询保真

**结论：{decision['status']}。** A4 相对 A2 的未见探针 RMSE 下降比例为 {decision['fraction_RMSE_reduction']:.6%}，正向种子 {decision['positive_seeds']}/10。锁定门要求下降至少15%、至少8个种子正向，并同时满足 logZ、质量结构及拓扑反例条件。

| 组别 | 探针 RMSE 均值 | 真实查询 RMSE 均值 | 探针 logZ MAE 均值 |
| --- | ---: | ---: | ---: |
{table}

本轮有效训练为20次压缩器拟合、10次训练侧全局 KMeans 拟合；优化步数为{ledger['qualified_optimizer_updates']}。所有10个种子均报告。A0为完整读取，A5仅为去质量推理诊断，不计新增拟合。

范围：outer1训练患者内的328人拟合、81人用于无结局验证；synthetic A/B/C均按预锁训练、验证、测试拆分。真实患者验证也用于选轮，因此这些是内部资格证据，不是独立医学检验。

软件说明：首次尝试的零质量 log(0) 反向错误已用独立最小例复现。修复只改变空摘要的数值求值，正质量前向等价；原失败日志、旧版本和不完整计算量边界全部保留，未计入有效科学比较。有效版本完成20次真实落盘并重新加载的检查点回放；临时 G1 权重随后删除，保留其SHA及回放证据。

优势：固定教师、固定摘要预算和全种子比较直接检验当前查询保持压缩器。局限：低维PCA、单位球内查询和固定损失权重限定了结论范围；结果不能推导为所有空间结构或所有压缩方法无效。不计算置信区间、bootstrap或p值。
"""
    (p/'RESULTS.md').write_text(text,encoding='utf-8')


def gate2(reports):
    p=reports/'gate2';d=json.loads((p/'decision.json').read_text());cost=json.loads((p/'TRAINING_COST.json').read_text())
    rows=[{k:v for k,v in c.items() if k!='wrong_query_mapping'} for c in cost];csvwrite(p/'training_effectiveness.csv',rows)
    text='\n'.join(f"| {arm} | {m['Harrell']:.6f} | {m['Uno60']:.6f} |" for arm,m in d['mean_metrics'].items())
    differences='\n'.join(f"| B3−{arm} | {m['Harrell']:+.6f} | {m['Uno60']:+.6f} |" for arm,m in d['paired_mean_differences_B3_minus'].items())
    results=f"""# G2：完整真实空间关系的 DSS 条件增量

**结论：{d['status']}。** 本轮只使用outer1训练患者内的3个内层分区和2个种子，不读取outer1测试结局。每个种子先汇总3个分区的折内有效患者对分子/分母，再对两个种子等权平均。

| 模型 | Harrell | Uno60 |
| --- | ---: | ---: |
{text}

| 同患者配对比较 | ΔHarrell | ΔUno60 |
| --- | ---: | ---: |
{differences}

主门要求B3−B0至少+0.005 Harrell、+0.003 Uno60，至少2/3分区及2/2种子Harrell正向；还须超过无空间、错拓扑、错RNA、无RNA和无条件对照。实际正向分区{d['positive_inner_splits']}/3、种子{d['positive_seeds']}/2。

实际执行36次神经拟合、2880次完整训练风险集优化更新；每次均执行80轮，选轮仅比较0/10/20/40/80的内层验证Efron Cox。即使选中初始化或早期快照，也完整保留后续真实训练结果。B0另外新拟合156个Cox模型及12个标量模型，交叉拟合整个锚点流程，包含监督标量系数；所有组共享本人Clinical/病理/RNA主风险。

风险读出使用显式L2（系数1e-4）与AdamW解耦weight decay（1e-4），二者均在训练前锁定；另有0.01归一化风险修正平方惩罚，没有用tanh硬截断代替。第一阶段仅训练新增头，后60轮联合训练Query。

评分器：逐单元独立调用scikit-survival 0.25.0官方Harrell与IPCW Uno60，训练侧估计删失分布。最大评分差为{d['official_scorer_maximum_absolute_difference']:.3g}。训练使用确定性两遍前向的完整Efron风险集，未用微批Cox替代。配对修复/损伤与净Harrell变化做恒等式核验。

优势：原始真实边、冻结共同强锚点、真实重训负对照，可区分空间关系与额外容量/主效应。局限：B2读取节点而B3读取边，输入数量和计算量不同；B1/无RNA组的有效Query参数量也不同，已披露。错RNA使用训练患者供体，优先癌种/分期匹配，回退比例另列；它与本人Query的比较是特定错配设计证据。全部是长期开发过的TCGA内部证据，不能称外部泛化或因果机制确认。不计算置信区间。
"""
    (p/'RESULTS.md').write_text(results,encoding='utf-8')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--reports',required=True);a.add_argument('--stage',choices=['gate1','gate2'],required=True);x=a.parse_args()
    globals()[x.stage](Path(x.reports))
