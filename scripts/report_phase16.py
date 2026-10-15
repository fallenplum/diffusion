"""Publish honest completion boundaries and anonymous aggregate evidence."""
import sys,json,re,subprocess,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from src.phase16_routing.core import *

def main():
    result=json.loads((R/'INDEPENDENT_AUDIT.json').read_text());gate=json.loads((R/'HARD_GATES.json').read_text());pilot=json.loads((R/'PILOT_GATE.json').read_text())
    if result['status']!='PASS' or gate['status']!='PASS' or result['source_hash']!=source() or pilot['source_hash']!=source():raise ValueError('not accepted')
    s=pd.read_csv(R/'SEED_OOF_METRICS.csv');pair=pd.read_csv(R/'PAIRED_COMPARISONS.csv');ms=pd.read_csv(R/'MODEL_SUMMARY.csv');training=pd.read_csv(R/'TRAINING_DIAGNOSTICS.csv');g1=pd.read_csv(R/'G1_COVERAGE.csv');asset=json.loads((P/'ASSET_SUMMARY.json').read_text());coord=json.loads((R/'COORDINATE_AUDIT.json').read_text());means=s.groupby(['endpoint','arm'])[['Harrell','Uno60']].mean();summary=means.add_suffix('_3seed_mean').join(s.groupby(['endpoint','arm'])[['Harrell','Uno60']].std().add_suffix('_seed_std')).reset_index();summary.to_csv(R/'AGGREGATE_METRICS.csv',index=False)
    anchors=[]
    from validate_phase16_independent import cs_independent
    from src.phase15_complement.core import cs_features
    d=parent.load()
    for seed in cfg()['seeds']:
        for fold in cfg()['folds']:
            for role in ['inner','outer']:
                a,p=anchor(seed,fold,role);x,st=cs_features(d,a['fit']);v=cs_independent(d,a['fit']);err=float(np.max(abs(x-v)));riskerr=float(np.max(abs(v[a['test']]@a['w']-a['test_risk'])))
                if err>1e-10 or riskerr>1e-10:raise ValueError('anchor elementwise failure')
                anchors.append({'seed':seed,'fold':fold,'role':role,'all_element_feature_error':err,'risk_error':riskerr,'parameters':len(a['w']),'fit_n':len(a['fit']),'file_sha256':sha(p)})
    pd.DataFrame(anchors).to_csv(R/'ANCHOR_ELEMENTWISE.csv',index=False)
    active=[]
    for arm in arm_names():active.append({'arm':arm,'status':'HISTORICAL_REFERENCE' if arm in ['C_LEGACY','CS_LEGACY','CS_P15','C4_LEGACY'] else 'COMPLETE','outer_fold_records':15,'actual_new_outer_fits':0 if arm in ['C_LEGACY','CS_LEGACY','CS_P15','C4_LEGACY'] else 15})
    for selector in cfg()['blocked_selectors']:
        for m in cfg()['budgets']:active.append({'arm':f'{selector}_{m}','status':'BLOCKED_CONCH_Z_ACCESS','outer_fold_records':0,'actual_new_outer_fits':0})
    pd.DataFrame(active).to_csv(R/'ARM_STATUS.csv',index=False)
    # All decisions remain bounded by the missing semantic and raw-image gates.
    positive=[]
    for comparison,g in pair[(pair.endpoint=='DSS')&(pair.status=='COMPLETE')].groupby('comparison'):
        if len(g)==2 and (g.corrected_low>0).all() and (g.positive_seeds==3).all():positive.append(comparison)
    decision={'status':'BLOCKED','nonsemantic_status':'EXPLORATORY_CORRECTED_INCREMENT' if positive else 'NO_CONFIRMED_INCREMENT','semantic_route':'BLOCKED_CONCH_Z_AUTHORIZATION_STILL_PENDING_AUTHOR_REVIEW','dual_visual':'BLOCKED_RAW_WSI_QA','G3':'BLOCKED_SEMANTIC_GATE','G5':'BLOCKED_SEMANTIC_G2_GATE','completed_arms':14,'new_formal_outer_fits':150,'historical_reference_records':60,'secondary_two_metric_positive_comparisons':positive,'primary_tests_run':0,'primary_tests_reserved':16,'source_hash':source(),'candidate_pool_limited':True,'original_slides_recovered':'519/520','execution_confidence':'high within completed scope','patient_specific_semantic_mechanism_confidence':'not estimable;not trained','claim':'no decision against semantic routing idea from blocked arms'}
    decision.update(formal_ROI_budget_ceiling_cells=int((ms.selected_epoch==80).sum()),formal_ROI_cells=120,training_budget_sufficiently_excluded=False,posthoc_budget_diagnostic_outer_fits=0)
    write(R/'decision.json',decision)
    text=['# Phase16 — 患者条件化病理语义路由','', '**当前结论：BLOCKED（语义主实验未运行）。已完成可执行的非文本对照；不能据此否定患者条件化语义路由。**','',
    f'**candidate-pool-limited**：原始520张切片中519张有CONCH，覆盖{asset["CONCH_patients"]}名患者、{asset["CONCH_original_patches"]:,}个patch，全部匹配原始H5哈希。缺失切片的患者已有另一张匹配切片，不替换或重命名。历史128缓存只有{asset["historical_128_patches"]:,}个patch。本轮使用所有可用CONCH原始行，不是128缓存内检索，也不声称完整原始WSI/组织掩码覆盖。','',
    '## 完成范围','',
    '- RAND/GRID/DIVERSE/MIL × 128/512：8 arms × 15；另补2种global表征对照，共150个新的outer-fold训练流程。每流程包含inner训练/选择及fresh outer重训，不把内部优化trial谎报独立正式cell。',
    '- Clinical-only、历史CS、Phase15嵌套CS、历史C4：60个引用记录；没有重复训练。总210个fold单元，DSS与PFI各210个指标记录。',
    '- 语义8 arms、G3错Prompt对照和G5未运行：新令牌返回403，明确“等待仓库作者审核”。授权申请不是已可下载；本轮没有绕过模型门禁。',
    '- G4不运行：原始WSI在服务器及目前可访问E盘未找到；只有计算坐标映射，没有30组原图QA。',
    '- MIL实现为同容量端到端top-M attention-MIL；没有单独Stage-A selector训练，额外selector训练计数0。独立两阶段MIL及其预算匹配消融尚未完成；不宣称该部分完整满足机制验收。','',
    '## 统一训练及预算','',
    'Patch逐行L2归一化只用于视觉建模，不意味着图文对齐。RAND为固定患者/切片随机排名；GRID为归一化坐标farthest-point空间均匀覆盖；DIVERSE为固定随机32维视觉投影+0.1坐标后的无监督k-center。GRID是空间覆盖代理，不是组织学验证的规则网格。三个selector均不查看终点标签。',
    '预算按患者总数分配给全部WSI，按原始patch数比例的确定性round-robin生成嵌套配额；每张WSI至少1块，不跨患者合并。候选不足M时只用所有真实patch并显式mask，实际不足数量见G1表；没有把512私自替换成128。多WSI先各自attention聚合，再等权平均到患者。',
    '所有8个核心arm使用相同的线性768D attention和768D风险头，加上真实forward中的Clinical+S32。最终风险为联合可训练CS线性主效应+0.25*tanh(ROI风险)；初始化使用本角色训练集内嵌套选择的Phase15 CS。CS主效应同步优化，避免把in-sample预测当新stacking特征。输入6个合法临床字段和32维S32。',
    'Adam lr0.001、ROI L2=0.001、inner最多80epoch；按inner-val Cox loss的预定义EMA(0.3)选择epoch，外层从同一初始化fresh refit相同epoch。没有用outer-test选择方案或扩展预算。所有ROI/attention梯度在pilot真实核验；S32具有非零梯度。',
    'Global对照为train-local PCA32后与CS联合Ridge Cox，global惩罚[0.1,1,10,OFF]只由inner-val选；不是神经ROI架构的等参数对照。','',
    '## DSS结果','', '|模型|Harrell：3个seed OOF均值|Uno60：3个seed OOF均值|','|---|---:|---:|']
    for arm in arm_names():
        row=means.loc['DSS',arm];text.append(f'|{arm}|{row.Harrell:.6f}|{row.Uno60:.6f}|')
    text+=['','先在每个fold保存同折可比较对分子/分母，再按seed拼5-fold OOF汇总，最后均值3个seed；不是直接平均15个cell。不同fold风险不互相比较，seeds不合并为新患者。Clinical确定性参考跨seed重复不代表三个独立模型证据。C4历史模型还含RNA交互，预算为每张WSI128，不等于本轮每名患者128；仅作保留历史强参考，不能据差值独占归因ROI选择。','',
    '## 预注册配对比较','',
    '语义8个主要contrast及两指标共16个检验仍保留为BLOCKED，没有以可执行分支的胜出结果替换primary。非文本预定义26个次要contrast×2指标采用Bonferroni52；30,000次患者共享权重bootstrap。全部区间见PAIRED_COMPARISONS.csv。bootstrap固定模型，不含全流程重训方差；极端校正分位数仅约14个tail draws，需保留蒙特卡洛不确定性。','',
    '|DSS次要比较|指标|Δ|95% CI|Bonferroni52 CI|正向seed|','|---|---|---:|---|---|---:|']
    focus=pair[(pair.endpoint=='DSS')&(pair.status=='COMPLETE')&pair.comparison.str.endswith('-CS_P15')]
    for row in focus.itertuples():text.append(f'|{row.comparison}|{row.metric}|{row.delta:+.6f}|[{row.CI_low:+.6f},{row.CI_high:+.6f}]|[{row.corrected_low:+.6f},{row.corrected_high:+.6f}]|{int(row.positive_seeds)}/3|')
    text+=['','## 训练有效性','', '|arm|选中epoch中位数|选中80epoch的cells|峰值分配显存MiB|','|---|---:|---:|---:|']
    for arm,g in ms[ms.arm.str.contains(r'_(128|512)$',regex=True)].groupby('arm'):text.append(f'|{arm}|{g.selected_epoch.median():.1f}|{int((g.selected_epoch==80).sum())}/15|{g.peak_GPU_MiB.max():.1f}|')
    text+=['','各epoch损失、inner Harrell与梯度见TRAINING_DIAGNOSTICS.csv。早期最佳epoch只说明此锁定协议下inner验证选择较早，不证明训练充分或不充分；触及80epoch的单元需视为预算尚不能排除的限制。正式外折预算保持不变，另列不改变正式预测的内层事后诊断。ROI校正幅度、base-only风险保留在私有checkpoint，不能把CS优化产生的变化全部归因于ROI。','',
    '## 坐标及空间硬门槛','',
    f'已计算{coord["computationally_mapped_slides"]}张切片CONCH→UNI2的真实矩形重叠，不按行号配对。配对切片含{coord["uni2_patches_in_paired_slides"]:,}个UNI2子块；切片均值面积覆盖{coord["area_mean_across_slides"]:.3%}，CONCH行未匹配均值{coord["unmatched_mean_across_slides"]:.3%}。保存30组坐标例子，但原图QA完成0组。因此不能将这些计算比例当作组织学对齐正确率，G4保持BLOCKED。',
    'TITAN native输出需视觉投影到其slide-level文本空间；历史S32确实使用该投影，未发现“直接raw native cosine”问题。但TITAN slide文本不能直接余弦raw CONCH1.5 patch。官方CONCH-Z提供project_image_features和独立匹配文本塔；下载并成功功能核验前，只能说接口路径可行，不能说本队列patch图文兼容已PASS。','',
    '## 五个研究问题','',
    '1. 官方图文空间能否路由：找到CONCH-Z合法接口；权重访问等待作者审核，本地功能验证未完成。',
    '2. 患者条件路由是否胜过FIXED/GRID/MIL：未运行语义arm，不能回答。',
    '3. 真RNA是否胜过错RNA：G3被阻塞，不能回答。',
    '4. UNI2正确匹配后是否净增量：尚缺原图QA，未训练双视觉，不能回答。',
    '5. 是否共同超过强CS和C4且校正/跨seed稳定：本轮非文本比较见上；语义主假设未获检验，不作SOTA或病理机制成立声明。','',
    '## 优势、风险与置信度','',
    '**优势：** 全量H5候选、统一患者预算、同容量非文本比较；fold-local clinical/PCA、真实S32 forward、独立原始终点和逐checkpoint回放。故障注入确实拒绝跨患者ROI、错尺度、未经验证双视觉、跨分区/身份Prompt、删失时间篡改、反向风险、不匹配checkpoint；外折标签/时间变化不改变训练及内层选择。',
    '**劣势/风险：** 单队列65事件且多阶段反复探索；无ROI金标准/原图组织掩码验证；固定训练配置与极小线性聚合器不能排除其他模型的收益；C4面积不严格等价；MIL两阶段消融未完成；语义负对照和双视觉缺失。负结果只能限定当前执行范围。',
    '**置信度：** 已执行分支的实现与指标核验为高；患者特异性语义机制未检验，不能评级为成立。PFI为DSS模型在510/129有效集上的冻结迁移敏感性，不是PFI重新训练，也不是独立外部确认。',
    '## 拓展思考','', '当前主要缺口是检验语义路由所需的匹配文本模型授权，而非据非文本结果决定继续堆叠融合模块。','',
    '## 你可能想知道','', 'E盘已重新接入，保留原始CONCH/TITAN大文件；服务器重用已有完整H5，新增约3.7GiB标准化缓存放在易失/dev/shm，重启可从只读H5重建。权重、患者级OOF、切片坐标和所有输入关联不公开。',
    '接口依据：[官方LIMIT/CONCH-Z](https://github.com/mahmoodlab/LIMIT)、[TITAN维护者说明](https://github.com/mahmoodlab/TITAN/issues/17)、[CONCH投影接口](https://github.com/mahmoodlab/CONCH)。']
    (R/'RESULTS.md').write_text('\n'.join(text),encoding='utf8');(R/'DECISION.md').write_text('# 研究决策\n\n'+json.dumps(decision,indent=2,ensure_ascii=False)+'\n\n详见RESULTS.md；未执行的语义路线不能判定为NO-GO。',encoding='utf8')
    if (R/'BUDGET_DIAGNOSTIC.csv').exists():
        bd=pd.read_csv(R/'BUDGET_DIAGNOSTIC.csv');n80=int((ms.selected_epoch==80).sum());n320=int((bd.inner_selected320==320).sum());improved=int((bd.inner_min_EMA320<bd.inner_min_EMA80-1e-8).sum())
        c_up=int((bd.inner_val_C320>bd.inner_val_C80+1e-8).sum());c_down=int((bd.inner_val_C320<bd.inner_val_C80-1e-8).sum())
        appendix=['','## 仅内层的预算诊断（事后，非确认）','',f'正式ROI流程中{n80}/120个单元选中80epoch上限。固定seed31、fold1/2的16个内层流程重跑到320epoch，不生成或保存外折预测/模型，未改变任何正式结果。其中{improved}/16个内层最小EMA loss进一步下降，{n320}/16个仍在320epoch上限；inner Harrell提高{c_up}个、下降{c_down}个、相同{len(bd)-c_up-c_down}个。loss下降不等于排序增益。详细结果见BUDGET_DIAGNOSTIC.csv。',
          '这只能诊断当前优化日程，不能视为320epoch的泛化改进证据。预算诊断在正式OOF评估之后进行，明确为事后探索；未用外折优劣选择预算，未追加outer fit。若内层继续下降，原80epoch结果不能支持“充分训练后方法仍无收益”。即使内层改善也不能自动外推至DSS held-out增量。',
          '算力表COMPUTE_PROFILE.csv含实测批量前向时延和线性算子解析MAC×2。torch profiler会漏计MV，故同时给出解析注意力FLOPs；top-k/softmax/mask和I/O不完整计入，不称为完整训练FLOPs。',
          'RISK_COMPONENTS.csv与INFERENCE_ABLATION_SEED_METRICS.csv分别记录ROI校正/CS系数漂移与冻结checkpoint去ROI、置零S32的预测变化。它们是实际forward的推理诊断，不是重新训练的公平对照，不计入新formal fits，也不用于新主假设选择。']
        with (R/'RESULTS.md').open('a',encoding='utf8') as f:f.write('\n'.join(appendix))
    reproduce='''# 复现\n\n起点04150dc976057f825702d5fe3502696060790dd7；分支codex/phase16-semantic-roi-routing。需已有Phase11–15私有只读资产与原始H5；公开仓库不包含患者数据/权重。\n\n```bash\nexport PYTHONPATH=/root/phase10_trimodal_eval_deps\nexport CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\nPY=/root/miniconda3/bin/python\n$PY scripts/audit_phase16.py\n$PY scripts/prepare_phase16.py\n$PY scripts/coordinate_audit_phase16.py\n$PY scripts/pin_phase16_inputs.py\n$PY scripts/launch_phase16.py --pilot\n$PY scripts/execute_phase16.py\n$PY scripts/pin_phase16_inputs.py\n$PY scripts/reconcile_phase16_scope.py\n$PY scripts/report_phase16.py\n$PY scripts/publish_phase16.py\n```\n\n原始H5只读，prepare缓存易失；已有同源cells可恢复，但输入哈希/协议不符时报错。所有新的正式outer fit有inner/outer checkpoint和实际ROI索引；仅正确完成且独立验收后发布。Pilot只存资源/梯度摘要，不存探索性预测或权重。\n\n访问权生效后应作为新的预注册执行批次，先做官方projection/text功能核验、实际Prompt sanity和1seed×2fold数值pilot，再启动语义/G3；本分支不能伪装已完成这些实验。G4必须补原始WSI与30组原图QA。\n'''
    (R/'REPRODUCE.md').write_text(reproduce,encoding='utf8')
    with (R/'REPRODUCE.md').open('a',encoding='utf8') as f:f.write('\n一键重建当前受限范围：`/root/miniconda3/bin/python scripts/pipeline_phase16.py`；加`--publish`仅在完整验收后提交公开清单。该命令不会解锁被模型授权或原图QA阻塞的分支。\n')
    env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'source_hash':source(),'workers':cfg()['workers'],'GPU_total_MiB':torch.cuda.get_device_properties(0).total_memory/1024**2};write(R/'ENVIRONMENT.json',env)
    public=sorted((ROOT/'configs').glob('phase16_*.json'))+[ROOT/'PHASE16_SEMANTIC_ROI_ROUTING_CODEX_PROMPT.md']+sorted((ROOT/'src/phase16_routing').glob('*.py'))+sorted((ROOT/'scripts').glob('*phase16*.py'))+[p for p in sorted(R.iterdir()) if p.suffix in ['.md','.csv','.json'] and p.name not in ['ARTIFACT_MANIFEST.json','DELIVERY.json']]
    pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----');files=[]
    for p in public:
        b=p.read_bytes()
        if pattern.search(b):raise ValueError('protected identifier in '+str(p))
        columns=pd.read_csv(p,nrows=0).columns.tolist() if p.suffix=='.csv' else []
        if set(columns)&{'patient_id','patient_ids','risk','test_indices','event','time'}:raise ValueError('protected patient rows')
        files.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':len(b)})
    private=[{'kind':'all_patient_inputs_OOF_checkpoints_full_ROI','location':'private server directories;not public','source_inputs_pin_sha256':sha(P/'INPUT_PINS.json'),'candidate_cache_sha256':manifest()['cache_sha256'],'retained':True}]
    write(R/'ARTIFACT_MANIFEST.json',{'status':'PASS','public_files':files,'private_artifacts':private,'historical_results_overwritten':False,'source_hash':source()});delivery={'status':'PASS_PARTIAL_G0_SCOPE','verified_formal_cells':210,'actual_new_outer_fits':150,'reference_records':60,'public_files':len(files),'manifest_sha256':sha(R/'ARTIFACT_MANIFEST.json'),'decision':'BLOCKED','sensitive_data_public':False};write(R/'DELIVERY.json',delivery);write(P/'DONE.json',delivery);print(json.dumps(delivery),flush=True)
if __name__=='__main__':main()
