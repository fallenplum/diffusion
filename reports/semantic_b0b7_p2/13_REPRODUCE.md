# P2 复现与交付命令

本文记录已锁定的接口和执行顺序，不代表正式矩阵、软件资格、历史风险核查或发布已经完成。终态数值、实际 fit 数、运行时间和科学判断必须引用本轮真实生成的 05–11 报告，不能从下述结构预算推填。所有代码块均为复现说明，本文件的生成没有执行其中的训练或读取 outer 风险。

## 1. 固定版本与授权资产

| 对象 | 固定标识 |
|---|---|
| 科学主锁 `configs/semantic_b0b7/p2_lock.json` | `b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c` |
| 完整 P1R 发布父提交 | `d2568ad4d57bb9c83366029cacee6baea6c3522d` |
| P1R publication manifest | `2e914a8326614843520e9fbae4cd6b5b6a13608ebcc9eb0dd6518d5efd16d598` |
| 原无 DSS、2×50 次更新资格脚本 | `ddf2176ed7738cdd0b78b42d5f9dea06b053753376ab59678743f2c137627fb2` |
| 完整 DSS 管线标签隔离资格脚本 | `84bf92571e00288bbcc41e808ef1e866f8900778dd7e85dd8f6b41d30937d7b8` |
| R4 历史 seed31 风险核查脚本 | `4d85be94306fd5a4596b3ad70c0d6406d2434500300ef46da823f511b4d7cc39` |
| 完整训练预算与曲线审计脚本 | `31924339c2f9b3cb346a29d3823f1a158c23eae6adbbfe0ba9d6bc04fef6c8e7` |

三个资格/历史核查子脚本均独立于主锁的20份科学源码，不能据其新增而改写正式模型、已冻结 epoch 或 lambda。主锁保存的准备时 HEAD 为 `98f42e298975c39e9e3bde70f2250b2334caca8b`；P1R 当时的资格内容已按哈希绑定，后续完整发布父提交另行绑定。详见 `00_PARENT_AND_PROTOCOL_LOCK.md`，不能把这两个时间点伪写成同一提交。

完整数据复现需要研究者具有以下资产的合法访问权限，GitHub 交付不包含这些数据：canonical 512 人组学/临床容器、独立 DSS endpoint 容器、患者折映射、原 GRID-local/full-CONCH 均值及其收据、REGIONS512 patch/slide/mask 注册表、TITAN 缓存、CONCH-Z 官方匹配投影、固定文本/关系与 Hallmark/旧八程序成员文件、P1R 已核验输入以及旧 R4 五折私有模型与 COMPLETE 收据。受限模型权限不能由另一模型的权限替代。

本轮不是从原始 SVS 到所有 FM 权重的全链路复现：使用已核验冻结特征与权重，当前无原始 WSI 组织学 QA。原始资产和模型使用条款需独立满足。Python 运行环境需支持本轮锁定源码实际使用的 PyTorch、NumPy、SciPy、scikit-learn、h5py 与 safetensors；官方 scikit-survival 0.23.1 的评分核查使用已有隔离依赖目录。具体版本以实际 runtime/profile 收据为准，不能把合成 API 资格视为完整估计器环境复现。

## 2. 路径约定与字节核验

以下为服务器 Bash 示例。先将所有带尖括号的路径替换为已授权位置；不把令牌、口令、患者路径写入公开文件。所有新输出目录都必须尚不存在。正式运行、软件资格、报告与元数据发布使用不同目录；不要覆盖历史结果。

```bash
PYTHON="<PINNED_PYTHON_EXECUTABLE>"
REPO="<PRIVATE_REPO>"
LEGACY="<PINNED_LEGACY_REPO>"
ASSETS="<PRIVATE_ASSET_MAPPING_JSON>"
FROZEN="<PINNED_ALL_512_FROZEN_IMAGES_PT>"
RUN="<PRIVATE_RUN>"
AUX="<PRIVATE_AUXILIARY_OUTPUT_ROOT>"
REPORT="<NEW_ANONYMOUS_REPORT_OUTPUT>"
PRESERVED="<PRESERVED_HISTORY_AND_P1R_TREE>"
OFFICIAL_PKGS="<ISOLATED_SCIKIT_SURVIVAL_PACKAGE_DIRECTORY>"
COHORT="<PINNED_CANONICAL_COHORT_NPZ>"
DSS="<PINNED_DSS_ENDPOINT_NPZ>"
HISTORY="<PRIVATE_HISTORICAL_R4_ROOT>"
PROFILE="<PINNED_SIX_CELL_ZERO_UPDATE_PROFILE_JSON>"
PROFILE_SHA="becf224dece6ec249dd00966f1b174a5ec05be3aabb65f8da4a84217d4ce8914"
LOCK_SHA="b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c"
PARENT="d2568ad4d57bb9c83366029cacee6baea6c3522d"
cd "$REPO"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
```

在首次执行前，核对主锁、20份科学源码和三个子脚本的原始字节，不通过换行重写来“修复”哈希：

```bash
"$PYTHON" - <<'PY'
import hashlib, json
from pathlib import Path
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock = Path('configs/semantic_b0b7/p2_lock.json')
assert sha(lock) == 'b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c'
value = json.loads(lock.read_text(encoding='utf-8'))
assert len(value['frozen_sources']) == 20
for name, expected in value['frozen_sources'].items():
    assert sha(name) == expected, name
for name, expected in {
 'scripts/semantic_b0b7_p2_leakage_audit.py':'ddf2176ed7738cdd0b78b42d5f9dea06b053753376ab59678743f2c137627fb2',
 'scripts/semantic_b0b7_p2_dss_isolation_audit.py':'84bf92571e00288bbcc41e808ef1e866f8900778dd7e85dd8f6b41d30937d7b8',
 'scripts/semantic_b0b7_p2_r4_history_audit.py':'4d85be94306fd5a4596b3ad70c0d6406d2434500300ef46da823f511b4d7cc39',
}.items():
    assert sha(name) == expected, name
print('SOURCE_BYTES_VERIFIED_NO_DATA_OR_RESULT_READ')
PY
```

若需要重新构造纯 FM 缓存，使用单独新目录。此步骤不训练、无结局输入；不在任何 outer fold 之间共用已训练 adapter、PCA 或临床背景：

```bash
"$PYTHON" scripts/semantic_b0b7_p2_prepare_frozen.py \
  --assets "$ASSETS" --legacy-root "$LEGACY" \
  --prior-input "<PINNED_P1R_INPUT_PT>" --prior-input-sha256 "<PINNED_P1R_INPUT_SHA256>" \
  --conch-model "<AUTHORIZED_PINNED_CONCH_Z_MODEL>" \
  --titan-cache "<PINNED_TITAN_PATIENT_CACHE_NPZ>" \
  --output-newdir "<NEW_FM_ONLY_CACHE_DIRECTORY>"
```

原主锁要求冻结缓存容器 SHA 为 `eb12386a1f51b65635893e47a60f79867a092eb663615dcce932b6db0cf41da5`。重建后的序列化容器若不同，不能直接改旧主锁或声称已经复现原封存字节；应保留新产物、核对张量与官方投影误差并另行审计版本。最直接的本轮字节复现使用原封存、已授权缓存及其收据。

## 3. 正式有限矩阵

正式预算绑定已有六单元真实零更新GPU profile。若需要在新机器核查资源，先生成仅含行索引的独立scope，再在独立进程profile；planner会读取DSS进行原合同的分层划分，profile进程本身不读取结局。这些是资源资格，不是新模型训练或性能结果。E0完整锚点CPU profile同样单列成本，不能从正式ledger中免费扣除：

```bash
"$PYTHON" scripts/semantic_b0b7_p2_profile_scope.py \
  --assets "$ASSETS" --legacy-root "$LEGACY" --output "<NEW_PRIVATE_PROFILE_SCOPE_JSON>"

"$PYTHON" scripts/semantic_b0b7_p2_profile.py \
  --assets "$ASSETS" --legacy-root "$LEGACY" --frozen-images "$FROZEN" \
  --expected-frozen-sha256 eb12386a1f51b65635893e47a60f79867a092eb663615dcce932b6db0cf41da5 \
  --scope "<NEW_PRIVATE_PROFILE_SCOPE_JSON>" --expected-scope-sha256 "<ACTUAL_NEW_SCOPE_SHA256>" \
  --output-newdir "<NEW_ZERO_UPDATE_PROFILE_DIRECTORY>"

"$PYTHON" scripts/semantic_b0b7_p2_anchor_profile.py \
  --assets "$ASSETS" --legacy-root "$LEGACY" --output "<NEW_SEPARATE_E0_PROFILE_DIRECTORY>"
```

新profile有新的计时与收据字节，不能替换原锁定收据后冒称其SHA不变。若资源资格与原并发预算不同，先保留差异并审查运行资源安排，不能据测试性能变更科学模型。然后运行源码/合成测试；这一步同样不代表真实科研结果完成：

```bash
"$PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_semantic_b0b7_p2_data.py tests/test_semantic_b0b7_p2_anchor.py \
  tests/test_semantic_b0b7_p2_models.py tests/test_semantic_b0b7_p2_survival.py \
  tests/test_semantic_b0b7_p2_schedule.py tests/test_semantic_b0b7_p2_report.py \
  tests/test_verify_semantic_b0b7_p2.py tests/test_semantic_b0b7_p2_leakage_audit.py \
  tests/test_semantic_b0b7_p2_dss_isolation_audit.py \
  tests/test_semantic_b0b7_p2_r4_history_audit.py \
  tests/test_semantic_b0b7_p2_training_budget_audit.py
```

经实际 profile 资格后执行同一 scheduler。它依次完成 E1、其余 inner、所有 outer 的选择冻结屏障和 final；不要另起第二个 scheduler 抢占同一个 RUN：

```bash
"$PYTHON" scripts/semantic_b0b7_p2_schedule.py \
  --lock configs/semantic_b0b7/p2_lock.json --expected-lock-sha256 "$LOCK_SHA" \
  --private "$RUN" --assets "$ASSETS" --legacy-root "$LEGACY" --frozen-images "$FROZEN" \
  --stage all --max-parallel 6 --cpu-parallel 2 --threads 2 --poll-seconds 5
```

按阶段执行的等价接口为 `--stage e1`、`--stage inner_remaining`、`--stage final`，顺序与资格校验不能跳过。已有单元仅在其完整源/输入/checkpoint/回放收据均匹配时复用；没有“复制输出填满矩阵”的恢复方式。并发上限为已 profile 的6，实际 admission 仍受显存、容器内存和磁盘约束，不保证始终六路同时运行。

结构预算是480个 source neural fits＝360 inner＋120 final；1800个下游 Cox＝1620 inner lambda fits＋180 final（105主臂、60额外控制、15原结构诊断）；20个共同锚点分区包含1040个 teacher Cox＋80个 scalar。B0五折复用对应 full R4 teacher，不能再虚增为三seed或额外重复拟合。MAIN 可由五个读出共享，但实际背景和每个 lambda 的 PCA 重拟合均须按09实际 ledger 计费。最终 optimizer steps 依赖冻结 inner 所选 final epoch，不能用“480×400”代填。

## 4. 正式完成后的三项独立资格/历史核查

以下显式授权 flag **只在正式单元全部完成后使用**；它不是解除源码内完整性检查的开关。原F1与完整DSS管线资格还要求GPU空闲。它们不更改主预测、模型、epoch、lambda 或科学版本。

原F1：两次真实50更新无DSS Phase A，检查 held-out 协变量/隔离标签扰动不改变训练状态。新增2个神经拟合、100次更新，独立于480正式 fits：

```bash
"$PYTHON" scripts/semantic_b0b7_p2_leakage_audit.py run \
  --master-lock configs/semantic_b0b7/p2_lock.json \
  --qualification-lock configs/semantic_b0b7/p2_leakage_qualification_lock.json \
  --expected-qualification-sha256 7a6b324c9bb6a78733ac1dc566277404204cc25df309d497073cc239d45ccf85 \
  --assets "$ASSETS" --legacy-root "$LEGACY" --frozen-images "$FROZEN" --formal-private "$RUN" \
  --private-output "$AUX/phase_a_mutation_v1" --public-output "$AUX/F1_PHASE_A_QUALIFICATION.json" \
  --formal-complete-authorized
```

完整DSS管线资格：固定 outer1/MAIN/B5/seed31 的既有 final checkpoint 和冻结 lambda，仅置换held-out DSS `(time,event)` 对。两次重新拟合 CPU 教师、背景、PCA和风险头；共享一次冻结encoder提取，不进行任何神经优化。新增106 Cox、8 scalar、32个完整teacher preprocessing、6个risk PCA、8个多输出背景Ridge，单列软件资格，不替代原F1：

```bash
"$PYTHON" scripts/semantic_b0b7_p2_dss_isolation_audit.py \
  --master-lock configs/semantic_b0b7/p2_lock.json \
  --qualification-lock configs/semantic_b0b7/p2_dss_pipeline_isolation_lock.json \
  --expected-qualification-sha fa0f399a6315770df5089a07635a36e78a5a0304c8aca28765494819611f2e94 \
  --assets "$ASSETS" --legacy-root "$LEGACY" --frozen-images "$FROZEN" --formal-private "$RUN" \
  --private-output "$AUX/dss_pipeline_mutation_v1" --public-output "$AUX/F1_DSS_PIPELINE_QUALIFICATION.json" \
  --feature-device cuda --formal-complete-authorized
```

R4历史风险核查：只读取五折旧 `COMPLETE.json/models.pt` 与当前anchor，经哈希、来源和行序验证后比较seed31风险、beta及成员，并用统一DSS评分器重算；不新增训练/前向，不追调历史三seed总分。旧Torch blob可能同时解码已有PFI聚合结果字段，但不使用这些字段，也不读取原始PFI标签：

```bash
"$PYTHON" scripts/semantic_b0b7_p2_r4_history_audit.py \
  --master-lock configs/semantic_b0b7/p2_lock.json --expected-master-sha "$LOCK_SHA" \
  --assets "$ASSETS" --legacy-root "$LEGACY" --historical-root "$HISTORY" --formal-private "$RUN" \
  --output "$AUX/B0_HISTORICAL_SEED31_AUDIT.json" --formal-complete-authorized
```

资格失败或历史风险不精确时保留真实失败/差异收据并定位原因，不能修改数据、补调系数或删除反例使其“通过”。

## 5. 完整报告、独立审计与科学裁决

报告器读取正式已保存风险；预算旁路只读取训练收据；独立审计器重算保存线性头、完整风险分子/分母与官方API，不调用神经模型，也不冒称从原始H5重建全链路。先生成到独立新目录，再把经过审核的实际匿名文件纳入最终报告目录，保留00–03预运行事实文档及后续资格附录：

```bash
"$PYTHON" scripts/semantic_b0b7_p2_report.py \
  --private "$RUN" --lock configs/semantic_b0b7/p2_lock.json --expected-lock-sha256 "$LOCK_SHA" \
  --output "$REPORT" --resource-profile "$PROFILE" --expected-resource-profile-sha256 "$PROFILE_SHA" \
  --allow-completed-outer-results

"$PYTHON" scripts/semantic_b0b7_p2_training_budget_audit.py \
  --private "$RUN" --lock configs/semantic_b0b7/p2_lock.json --expected-lock-sha256 "$LOCK_SHA" \
  --output-newdir "$AUX/budget_report_v1" --allow-completed-formal-receipts

"$PYTHON" scripts/verify_semantic_b0b7_p2.py \
  --private "$RUN" --lock configs/semantic_b0b7/p2_lock.json --expected-lock-sha256 "$LOCK_SHA" \
  --cohort "$COHORT" --dss-endpoints "$DSS" --report-dir "$REPORT" \
  --sksurv-target "$OFFICIAL_PKGS" --output "$AUX/independent_audit_v1" \
  --allow-completed-outer-results
```

主指标必须先逐seed加总五折内部可比较对的分子/分母，再对三个seed等权平均；Harrell和Uno各用自己的分母，不跨折risk排序。不计算CI、bootstrap或p值。`06_paired_comparisons`保留12个核心比较，额外B0/B1参考与B5−B3位于独立supplementary表；逐fold×seed Δ、逐fold方向及两指标各自worst位置均为已有风险的派生诊断。08训练指标属于“在crossfit teacher offset之上的in-sample head拟合”，不是完整管线OOF性能。

`11_decision.json`中的运行完成状态不等同于科学机制资格。最终五类研究裁决记录于独立 `11_RESEARCH_DECISION.md/json`，应结合完整点估计、seed/fold方向、预算充分性、真实控制与独立核验；不能由单一总分自动选winner。负结果、NA和未完成比较须保留。

## 6. 只发布精确匿名文本包

公开候选清单必须人工审核且仅含本轮源码、配置、测试、原方案及匿名报告。每项是`file/bytes/sha256`，读取原UTF-8字节，不经文本模式改变CRLF。原始数据、患者/spot标识、风险数组、表达/特征、模型与优化器权重均不入包。清单不能由“递归添加整个RUN”替代。

```bash
META="<NEW_PUBLICATION_METADATA_DIRECTORY>"
CANDIDATES="<EXACT_REVIEWED_PUBLIC_CANDIDATES_JSON>"
PLAN="$META/P2_PLAN.json"
ARCHIVE="$META/P2_STAGE.zip"
P1R_MANIFEST="$PRESERVED/reports/semantic_b0b7_p1r/PUBLICATION_MANIFEST.json"
DENYLIST="<EXISTING_SERVER_PRIVATE_IDENTIFIER_REGISTRY>"
file_sha() { "$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }

"$PYTHON" scripts/semantic_b0b7_p2_publication.py plan \
  --root "$REPO" --preserved-root "$PRESERVED" --phase P2 \
  --base-commit "$PARENT" --prior-p1r-commit "$PARENT" \
  --prior-p1r-manifest "$P1R_MANIFEST" \
  --expected-prior-p1r-manifest-sha256 2e914a8326614843520e9fbae4cd6b5b6a13608ebcc9eb0dd6518d5efd16d598 \
  --candidates "$CANDIDATES" --plan "$PLAN"
PLAN_SHA="$(file_sha "$PLAN")"

"$PYTHON" scripts/semantic_b0b7_p2_publication.py stage \
  --root "$REPO" --preserved-root "$PRESERVED" --plan "$PLAN" --expected-plan-sha256 "$PLAN_SHA" \
  --prior-p1r-manifest "$P1R_MANIFEST" --archive "$ARCHIVE"
ARCHIVE_SHA="$(file_sha "$ARCHIVE")"

"$PYTHON" scripts/semantic_b0b7_p2_publication.py final-scan \
  --preserved-root "$PRESERVED" --prior-p1r-manifest "$P1R_MANIFEST" \
  --plan "$PLAN" --expected-plan-sha256 "$PLAN_SHA" \
  --archive "$ARCHIVE" --expected-archive-sha256 "$ARCHIVE_SHA" \
  --denylist "$DENYLIST" --output-newdir "$META/final_scan_v1"
```

`META`及其父目录需预先创建，但`final_scan_v1`必须不存在。清单JSON包装必须精确为`{"files":[...]}`；路径与字节由实际plan/stage校验。必须在持有原私有名单的服务器完成最后一步，本地无标识匹配pattern的预检不能替代它。原scanner SHA为 `4c70929d57459a9e97774e6319bd455deedc29fbcedf077131c918c53db60e98`；原名单SHA为 `d73b70939a38277f8f9f693e9958afcc9435893cd05f5c1d6a6d08a100ffe867`，名单本身不得导出。每文件2MiB、CSV含header≤20000行、JSON list≤1000、禁止大型纯数值数组等原规则不放宽；长曲线用正常CSV分片与完整索引，不用opaque编码规避规则。

这些命令不写Git。通过终检后，本轮使用已连接GitHub插件的实际操作序列为：`github_fetch_commit`读取父tree与现branch HEAD；对精确allowlist原字节逐一`github_create_blob`；`github_create_tree`基于完整P1R父tree只追加这些path；`github_create_commit`指定父提交；`github_update_ref`用已核对的expected HEAD更新 `research/semantic-b0b7-p2-dss`。如branch HEAD已前进，应先核对其变更，不强制覆盖。提交后逐路径核对远程blob SHA/size及原字节SHA256，确认旧79＋121及完整P1R105文件未变。最终commit、manifest和终检收据以实际交付记录为准；本文没有虚构最终P2 commit。

优势是源、输入、选择、实际拟合、评分、匿名发布均有独立可核对边界；限制是私有资产授权、封存字节、匹配运行环境和原私有标识名单不能由公开代码独立提供。无这些条件时只能验证公开代码与合成反例，不能声称完整数据复现。


## 实际执行的运行时并发安全调整

正式最终阶段启动后，6个409人源模型进程的实际NVML常驻显存为每进程5,136–5,138 MiB，高于原每进程5,200,000,000 B（约4,959 MiB）的准入预测。一轮实测空闲1,914 MiB，短暂低于2,048 MiB预留；没有观察到OOM或训练worker失败。

因此仅暂停调度器父进程的新任务准入，让当时6个worker全部自然完成（退出码均0、完整风险束齐），再退出旧父进程，用**原20份冻结源码、相同主锁和原全五折选择**执行同一final阶段，运行时上限由6降至5。没有给worker发中止信号，没有恢复部分优化器、改变轮数、重选超参或重训已完成模型。新调度器按原hash门禁复用14个完整最终源单元；历史内层360个拟合仍保持。

恢复命令对应上文schedule的同一输入参数，运行控制选项为 `--stage final --max-parallel 5 --cpu-parallel 2 --threads 2`。原始CLI允许1–6并发，此调整属于运行资源控制，不是模型修复或新的性能候选。旧父进程的SIGINT会由冻结调度器生成通用FINAL_FAILURE控制退出记录；原件保留，不能将其误计为模型拟合失败，也不能删除或隐藏。其他真正失败仍须独立处理。

实际证明见 `09_RESOURCE_ADMISSION_ADJUSTMENT.json`，其绑定主锁和未变化的全局选择SHA。新旧调度器日志与私有控制记录仅留在私有环境。CUDA peak-allocated profile与NVML的总常驻/缓存/上下文占用是不同统计量，本报告未声称显存预算精确匹配。


## 原R4运行依赖在公开父提交中的可获得性

已从公开父提交 `d2568ad4d57bb9c83366029cacee6baea6c3522d` 的真实Git对象逐字节核验：`src/bgb_en/{baselines,data,protocol,survival}.py` 与 `src/phase10_trimodal/{math,reference}.py` 六份代码均存在，并与实际P2锁定运行的legacy源码SHA256完全相同。证明见 `13_PUBLIC_PARENT_RUNTIME_SOURCE_AUDIT.json`（0模型拟合、0前向、0外测成绩读取）。因此legacy源码依赖本身不是未公开实现；在源码字节符合锁时，可将同一完整检出作为legacy-root。私有原始资产、冻结特征、患者结果和模型权重仍须按原授权在私有环境提供，不进入Git。

历史R4核查公开报告只保留学习系数β的一致性SHA与差值，β本身仍留私有；此执行前导出细化不改变任何风险、训练或比较，证据见 `13_HISTORY_AUDIT_PRIVACY_REFINEMENT.json`。


## 2026-10-31 正式完成后的实际状态补充

本文件上文保留编写当时的预运行事实，不把当时NOT_RUN追溯改写为PASS。当前480个正式源神经拟合、185个外折风险与实际独立核验均已完成；无DSS两次50更新的held-out协变量扰动、另106个Cox的完整DSS管线标签扰动均真实通过。详见 `00_FINAL_PROTOCOL_AND_QUALIFICATION_STATUS.json`、两份`03_LIVE_*`及`10_INDEPENDENT_REPLAY_AUDIT.json`。

完整曲线仍显示Phase-A预算上界活跃，故最终研究裁决为 `INCONCLUSIVE_DUE_TO_IMPLEMENTATION_OR_BUDGET`，具体是训练充分性未闭合；固定预算下B5未见DSS增量这一负向事实不变。最终解释见 `11_RESEARCH_DECISION.md` / `FINAL_RESULTS.md`，未计算CI或新增架构训练。原生成器42报告及其index按原字节保留。

完整DSS资格首次plain启动在输入哈希处停止，尚未加载模型或拟合。正式scheduler在解释器启动前设置`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8`；复现时必须在NumPy初始化前沿用这组数值环境。仅调用`torch.set_num_threads(1)`不能设置已初始化的NumPy BLAS。使用正式启动环境后原`84bf…`资格源码、原`fa0f…`子锁和输入哈希门禁均不变，CPU教师/头fit仍由原`threadpool_limits(limits=1)`执行。实际通过报告SHA为 `2bee33bf17cc91a6a4bb953baf985fc946a2191dce65e0313036708e180a0130`；原失败和环境诊断均保留，不将失败尝试计为额外fit。

新增匿名交付视图：`scripts/semantic_b0b7_p2_delivery_views.py --help` 描述接口；实际视图来源SHA见`DELIVERY_FILE_MAP.json`。最终科学解释生成器 `scripts/semantic_b0b7_p2_final_delivery.py`只读取已公开格式的实际报告，以追加方式写科学裁决、最终结果和独立证据索引，无私有数据或训练入口。原生成器`RESULTS.md`、`11_decision.json`及index不覆盖。

五折历史数值差异另有0新PCA/SVD/Cox/NN拟合的只读诊断，见`00_R4_PRECISION_LAYOUT_DIAGNOSTIC.md/json`及对应p2_r4_precision_audit脚本。完整源码与当前报告的最终字节由发布manifest统一绑定。测试记录分版本保留：原146测试套件、历史系数隐私细化13测试、只读精度诊断5测试和交付视图5测试，不冒称这些是同一版本的一次169测试运行。
