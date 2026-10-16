# Phase18A：独立验证资源准备清单

日期：2026-10-16。状态：**元数据与来源准备，不是已经取得外部验证结果**。本轮没有在下列资源上拟合、选择配置或计算模型分数；不按外部队列上的效果挑资源。不能将浏览到的资源标题、登记病例数或可下载性当作已验证的三模态患者交集。

## 1. 不可混淆的独立性

- 当前固定 COAD/READ 512 人换 seed、换折、PFI 冻结转移，都不是独立外部验证。
- 同一 TCGA 病例的其他数据库镜像、重新提取的 patch 或不同数据发布版本，也不是独立患者。
- 其他 TCGA 癌种可以是跨癌种复现，不是同癌种外部临床验证；其基础模型预训练暴露仍需单独记录。
- CPTAC 曾在本项目 `phase6_conch_svs_availability_audit` 的 HF 特征可用性清单中出现。它没有因此成为已用的本轮验证集，但不能声称此前从未接触过其资源信息；需核对本地/服务器 manifest 和过往运行产物后才能确认患者级独立性。

## 2. 候选资源：按来源和模态匹配登记

|资源|实际核验的原始来源与原始模态|本轮未确定、必须先审计的项目|资格与下载边界|
|---|---|---|---|
|**Prospective CPTAC colon：TCIA CPTAC-COAD + PDC prospective colon**|[TCIA 资源页](https://www.cancerimagingarchive.net/collection/cptac-coad/)的索引公开登记 WSI/SVS、Clinical/Genomics/Proteomics，病例数 106、373 张图、66.69 GB；[TCIA 总目录](https://www.cancerimagingarchive.net/browse-data/)核对该条目。实际集合页直开本次返回内部抓取错误，不能称完整下载成功。[PDC000116](https://proteomic.datacommons.cancer.gov/pdc/study/PDC000116)登记 Prospective Colon PNNL study，102 个 cases，链接 imaging/genomics 与 SRA PRJNA514017；这是不同资源口径，不能直接取交集为 102 或 106|实际 WSI∩RNA∩CNA∩合法 Clinical∩DSS/PFI 的患者数未知；随访时间原点、死亡原因、复发/进展定义、删失与事件数未知；TCIA 和 PDC 具体病例交集、与训练 512 人去重结果未知；可复用预提取 backbone 特征和坐标兼容性未知|优先做来源与字段审计的同癌种候选，不预先认定能验证 DSS。TCIA 索引为 CC BY 4.0；关联组学、临床及上游测序按其 DUA/文件级权限分别审计。资源页总大小不是本项目实际需要下载量；本轮未下载大文件|
|**GSE17536（Moffitt）**|实际读取 [NCBI GEO 正式 series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17536)：177 个 array 样本，GPL570 表达平台，研究涉及复发与结肠癌死亡，series matrix 与 SOFT/原始 CEL 可取得|本次未提取患者级 endpoint 字典，故不能据论文叙述直接把字段命名为 DSS/PFI；精确 time/status 编码、事件数、同人多样本与其他 series 重复未知；未证实可逐患者取得 WSI 和 CNA|可准备为 RNA 表征/排序的独立敏感性资源，**不是已就绪的三模态 S6 外部验证集**。优先处理矩阵与 metadata，无需先取约 1.5 GB 的原始 CEL；RNA 平台转换须训练端冻结|
|**GSE39582**|[NCBI GEO 正式 series 入口](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39582)及 GEO 官方 sample 索引可定位；本次直接访问 series 被 reCAPTCHA 拦截，未取得完整字段表|病例数、去重、endpoint/time origin、WSI/CNA、合法 Clinical 交集全部待正式字段审计；不复制二手文献对“外部验证集”的称呼作为本项目资格|登记为后备 RNA/Clinical 来源；本轮不能将它标作完整核验或三模态可用。访问障碍如需人工处理再单独说明，不因资料未取得而补写病例交集|
|**预先另立协议的非 COAD/READ TCGA 癌种**|GDC 可检索 clinical、biospecimen、表达、CNA 和数字病理来源；官方 [开放/受控数据说明](https://gdc.cancer.gov/access-data/data-access-processes-and-tools)区分访问权限|本轮不根据癌种试验效果选癌种；所选癌种、patient manifest、DSS/PFI 定义/覆盖、RNA 平台/病理特征均尚未锁定|只能登记为跨癌种复现方向，不能替代独立 CRC 临床验证；不自动启动所有癌种大矩阵|

## 3. 在任何新队列建模前的资格锁定

按下面顺序形成匿名 `ELIGIBILITY_AUDIT.json`，资格判断不读取模型表现：

1. **来源与独立性**：固定 archive accession、下载时间、文件摘要、原始采集机构与招募阶段；在私有目录对病例标识、公开交叉引用以及可取得的样本/切片指纹去重。公开只发布计数和摘要，不发布交叉病例名单。不能用不同 ID 前缀代替真正去重。
2. **模态交集**：区分肿瘤/正常、原发/转移、手术/复发取样；每患者 RNA、CNA、WSI 与 Clinical 的交集与缺失表。预定义多切片、重复组学和多时间点规则；原发/复发取样不能无意变为基线预测泄露。
3. **终点同义性**：逐字段写明时间原点、单位、cause-specific death、other-cause death、删失、失访、进展、复发与治疗后状态。DSS 与 OS、DFI/RFS 与 PFI 不互换。若缺 DSS，登记“不支持原主要终点”，不能靠重命名补足。
4. **输入可执行性**：基因 ID/平台映射、表达单位、CNA 编码、临床类别、分期版本；病理 backbone、patch 尺度、坐标与提取流程。已冻结的外部推理预处理与允许的适配层单独登记。
5. **权限与预算**：逐资源审计 DUA、署名、CC 条款及受控文件授权。GDC 开放数据不需要登录不代表所有测序和临床字段都开放；[官方权限说明](https://gdc.cancer.gov/access-data/data-access-processes-and-tools)已核验。GEO 鼓励开放使用但提示提交者知识产权，不能推定所有来源无条件可再发布；见 [GEO 官方声明](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html)。
6. **一次性评测锁定**：先固定候选模型、主比较、必要模态、终点、缺失处理和是否允许校准，再解封外部 outcomes。探索性平台适配若使用外部 labels，应另设训练分区和真正未见测试分区，不叫零样本外部确认。

## 4. 当前资格结论

**尚无被本轮证实已就绪的、与当前 512 人独立且完整具备 Clinical/WSI/RNA/CNA/DSS 的外部队列。** CPTAC prospective colon 是来源较匹配的资格审计候选；GEO 两项主要用于 RNA 独立性敏感性，不以缺失 WSI 的结果替代本轮局部读出验证。

优势：按模态、终点与独立性先定资格，避免挑外部分数及 TCGA 镜像伪外部验证。风险：CPTAC 随访与 DSS 覆盖可能不足；array RNA 与 TCGA RNA-seq 存在平台偏移；原始 WSI 或受控文件需要额外提取与权限。资源清单的存在不能提高当前内部结果的证据等级，也不授权自动启动下一阶段。
