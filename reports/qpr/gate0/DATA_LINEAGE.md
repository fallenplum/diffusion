# G0 原生空间证据资格

2026-11-02 实测：固定 512 名患者、519 张原生 CONCH H5 全部通过文件字节、SHA256、特征维度与整数坐标一一对应检查。共 1,278,642 个原生 patch、4,880,849 条不跨切片的无向八邻边；512/512 患者均有有效边，无无效切片。每切片最多4096条真实边，合计保留1,855,382条。完整参考严格指此边预算内的完整读取，不是全WSI全边真值。

原 H5 的 attrs 给出 patch_size=512、patch_size_level0=1024、level0_magnification=40、target_magnification=20、overlap=0。全部原生坐标与1024 level0像素格网兼容。使用元数据步长构边，不能把稀疏点的 GCD 自动当作更大邻接间距。未提供的 mpp 记为未知，未把像素解释为微米或细胞接触距离。

来源接口是 src/bgb_en/data.py 的原生H5读取、坐标排序、native_conch_catalog；新审计独立读取原 H5，而非 GRID512 缓存。TRIDENT 的 WSI 坐标导出接口与该元数据约定相符：[官方源码](https://github.com/mahmoodlab/TRIDENT/blob/main/trident/wsi_objects/WSI.py)。历史库存与实际 H5 的519个哈希逐一一致；匿名逐切片统计见 SPATIAL_COVERAGE.csv，私有路径/患者对应表留在服务器。

10张不同节点密度切片实际行置乱后，坐标、真实边、选边及特征对应均一致。软件另测试稀疏格网不能凭空生成邻边、平移和重排行为。几何资格不等于组织学核验；当前没有原始WSI图像与独立空间分子真值。

审计仅解码 cohort 的 patient_ids、folds；未打开 DSS 终点文件，未解码 DSS/PFI 列。65个 DSS 事件目前是历史终点合同，尚未在本轮重新解码计数；只有 G2 合法训练侧终点入口才能核验。禁止在 G1 用生存事件分层或读取结局。

既往 phase10_p0_spatial_niche 的拓扑差值为负，phase25_b 局部路由未通过主门，RSCIL 合成失败；这些背景不替代本轮 fresh B0，也不预设空间一定有收益。
