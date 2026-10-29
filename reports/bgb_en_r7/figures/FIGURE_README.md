# R7 普通科研诊断图

每个PNG/SVG配对来自同一个matplotlib figure；每图只有一个坐标轴，没有CI/p值、患者图像、热图或模型调用。

- interface_trajectory：只读旧R6的36个保存状态。每epoch先等权平均9个模型cell，再除以该series自己的epoch0均值；只有这张图使用log Y。所有原值和均值必须严格大于0，不加伪计数，不替换零值。训练探针重复使用，不是独立患者证据。
- stage_Harrell / stage_Uno60：固定B1、ADD、RAW、NORM、INDEP、SLOT、GATE1、GATE2顺序。3个点为已报告的每seed分区组件汇总值，黑菱形为3seed等权均值，不在图中跨分区拼风险。
- query_pairing_deltas：四个固定正确/错患者重训比较；Harrell圆点和Uno60方点分别表示各seed差值及其均值，不混成总体排名。
- interaction_risk_SD：11个模型在第80轮validation的cross SD / anchor SD，9个cell点和等权均值，linear坐标，保留ADD结构零；不称为variance explained。
- program_cancellation：7个matched模型的path/gene读出，直接使用已保存的病人级比值平均：mean_i[abs(sum_k contribution_ik) / sum_k abs(contribution_ik)]，只纳入分母非零的病例。图中P/G n/9是该统计可用的模型cell数；缺失为NA而非0。较高比值不自动代表更好预测。

`R7_FIGURE_BINDINGS.json`绑定严格35份匿名输入（接口audit、R7_RESULTS、33份诊断分片）的字节数/SHA、绘图器SHA与12张图文件的字节数/SHA。SVG为静态向量，禁止image或外部href。已有同名产物拒绝覆盖。本脚本不训练、不前向、不读取数组、标签或权重，也不选择模型、epoch或预算。
