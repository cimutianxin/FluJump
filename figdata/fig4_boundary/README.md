# figdata/fig4_boundary — 失效边界 = H7/H10 亚支（Results 3.3，Figure 4）

- `panelb_geometry_points.csv`（既有，10-04 服务器导出）：11,060 株逐株 PC1/PC2/proj_w
  （L28、jump_human、raw logit 口径；meta 的 dev_note：旧 geometry_analysis.json 有
  ≤0.1 标签位移偏差，以本 csv 为准）。
- `panelb_unit_transform.json`（本脚本派生）：raw logit → 单位 w 投影（rel. train 质心）
  的线性映射，由 meta 的 8 个校验均值解出（max|残差|<2e-6）。caption 的 −5.2/−3.3 与
  "negative half-space" 以变换后/原始 logit 两个口径分别核对（H7 负半空间比例 0.994）。
- `panelc_profiles.csv`：H10/H4 × 双标签 × 31 层逐层 isolate/cluster AUC
  （方向检验；符号检验口径 = jump 的 cluster_auc，已复算核对 24/31、3/31）。
- `panelc_preregistered.csv`：预注册层结果（jump 17/28；jump_human 13/28；含 bootstrap
  CI 与 P(AUC>0.5)）。H4 无 jump_human 阳性簇，其 jump_human cluster AUC 未定义（全 null）。
- `panelc_signtest_rank.json`：符号检验计数/p、人源 H10N8 簇（H10_197，江西 2013）
  rank 313→9/321（jump_human L13 翻转）、H10/H4 评估规模。
- `paneld_columns.csv`：逐位点驱动散点（jump L28 格，与结构图同格；region/共享核心标记
  来自 structure_colors.csv，d_pos 与 g3 输出逐列一致）。
- `paneld_overlap_summary.csv`：6 个 layer×label 格的 ρ(H7,H10)/p/top-20 Jaccard 及
  H4/H5 对照（H4 无 jump_human 阳性 → 对照仅 jump 三格）。
- `panele_summary.json` + `structure_colors.csv`（既有）+ `assets/panele_1hgg_*_v2.png`
  （既有位图）：共享核心 17 列（0 落 RBS）、JS 百分位、亚型身份对照探针。

由 figs/fig4_boundary/prepare_figdata.py 生成（本地重打包 + 断言，无新实验）。
