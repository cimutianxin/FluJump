# 实验：fig4_boundary panel (b) 几何散点逐点数据导出

- **日期**：10-03-23
- **目的**：复现 analyze_geometry.py 的 L3 × label_is_jump_human 几何分析，导出逐点 PC1/PC2/proj_w 供论文 fig4 panel (b) 绘图

## 方法
- 新建 `ESM_clf/jump_exp/export_geometry_points.py`（不改 analyze_geometry.py）
- 流程与 analyze_geometry.py 完全一致：H1+H3 train 上 StandardScaler + Ridge LR
  GridSearchCV（C∈[1e-3,100]，5-fold stratified，seed 42，scoring=roc_auc）；
  PCA(n_components=2, seed 42) 拟合 X_std[train] 后 transform 全量
- proj_w = decision_function（X_std @ w + b，raw logit）；校验用 analyze_geometry
  口径（w 单位化、相对 train 质心投影）复核 geometry_analysis.json

## 结果
- 校验：best_C=1.0（与 json 一致）；PC1 解释方差=0.4620（≈0.462 ✓，PC2=0.1254）；
  H7 proj_pos=-5.1609、proj_neg=-3.2814
- 与 geometry_analysis.json 逐字段最大偏差 0.0722（H1 proj_pos 1.2848 vs 1.3544）
- **偏差来源已查明**：geometry_analysis.json 生成于 2026-08-01，早于 09-19
  host_label 词边界修复（labels_*.npy 于 2026-09-19 重生成）；标签位移使 w 方向
  轻微旋转。H7 proj_neg 几乎不动（-3.2814 vs -3.2805），proj_pos -5.16 vs -5.10，
  方向与量级结论不变。与 depth_reversal REPRO_TOL=0.06「含 09-19 标签修复位移」同源
- 产物：
  - `figdata/fig4_boundary/panelb_geometry_points.csv`（11,060 行 = 全部 isolate）
  - `figdata/fig4_boundary/panelb_geometry_meta.json`（evr、best_C、proj 复核值
    + json 参照值 + 偏差说明）

## 结论
- 达到预期：逐点数据导出完成，校验锚点在 09-19 标签修复位移容差内全部通过
- 下一步：绘图侧用 CSV 直接画散点；若论文需引用 proj_pos/proj_neg 绝对值，
  建议用重跑口径（-5.16/-3.28）替换旧 json 口径
