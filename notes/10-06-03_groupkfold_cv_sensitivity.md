# 实验：Cluster-grouped 超参 CV 敏感性（S4，Major 12c）

- **日期**：10-06-03
- **目的**：probe ridge 超参选择原为 isolate 级 StratifiedKFold，与论文自查的 +0.11 近重复泄漏口径不一致；验证改 GroupKFold(5)（groups=(subtype,cluster_id)）后 headline 是否变化。

## 方法
- `ESM_clf/jump_exp/groupkfold_cv_sensitivity.py`：复制 train_probe/target_val_layer_select 全流程
  （数据、C 网格 1e-3…1e2、scoring、seed 42、target-val 五 seed cluster 臂选层），仅 cv 换 GroupKFold(5)。
- train 组数 1202 / 5933 行。基线读既有产物（g1_layer_geometry.json 逐层 best_C/AUC、target_val_layer_select.json）。

## 结果
- **选中 C 变化面广**：jump 19/31 层、jump_human 23/31 层的 best_C 改变（多向更小 C 移动，L17–19 例外向 10 移动）。
- **关键层全 holdout 原始 AUC**：L28 两标签完全不变（C 未变）；jump L17 H7 0.9246→0.9311；
  jump_human L13 H7 0.8632→0.6849、L17 H7 0.6277→0.4757（C 变小后中层 H7 正向幅度收缩）。
- **target-val headline 四格（cluster 臂 mean±SD）**：
  - jump H5：0.836±0.018 → 0.799±0.066（Δ−0.036；1/5 seed 改选 L11）
  - jump H7：0.918±0.023 → 0.919±0.026（Δ+0.002，选层不变 L17）
  - jump_human H5：0.688±0.103 → 0.681±0.133（Δ−0.007）
  - jump_human H7：**0.854±0.060 → 0.668±0.086（Δ−0.186）**；选层从 L13×5 漂为 L14/L22 混合

## 结论
- 结果**非"不变"**：3/4 格在噪声内稳定，但 jump_human H7 格对 CV 口径敏感（−0.186），且该格原本是
  CNN 不落下风之外 ESM 占优的关键格之一。论文 §Methods 2.2 末句需如实写：
  cluster-grouped CV 改变多数层的选中 C；四格中 jump H5/H7 与 jump_human H5 稳定（|Δ|≤0.036），
  jump_human H7 降至 0.668±0.086——该格结论需按敏感性口径限定（Supplementary Notes 全表）。
- 产物：`ESM_clf/jump_exp/output/groupkfold_cv_sensitivity.json`；RESULT.md §1 已追加。下一步：论文侧如实回填。
