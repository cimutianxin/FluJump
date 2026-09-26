# 实验：interval 预测重构（对齐修复 + 4 类任务 + 规范化流程）

- **日期**：2026-08-02-02
- **目的**：修复 interval 实验的 split/embedding 位置对齐 bug；把 clamped 3 类标签重定义为 4 类（human_first 独立成类）；规范模型选择流程（val 选择、test 只评一次）；回归目标改造（log1p / hurdle）；补统计报告（bootstrap CI + 基线）

> **2026-08-02 下午更新**：当天项目审阅又发现 `build_isolate_dataset.py` 的 acc2cid key bug
> （267 行 isolate 继承了错误亚型的 cluster 标签），修复重跑数据后本实验全部重跑。
> 数值仅有小幅变化（分类 bal_acc 0.705→0.697，回归 Spearman 0.729→0.719，winner 回归变为 hurdle|L1L3），
> 结论不变。最终数值以 `notes/08-03-13-项目审阅总结.md` 为准。

## 方法

### Bug 修复（P0）
- 根因：`all_isolates_clean.csv`（11261 行，07-12 17:26 重建）与 `isolate_split.csv` / embedding（11060 行，07-10）行序不一致；旧脚本 `df["split"] = split_df["split"]` 按位置赋值、`emb[idx]` 按当前行号切片 → split 与特征双双错位，07-17 的 ordinal 等结果全部作废。
- 修复：新建共享 loader `interval_data.py`，split 按 **(accession, subtype)** inner merge（同一 accession 可跨亚型重复 264 个，单 accession 不是唯一键）；embedding 行序与 split CSV 行序逐行一致（已验证），据此建 (acc, subtype)→行号映射切片。
- 最终样本 662 条（663 − 1 条无 split）。

### 任务重定义（P1）
- clamp 发生在 `datascripts/build_isolate_dataset.py:71`，从 clean CSV 现有列按同公式重算**未 clamp** 的有符号天数。
- 新 4 类（有序，风险递减）：`human_first`（raw<0）417 / `<1yr` 129 / `1-3yr` 50 / `3yr+` 66；test 分布 57/23/6/12。
- 流程规范：train GridSearchCV → val 比 balanced_acc 选唯一 winner → train+val 重训 → **test 只评估一次** + bootstrap 95% CI（1000 次）+ majority / stratified random 基线。

## 结果

### 分类（662 条，test=98）— `eval_interval_clf_final.json`

val 选择（18 组合前 3）：

| 组合 | val bal_acc | val macro_f1 |
|------|------------|--------------|
| SVM_RBF\|L3 ★ | 0.786 | 0.683 |
| RidgeLR\|L1L3 | 0.749 | 0.628 |
| SVM_RBF\|L1 | 0.768 | 0.663 |

**TEST（SVM_RBF|L3, C=10, γ=0.001, train+val 重训）**：
- acc=0.643 [0.541, 0.745]，**bal_acc=0.705 [0.572, 0.815]**，macro_f1=0.621 [0.486, 0.736]
- 基线 bal_acc：majority 0.250 / random 0.250
- per-class recall：human_first 0.632 / <1yr 0.522 / 1-3yr 0.833 / 3yr+ 0.833
- H1: bal_acc 0.691 (n=88)；H3: 0.600 (n=10)
- OrdinalAT 全面落败（val bal_acc 0.45–0.61），不支持 class_weight 时小类全灭的问题用 sample_weight 也未救回。

### 回归（clamped days）— `eval_interval_reg_final.json`

**TEST（log1p|L3, α=10, train+val 重训）**：
- MAE=188 [109, 269]，RMSE=425，**Spearman ρ=0.729 [0.61, 0.81]**，Pearson r=0.542，R²=0.240
- 基线 MAE：mean 332 / median 242
- val 上 hurdle R² 最高（0.63）但 MAE 略高于 log1p（104.7 vs 97.4），按 MAE 选了 log1p。
- 对比修复前（错位数据）：R²=0.10、Pearson r=0.32、MAE≈baseline → 修复后信号大幅增强，说明旧结果确实被错位摧毁。

### JumpScorer（ranking）— `eval_jumpscore.json`

- 最佳 L3 h512×256：test **Spearman ρ=0.723**，Kendall τ=0.566，Pearson r=0.525（3 seeds）
- 与回归 log1p|L3 的 ρ=0.729 相当，排序信号一致。

### jump_exp 影响排查（只查未修）
- jump_exp 核心结果 `eval_results.json`（07-10 22:12）生成于 aligned CSV 重建（07-12 17:27）之前，**不受影响**。
- 近期脚本（layer_probing / loso_matrix / interventions / finetune_lora / extract_all_layers）用 split_df 或 accession 索引，安全。
- 若重跑 `train_probe.py` / `train_probe_compare.py` / `verify_h7_auc.py` / `diagnose_h7.py`：它们按位置取 `data_df["subtype"]`（11261 行），与 11060 长 mask 长度不匹配会直接报错——需要时按本次同法修复。

## 结论

- 对齐 bug 修复后 interval 任务信号显著：4 类 bal_acc 0.705（基线 0.25），回归 Spearman 0.73（修复前 0.32），**旧数值全部作废，以本次为准**。
- human_first 独立成类后语义清晰，且该模式（4 类中最高危类）recall 0.63 可学习。
- 注意：cluster 跨 split 泄漏按项目设定保留，绝对数值偏乐观，结论限定在该数据集构建方式内。
- **下一步行动**：继续 — 可将 SVM_RBF|L3（分类）与 log1p|L3（回归）固化进 mainpipeline；如需对外报告，先做 cluster 级敏感性分析。

## 涉及文件

| 文件 | 操作 |
|------|------|
| `ESM_clf/interval_exp/interval_data.py` | 新建 — 共享 loader（(acc,subtype) 对齐 + raw days 重算） |
| `ESM_clf/interval_exp/config.py` | 重写 — 4 类标签、删 USE_CLAMPED_DAYS、协议常量 |
| `ESM_clf/interval_exp/prepare_interval_data.py` | 重写 — 薄封装 |
| `ESM_clf/interval_exp/train_interval_clf_final.py` | 新建 — 统一分类（val 选择 + test 一次 + CI + 基线） |
| `ESM_clf/interval_exp/train_interval_reg_final.py` | 新建 — 统一回归（raw/log1p/hurdle） |
| `ESM_clf/interval_exp/train_jumpscore.py` | 修复 — 改用共享 loader |
| `ESM_clf/interval_exp/legacy/` | 归档 11 个作废脚本；`output/legacy/` 归档旧 JSON |
| `AGENTS.md` | env1 依赖 +mord/xgboost；4.3 更新对齐与标签注意事项 |
| `output/eval_interval_{clf,reg}_final.json`、`eval_jumpscore.json`、`interval_data_info.json` | 新结果 |
