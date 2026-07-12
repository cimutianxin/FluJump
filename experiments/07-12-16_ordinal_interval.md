# 实验：Ordinal Regression — Interval 预测

- **日期**：2026-07-12
- **目的**：利用 interval 类别的有序性（<1yr < 1-3yr < 3-5yr < 5yr+），用 ordinal regression 替代 multinomial 分类，对比效果

## 方法

- **数据**：H1+H3 jump=1，729 条，split 不变
- **Embedding**：复用 jump_exp L1/L3/L1L3
- **Linear Ordinal**：`mord.LogisticAT`（proportional odds model, L2 正则化） + GridSearchCV(alpha)
- **MLP Ordinal**：自实现 `OrdinalMLP` — 单 score 输出 + K-1 个可学习单调阈值 + NLL loss（Adam/lr=1e-3/patience=20）
- **对比基线**：multinomial Ridge（`eval_interval_clf.json`）

## 结果

### Linear Ordinal vs Multinomial

| Embedding | Model | Test Acc | Balanced Acc | Macro F1 |
|-----------|-------|----------|-------------|----------|
| L1 | Multinomial (Ridge) | **0.670** | **0.579** | **0.599** |
| L1 | Ordinal (mord) | 0.639 | 0.504 | 0.512 |
| L3 | Multinomial (Ridge) | **0.711** | **0.597** | 0.469 |
| L3 | Ordinal (mord) | 0.608 | 0.453 | 0.448 |
| L1L3 | Multinomial (Ridge) | 0.629 | 0.510 | 0.393 |
| L1L3 | Ordinal (mord) | 0.619 | 0.475 | **0.478** |

- Ordinal 在所有指标上**不如** multinomial（acc 低 1-10pp，bal_acc 低 3-10pp）
- L1L3 的 macro F1 略优（0.478 vs 0.393），但准确率更低

### MLP Ordinal

| Embedding | Hidden | Test Acc | Balanced Acc | Macro F1 |
|-----------|--------|----------|-------------|----------|
| ALL | ALL | **0.546** | **0.333** | ~0.18 |

- **完全坍缩**：所有配置、所有种子均预测 majority class（<1yr = 54.6%）
- 单 score 瓶颈（640→1 维）导致模型无法区分类别
- 学到的阈值合理（~[0.7, 1.5, 2.3]），但 score 无区分度

## 结论

- **Ordinal regression 未改善 interval 预测**：Multinomial 在所有指标上优于 mord，OrdinalMLP 完全失败
- **原因分析**：
  1. 类别严重不均衡（<1yr 占 59%，5yr+ 仅 1%），ordinal 假设可能不成立
  2. 单 score → 阈值架构对 640 维输入而言过于压缩
  3. 小样本下 proportional odds 约束可能过于严格
- **下一步行动**：**放弃 ordinal 方向**，转向 binary target（short vs long）或特征工程

## 涉及文件

| 文件 | 操作 |
|------|------|
| `ESM_clf/interval_exp/ordinal_model.py` | **新建** — OrdinalMLP 模型 |
| `ESM_clf/interval_exp/train_interval_ordinal.py` | **新建** — Linear + MLP ordinal 实验 |
| `ESM_clf/interval_exp/output/eval_interval_ordinal.json` | **新建** — 结果 |
