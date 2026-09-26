# 实验：MLP vs Linear — Interval 预测（H1+H3 jump=1）

- **日期**：2026-07-12
- **目的**：在 ESM-2 150M embedding 基础上，用 MLP probe 替换 Linear probe（Ridge），对比 interval 分类和回归效果

## 方法

- **数据**：`all_isolates_clean.csv`（替换旧 `all_isolates_aligned.csv`，修复了 5 条缺失 `jump_interval_cat` 的样本）
- **筛选**：H1+H3，`label_is_jump=1`，`jump_interval_cat` 非空 → **729 条**（旧 724 条）
- **Embedding**：复用 `jump_exp/output/esm_emb_150M_{L1,L3,L1L3}.npy`（原始 `ha_sequence`，无 gap）
- **MLP 架构**：`Linear(in_dim, h1) → BatchNorm → ReLU → Dropout(0.3) → Linear(h1, h2) → ... → Linear(h_last, out_dim)`
  - 3 种 hidden dims: `[256,128]`, `[128]`, `[512,256]`
  - 3 seeds: 42, 123, 456
- **Linear baseline**：Ridge (L2 logistic regression) + GridSearchCV，与旧实验一致
- **训练**：Adam (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau, EarlyStopping (patience=20)
- **Split**：沿用 `isolate_split.csv`（489/113/97 train/val/test），37 cluster 跨 split

## 结果

### 分类（4 类：jump_interval_cat，729 条，test=97）

| Model | Test Acc | Balanced Acc | Macro F1 |
|-------|----------|-------------|----------|
| L1 (Ridge) | **0.670** | **0.579** | **0.599** |
| L1 MLP [256,128] | 0.663 ±0.013 | 0.511 ±0.019 | 0.512 ±0.026 |
| L3 (Ridge) | **0.711** | **0.597** | 0.469 |
| L3 MLP [256,128] | 0.660 ±0.022 | 0.515 ±0.026 | **0.516** ±0.033 |
| L1L3 (Ridge) | 0.629 | 0.510 | 0.393 |
| L1L3 MLP [512,256] | 0.656 ±0.027 | 0.494 ±0.036 | 0.481 ±0.051 |

- **Linear 在准确率上全面优于 MLP**（acc 优势 1-5pp）
- MLP 在 L3 的 macro F1 上略优（0.516 vs 0.469），但准确率低 5pp
- MLP 种子间波动较大（std 0.01-0.05），不如 Linear 稳定
- test 中无 5yr+ 样本（3 train, 2 val, 2 missing due to split）

### 回归（iso_interval_days，663 条有效，test=93）

| Model | R² | Pearson r | MAE | RMSE |
|-------|-----|-----------|-----|------|
| L1 (Ridge) | 0.095 | 0.308 | 559 | 803 |
| L1 MLP [512,256] | **0.130** ±0.008 | **0.387** ±0.016 | **513** ±4 | **788** ±4 |
| L3 (Ridge) | 0.089 | 0.300 | 548 | 806 |
| L3 MLP [512,256] | 0.095 ±0.008 | 0.325 ±0.005 | 524 ±4 | 803 ±3 |
| L1L3 (Ridge) | 0.102 | 0.320 | 559 | 800 |
| L1L3 MLP [512,256] | **0.130** ±0.008 | **0.387** ±0.016 | **513** ±4 | **788** ±4 |

- MLP 在回归上**略优于 Ridge**（R² +0.03, MAE -46, Pearson r +0.08）
- 但绝对效果仍然很弱（R²=0.13 意味着只能解释 13% 方差）
- L1 和 L1L3 的 MLP 结果完全相同（同一 best config），说明 MLP 从更宽 embedding 中未能提取更多信号
- 所有模型 pearson r < 0.4，预测能力有限

## 结论

- **MLP 未能显著超越 Linear**：分类上 Linear 更好，回归上 MLP 略优但差异不大
- **数据瓶颈是主要限制**：仅 729 条（663 回归有效），37 cluster，类别极度不均衡（5yr+ 仅 7 条）
- **MLP 种子不稳定性**：小数据集上 2-3 层 MLP 容易过拟合，需要更多正则化或更小模型
- **下一步行动**：调整 — 不继续深挖 MLP，建议回归到特征工程方向（如加入 cluster 特征、使用 ordinal regression）或扩大数据量

## 涉及文件

| 文件 | 操作 |
|------|------|
| `ESM_clf/interval_exp/config.py` | **修改** — 新增 MLP 参数，改 DATA_CSV 为 all_isolates_clean.csv |
| `ESM_clf/interval_exp/mlp_model.py` | **新建** — MLP 模型 + train/eval 工具 |
| `ESM_clf/interval_exp/train_interval_mlp_clf.py` | **新建** — MLP 分类实验 |
| `ESM_clf/interval_exp/train_interval_mlp_reg.py` | **新建** — MLP 回归实验 |
| `ESM_clf/interval_exp/compare_linear_mlp.py` | **新建** — Linear vs MLP 对比 |
| `ESM_clf/interval_exp/output/eval_interval_mlp_clf.json` | **新建** — MLP 分类结果 |
| `ESM_clf/interval_exp/output/eval_interval_mlp_reg.json` | **新建** — MLP 回归结果 |
| `ESM_clf/interval_exp/output/compare_linear_vs_mlp.json` | **新建** — 对比汇总 |
| `ESM_clf/interval_exp/output/labels_interval_*.npy` | **更新** — 重新生成（729 条） |
| `ESM_clf/interval_exp/output/eval_interval_clf.json` | **更新** — 重跑 Linear baseline |
| `ESM_clf/interval_exp/output/eval_interval_reg.json` | **更新** — 重跑 Linear baseline |
