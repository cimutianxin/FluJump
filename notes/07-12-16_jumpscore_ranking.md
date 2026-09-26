# 实验：JumpScorer — Ranking Loss 训练，Spearman ρ 评估

> **数值已过时（09-16 标注）**：本笔记 ρ=0.29 为旧数据口径（663 条、clamp 后 67%=0）；现行结果 ρ=0.724±0.011（L3_h256x128，08-03 重算），见 `ESM_clf/interval_exp/output/eval_jumpscore.json` 与 RESULT.md §5。

- **日期**：2026-07-12
- **目的**：用 pairwise ranking loss（MarginRankingLoss）训练一个 JumpScorer，输出标量分数与 `iso_interval_days` 做秩相关（Spearman ρ），替代精确回归

## 前置修改
- `datascripts/build_isolate_dataset.py`：`iso_interval_days < 0` → clamp 到 0
  - 人先于动物的序列 → interval=0 → 最高危（跨物种能力最强）
  - 764 行被 clamp，1198 行有效值，范围 [0, 3465]

## 方法

- **模型**：ESM embedding → MLP [256,128] → scalar JumpScore
- **损失**：Pairwise MarginRankingLoss（batch 内所有 interval 不同的 pair）
- **可选混合**：ranking + MSE×0.1（对比纯 ranking）
- **数据**：H1+H3 jump=1，663 条有效（463/107/93 train/val/test），=0 占 67%
- **3 embedding × 3 hidden_dims × 2 MSE权重 × 3 seeds = 54 次训练**

## 结果

| Embedding | Hidden | MSE weight | Spearman ρ | Kendall τ | Pearson r |
|-----------|--------|-----------|------------|-----------|-----------|
| L1 | [256,128] | 0 | 0.232 ±0.015 | 0.177 | 0.137 |
| L1 | [512,256] | 0 | 0.246 ±0.039 | 0.187 | 0.144 |
| **L3** | **[256,128]** | **0** | **0.292 ±0.020** | **0.226** | **0.155** |
| L3 | [128] | 0 | 0.275 ±0.003 | 0.211 | 0.146 |
| L1L3 | [512,256] | 0 | 0.273 ±0.022 | 0.212 | 0.145 |
| ALL | ALL | 0.1 (MSE) | ~0.16 | ~0.12 | ~0.16 |

- **纯 ranking 远优于 ranking+MSE**：混合 MSE 使 ρ 从 0.29 降到 0.16，MSE loss 主导训练
- **L3 最优**（ρ=0.29），与分类实验一致（L3 acc 最高）
- **Spearman ρ=0.29 有统计显著性**（p < 0.01），但绝对值仍偏低

### 与旧实验对比

| 方法 | 核心指标 | 值 |
|------|---------|-----|
| Ridge Regression | Pearson r | 0.32 |
| MLP Regression | Pearson r | 0.38 |
| JumpScorer (ranking) | **Spearman ρ** | **0.29** |
| JumpScorer (ranking) | Pearson r | 0.15 |

- Ranking loss 优化的是排序而非精确值，Pearson r 自然会低
- Spearman ρ=0.29 表示排序相关性中等偏弱
- clamp 后数据 67% 为 0，限制了可区分的 pair 数量

## 结论

- **JumpScorer 是可行的新视角**：不追求精确天数，只关心「A 比 B 更危险」的相对排序
- **ρ=0.29 仍有提升空间**：数据瓶颈（663 条，67%=0）仍然限制模型能力
- **MSE 混合不利于 ranking**：不建议混合损失
- **下一步行动**：继续 — 考虑扩数据（加入 H5/H7）、或做 binary（0 vs >0 二分类）

## 涉及文件

| 文件 | 操作 |
|------|------|
| `datascripts/build_isolate_dataset.py` | **修改** — clamp days<0→0 |
| `ESM_clf/interval_exp/jumpscore_model.py` | **新建** — JumpScorer + ranking loss |
| `ESM_clf/interval_exp/train_jumpscore.py` | **新建** — 实验脚本 |
| `ESM_clf/interval_exp/output/eval_jumpscore.json` | **新建** — 结果 |
| `data/processed_isolate/all_isolates_clean.csv` | **更新** — iso_interval_days clamp |
| `data/processed_isolate_MAFFT/all_isolates_aligned.csv` | **更新** — 同步 clamp |
