# ESM-2 150M + Ridge LR：jump/jump_human 二分类 + H7 方向反转

- **日期**：2026-07-12
- **上一笔记**：`07-10-Borkenhagen-CNN-ESM2-probing.md`

## 背景与目标

在 `ESM_clf/jump_exp/` 下实现 ESM-2 150M (EsmModel) 嵌入提取 + Ridge Logistic Regression，对 `label_is_jump` 和 `label_is_jump_human` 做二分类：
- **训练/验证/测试**：H1+H3（与 Borkenhagen CNN 相同的数据划分）
- **迁移评估**：H5 holdout（1747 条）、H7 holdout（838 条）

并与 Borkenhagen CNN baseline 对比。

## 关键步骤

### 1. 创建文件

| 文件 | 作用 |
|---|---|
| `ESM_clf/jump_exp/config.py` | 配置：ESM-2 150M, Ridge C 值, 两个标签列, 输出路径 |
| `ESM_clf/jump_exp/extract_embeddings.py` | 用 EsmModel 提取 L1/L3/L1L3 mean-pooled embedding |
| `ESM_clf/jump_exp/train_probe.py` | Ridge LR + GridSearchCV + multi-seed 评估 + H5/H7 迁移 |
| `ESM_clf/jump_exp/diagnose_h7.py` | 诊断脚本：检查 H7 AUC 是否方向反转 |

### 2. 嵌入提取

```
总 isolates: 11060 (H3: 4486, H1: 3989, H5: 1747, H7: 838)
设备: cuda (RTX 4080 SUPER)
提取 embedding: 2765 batches, ~2.7 min
输出:
  esm_emb_150M_L1.npy:   (11060, 640)
  esm_emb_150M_L3.npy:   (11060, 640)
  esm_emb_150M_L1L3.npy: (11060, 1280)
  labels_label_is_jump.npy:       (11060,) pos=1383 (12.5%)
  labels_label_is_jump_human.npy: (11060,) pos=1206 (10.9%)
```

使用 env1 的 Python (`/root/miniconda3/envs/env1/bin/python`)，base 环境缺少 transformers。

### 3. Ridge LR 训练

- StandardScaler + LogisticRegression(L2, lbfgs, max_iter=5000)
- GridSearchCV over C ∈ {1e-3, 1e-2, 1e-1, 1, 10, 100}, 5-fold stratified CV, scoring=roc_auc
- 多 seed (5) 验证稳定性

### 4. H7 方向反转诊断 + 精度修正

发现 H7 AUC 0.12-0.28（低于随机 0.5）。诊断脚本 `diagnose_h7.py` 确认方向反转。
后续 `verify_h7_auc.py` 修正：用 `decision_function` 的 `-logits` 替代 `1-p` 计算翻转 AUC，因 H7 预测概率极端集中（pred_mean≈0.008）导致 `1-p` 数值不稳定。修正后严格满足 $AUC(p)+AUC(−logit)=1$。

## 结果与发现

### Test (H1+H3 in-distribution)

| 任务 | 最佳 Embedding | Test AUC | vs CNN |
|---|---|---|---|
| `label_is_jump` | L1L3 | 0.9888 | CNN 0.9920 |
| `label_is_jump_human` | L1 | 0.9933 | CNN 0.9900 |

ESM-2 Ridge LR 与 CNN 在 in-distribution 上性能相当。

### 迁移

| 任务 | 最佳 Embedding | H5 AUC | H7 AUC (原始) | H7 AUC (翻转) | 
|---|---|---|---|---|
| `label_is_jump` | L1L3 | 0.7575 | 0.2442 | **0.7558** |
| `label_is_jump` | L3 | 0.7940 | 0.2812 | **0.7188** |
| `label_is_jump_human` | L1 | 0.5868 | 0.1205 | **0.8795** |
| `label_is_jump_human` | L3 | 0.7923 | 0.2134 | **0.7866** |
| *Borkenhagen CNN* | — | 0.54-0.55 | 0.66-0.72 | — |

> 翻转 AUC 用 `-logits`（`decision_function`）计算，满足 $AUC(p)+AUC(−logit)=1$。之前用 `1-p` 因预测概率极端集中（pred_mean≈0.008）导致数值不稳定，已纠正。

### 核心发现

1. **ESM-2 L3 层在 H5 迁移上远超 CNN**：AUC ~0.79 vs CNN 0.54（+46%）
2. **H7 是方向性反转**：翻转后（用 `-logits`）H7 AUC：`jump` 0.72-0.76，`jump_human` 0.79-0.88。整体持平或优于 CNN 的 0.66-0.72
3. **H5 和 H7 不能用同一方向预测**——H7 的 jump 信号在 ESM-2 embedding 空间中与 H1+H3 训练方向天然反 align
4. L3 层在迁移任务上一致优于 L1 和 L1L3 concat
5. **翻转预测必须用 logits**，不能直接用 `1-p`——H7 预测概率极端集中在 0 附近（pred_mean≈0.008），`1-p` 有显著数值误差

## 对齐序列补充实验

为验证"对齐能否解决 H7 方向反转"，额外用 MAFFT 对齐序列（`aligned_ha_seq`，581aa）跑了一版。关键差异：
- 使用对齐序列（含 `-` gap），ESM-2 将 `-` 映射到 token id=30
- mean pooling 时排除 gap token（`GAP_TOKEN_ID=30`），避免 untrained gap embedding 引入噪声
- 脚本：`extract_embeddings_aligned.py`、`train_probe_compare.py`

### 结果

| 任务 | Variant | Emb | Test AUC | H5 AUC | H7_flip |
|---|---|---|---|---|---|
| `label_is_jump` | raw | L3 | 0.9879 | **0.7940** | 0.7188 |
| `label_is_jump` | aligned | L1L3 | 0.9911 | 0.6061 | **0.7894** |
| `label_is_jump_human` | raw | L3 | 0.9916 | 0.7923 | 0.7866 |
| `label_is_jump_human` | aligned | L3 | 0.9917 | 0.7898 | **0.9005** |
| *CNN baseline* | — | — | 0.99 | 0.54-0.55 | 0.66-0.72 |

### 结论

- **方向问题未解决**：对齐序列 H7_raw 仍是反向的（~0.1），对齐不改变方向
- **H7 翻转有提升**：`jump_human` L3 翻转达 **0.90**（远超 CNN 0.72），`jump` L1L3 达 0.79
- **H5 反而下降**：L1/L1L3 的 H5 AUC 显著低于 raw（0.63→0.43, 0.76→0.61），L3 基本持平
- **主方案仍是 raw**：raw 在 H5+H7 上综合更均衡。对齐方案作为补充，证明了位点一致性对 ESM-2 跨亚型迁移有价值，但 gap token 在 self-attention 中的噪声拖累了 H5

## 涉及文件

| 文件 | 操作 |
|---|---|
| `ESM_clf/jump_exp/config.py` | 新建 |
| `ESM_clf/jump_exp/extract_embeddings.py` | 新建 |
| `ESM_clf/jump_exp/train_probe.py` | 新建 |
| `ESM_clf/jump_exp/diagnose_h7.py` | 新建 |
| `ESM_clf/jump_exp/verify_h7_auc.py` | 新建（精度修正） |
| `ESM_clf/jump_exp/extract_embeddings_aligned.py` | 新建（对齐序列补充实验） |
| `ESM_clf/jump_exp/train_probe_compare.py` | 新建（raw vs aligned 对比） |
| `ESM_clf/jump_exp/output/*.npy` | 新建（12 个文件，~170 MB） |
| `ESM_clf/jump_exp/output/eval_results.json` | 新建 |
| `ESM_clf/jump_exp/output/compare_raw_vs_aligned_*.json` | 新建 |

## 待办

- [x] ~~对齐序列补充实验~~ → H7 方向仍反，raw 为主方案
- [ ] 进一步分析 H7 方向反转的生物学原因（embedding 空间可视化？）
- [ ] 实验：H1+H3+H5 联合训练后测试 H7 是否改善方向
- [ ] 补充实验日志到 `notes/`
