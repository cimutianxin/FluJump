# Borkenhagen CNN Baseline + ESM-2 Binding Probing 实验记录

> 2026-07-10 | env: borkenhagen (CNN), env1 (ESM-2)

---

## 一、CNN Baseline（Borkenhagen 2024 复现）

### 目录

```
Borkenhagen/cnn_baseline/
├── config.py          # 超参配置
├── data_prep.py       # one-hot 编码 + 数据加载
├── model.py           # 5层 Conv1D (1.7M 参数)
├── train_stage1.py    # Stage 1: host 预训练
├── train_stage2.py    # Stage 2: jump 微调
├── evaluate.py        # 评估 + per-subtype + holdout
└── output/
    ├── models/        # stage1_best.pt, stage2_*_best.pt
    ├── eval_results.json
    └── roc_*.png
```

### 数据

- **输入**: `data/processed_isolate_MAFFT/all_isolates_aligned.csv` (11,261 isolates, 581-aligned)
- **划分**: isolate 级 70/15/15, H1+H3 only, H5/H7 全量 holdout
- **划分文件**: `data/splits/isolate_split.csv`

### 最终超参（修复后）

| 参数 | 值 | 说明 |
|---|---|---|
| S1_LR | 1e-3 | Stage 1 学习率 |
| S2_LR | **1e-3** | 从 0.01 降到 0.001，防止梯度振荡 |
| clip_grad_norm | **1.0** | 防止高样本权重导致的梯度爆炸 |
| class weight | subtype + class 双重加权 | Borkenhagen 原文方法 |

### 结果

| 任务 | Test AUC | Test Acc | Test F1 | 备注 |
|---|---|---|---|---|
| Stage 1 (host) | 1.000 | 1.000 | — | avain vs human+swine |
| `label_is_jump` | **0.992** | 0.969 | 0.837 | H1+H3, 70/15/15 split |
| `label_is_jump_human` | **0.990** | 0.974 | 0.851 | H1+H3, 70/15/15 split |

| Holdout | jump AUC | jump_human AUC |
|---|---|---|
| H5 (1,757) | 0.541 | 0.550 |
| H7 (838) | 0.662 | 0.724 |

### 关键发现

1. **LR=0.01 导致 jump 任务坍缩**（AUC=0.50）。根因: 41 条 avian jump=1 的正例与 Stage 1 的 "avian=label0" 信号冲突，高 LR × 高权重 → 梯度振荡
2. **LR=0.001 + grad_clip=1.0 修复**：jump AUC 从 0.50 → 0.99
3. **跨亚型泛化有限**：H5/H7 holdout AUC 仅 0.5-0.7
4. **`label_is_jump` 学不到的原因非数据质量问题**：序列多样性高（31 unique/41，9 clusters），是优化问题

---

## 二、ESM-2 Binding Probing（外部验证）

### 目录

```
ESM_clf/binding_exp/
├── config.py              # 配置（模型选择、超参）
├── extract_embeddings.py  # ESM-2 推理提取 embedding
├── train_probe.py         # Ridge LR + MLP 训练
└── output/
    ├── esm_emb_8M_L1.npy
    ├── esm_emb_150M_esmmodel_L1.npy
    ├── esm_emb_150M_esmmodel_L3.npy
    ├── esm_emb_150M_esmmodel_L1L3.npy
    ├── probe_results_8M.json
    └── probe_final.json
```

### 数据

- **输入**: `data/dataset_borkenhagen/borkenhagen_clean.csv` (402 条, H1-H16)
- **标签**: `binding_label` (0=α2,3, 1=α2,6)
- **划分**: 358 train + 44 test（stratified by subtype+binding_label）
- **序列**: 原始序列去 gap → ESM-2 tokenizer
- **模型下载**: 150M 通过 `hf-mirror.com` 镜像下载（HF xet 有 401 错误）

### 方法

| 方法 | 输入 | 分类器 | 说明 |
|---|---|---|---|
| CNN (Borkenhagen 原文) | one-hot (581×21) | 5-layer Conv1D | 原文结果，直接引用 |
| ESM-2 8M probe | 320-dim embedding (L1, mean pool) | Ridge LR | 冻结 ESM-2 |
| ESM-2 150M probe | 640-dim embedding (L1/L3, mean pool) | Ridge LR | 冻结 ESM-2, EsmModel |

### 结果

| 方法 | 维度 | AUC | Acc |
|---|---|---|---|
| Borkenhagen CNN (ref) | 581×21 | 0.930 | 0.940 |
| **ESM-2 8M + Ridge LR** | 320 | **0.958** | 0.886 |
| **ESM-2 150M + Ridge LR** | 640 | 0.951 | 0.932 |
| ESM-2 150M L3 + Ridge LR | 640 | 0.951 | 0.909 |
| ESM-2 150M L1+L3 + Ridge LR | 1280 | 0.948 | 0.909 |

### 关键发现

1. **ESM-2 小模型超越特化 CNN**：8M 参数 + 线性分类（AUC 0.958）> 1.7M 5层卷积（AUC 0.930）
2. **线性 probe 即可**：MLP 无增益（甚至下降），说明 ESM-2 学到的 HA 特征中 binding 信息是线性可分的
3. **8M > 150M**：小维度在小样本上泛化更好（Ridge 正则的经典现象）
4. **多层无增益**：L1、L3 相同，L1+L3 concat 因维度翻倍而轻微过拟合

### 生物学意义

- ESM-2 的无监督预训练隐式学到了 HA 受体结合特征
- 跳跃预测 vs 结合偏好的区别符合多基因适应假说（HA 结合是必要条件非充分条件）
- 该实验作为外部 probing，验证了 PLM 表征的通用性

---

## 三、数据划分（全局统一）

### 文件

```
data/splits/
├── isolate_split.csv      # accession,subtype,cluster_id,split
└── cluster_split.json
```

### 划分策略

- **H1+H3**: isolate 级 70/15/15 分层（stratify by `(subtype, label_is_jump_human)`）
- **H5/H7**: 全量 holdout（跨亚型泛化验证）
- **划分脚本**: `datascripts/build_splits.py`

### 分布

| Split | H1 | H3 | 总计 |
|---|---|---|---|
| train | 2,848 | 3,218 | 6,066 |
| val | 602 | 698 | 1,300 |
| test | 619 | 681 | 1,300 |
| h5_holdout | — | — | 1,757 |
| h7_holdout | — | — | 838 |

---

## 四、环境

| 环境 | 用途 | 主要依赖 |
|---|---|---|
| `borkenhagen` | CNN baseline | PyTorch 2.5.1, sklearn, pandas |
| `env1` | ESM-2 probing | PyTorch 2.11, transformers 5.13 |

---

## 五、论文叙事建议

1. **CNN baseline**：两阶段迁移学习迁移到 jump 预测，AUC 0.99，作为 PLM 对照组
2. **ESM-2 probing**：冻结 PLM + 线性 probe 在 binding 上超越特化 CNN（0.958 vs 0.93），证明无监督预训练学到了更通用的 HA 特征
3. **生物学洞察**：跳跃 ≠ 结合偏好，符合多基因适应假说
