# jev_exp — JEV 判别器实验

> 状态：首轮实验完成（2026-09-26）。**结论：未达预期** —— jev 头四格 cluster 臂 test AUC（0.530/0.671/0.622/0.645）全部低于 attention pooling 臂，更远低于 linear probe（0.836/0.918/0.688/0.854）；分布内 H1+H3 test 正常（0.96–0.98），问题在跨亚型迁移。详见 `notes/09-26-13_jev_class_query_head.md`。

## 1. 实验目标

- 使用 **jev 作为判别器（分类器）** 进行流感病毒跨物种传播（jump）预测实验。
- 在现有任务设定下评估 jev 分类器的表现，观察是否能取得比现有主线方法更好的结果。

## 2. 硬性约束

- **不改动现有 workflow 的任何代码**：本目录完全自包含，只读复用其他目录的资产（数据、embedding、划分文件等），所有新增/修改都限制在本目录内。
- 现有主线代码（`ESM_clf/`、`ESM_tf_clf/`、`mainpipeline/`、`Borkenhagen/`、`validation_exp/` 等）保持原样，不作任何变更。

## 3. 任务设定与对比基准

沿用项目核心评估设定，便于与主线方法直接对比：

- **任务**：`label_is_jump` / `label_is_jump_human` 二分类（后续可扩展到 interval 预测）。
- **核心设定**：H1+H3 训练 → H5/H7 holdout 亚型外推迁移。
- **指标**：AUC、ACC（多 seed 重复确认稳定性）。
- **主线基准**（详见 `RESULT.md` 与 `notes/`）：
  - ESM-2 150M 冻结 embedding（L3 层）+ Ridge Logistic Regression probe：H5 迁移 jump AUC ~0.79；
  - CNN baseline（Borkenhagen 2024 复现）：H5 迁移 jump AUC ~0.46。
- **评估口径注意**：
  - 一切 AUC/排序评估统一用 raw logits（禁用 sigmoid 概率，H7 上概率饱和会退化排序）；
  - H7 存在方向反转，需用 `-logits` 口径评估。

## 4. 算法方案（2026-09-26 确定）

设计来源：`jev_exp/算法实现考虑.md`（JEV 启发的"查询式概率分类头"，在冻结 ESM-2 之上替代 Linear/MLP 分类模块）。

```
x (B, L, 640)  →  Linear(640→256) 投影
2 个可学习类别查询 q₀/q₁ 经 cross-attention 并行读取残基表示 → z₀/z₁
2 个可学习类别原型 e₀/e₁，分数 s_c = cos(z_c, e_c)/τ（τ 可学习，CLIP 式 log 参数化，init 0.1）
logits = (s₀, s₁) → CrossEntropyLoss；训练后温度缩放校准（T 拟合于 H1+H3 val）
```

关键决策：

- **输入特征**：冻结 ESM-2 150M per-residue embedding（只读复用 `ESM_tf_clf/output/residue_emb_L{13,17,28}.npy`，行序经 accessions.npy 锚定）；
- **容量档位**：无 transformer encoder、无 FFN，与 attention pooling 臂相当（tf 头容量过大迁移反而退化，见 §3）；
- **训练协议**：Adam lr=1e-4 wd=1e-4、batch 16、≤100 epoch、early stop on H1+H3 val AUC、3 种子 logit 集成——逐项对齐 `ESM_tf_clf/train_tf_probe.py`；
- **公平对比协议**：target-val（h5/h7 val 选层 → test 只评一次，cluster + isolate_random 两臂 × 5 seeds），与 linear / attnpool / tf 三臂同协议并排；
- **评估口径**：AUC/排序一律 logit（s₁−s₀），增报 AUPRC、MCC（阈值=val Youden）；校准报 NLL/Brier/ECE(15 bin)；
- **已知限制**：类别查询不自动具备生物学含义，不做位点解读；单二分类下 JEV"并行结构化输出"额外价值有限，实验假设定位为"查询式读出+原型打分是否优于 mean-pool/attention-pool 读出"。

## 5. 目录结构

```
jev_exp/
├── README.md            # 本文件
├── 算法实现考虑.md       # 设计来源文档（用户提供）
├── config.py            # from ESM_tf_clf.config import * 复用 + jev 专有超参
├── model.py             # ClassQueryPrototypeHead（类别查询 cross-attn + 原型 cos 打分）
├── train_jev.py         # 训练 + target-val 评估（协议复刻 ESM_tf_clf/train_tf_probe.py）
├── calibrate.py         # 温度缩放校准 + NLL/Brier/ECE
├── compare_summary.py   # linear / attnpool / tf / jev 四臂并排对比
└── output/              # jev_results.json / jev_calibration.json / scores/ / models/
```

## 6. 环境

- 预计使用 conda 环境 `env1`（`/root/miniconda3/envs/env1`），从项目根目录 `/root/autodl-tmp/FluJump` 运行。
- 若涉及 ESM-2 模型加载，须加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

## 7. 实验记录

- 每次实验完成后须在 `notes/` 写实验日志（`MM-DD-HH_<描述>.md`）；
- 产出论文级结果后须更新 `RESULT.md`。
