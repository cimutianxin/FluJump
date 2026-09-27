# 实验：规模复现 — ESM-2 650M / 3B 上的层方向翻转与 target-val 选择

- **日期**：08-09-23
- **目的**：回应"只有 150M 单一规模"的审稿质疑——检验层深度方向翻转是否随模型规模复现。

## 方法

- `reviewer_exp/scale_replication/`：`extract_all_layers.py`（hf-mirror 下载 650M/3B，11,060 条 raw 序列全层 mean-pooled，fp16，余弦抽查=1.0000）；`run_scale_select.py`（复刻 target-val 协议：probe 训一次 → 10 切分文件 × 4 mask 评估、val argmax 选层 → test 评一次、cluster_seed42 bootstrap；同时输出全层 holdout 翻转曲线）。
- 输出：`output/scale_{650m,3b}_{select,sweep}.json`

## 结果

### target-val 选择（cluster 臂 5 seeds，H7 为重点）

| 规模 | 标签 | 选层 | test AUC | bootstrap P(>0.5)（seed42） |
|---|---|---|---|---|
| 150M（旧） | jump H7 | L17 ×5 | 0.892 ± 0.027 | 0.97 |
| 150M（旧） | jump_human H7 | L13 ×5 | 0.854 ± 0.060 | 1.00 |
| 650M | jump H7 | **L5 ×5** | 0.579 ± 0.080 | 0.93 |
| 650M | jump_human H7 | **L3 ×5** | 0.645 ± 0.039 | 0.91 |
| 3B | jump H7 | **L2 ×4** | 0.634 ± 0.057 | 0.98 |
| 3B | jump_human H7 | L1/L2 | 0.629 ± 0.028 | 0.90 |

### 方向翻转曲线（全量 H7 holdout AUC，argmax 层 vs 末层）

| 规模 | 标签 | argmax 层（相对深度） | 末层 AUC | 差 |
|---|---|---|---|---|
| 150M | jump | L17 0.893（0.55） | 0.251 | +0.64 |
| 650M | jump | L5 0.624（0.15） | 0.163 | +0.46 |
| 3B | jump | L2 0.708（0.06） | 0.305 | +0.40 |
| 150M | jump_human | L13 0.863（0.42） | 0.121 | +0.74 |
| 650M | jump_human | L3 0.664（0.09） | 0.212 | +0.45 |
| 3B | jump_human | L2 0.670（0.06） | 0.275 | +0.40 |

## 结论

1. **方向翻转定性复现**：三个规模都存在"存在某个非末层方向正确（AUC 0.6–0.9）+ 末层系统性反转（AUC 0.1–0.3）"的结构——"末层反转"是普适现象，不是 150M 的偶然。
2. **最优相对深度随规模前移**：0.55（150M）→ 0.15（650M）→ 0.06（3B）。论文表述需从"中层现象"修正为"非末层现象，最优深度随规模前移"——这本身是一个新的可报道规律。
3. **规模放大无迁移收益**：峰值 AUC 0.89→0.62/0.71（jump H7），150M 反而是最佳规模；650M 中后层反转更极端（jump_human 0.08–0.18）。回应"为何选 150M"：不是算力限制，是实证最优。
4. 650M/3B 的 val 选层同样稳定（L5/L3 5/5、L2 4/5），target-val 协议在其他规模上行为一致。
5. 下一步行动：**继续**——论文新增"规模-深度"分析小节（三规模翻转曲线图）；主线模型维持 150M。

## 涉及文件

- 新建 `reviewer_exp/scale_replication/`（config / extract_all_layers / run_scale_select）
- 输出 `reviewer_exp/scale_replication/output/`（esm_emb_{650m,3b}_all_layers.npy、scale_{650m,3b}_{select,sweep}.json、extract.log、scale_select.log）

## 追加（09-27）：09-19 标签修复后重跑版本

09-19 host 标签修复后本实验被重跑（select/sweep JSON mtime 09-19 13:53/14:11）。修复后 cluster 臂 H7：650M jump **0.548±0.091**（L5×5，seed42 bootstrap P(>0.5)=0.82）、3B jump **0.624±0.056**（L2×4/L1×1，P=0.96）；jump_human 两格不变（0.645±0.039 / 0.629±0.028，P=0.91/0.90）；150M 修复后数值见 `08-09-00_target_val_layer_select.md` 追加节。选层结论两版本一致（L5/L3/L2 稳定）；末层 sweep 原始 AUC（修复后）：650M L33 0.149/0.212、3B L36 0.294/0.275。RESULT.md §4 表已于 09-27 刷新为修复后数值（协议审计 `09-27-01_h7_flip_protocol_audit.md`）。
