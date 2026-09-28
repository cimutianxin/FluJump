# 实验：多年份前向复现（2024/2025）——TODO-B7 升档关键②

- **日期**：09-26-14
- **目的**：把 §5 前向排序验证从"2026 单年 2 簇"扩展为三年（2024/2025/2026）可复现规律；检验 probe 排序在新年份（含 6 个 2024 首跳簇，含 (H5,70) 美国奶牛/人暴发簇）是否稳定优于朴素基线

## 方法

- 新脚本 `validation_exp/temporal_validation/run_multiyear_forward.py`（自包含，不改现有文件；CPU 全程，embedding/对齐序列现成）。
- 协议复刻 2026 前向 + 加固，**关键新增**：训练标签按 ≤cutoff 簇成员重算（现行标签由 ≤2025 全量数据算出，对 2024/2025-cutoff 构成未来信息泄漏；口径同 build_jump_labels，簇键 (subtype, cluster_id)）。
- 每年：train = split==train 且 year≤cutoff（2024: 3,426 条 / 2025: 5,262 条）→ Ridge LR probe（同 2026 超参）→ 评 year==Y 全亚型 isolate（2024: 2,677 条/136 簇，7 阳性簇；2025: 1,020 条/201 簇，8 阳性簇）。
- 指标：isolate/cluster AUC、cluster bootstrap CI + P(>0.5)（B=10,000）、cluster permutation p（B=10,000）、双口径标签（headline=全量 ever-jump / 因果=≤cutoff）、首跳簇名次、L1 末层对照、分层命中（阈值仅 ≤cutoff train 定）、朴素基线（max identity to ≤cutoff train 人源参考）。
- H5 sanity 改为同口径（≤cutoff 标签）断言且仅 L3 硬断言（>0.5 方向性门槛）：jump_human H5 是已知最弱格，L1 为对照层其失效本身即证据。

## 结果

**L3 主线（cluster 级 AUC，headline 口径）：**

| 年份 | 标签 | isolate | cluster [95% CI] | perm p | 因果口径 | naive isolate |
|---|---|---|---|---|---|---|
| 2024 | jump | 0.857 | **0.907** [0.825, 0.973] | <1e-4 | 0.993 | 0.275 |
| 2024 | jump_human | 0.634 | **0.744** [0.585, 0.881] | 0.018 | 0.681 | 0.275 |
| 2025 | jump | 0.996 | **0.972** [0.934, 0.996] | <1e-4 | 0.989 | 0.170 |
| 2025 | jump_human | 0.951 | **0.955** [0.909, 0.990] | <1e-4 | 0.952 | 0.170 |

（四年 P(>0.5)：2024 jump_human=0.996，其余三格=1.000）

- **首跳簇 anticipation（jump 标签，簇级名次/总簇数）**：2024 六首跳簇全部排前——(H5,70) **3/136**、(H5,153) 10、H1/36 13、H1/42 20、H1/34 28、H1/15 36（全进 top 27%；最差名次 36/136=26.5%，09-28 校订）；2025 两首跳簇 H1/42 15/201、H1/15 31/201。jump_human 较弱（2024 (H5,70) 55/136）。
- **阳性 isolate 名次**：2024 jump 前 10 名为 1,2,5,6,7,8,9,10,11,12/2,677；2025 两标签前 10 名均从第 1 名起。
- **朴素基线两年方向均反**（0.275 / 0.170，同 2026 的 0.046）：前向年份主体为季节性人源株，identity 排序把阴性排最上。
- **L1 末层对照**：jump 两年仍强（cluster 0.872 / 0.941，perm p ≤ 1e-4）；jump_human 复现末层失效——2024 cluster 0.663（P(>0.5)=0.896，perm p=0.076 n.s.，因果口径 0.333）、2025 isolate 0.419 反向（cluster 0.874 分歧）——与 §4 末层反转自洽，jump_human 前向结论依赖中层选择再次成立。
- **分层命中（阈值仅 ≤cutoff train 定）**：2025 jump 高层 26 条 precision 0.96 recall 0.50 FPR 0.001；中+ 61 条 recall 0.94；2024 jump 中+ 82 条 precision 0.79 recall 0.34 FPR 0.007；2024 jump_human 中+ 0 命中（该年 jump_human 排序弱一致）。【09-28 校订（对账 output/forward_multiyear.json）】高档逐年全口径：jump 高档 2024 触发 3 条（tp 2，precision 0.67 recall 0.01 FPR 0.0004）/ 2025 26 条（tp 25）/ 2026 1 条（tp 1，见 09-19-19）；jump_human 高档 2025 触发 4 条（全 TP，recall 0.08 FPR 0），2024/2026 为 0。原"2025 高层首次前瞻触发"不成立（2024 jump 高档已有 3 条）；"jump_human 高档三年从未触发"亦不成立（2025 有 4 条）。论文 FluJump.tex 已按此口径表述。
- H5 sanity（因果口径）：jump 0.728/0.803；jump_human 0.592/0.657（最弱格如预期打折，方向未崩塌）。剔 H7 敏感性：AUC 变化 <0.003，忽略。
- 训练标签口径影响小：2024 train 阳性 headline vs causal = 439 vs 432（jump）、391 vs 385（jump_human）。
- 新簇占比高：2024 无 ≤cutoff 成员簇 81/136，2025 165/201——前向评估大部分落在全新簇上，排序能力非"老簇延续"驱动。

## 结论

- **达到预期，TODO-B7 完成**：前向排序在 2024/2025 复现（jump 两年 cluster ≥0.90、jump_human 0.744–0.955，perm p ≤ 0.018 全显著），加 2026 共**三年 17 个阳性簇**（2024×7 + 2025×8 + 2026×2）的可复现规律；朴素基线三年方向全反。
- 新增超 2026 的证据：**首跳簇 anticipation**——2024 六个首跳簇（含 (H5,70) 奶牛暴发簇，簇级名次 3/136）在 jump 标签下全部排前，说明 w 方向不仅排序已知跳跃谱系，对"该年首次跳跃"的谱系也有前瞻性排序能力（jump_human 较弱，如实并置）。
- 弱格披露：2024 jump_human（cluster 0.744）为六年格中最弱，与 §1 "jump_human H5 双线最弱格"一脉相承；2025 强（0.955）。
- **下一步行动**：进 RESULT.md §5（TODO-B7 勾掉）；L1 jump_human 两年失效进 §4 佐证链。

**产物**：`validation_exp/temporal_validation/run_multiyear_forward.py`、`output/forward_multiyear.json`、`output/forward_{2024,2025}_ranking.csv`
