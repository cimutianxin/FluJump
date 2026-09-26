# 实验：时间维度验证——2026 前向外推 + H7N9 回溯排序

- **日期**：08-28-18
- **目的**：验证 probe 在时间维度上的实用性（§5 待补高优先级项）：
  1. 前向：用 2026 新分离株做"真实世界"前向外推排序（兼做 §5 待补的 ranking vs §1 朴素基线正面对比）；
  2. 回溯：pre-2013 训练 → 2013 H7N9 暴发簇排序（兼做 §3 待补的时间验证）。

## 方法

- 目录：`validation_exp/temporal_validation/`（自包含，config.py 集中路径与常量）
- **前向（run_2026_forward.py）**：
  - 训练：H1+H3 train split 重训 Ridge LR probe（L3 embedding，StandardScaler + GridSearchCV 5-fold）；先断言 H5 holdout 复现 AUC≈0.80 再继续；
  - 评估：201 条 2026 分离株（`emb_2026_L3.npy`，extract_2026_embeddings.py 提取，L28=倒数第三层），label_is_jump / label_is_jump_human 双标签；
  - 阳性：10 条（cluster 70 的 9 条美国野鸟 H5N1 + cluster 15 的 1 条猪源 H1）；
  - 朴素基线：max identity to H1+H3 train 人源参考（MAFFT 对齐序列）。
- **回溯（run_h7n9_retrospective.py）**：
  - 训练：H1+H3 且 0 < collection_year < 2013（1048 行、158 阳性），**用 label_is_jump**（H3 pre-2013 jump_human 仅 3 阳性，不可用——口径与全量实验不同，注意）；
  - 评估：H7 且 year ≥ 2013（478 行、90 cluster），阳性 = 7 个 H7N9 簇 {109,149,150,151,154,155,198}，原始与翻转（-logit）方向都报告。
- **补充对照（run_retro_controls.py）**：pre-2013 probe 的 train CV AUC + 同亚型跨时间（2013+ H1+H3）AUC，用于区分"时间外推失效"与"跨亚型失效"。

## 结果

### 前向 2026（GO，超预期）

| 指标 | probe | 朴素基线 | CNN（stage2，Borkenhagen 架构） |
|------|-------|----------|-------------------------------|
| 2026 AUC（jump） | **0.9995** | 0.0461 | 0.8649（cluster 级 0.8651） |
| 2026 AUC（jump_human） | **0.9995** | 0.0461 | **0.2293**（cluster 级 0.6508） |
| cluster 级 AUC | 0.9921 / 1.0000 | — | 见上 |
| 10 条阳性名次（/201，jump） | [1,2,3,4,5,6,7,8,9,11] | — | [1,25,26,33,34,35,36,40,41,42] |
| 10 条阳性名次（jump_human） | [1–9, 11] | — | [1,144,150,164,173–176,183,187] |

- CNN 复现断言通过：H5 holdout AUC 0.4588（jump）/ 0.7020（jump_human），与 evaluate.py 已知水平一致。
- CNN 解读：jump 标签下 CNN 有中等排序能力（0.865，但阳性散布在 25–42 名，probe 全部 top 11）；**jump_human 标签下 CNN 方向反转（isolate AUC 0.229）**，9/10 阳性被排到 144 名之后——CNN 学到的人适应信号在前向设定下不成立。
- 三方对比结论：probe（0.9995）≫ CNN jump（0.865）≫ 朴素基线（0.046）；jump_human 上 probe 是唯一有效方法。

- H5 复现断言通过：0.798 / 0.794（vs 全量实验 0.80）。
- 朴素基线方向完全反了：2026 数据主体是季节性人源株，与人源参考 identity 中位 0.995，被人源相似度排到最上面；真正的暴发簇（H5N1 禽源）被排到最低 → AUC 0.046。
- 前 9 名 = 2026-01 美国野鸟 H5N1 暴发簇（cluster 70），第 11 名 = 猪源 H1（cluster 15）。
- 功效限制：阳性仅 10 条 / 2 簇，AUC 应表述为"排序验证"而非分类性能。

### 回溯 H7N9（NO-GO，阴性）

| 指标 | probe raw / 翻转 | 朴素基线 raw / 翻转 |
|------|------------------|---------------------|
| isolate AUC | 0.570 / 0.430 | 0.250 / 0.750 |
| cluster AUC | 0.542 / 0.458 | 0.352 / 0.648 |

- 7 个 H7N9 阳性簇 cluster 级名次双向都在 33–58/90（中段），pre-2013 模型**无法**把 H7N9 簇排前。
- 重要现象：pre-2013 训练下 **H7 反转本身消失**（全量训练时 H7 强反转，翻转 AUC 0.72–0.90；此处翻转后 0.43–0.46 仍近随机）→ H7 信号依赖 post-2013 的 H1+H3 训练数据。

### 补充对照（阴性结果的归因）

- pre-2013 probe train CV AUC = **0.9648**（训练本身有效，best C=1.0）；
- 同亚型跨时间（2013+ H1+H3，7343 行/571 阳性）AUC = **0.7873**（同亚型跨时间仍有效）；
- → 回溯失败**特异于 H7（Group 2 跨亚型）**，不是"时间外推失效"。与 §3 "Group 边界"叙事自洽。

## 结论

- **前向 GO**：probe 在真实 2026 数据上把全部 10 条 jump 阳性排进 top 11/201，朴素基线完全失效（AUC 0.046 反向）——§5 前向叙事成立，且顺带完成 "ranking vs 朴素基线正面对比" 待补项。
- **回溯 NO-GO**：pre-2013 模型无法识别 H7N9 暴发簇；但对照显示失败特异于 H7/Group 2，与同亚型跨时间有效（0.787）不矛盾——如实记录为阴性结果，支撑 §3 的失效边界叙事。
- **下一步**：剩 §3 H10 扩展（最后一个投稿前置决定性项）与 §1 CNN 预算对齐。

## 涉及文件

- `validation_exp/temporal_validation/{config.py, extract_2026_embeddings.py, run_2026_forward.py, run_2026_forward_cnn.py, run_h7n9_retrospective.py, run_retro_controls.py}`
- 输出：`output/{emb_2026_L3.npy, forward_2026.json, forward_2026_ranking.csv, retro_h7n9.json, retro_h7n9_cluster_ranking.csv, retro_h7n9_controls.json}`
