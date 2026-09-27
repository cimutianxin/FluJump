# 审计：前向监测评估（2024/2025/2026）的阳性组成与数字出处

- **日期**：2026-09-27-16
- **目的**：论文摘要/Results 需标注每年前向评估的阳性簇数与组成。本审计只做 CPU 端 CSV/JSON 计数聚合，逐数字钉出处；不训练、不推理、不修改任何文件。
- **范围**：`validation_exp/temporal_validation/output/` 下 5 个前向/workflow 脚本的输出（retro_h7n9* 为回溯实验，不在本次范围）。

---

## 0. 数据口径备忘（审计前提）

- 标签为**簇继承标签**（`datascripts/build_jump_labels.py`）：`label_is_jump`=簇内非 unknown 宿主类别 ≥2；`label_is_jump_human`=簇内含 human 且 ≥2 类 ⇒ jump_human 阳性簇 ⊂ jump 阳性簇。
- 评估标签为 **headline 口径**（全量 ever-jump 簇继承）；**因果口径**（≤cutoff 成员重算）只用于训练标签与敏感性对照（`forward_multiyear.json: protocol`）。
- 簇键一律 **(subtype, cluster_id)**（cluster_id 亚型内编号，跨亚型复用）。
- 三年全部评估行内核验通过：簇内标签唯一（无非一致簇），且 **每年 jump 与 jump_human 的阳性 isolate/簇集合完全相同**（crosstab 无 jump=0&jump_human=1 行）——因为三年评估年的阳性簇全部含人源成员。
- 各年阳性数与 JSON `positive_ranks` 列表长度一致：2024=193、2025=50、2026=10（probe/CNN/hardening 三处互验）。

---

## 1. (b-1) output JSON 逐文件关键字段（Q1）

### `forward_2026.json`（`run_2026_forward.py` 写；"cnn" 节由 `run_2026_forward_cnn.py` 合并写入）
| 字段路径 | 值 |
|---|---|
| `n_2026` / `n_positive_jump_human` | 201 / 10 |
| `probe.label_is_jump.h5_repro_auc` / `.auc_2026` / `.auc_2026_cluster` | 0.798 / **0.9995**（isolate）/ **0.9921**（cluster） |
| `probe.label_is_jump.positive_ranks_of_201` | [1–9, 11]（10 条全进 top 11/201） |
| `probe.label_is_jump_human.*` | 0.7936 / **0.9995** / **1.0** / 同名次 |
| `naive.label_is_jump.auc_2026` / `naive.label_is_jump_human.auc_2026` | **0.0461** / 0.0461（isolate 级，方向反） |
| `cnn.label_is_jump.h5_repro_auc` / `.auc_2026` / `.auc_2026_cluster` | 0.4588 / **0.8649** / 0.8651 |
| `cnn.label_is_jump.positive_ranks_of_201` | [1, 25, 26, 33, 34, 35, 36, 40, 41, 42] |
| `cnn.label_is_jump_human.h5_repro_auc` / `.auc_2026` / `.auc_2026_cluster` | 0.702 / **0.2293**（反转）/ 0.6508 |
| `cnn.label_is_jump_human.positive_ranks_of_201` | [1, 144, 150, 164, 173, 174, 175, 176, 183, 187]（9/10 在 144 名后） |

### `forward_2026_hardening.json`（`run_2026_forward_hardening.py`）
| 字段路径 | 值 |
|---|---|
| `n_2026` / `n_clusters_2026` | 201 / 65 |
| `label_change_pre2025` | jump=0, jump_human=0（≤2025 口径标签零变化，循环风险排除） |
| `layers.L3.label_is_jump` | h5_repro 0.8255；isolate 0.9995；cluster 0.9921；CI95 [0.9531, 1.0]（B_valid=8727，P(>0.5)=1.0）；**perm p=0.0008**；pre2025 口径 0.9921；n_positive_clusters=2 |
| `layers.L3.label_is_jump_human` | h5_repro 0.7936；isolate 0.9995；cluster 1.0；CI95 [1.0, 1.0]（B_valid=8729）；**perm p=0.0002** |
| `layers.L1.label_is_jump`（对照层） | isolate 0.8969；cluster 0.9841；CI [0.9365,1.0]；p=0.002 |
| `layers.L1.label_is_jump_human`（对照层） | isolate 0.3141（反向）；cluster 0.7302；CI [0.381,1.0]；**p=0.1494 不显著** |

注：本文件 `h5_repro_auc`（jump 0.8255）与 `forward_2026.json`（0.798）不同，因为两跑之间隔了 09-19 宿主审计重建标签（7 簇 jump 1→0，见 `data/processed_isolate/host_audit_label_impact.json`；关键簇 (H5,70) touched 但未翻转）；两阳性簇标签不变 ⇒ 2026 指标两跑完全一致。

### `forward_multiyear.json`（`run_multiyear_forward.py`，2024/2025 两年）
| 字段路径 | 值 |
|---|---|
| `years.2024.cutoff` / `.n_train` / `.n_eval` | 2023 / 3426 / **2677**（亚型 H3 1689、H1 857、H5 108、H7 23） |
| `years.2024.n_train_pos_headline_vs_causal` | jump [439, 432]；jump_human [391, 385] |
| `years.2024.layers.L3.label_is_jump` | isolate **0.8566**（剔 H7 0.8553）；cluster **0.9070**，CI [0.825, 0.973]（B_valid=9991），**perm p=0.0**；因果口径 0.9926；naive isolate 0.2747 / cluster 0.4402；n_eval_clusters=136，无≤2023成员 81，阳性簇 7（因果 1 + 首跳 6） |
| `years.2024.layers.L3.label_is_jump_human` | isolate 0.6339；cluster **0.7442**，CI [0.585, 0.881]，**p=0.0176**；因果 0.6815 |
| `years.2024.layers.L1.*`（对照层） | jump：isolate 0.8334 / cluster 0.8715 / p=0.0001；jump_human：isolate 0.5233 / cluster 0.6633 / **p=0.0756 n.s.**（因果 0.3333） |
| `years.2025.cutoff` / `.n_train` / `.n_eval` | 2024 / 5262 / **1020**（H1 745、H3 223、H5 48、H7 4） |
| `years.2025.layers.L3.label_is_jump` | isolate **0.9962**；cluster **0.9715**，CI [0.934, 0.996]（B_valid=9998），**perm p=0.0**；因果 0.9889；naive isolate 0.1703 / cluster 0.7131；n_eval_clusters=201，无≤2024成员 165，阳性簇 8（因果 6 + 首跳 2） |
| `years.2025.layers.L3.label_is_jump_human` | isolate 0.9511；cluster **0.9553**，CI [0.909, 0.990]，**p=0.0**；因果 0.9521 |
| `years.2025.layers.L1.*`（对照层） | jump：0.7939 / 0.9411 / p=0.0；jump_human：isolate **0.419（反向）** / cluster 0.8737 / p=0.0002 |
| `years.2024.tiers` / `years.2025.tiers` | 见第 5 节（Q5） |

### `workflow_2026_eval.json`（`run_2026_workflow_eval.py`）
| 字段路径 | 值 |
|---|---|
| `n_2026` / `positives` | 201 / "10 条 = 2 簇（H5 cluster70 ×9 + H1 cluster15 ×1）" |
| `probe_thresholds` | jump {high 2.1275, mid −2.3618}；jump_human {high 2.1917, mid −2.8479}（仅 ≤2025 train 定） |
| `logit_crosscheck` | ρ=1.0，Δmax=5e-05（与 forward_2026 logits 一致） |
| `jump.probe.high` / `.mid_plus` | n=1, tp=1（P=1.0, R=0.1, FPR=0）/ **n=10, tp=9（P=0.9, R=0.9, FPR=0.0052）** |
| `jump.probe.pos_cluster_best_tier` | {15: 低, 70: 高} |
| `jump_human.probe.high` / `.mid_plus` | **n=0（高层不触发）** / **n=8, tp=8（P=1.0, R=0.8, FPR=0.0）** |
| `jump_human.probe.pos_cluster_best_tier` | {15: 中, 70: 中} |
| `naive.*` 对照 | jump 高档 79 条 0 命中（FPR 0.4136）；jump_human 高档 50 条 0 命中（FPR 0.2618） |

---

## 2. (a) 逐年前向评估组成表（Q2；可直接改写为 Table S2 脚注）

三年阳性簇合计 7+8+2=**17 个**（与 RESULT.md "三年 17 阳性簇"一致）。每年 jump 与 jump_human 阳性集合**完全相同**（见第 0 节核验），下表按年×标签列出行但只写一次构成。

| 年（train ≤） | 标签 | 评估 isolate | 评估 cluster | 阳性 cluster | 阳性 isolate | 阳性簇构成（簇键: 年内条数, 年内宿主组合, 国家, 月份窗） |
|---|---|---|---|---|---|---|
| 2024（≤2023） | jump | 2677（H3 1689/H1 857/H5 108/H7 23） | 136 | 7 | 193 | (H5,81): 39, bovine25+human14, USA, 03–12；(H5,153): 27, bovine16+human11, USA, 10–12；(H5,70): 23, human16+avian7, USA, 09–12；(H1,34): 65, human64+swine1, USA, 01–10；(H1,36): 29, human28+swine1, USA, 01–10；(H1,42): 9, 全 human, USA, 07–12；(H1,15): 1, human, USA, 11 |
| 2024 | jump_human | 同上 | 136 | 7（同 jump） | 193（同 jump） | 同 jump |
| 2025（≤2024） | jump | 1020（H1 745/H3 223/H5 48/H7 4） | 201 | 8 | 50 | (H5,70): 36, avian29+human5+bovine2, USA34+Canada2, 02–12；(H1,208): 4, 全 human, India, 09；(H1,42): 3, swine2+human1, USA, 03–09；(H1,34): 2, human, USA；(H5,81): 2, human, USA, 01；(H1,15): 1, swine, USA, 12；(H1,41): 1, human, USA；(H1,113): 1, human, India |
| 2025 | jump_human | 同上 | 201 | 8（同 jump） | 50（同 jump） | 同 jump |
| 2026（≤2025） | jump | 201（H3 111/H1 80/H5 10） | 65 | 2 | 10 | (H5,70): 9, **全 avian 野鸟**（Canada Goose 6/Red-tailed Hawk 2/Snow Goose 1）, USA（ND/MS）, 01–02；(H1,15): 1, swine, USA（NC）, 02 |
| 2026 | jump_human | 同上 | 65 | 2（同 jump） | 10（同 jump） | 同 jump |

单事件性判定（"同一暴发/同一宿主类别组合"口径）：
- **2026 (H5,70) = 单事件**：9 条全为美国野鸟、同一 2 个月窗口、同一 CD-HIT 簇——一次暴发/监测采样的延续；**2026 (H1,15) = 单条**（簇全期仅 3 条：human 2024 + swine 2025/2026）。
- **2024 (H5,81)/(H5,153)/(H5,70) = 单一暴发复合体（美国奶牛 H5N1 暴发及其人/禽溢出）**，宿主类别混合但流行病学上属同一事件；(H1,34)/(H1,36) **不是**单事件（年内 64+1 / 28+1 的 human+swine 混合，时间跨 10 个月，"跳跃"信号各由 1 条猪源株锚定）；(H1,42) 年内全 human（簇阳性由 2025 年猪源成员继承而来）；(H1,15) 单条。
- **2025**：(H5,70) 为奶牛暴发复合体的延续（三宿主、跨 11 个月、美加两地）→ 持续暴发而非单事件；其余 7 簇年内条数 ≤4 且多为单宿主小簇（(H1,42) 为 swine2+human1 混合）。
- **重要口径提示**：阳性身份来自簇继承——例如 2026 年 10 条阳性**无一条是人源分离株**（9 禽 + 1 猪），其 jump_human=1 是因为簇内 2024–2025 年已有人源成员；2024/2025 年亦有若干年内全 human 的阳性簇（(H1,42)、(H1,208)、(H1,41)、(H1,113) 等）。

### 2026 十条阳性逐条清单（确认"9 条同一 H5N1 暴发簇 + 1 条猪 H1"）

| probe 名次 | accession | subtype/serotype | (subtype, cluster) | host_category | strain | 地点/日期 |
|---|---|---|---|---|---|---|
| 1 | YEO13266 | H5/H5N1 | (H5, 70) | avian（Canada Goose） | A/Canada Goose/Mississippi/876/2026 | USA-MS, 2026-01 |
| 2 | YEO13224 | H5/H5N1 | (H5, 70) | avian（Canada Goose） | A/Canada Goose/North Dakota/2731/2026 | USA-ND, 2026-02 |
| 3 | YEO13386 | H5/H5N1 | (H5, 70) | avian（Canada Goose） | A/Canada Goose/North Dakota/2729/2026 | USA-ND, 2026-02 |
| 4 | YEO13314 | H5/H5N1 | (H5, 70) | avian（Canada Goose） | A/Canada Goose/North Dakota/2730/2026 | USA-ND, 2026-02 |
| 5 | YEO13276 | H5/H5N1 | (H5, 70) | avian（Red-tailed Hawk） | A/Red-tailed Hawk/Mississippi/877/2026 | USA-MS, 2026-02 |
| 6 | YEO13234 | H5/H5N1 | (H5, 70) | avian（Red-tailed Hawk） | A/Red-tailed Hawk/Mississippi/890/2026 | USA-MS, 2026-02 |
| 7 | YDK26866 | H1/H1N1 | (H1, 15) | **swine** | A/swine/North_Carolina/014073_(A02859186)_H1/2026 | USA-NC, 2026-02 |
| 8 | YEO13359 | H5/H5N1 | (H5, 70) | avian（Canada Goose） | A/Canada Goose/North Dakota/2733/2026 | USA-ND, 2026-02 |
| 9 | YEO13382 | H5/H5N1 | (H5, 70) | avian（Snow Goose） | A/Snow Goose/Mississippi/888/2026 | USA-MS, 2026-01 |
| 11 | YEV12912 | H5/H5N1 | (H5, 70) | avian（Canada Goose） | A/Canada Goose/North Dakota/2725/2026 | USA-ND, 2026-01 |

构成确认：**9 条 = 同一 (H5,70) 暴发簇的美国野鸟分离株（2026-01×3 + 2026-02×6，ND/MS），1 条 = (H1,15) 猪源 H1（NC）**。⚠️ RESULT.md 现表述"9 条 2026-01 美国野鸟"不精确：实际 2026-01 仅 3 条、2026-02 有 6 条，建议改"2026 年 1–2 月"。

---

## 3. (b-2) 2024 首跳簇名次出处与簇键核对（Q3）

- **出处**：`forward_multiyear.json → years.2024.layers.L3.label_is_jump.first_jump_cluster_ranks`：
  `(H5,70)=3、(H5,153)=10、(H1,36)=13、(H1,42)=20、(H1,34)=28、(H1,15)=36` —— 即"3/10/13/20/28/36"。
- **口径**：首跳簇 = headline=1 且 ≤2023 因果标签=0（`n_first_jump_clusters=6`）；名次为**簇级**（簇内 mean logit 降序在 136 个评估簇中的排名）。已用 `forward_2024_ranking.csv` 的 `logit_L3` 独立复现 jump_human 首跳名次（15/17/20/25/55/83）与 JSON **逐值一致** ✓（jump 名次用 jump-probe logits，CSV 未存，以 JSON 为准）。
- 六簇键与大小（clean 两表合并行数 / 表内 cluster_size 字段；括号为 2024 评估年内条数）：

| 簇键 | 全期行数 | cluster_size 字段 | 2024 eval 内 | 全期宿主组合 | 全期年份/国家 |
|---|---|---|---|---|---|
| (H5, 70) | 74 | 103 | 23 | avian 51 + human 21 + bovine 2 | 2022–2026, USA 72 + Canada 2 |
| (H5, 153) | 27 | 27 | 27 | bovine 16 + human 11 | 全 2024, USA |
| (H1, 36) | 35 | 35 | 29 | human 34 + swine 1 | 2023–2024, USA 34 + Nicaragua 1 |
| (H1, 42) | 12 | 12 | 9 | human 10 + swine 2 | 2024–2025, USA |
| (H1, 34) | 69 | 69 | 65 | human 68 + swine 1 | 2023–2025, USA |
| (H1, 15) | 3 | 3 | 1 | swine 2 + human 1 | 2024/2025/2026 各 1 条, USA |

- **"US dairy-cattle/human H5 簇 rank 3" = (H5, 70)** ✓（serotype 全 H5N1）。组成注记：该簇 clean 行 avian 占多数（51），bovine 锚点 2 条（2025），human 21 条（2024–2025 奶牛场相关人病例）；同年的 (H5,81)（bovine25+human18+avian72）与 (H5,153)（bovine16+human11）也是奶牛暴发相关簇——论文若要精确，宜写"(H5,70) 美国奶牛暴发相关 H5N1 簇（含 bovine/human/avian 成员）"。
- cluster_size 字段（103/179 等）= 全量表 `all_isolates.csv` 的行数（含清洗中剔除的 2022 年旧记录等）；分析用 clean 表行数更少（如 (H5,70) 74）。两个口径都已列出，引用时勿混。
- 对照：2025 年首跳簇 2 个（(H1,15)、(H1,42)；`years.2025...first_jump_cluster_ranks`）；2026 年首跳簇 0 个（两阳性簇 ≤2025 已为阳性，`label_change_pre2025=0`；65 簇中 30 簇为 2026 新簇，本审计独立复核一致 ✓）。

---

## 4. (b-3) CNN 与 naive 基线数字出处（Q4）

| 数字 | 出处（文件 → 字段） | 口径 |
|---|---|---|
| CNN 2026 jump **0.865** | `forward_2026.json → cnn.label_is_jump.auc_2026` = 0.8649（cluster 级 `auc_2026_cluster` = 0.8651） | isolate 级；由 `run_2026_forward_cnn.py` 合并写入 |
| CNN 2026 jump_human **0.229** | `forward_2026.json → cnn.label_is_jump_human.auc_2026` = 0.2293（cluster 级 0.6508） | isolate 级，方向反转 |
| naive 2024 **0.275** | `forward_multiyear.json → years.2024.layers.L3.label_is_jump.auc_isolate_naive` = 0.2747 | isolate 级；同值重复于 4 个层×标签格（阳性集合两标签相同）；cluster 级 `auc_cluster_naive` = 0.4402 |
| naive 2025 **0.170** | `forward_multiyear.json → years.2025.layers.L3.label_is_jump.auc_isolate_naive` = 0.1703 | 同上；cluster 级 0.7131 |
| naive 2026 **0.046** | `forward_2026.json → naive.label_is_jump.auc_2026` = 0.0461 | isolate 级（该脚本未算 cluster 级 naive） |

---

## 5. (b-4) 风险分层数字核对（Q5，逐项给出分母）

| 声称 | 出处（文件 → 字段） | 原始值 | 分母拆解（本审计复核） |
|---|---|---|---|
| 2026 jump_human 中档+ **8/10 命中、零误报** | `workflow_2026_eval.json → jump_human.probe.mid_plus` | n=8, tp=8, P=1.0, R=0.8, FPR=0.0 | 阳性分母 10（8/10）；阴性分母 191（0/191）。漏报 2 条均为 (H5,70) 野鸟（YEV12912、YEO13382） |
| 2026 jump 中档+ **9/10、1 误报** | `workflow_2026_eval.json → jump.probe.mid_plus` | n=10, tp=9, P=0.9, R=0.9, FPR=0.0052 | 阳性 10（9/10）；阴性 191（1/191=0.0052）。误报 1 条 = YDL35783（(H5,462) avian，孟加拉蛋鸡，trainpct 0.891）；漏报 = 猪源 H1 YDK26866 |
| 2026 jump 高档（补充） | `jump.probe.high` | n=1, tp=1（即 (H5,70) 最佳成员），P=1.0, R=0.1 | ⚠️ RESULT.md"2026 高层不触发"仅对 **jump_human** 成立（`jump_human.probe.high` n=0）；jump 标签高层触发了 1 条且命中——引用时需写明标签 |
| 2025 jump 高档 **26 条（0.96/0.50）** | `forward_multiyear.json → years.2025.tiers.jump.high` | n=26, tp=25, P=0.9615, R=0.5, FPR=0.001 | 阳性分母 50（25/50）；阴性分母 970（1/970≈0.001） |
| 2024 jump 中档+ **82 条（0.79/0.34）** | `forward_multiyear.json → years.2024.tiers.jump.mid_plus` | n=82, tp=65, P=0.7927, R=0.3368, FPR=0.0068 | 阳性分母 193（65/193）；阴性分母 2484（17/2484≈0.0068；65+17=82 ✓） |

阈值出处（均仅用 ≤cutoff train 定界：高=train 阳性 logit 中位，中=train 阴性 P95）：2024 `years.2024.tiers.jump.thresholds` {2.146, −1.920}；2025 {2.115, −2.119}；2026 `workflow_2026_eval.json probe_thresholds` jump {2.127, −2.362} / jump_human {2.192, −2.848}。三年 `threshold_degenerate=false`。

---

## 6. (c) 建议的论文表述（仅建议，不改稿）

摘要（或 Results 首句）建议一句话标注功效：

> "前向评估的阳性集合高度集中且以簇继承标签定义：2024/2025/2026 年评估集分别为 2677/1020/201 条分离株（136/201/65 簇），阳性为 7/8/2 个簇（193/50/10 条分离株，jump 与 jump_human 阳性集合逐年相同）；其中 2026 年的 10 条阳性为同一 H5N1 暴发簇 (H5, cluster 70) 的 9 条美国野鸟分离株（2026 年 1–2 月）加 1 条猪源 H1（cluster 15），故三年结果应解读为排序/富集验证（cluster 级 AUC 0.907/0.972/0.992，permutation p≤8×10⁻⁴），而非独立事件层面的分类性能。"

Results/方法可再加一句口径说明：

> "阳性身份按簇继承：2026 年 10 条阳性分离株本身均非人源（9 禽 + 1 猪），其 jump_human 阳性来自簇内 2024–2025 年人源成员；≤2025 口径重算标签对 2026 评估零影响（0/65 簇变化）。"

---

## 7. 附带审计发现（建议作者知悉，均未改任何文件）

1. RESULT.md"9 条 2026-01 美国野鸟 H5N1 暴发簇"日期不精确（实为 2026-01×3 + 2026-02×6）——建议改"2026 年 1–2 月"。
2. RESULT.md"2026 高层不触发系分布漂移"只覆盖 jump_human；jump 标签 2026 高档触发 1 条且为真阳性（P=1.0）——并置时需写清标签。
3. "US dairy-cattle/human H5 簇"（(H5,70)）avian 成员占多数（51/74），bovine 直接锚点为 2 条；措辞建议"dairy-outbreak–associated"。
4. `forward_2026.json`（08-28）与 hardening（09-19）之间隔了 09-19 宿主审计（7 簇 jump 标签 1→0），H5 复现 AUC 因此 0.798→0.8255；2026  headline 指标不受影响（两阳性簇标签未变）。
5. 2024 年 7 个阳性簇中 (H1,34)/(H1,36)/(H1,42)/(H1,15) 的年内宿主组合以人源为主，跳跃信号由少数/其他年份猪源成员锚定——Table S2 若列组成，建议用本审计第 2 节的年内宿主拆分。

## 涉及文件（全部只读）

- `validation_exp/temporal_validation/output/{forward_2026.json, forward_2026_hardening.json, forward_multiyear.json, workflow_2026_eval.json, forward_2024_ranking.csv, forward_2025_ranking.csv, forward_2026_ranking.csv, workflow_2026_tiers.csv}`
- `validation_exp/temporal_validation/{config.py, run_2026_forward.py, run_2026_forward_cnn.py, run_2026_forward_hardening.py, run_2026_workflow_eval.py, run_multiyear_forward.py}`
- `data/processed_isolate/{all_isolates_clean.csv, 2026_isolates_clean.csv, all_isolates.csv, host_audit_label_impact.json}`
- `datascripts/build_jump_labels.py`、`RESULT.md`（核对声称，未改）

## 待办

- ~~作者决定是否采纳第 6/7 节的措辞与修正建议~~ **已采纳并落实（09-27 当日）**：§7 第 1–3 条已修订进 `RESULT.md`（第 99 行日期改"2026 年 1–2 月"；第 102 行高层不触发限定 jump_human 并补 jump 高档 1 条命中、(H5,70) 改"奶牛暴发相关 H5N1 簇（avian 占多数）"；第 156 行高层不触发补 jump_human 标签）。
- 若 Table S2 需要 per-cluster 全期组成，可直接复用第 3 节表格。
