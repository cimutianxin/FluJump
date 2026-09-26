# 实验：H10（+H4 对照）零样本方向检验 — Group 边界从单点到规律

- **日期**：09-11-21
- **目的**：RESULT.md §3 头号待补。H7 反转仅建立在 7 个阳性 cluster（全是 2013–2015 H7N9 单次暴发）上；纳入 H10（H7 姐妹亚支，H7/H10/H15 亚支）检验反转是否复现，H4（Group 2 但属 H3 亚支）做特异性对照，钉死失效边界位置。

## 方法

- 新自包含目录 `validation_exp/group_boundary/`（不动主数据目录）
- **数据下载**（`download_subtypes.py`）：发现 NCBI esearch 不把 "H10N8" 分词出 "H10"，原式 query `"Influenza A virus hemagglutinin H10"` 会漏掉全部只按全型标注的记录（含所有人源 H10N8/H10N3 HA），并带入 H5N6/H9N2 污染。改用 union query `(H10 OR H10N1..H10N9)[all fields] AND hemagglutinin[protein name] AND Influenza A virus[organism] NOT pdb[filter]`；clean 时加亚型一致性 post-filter（serotype/strain_name/definition 正则复核）
- **pipeline**：clean（同 clean_data.py 规则）→ CD-HIT 0.99（同 cluster_strains.py 参数）→ cluster 标签（同 build_jump_labels.py 定义），完全不动标签口径
- **评估**（`run_direction_test.py`）：重训 31 层 Ridge LR probe（H1+H3 train，同 target_val 协议），H10/H4 全量打分；预注册层（jump→L17/L28，jump_human→L13/L28，不用 H10/H4 选层）cluster bootstrap B=1000；全层剖面 + 几何投影 + 朴素基线（mafft --add --keeplength 并入既有对齐）

## 数据规模

| 亚型 | raw | clean | cluster | jump 阳性簇 | jump_human 阳性簇 |
|---|---|---|---|---|---|
| H10 | 2263 | 1828 | 321 | 13 | 2 |
| H4 | 3553 | 3198 | 408 | 25 | 0 |

- H10 人源簇核对通过：H10_197 = H10N8 江西 2013（4 条）、H10_22 = H10N3 江苏 2021（含 2021 人源株）；另发现 2023–2024 新中国人源 H10 记录 4 条（浙江/广西/南宁/云南），但所在簇无禽源共聚 → 按标签定义非阳性
- 水貂 1984 / 海豹 2014 事件因簇内无 99% 同源禽源序列，按 cluster 标签定义为非阳性（singleton 哺乳动物簇）
- 注意：H10 阳性簇中 10/13 是 avian|environmental（监测共检出，生物学弱）；1 个假阳性簇（H10_143：A/oystercatcher 的 strain_name 命中 `/cat/` 子串误判 feline，主 pipeline 同源 bug）

## 结果

**回归校验通过**：重训 probe 在 L28 复算 H5=0.798 / H7=0.282，与历史 0.7979/0.2823 完全一致。

**方向检验（label_is_jump，cluster 级全层剖面）**：

| 亚型 | <0.5 层数 | 符号检验 p | 预注册层 cluster AUC（B=1000 bootstrap） |
|---|---|---|---|
| H10 | **24/31** | **0.0033** | L17: 0.464 [0.357,0.582] P(>0.5)=0.29；L28: 0.468 P=0.37 |
| H4 | 3/31 | 4.6e-06（正向） | L17: 0.566 P=0.83；L28: 0.516 P=0.64 |

**几何（对照 H7 "整个亚型落入 w 负半空间且阳性更深"）**：
- H10 L17：frac_all<0 = 0.98，pos_mean −10.85 < neg_mean −10.59（阳性更深）→ **H7 模式复现**
- H4 L17：frac_all<0 = 0.96，但 pos_mean −3.69 > neg_mean −6.10（阳性更浅）→ 方向正常

**已知人源簇名次（jump_human L13 = H7 target-val 选层）**：H10N8 簇 raw 名次 **313/321**（几乎垫底），翻转后 9/321 —— 与 H7 阳性簇"深陷负端"行为一致。

**朴素基线**：H10 jump raw 0.403（同向偏负）；jump_human raw 0.145（强负向）；H4 jump 0.474（近随机）。H10 的弱负方向部分可由全局相似度解释（同 §1 H7 nuance）。

**jump_human（仅 2 阳性簇，功效不足，描述性）**：L13 cluster AUC 0.212 / L28 0.589；哺乳子集（4 簇）jump L17 0.583 / L28 0.457 —— 均不作判定。

## 结论

- **H10 复现了 H7 反转的方向**（cluster 级 24/31 层系统性 <0.5，p=0.003；几何同人源簇深陷负端），但**幅度远弱于 H7**（0.37–0.46 vs H7 的 0.1–0.28），预注册层 bootstrap 不显著 → 定性复现、定量打折
- **H4（Group 2 / H3 亚支）不反转**（3/31 层 <0.5，p=4.6e-06 正向；几何正常）→ 失效边界精确到 **H7/H10 亚支**，而非整个 Group 2（与 H3 属 Group 2 却可训练一致）
- 边界叙事从"n=7 单次事件"升级为"亚支级两点 + 对照一点"，但 H10 弱幅度 + 阳性簇构成（10/13 环境共检出）限制了强度
- **下一步**：完成（本待补项可勾）；RESULT.md §3 措辞改为"H7/H10 亚支边界"；若继续加固可考虑 H15（但纯禽源无阳性，不可行）或位点级对比 H7 vs H10 的反转驱动位点
