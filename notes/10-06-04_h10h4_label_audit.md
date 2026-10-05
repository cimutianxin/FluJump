# 实验：H10/H4 标签按 09-19 词边界口径重跑（S5，Major 7 彻底版）

- **日期**：10-06-04
- **目的**：group_boundary 的 H10/H4 标签仍是 09-19 修复前裸子串口径（H10_143 feline 误判源于此），需与主数据同步。

## 方法
- 移植 09-19 词边界逻辑到 `prep_clusters.py`（\b 词边界 + 垃圾 species 内嵌株名提取 + improve-only 覆盖）
  与 `download_subtypes.py`（下载端词边界推断，同 download_flu.py）。
- 新增 `relabel_word_boundary.py` 两阶段重跑：A) raw csv 下载端值就地修复（同主数据 09-19 对 raw 的处理）；
  B) clean 过滤 + improve-only + 簇级标签重算。**序列与簇划分不变**（不重跑 CD-HIT，保留集逐行断言相等），
  {ST}_isolates.csv 行序保持 = embedding 行序。原始文件备份 *.pre_wboundary（与 git HEAD 核对恢复）。
- 重跑 `run_direction_test.py` + `supplement_analysis.py`（probe 重训，embedding 复用）。

## 结果
- raw 修复：H10 9 行（含 AKM13954/AKM14062 "oyster catcher" feline→unknown——裸子串 cat 误伤）、H4 27 行。
- host_category 变化：H10 8 行、H4 27 行；**阳性簇 jump：H10 13→11**（H10_64、H10_263 转阴性，
  均为 avian|environmental 共检出簇），**H4 25→18**（7 个 avian|environmental 簇转阴）；jump_human 均不变（2/0）。
- **H10_143 核对：feline 成员已移除（hosts avian|feline|environmental → avian|environmental），
  但簇仍阳性**——genuine environmental 成员（AFY97343）维持 ≥2 宿主类别，不翻转为阴性（如实记录）。
- 修复后方向检验：
  - **H10 jump 符号检验 24/31→20/31（p=0.003→0.150，不再显著）**；预注册层 cluster AUC 0.460/0.463
    （P(>0.5)=0.27/0.33，旧 0.464/0.468 P=0.29/0.37）——幅度基本不变，计数掉 4 层。
  - jump_human H10 L13（2 阳性簇不变）：cluster AUC 0.212、P(>0.5)=0.00，反转显著性**完全保持**；
    H10N8 人源簇 H10_197 名次 313→9/321 不变。
  - **H4 对照 3/31→6/31（p=4.6e-6→8.8e-4）仍显著不反转**；预注册层 L17 cluster AUC 0.656、P=0.98（方向为正）。
  - 哺乳子集阳性簇 4→3（H10_143 退出哺乳口径）。

## 结论
- 方向结论**定性保持、定量降档**：H10 jump 符号检验由显著转为不显著（n.s.），jump_human L13 预注册格与
  H4 对照保持原结论——边界主张的表述需改为修复后口径（"20/31 层 n.s. + jump_human L13 显著 + H4 6/31 正向显著"）。
- 产物：group_boundary output 全量更新（direction_test*.json、relabel_report.json、{ST}_cluster_ranking.csv）；
  figdata/fig4_boundary panelc 三件已同步重打包；RESULT.md §3 已更新。
- 后续注意：reversal_sites 的 H10 d_pos 剖面用的是 13 簇旧阳性集，未随本次重跑（S5 范围外），
  其 ρ 数值可能有小幅漂移——如需论文级刷新应重跑 reversal_sites g1/g2。下一步：论文 §3.3 披露句与 Limitations (iv) 改为修复后口径。
