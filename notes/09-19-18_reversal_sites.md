# 实验：H7 vs H10 反转驱动位点对比 + 结构映射（B6，升档关键①）

- **日期**：09-19-18
- **目的**：H7 方向反转（§3 失效边界）是否由与 H10 相同的一组位点驱动？把"分类学边界"做成机制线索——逐位点对比 H7/H10 的"推负"剖面（H4/H5 对照），并做结构映射（RBS/抗原位点/HA2 stem）。

## 方法

三个小目标（目录 `validation_exp/reversal_sites/`，自包含）：

- **G1**（`g1_h7_driver_sites.py`）：主数据 L28 probe 逐位贡献分解（复用 site_attribution npz，校验 H5=0.8255 通过），定义 `d_pos_ST[col] = mean(ST,+) − mean(H1H3,+)`（亚型特异"推负"剖面）。
- **G2**（`g2_extract_h10h4_contrib.py`）：6 个 probe（L28/L17/L13 × 双标签）重训（回归断言过，对照 group_boundary 同层 AUC ±0.06）；H10/H4 isolate GPU 前向在线算逐位贡献，存 `output/g2_h10h4_contrib.npz`（5026×576 fp16 ×6 键 + lens + accession）。
- **G3**（`g3_compare_h7_h10.py`）：贡献按组聚合到 MAFFT 对齐列（1039 列同坐标系），算每 (label × layer) 的 Spearman(d_pos_H7, d_pos_H10) vs H7~H4（Group 2 阴性对照）/H7~H5（Group 1 对照）+ top-20 最负列 Jaccard；结构注释用 `col_to_h3.json` canonical H3 编号。

## 结果

**G1**：H7 推负是弥漫性的——jump 有效 569 列中 d_pos_H7 中位 −26.8（57% 列为负）、jump_human 中位 −63.2（63%）；H5 对照无偏（中位 −1.06/+1.63，~50%）。top 位点两标签共享核心集：HA1 117/129/116/124/93/33/196/279/45/91、HA2_124（均非经典 RBS 位点）。

**G3**（`output/g3_h7_vs_h10_overlap.json`，566/556 有效列）：

| label × layer | H7~H10 ρ | p | H7~H4 ρ | H7~H5 ρ | top20 Jaccard H10 / H4 |
|---|---|---|---|---|---|
| jump L28 | **0.691** | 2e-81 | 0.387 | 0.226 | 0.29 / 0.111 |
| jump L17 | **0.632** | 2e-64 | 0.369 | 0.315 | 0.25 / 0.081 |
| jump L13 | **0.718** | 6e-91 | 0.421 | 0.163 | 0.379 / 0.143 |
| jump_human L28 | **0.657** | 5e-70 | —（H4 无阳性） | 0.245 | 0.29 / — |
| jump_human L17 | **0.654** | 4e-69 | — | 0.320 | 0.176 / — |
| jump_human L13 | **0.683** | 1e-77 | — | 0.176 | 0.333 / — |

6/6 格一致：H7 推负剖面与 H10 强相关（ρ 0.63–0.72，p ≤ 4e-69），显著高于两个对照（H4 0.37–0.42，H5 0.16–0.32），top-20 共享 6–11 列（Jaccard 为 H4 的 ~2.6 倍）。

**结构映射**（top-20 共享列，canonical H3 编号）：**0 列落在经典 RBS**（130-loop/190-helix/220-loop/190/225/226/228）——与 §2 "w 不富集 RBS" 自洽；少量抗原位点（Ag-A 124/129、Ag-B 157/196、Ag-C 278、Ag-E 82）与 HA2 stem（HA2_79/80/83/88/163/182/211）；跨标签/层复现的共享核心：HA1_116/117/124/129（L28 双标签）、HA1_7/8/9 与 HA1_20/21（L17/L13 N 端）、HA1_93、HA1_282/285。机制重叠发生在分布式非经典位点，而非受体结合域。

## 踩坑（G3 四轮修复）

1. G2 npz 的 accession/subtype 为字符串 object 数组 → `np.load(..., allow_pickle=True)`。
2. 7 条 isolate 无对齐记录（H10×4 + H4×3）+ 10 条对齐序列与 isolate 序列版本不一致（各差 1 残基）→ 共剔除 17/5026（0.34%），不重新对齐（group_boundary 资产只读复用）。
3. **G2 lens 对非 batch 最长序列多计 EOS 一位**（`attention_mask[1:-1]` 切片 bug，475/5026 行 lens=真长+1）→ G3 以列映射真长切片（残基贡献在前、EOS 在尾，顺序不受影响）。
4. H4 无 jump_human 阳性（3198 行全 0）→ 该标签下 d_pos_H4 无定义记 None（数据事实）。

## 涉及文件

- `validation_exp/reversal_sites/{config.py, g1_h7_driver_sites.py, g2_extract_h10h4_contrib.py, g3_compare_h7_h10.py}`
- 产出：`output/g1_*`、`output/g2_h10h4_contrib.npz`、`output/g3_column_contrib_{label}_{layer}.csv` ×6、`output/g3_h7_vs_h10_overlap.json`

## 结论

**达到预期（阳性）**：H7 反转的位点驱动剖面与 H10 显著重叠、且重叠强度高于 H4/H5 对照——失效边界（H7/H10 亚支）存在共享的位点级机制线索，且落在非 RBS 的分布式位点上。H10 功效限制（jump 阳性 13 簇/哺乳 4 簇，jump_human 22 isolate）仍在，结论为方向性证据。

**下一步行动**：进 RESULT.md §3 确定结果；TODO-B6 勾掉。后续可选：共享位点的保守性/糖基化 motif 分析、三维结构距离统计（是否聚集于某一空间区域）。
