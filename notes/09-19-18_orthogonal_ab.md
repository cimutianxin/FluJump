# 实验：反转位点 (a) clade 指纹 vs (b) 功能机制 —— 四项正交检验（G4）

- **日期**：09-19-18
- **目的**：B6 发现 H7/H10 反转推负剖面位点级重叠（6/6 格 ρ 0.63–0.72），本实验判定重叠的解读——(a) 分类学信号（亚型诊断位点/共同祖先）还是 (b) 共享功能机制。判定规则预注册于计划（nebula-war-machine 计划文件）。

## 方法

`validation_exp/reversal_sites/g4_orthogonal_evidence.py`，输入 = g3 的 6 格共享驱动列（top20 d_pos_H7 ∩ top20 d_pos_H10）+ core 集（≥2 格共享，17 列）。四检验：

- **A**：列级 JS 分歧度（pooled(H7,H10) vs pooled(H1,H3,H5)），共享列百分位中位 + 10k 置换 p。判 (a) 需百分位中位 ≥95 且 p<0.05。
- **B**：1HGG Cα 空间聚集（chain A=HA1/B=HA2；锚点 chain A 226=LEU 验证通过），共享集 median pairwise 距离 vs 同 HA1/HA2 构成 10k 随机 null。
- **C**：N-X-S/T sequon 列级 fraction 差异 Δ=|H7H10−H1H3H5| 的百分位 + 置换 p。
- **D**：L28 mean-pooled 亚型身份 LR（H7 vs H1H3；H5 vs H1H3 对照），逐位贡献组差 d_id；判据 ρ(|d_id_H7|, −d_pos_H7)：≥0.7 坐实 (a)，<0.4 指向 (b)。

## 结果

| 检验 | 结果 | 判定 |
|---|---|---|
| A clade 标记 | 百分位中位 52.8–85.4（主线 L28 双标签仅 52.8，p≈0.09）；无一 ≥95 | **(a) 不成立** |
| B 空间聚集 | 全细胞 p 0.08–0.96；core 44.9Å vs null 34.3Å（反聚集） | 无结构证据 |
| C 糖基化 | L17 显著（89.3/95.9, p=1e-4）、L28 边缘（p~0.002, pct 56）、L13 不显著；core 4/17 列 Δ>0.3 | 混合 |
| D 身份 probe | CV AUC 0.995/0.997（sanity 通过）；ρ(\|d_id_H7\|, −d_pos_H7)=**0.147/0.113**（jump/jump_human），对照 ≤0.04 | **(a) 不成立**，<0.4 → (b) 方向 |

附加观察：A 中共享列 H7/H10 consensus 一致率 0.5–1.0（L28 jump = 1.0），即两亚支在同一批列上取相同残基，但这些列的分歧度仅中等——是"温和偏斜的非极端位点"。

## 结论

**判定（按预注册表）：(b) 方向——H7/H10 共享驱动位点不是 clade 指纹，但具体功能机制未锁定。**

- 两项 (a) 检验双双失败：位点不是极端亚型诊断位点（A），jump probe 的 H7 推负谱与亚型身份谱几乎无关（D，ρ~0.1 vs 阈值 0.7）。
- 支持 (b) 的直接证据弱：B 空间聚集阴性（core 反聚集），C 糖基化仅 L17 一致显著。
- 诚实表述：**排除 (a) 强于证明 (b)**——H7/H10 反转重叠携带超出亚型身份的分布式信号，机制（稳定性/糖基化/抗原表面）留作开放问题。
- 对论文的意义：§3 B6 条可从"机制重叠线索"升级为"重叠非平凡（排除 clade 指纹解读）"，但不可声称已定位功能机制。

**下一步行动**：按此判定更新 RESULT.md §3 B6 条（一句判定）；开放方向 = 已知 host-range 位点文献集 Fisher 富集检验、HA2 stem 稳定性位点专项核对。

## 涉及文件

- `validation_exp/reversal_sites/g4_orthogonal_evidence.py`
- 产出：`output/g4_shared_driver_sets.json`、`output/g4_column_scores.csv`、`output/g4_orthogonal.json`、`data/1HGG.pdb`（新下载，RCSB 可连通）
