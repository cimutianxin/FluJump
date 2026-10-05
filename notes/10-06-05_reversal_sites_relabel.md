# 实验：reversal_sites 按 S5 修复后口径重跑（S8）

- **日期**：10-06-05
- **目的**：S5（`notes/10-06-04_h10h4_label_audit.md`）把 H10/H4 标签切换到 09-19 词边界口径
  （阳性簇 H10 13→11、H4 25→18，jump_human 不变 2/0）后，`validation_exp/reversal_sites/`
  的位点剖面仍锚在旧阳性集，是论文 §3.3 唯一未刷新的数字；按依赖序重跑 g2→g1→g3→g4
  并重打包 figdata/fig4_boundary paneld/panele（服务器任务
  `notes/10-06-04-服务器任务_S8_reversal_sites重跑.md`）。

## 方法
- `g2_extract_h10h4_contrib.py`：REPRO_REF 基准由"09-11 修复前标签"改为
  direction_test.json regression_check（10-06 S5 重跑口径）6 格数值，容差 ±0.06 不变；
  重跑 H10/H4 逐位贡献提取（GPU，~2 min）。`g3_compare_h7_h10.py` 同步更新 L17/L13 基准。
- `g1_h7_driver_sites.py`、`g3_compare_h7_h10.py`、`g4_orthogonal_evidence.py` 依序重跑（CPU）。
- figdata 重打包：`export_structure_colors.py` 重导 structure_colors.csv + v1 pml
  （sanity check：核心 0 列落 RBS），PyMOL 重渲染 v1/v2 四张 PNG；
  paneld_columns/paneld_overlap_summary/panele_summary/summary_metrics 重建
  （summary_metrics 的 sign_test 与 H10 预注册 cluster AUC 同步至 S5 已提交 panelc 口径）。

## 结果
- **g2/g1 不变**：g2 回归断言全过（重训 probe H5/H7 AUC 与新基准逐位一致），
  贡献矩阵 npz 与 git HEAD 字节相同（贡献只依赖序列与 probe 权重，均不受标签重跑影响）；
  g1 全部输出字节相同（H7 属主数据，标签未变）——记"不变"。
- **g3（6 格，新→旧对照）**：

  | 格 | ρ(H7,H10) 新 | 旧 | Jaccard H10 新 | 旧 | ρ(H7,H4) 新 | 旧 |
  |---|---|---|---|---|---|---|
  | jump L13 | 0.7183 | 0.7184 | 0.333 | 0.379 | 0.4198 | 0.4206 |
  | jump L17 | 0.6311 | 0.632 | 0.25 | 0.25 | 0.3678 | 0.3693 |
  | jump L28 | 0.6905 | 0.6907 | 0.29 | 0.29 | 0.3863 | 0.3874 |
  | jump_human 三格 | 与旧完全一致 | 0.6829/0.6539/0.657 | 完全一致 | 0.333/0.176/0.29 | —（H4 无阳性） | — |

  n_valid_cols 不变（566/556）；H5 对照不变（0.1625–0.3196）；H4 Jaccard 不变（0.081–0.143）。
  ρ(H7,H10) 区间 0.632–0.718 → **0.631–0.718**（p ≤ 3.3e-64），Jaccard 区间
  0.176–0.379 → **0.176–0.333**，H4 对照 0.369–0.421 → **0.368–0.420**。
- **g4**：jump L13 共享集 11→10 列（列 669=HA1_274 在新 d_pos_H10 排 21/566，跌出 top-20）；
  **core 17 列成员完全不变，0 列落 RBS 不变**。JS 百分位：jump L13 74.0→78.7，其余格不变，
  区间 **52.8–85.4 不变**（<95 阈值结论不变）；空间聚集仍阴性（core p=0.963）；
  糖基化仍仅 L17 一致显著；亚型身份对照 probe 数值不变（ρ 0.1469/0.1132，CV AUC 0.995/0.997）。
  paneld_columns.csv 仅 d_pos_H10 列漂移（max |Δ|=11.85），其余逐列核验不变。

## 结论
- 论文 §3.3 位点段结论**定性全保持、定量微移**：6/6 格强相关 > H4/H5 对照、共享驱动位点
  0 列落经典 RBS、非 clade 指纹（JS <95 阈值、身份探针无关）、机制开放——全部不变；
  刷新区间：ρ 0.631–0.718 / Jaccard 0.176–0.333 / H4 对照 0.368–0.420。
- 至此 §3.3 全部数字（方向检验 S5 + 位点 S8）统一为 09-19 词边界修复后口径。
- 产物：reversal_sites output 更新（g3 三张 jump CSV + g3 json + g4 两个 json；g1/g2 不变）；
  figdata/fig4_boundary paneld/panele/structure_colors/summary_metrics + 四张 PNG 重打包，
  README 更新；RESULT.md §3 B6 条刷新。
- 下一步：**继续**——本地论文侧回填 §3.3 位点段 TODO(S8) 处、fig:boundary 图注 (d)(e)，
  Fig 4 整图重新生成。
