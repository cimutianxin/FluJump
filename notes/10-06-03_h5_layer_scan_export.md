# 实验：三规模 H5 逐层 AUC 导出 figdata（S7，Major 3 配套）

- **日期**：10-06-03
- **目的**：审稿 Major 3 指出深度/规模结论只有 H7 证据；A1 分析（09-26）已算过三规模 H5 逐层 AUC 但未进 figdata。纯数据导出，非新实验。

## 方法
- 自 `figdata/fig5_depth/panela_per_layer_auc.csv`（同源自 depth_reversal g1/g5 JSON，提交 3ea5ca3）提取
  scale, label, layer, rel_depth, h5_auc 五列，写 `figdata/fig5_depth/h5_layer_scan.csv`（204 行 = 31+34+37 层 × 2 标签）。
- 交叉核对：panela 与服务器侧 `validation_exp/depth_reversal/output/g1_layer_geometry.json`（150M）逐位一致；
  650M/3B 少数层与服务器 `g5_scale_geometry.json` 有差（最大 0.064，650M jump L1），系本地 prepare 重打包口径，以已提交 panela 为准。

## 结果
- 各规模 H5 最优层（rel_depth）：150M jump/jump_human 均 L28（rd 0.933，0.8255/0.7936）；
  650M jump L27（rd 0.818，0.6546）、jump_human L33（rd 1.000，0.6451）；
  3B jump L17（rd 0.472，0.7304）、jump_human L14（rd 0.389，0.8286）。
- 与 A1 结论一致：同 Group 1 的 H5 越深越好（best rd 0.82–1.00 为主，3B 例外在 0.39–0.47，如实记录）。

## 结论
- 达到预期（数据导出）。下一步：论文侧决定 Fig 5 备选面板或补充材料回填（§3.4 可补三规模 H5 曲线概要一句）。
