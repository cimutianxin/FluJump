# 实验：2026 朴素基线 cluster 级 AUC（S3，Major 6）

- **日期**：10-06-03
- **目的**：朴素基线前向 cluster 级口径缺 2026 值（2024=0.440、2025=0.713 已有），论文 Table S2 注需三年同口径。

## 方法
- 数据源：`forward_2026_ranking.csv` 的 `_naive`（2026 池 201 条 isolate 对 H1+H3 train 人源参考的 max identity），
  与 `2026_isolates_clean.csv` 按 (accession, subtype) merge 取双标签。
- 聚合口径同 `run_multiyear_forward.py` 的 `auc_cluster_naive`：按 (subtype, cluster_id) 分组，
  簇分数 = mean(naive)，簇标签 = max(label)（ever-jump），算 AUC。
- 校验：`_naive` 的 isolate 级 AUC 复现 0.0461（与 forward_2026.json 一致）。

## 结果
- 2026 cluster 级朴素基线 AUC = **0.4643**（65 簇、2 阳性簇；jump 与 jump_human 同值，阳性簇两标签一致）。
- 三年同口径：2024 = 0.4402 / 2025 = 0.7131 / 2026 = **0.4643**。

## 结论
- 达到预期。产物：`output/forward_2026.json` 的 `naive.<label>` 增加 `auc_2026_cluster_naive`/`n_clusters`/`n_pos_clusters`；
  RESULT.md §5 多年份条目已更新。论文回填：Table S2 注 3 补 2026 值 0.464。下一步：论文侧回填。
