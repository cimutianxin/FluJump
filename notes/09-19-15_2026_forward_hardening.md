# 实验：2026 前向加固（B4）——cluster bootstrap CI / permutation p / ≤2025 标签口径 / L1 对照层

- **日期**：09-19-15
- **目的**：完成 RESULT.md TODO-B4 三子项，并把当日早些时候的 ad-hoc 计算（`forward_2026_hardening.json`，无脚本留存）固化为可复现脚本。

## 方法

- 新脚本 `validation_exp/temporal_validation/run_2026_forward_hardening.py`（config.py 增补 `MEAN_EMB_L1` / `ALL_ISOLATES_CSV` / `B_BOOT` / `B_PERM` 常量）：
  - L3（主线层）+ L1（末层对照）× 双标签 Ridge probe，协议同 `run_2026_forward.py`（StandardScaler + GridSearchCV 6×C，5-fold，seed 42）；L3 处断言 H5 复现 AUC ∈ [0.65, 0.95]。
  - ① cluster 级统计推断：65 簇（簇内 mean logit、簇标签 max）bootstrap B=10000（弃退化）→ 95% percentile CI + P(AUC>0.5)；簇标签 permutation B=10000 → p = #(AUC_perm ≥ AUC_obs)/B。
  - ② ≤2025 标签口径：从 `all_isolates.csv` 仅用 year≤2025 有日期成员的非 unknown host 重算簇标签（口径同 build_jump_labels），报告变化数 + 新口径 cluster AUC。
  - ③ L1 对照层全指标（层选择敏感性的最小检查）。
- **踩坑（重要）**：`cluster_id` 是亚型内编号，all_isolates 中 693/908 个 id 跨亚型复用——首版按 cluster_id 单键聚合导致跨亚型宿主串扰（假标签变化 jump 29 / jump_human 12、pre2025 AUC 假降至 0.64/0.81）；改为按 **(subtype, cluster_id)** 聚合后恢复。手工核查阳性簇：(H5,70) ≤2025 成员 65 条含 human|avian|bovine，(H1,15) 2 条含 human|swine——两阳性簇标签均不依赖 2026 数据。

## 结果

| 层 | 标签 | H5 复现 | 2026 isolate | 2026 cluster | bootstrap 95% CI | P(>0.5) | perm p | pre2025 口径 cluster AUC |
|---|---|---|---|---|---|---|---|---|
| L3 | jump | 0.8255 | 0.9995 | 0.9921 | [0.953, 1.0] | 1.0000 | 0.0008 | 0.9921 |
| L3 | jump_human | 0.7936 | 0.9995 | 1.0000 | [1.0, 1.0] | 1.0000 | 0.0002 | 1.0000 |
| L1 | jump | 0.6565 | 0.8969 | 0.9841 | [0.937, 1.0] | 1.0000 | 0.0020 | 0.9841 |
| L1 | jump_human | 0.5779 | 0.3141 | 0.7302 | [0.381, 1.0] | 0.7959 | 0.1494 | 0.7302 |

- **≤2025 标签变化数 = 0（双标签）**——循环风险排除。附带事实：30/65 个 2026 簇无 ≤2025 成员（2026 新簇，两口径下均为阴性，不影响阳性结论）。
- 与当日 ad-hoc JSON 核对：所有确定性数值（AUC/CI/阳性名次）完全一致，随机统计量在噪声内吻合（perm p 0.0008 vs 0.0009 等）→ ad-hoc 结果确认有效，且现已可复现。
- H5 复现 L3 jump = 0.8255（08-28 `run_2026_forward.py` 记录 0.798）：差异在 GridSearchCV 选 C 的数值层面（C=10 vs 其他），2026 headline（0.9995/0.9921/1.0）四有效数字一致，断言通过。
- L1 对照：jump 仍强（cluster 0.984，perm p=0.002），但 **jump_human 在末层失效**（isolate 0.314 / cluster 0.730，perm p=0.149 不显著，阳性 9/10 掉到 147 名后）——与 §4"末层方向反转"自洽，同时说明 2026 jump_human 结论确实依赖中层（L3）选择，论文需如实披露。
- 功效限制不变：阳性仅 2 簇 / 10 条，所有推断表述为排序验证。

## 结论

- B4 三子项**全部完成**：① cluster bootstrap CI（L3 jump [0.953,1.0] / jump_human [1.0,1.0]）+ permutation p（0.0008 / 0.0002，均远小于 0.05）；② ≤2025 口径 0 标签变化、AUC 不变；③ L1 对照显示 jump 稳健、jump_human 末层失效（披露点而非反驳点）。
- **下一步**：更新 RESULT.md §5 与 TODO-B4（本次执行）；B 组剩余 B5（feline 误判排查）/ B6 / B7。
