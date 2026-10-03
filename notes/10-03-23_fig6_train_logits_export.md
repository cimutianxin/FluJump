# 实验：fig6_surveillance train logit 分布导出（2024/2025/2026 三协议）

- **日期**：10-03-23
- **目的**：对 temporal_validation 两脚本做「只加导出、不改逻辑」最小改动，导出
  三个前向协议的 train logit 向量供论文 fig6 分布图；重跑并核验既有 output 不变

## 方法
- 先备份 `output/forward_multiyear.json` 与 `output/workflow_2026_eval.json` 到 /tmp
- `run_multiyear_forward.py`：L3 块算出 lg_tr（≤cutoff train 的 decision_function）
  处追加导出 `figdata/fig6_surveillance/train_logits_{2024,2025}_{jump,jump_human}.csv`
  （列 accession, train_label, logit；train_label = ≤cutoff 簇成员重算标签）
- `run_2026_workflow_eval.py`：复用 RiskScorer 内重训的 probe（risk_scorer.py 未改），
  对 split==train 全量算 logit，导出 `train_logits_2026_{jump,jump_human}.csv`
- 重跑两脚本，核验 AUC 与提交版一致、备份 json diff 为空

## 结果
- 终端 AUC 与提交版完全一致：2024 jump cluster=0.9070、2025 jump=0.9715、
  2024 jump_human=0.7442、2025 jump_human=0.9553；
  2026 分层 jump_human 中档 8/10 命中、0/191 误报
- **git diff 核验：两 json md5 与备份逐字节一致（diff 为空）**
- 行数核验：3426 / 5262 / 5933 行，与 json 的 n_train 完全一致
- 产物：figdata/fig6_surveillance/ 6 个 CSV

## 结论
- 达到预期：导出与回归核验全部通过
- 下一步：本地 figs/fig6_surveillance/ 绘图
