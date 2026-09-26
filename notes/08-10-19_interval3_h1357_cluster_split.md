# 实验：interval 3 类 — H1357 全体 + cluster 级 70/15/15 划分（Transformer 头）

- **日期**：2026-08-10-19
- **目的**：interval 3 类任务从「H1+H3、isolate 级划分」切换为「H1+H3+H5+H7 全体、cluster 级 70/15/15 划分」（消除 cluster 泄漏），用同一 Transformer 头协议重训评估

## 方法

- 新划分 `data/splits/interval_h1357_cluster_split.csv`（新文件，不动 isolate_split.csv）：
  - 总体：H1357 & jump=1 & interval 有效 = 1191 条 / 44 clusters
  - cluster 大小悬殊（2–115），两阶段分层切分会得到 55/19/26 且 val 无 3yr+ → 改为**随机搜索均衡划分**（5000 候选，约束 val/test 三类+四亚型俱全，最小化样本/类别占比偏差，seed=42 可复现）
  - 结果：train 834 (70.0%) / val 172 (14.4%) / test 185 (15.5%)，三类分布 train 708/84/42，val 154/11/7，test 159/7/19
- `interval_data.load_interval_data` 新增 `subtypes` / `split_csv` 参数（默认行为不变；embedding 行号映射固定由 isolate_split.csv 建立）
- 训练协议与上一版完全相同：L13/17/28 × 3 seeds，CE + balanced class_weight（[0.393, 3.31, 6.619]），val bal_acc early stop + 选层，test 单次评估

## 结果

val（3 seed 集成）：L13 0.459 / **L17 0.520 ★** / L28 0.351

**TEST（L17，n=185）**：
- acc=0.335 [0.270, 0.400]，**bal_acc=0.424 [0.402, 0.446]**，macro_f1=0.237 [0.189, 0.281]
- 基线 bal_acc：majority 0.333 / random 0.333
- per-class recall：<1yr 0.270 / **1-3yr 0.000** / 3yr+ 1.000（模型塌缩到高权重类 3yr+）
- 分亚型：仅 H1 在 test 有多类（bal_acc 0.333，n=122）；H3/H5/H7 test 子集单类无法算

对比上一版（H1+H3、isolate 级划分、test n=100）：bal_acc 0.757 → **0.424**。

## 结论

- **cluster 泄漏是此前高指标的主要来源**：换成无泄漏的 cluster 级划分后，bal_acc 从 0.757 跌到 0.424（仅略超 0.333 随机基线），与 08-02 日志中"cluster 跨 split 泄漏保留、数值偏乐观"的警告一致。
- 训练动态不健康：balanced class_weight 下模型 13–19 epoch 即 early stop，test 上塌缩到 3yr+（recall=1.0）而 1-3yr 全灭；H5/H7 与 H1 的 interval 标签分布差异大（3yr+ 几乎全部在 H1），跨亚型泛化差也贡献了下坠。
- **下一步行动**：调整 — ① 缓解类别权重（sqrt 反比或 capped）；② 检查 H1-only vs 跨亚型训练的差异；③ 考虑 cluster 级划分下信号本就弱，可对比 mean-pooled 线性 probe 同协议数值，判断 transformer 头是否过拟合。

## 涉及文件

| 文件 | 操作 |
|------|------|
| `datascripts/build_interval_h1357_cluster_split.py` | 新建 — cluster 级随机搜索均衡划分 |
| `data/splits/interval_h1357_cluster_split.csv` / `.json` | 新划分文件（1,191 行） |
| `ESM_clf/interval_exp/interval_data.py` | 增量修改 — `subtypes` / `split_csv` 参数 |
| `ESM_tf_clf/train_tf_interval.py` | 修改 — 切到 H1357 cluster 划分 + 分亚型指标 |
| `ESM_tf_clf/output/tf_interval3_h1357_results.json` / `.log` | 新结果（旧 `tf_interval3_results.json` 保留） |
| `data/README.md`、`datascripts/README.md` | 登记新划分文件 |
