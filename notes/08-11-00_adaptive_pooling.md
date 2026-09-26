# 实验：jump 分类 — ESM 与 linear 之间换 attention pooling（全层扫描）

- **日期**：2026-08-11-00
- **目的**：在 mean-pool + Ridge LR（linear probe）与 transformer 头之间插入最小容量档——per-residue → attention pooling（可学习 query，masked softmax）→ 单层 linear，隔离"自适应池化"本身的贡献；候选层扩到全部 31 层

## 方法

- 模型 `AttentionPoolClassifier`（ESM_tf_clf/model.py）：可学习 query（640 维）打分 + masked softmax 加权求和 + Linear(640→1)，**仅 1281 参数**，与 Ridge LR 同量级。
- 候选层：全部 31 层 hidden states（与 linear probe 层扫描一致）。
- 资源约束：磁盘仅 5.6GB 无法落盘新层；容器 **cgroup 内存上限 62GB**（/proc/meminfo 的 500GB 是宿主机值，不可信）→ 分批常驻 RAM（每批 5 层 × 7 批，每批重新 ESM 前向 ~3 min），只写结果 JSON。
- 协议与 train_tf_probe.py 完全一致：H1+H3 train 训练 + val AUC early stop，3 seed logit 集成；10 个 h5h7 切分文件 × 2 臂；val AUC argmax 选层 → test 评一次；cluster_seed42 cluster bootstrap（B=1000）。
- 正确性抽查：每层 mean-pool 与 `esm_emb_150M_all_layers.npy` 同层余弦相似度 = 1.0000 ✓

## 结果

三方法并排（cluster 臂 test AUC mean，5 seeds）：

| 标签 | 亚型 | linear | **attnpool** | tf 头 |
|------|------|--------|--------------|-------|
| label_is_jump | h5 | **0.786** | 0.612 | 0.598 |
| label_is_jump | h7 | **0.892** | 0.776 | 0.768 |
| label_is_jump_human | h5 | **0.688** | 0.635 | 0.617 |
| label_is_jump_human | h7 | **0.854** | 0.790 | 0.439 |

- 选层：h5 倾向 L28/29，h7 一致选 L17（与历史规律一致）。
- 泄露量化（isolate_random − cluster）：+0.03 / −0.00 / +0.08 / −0.01，与 linear 臂量级相当。
- cluster bootstrap（cluster_seed42）：h5 jump L28 AUC=0.697 [0.586, 0.815]；h7 jump_human L17 0.783 [0.652, 0.881]。

## 结论

- **自适应池化带来轻微增益（相对 tf 头）但仍全面落后于 mean-pool linear probe**：attnpool > tf 头（4 项全优，尤其 jump_human/h7 0.79 vs 0.44），但 4 项全部输给 linear（差 0.06–0.17 AUC）。
- 容量阶梯结论明确：**linear(mean-pool) > attention pool > transformer 头**，跨亚型迁移下容量越大越差——可学习 pooling 会对训练分布（H1+H3）的序列位置/残基模式过拟合，损害泛化。ESM 冻结表示 + 最简单线性头仍是本任务最优。
- **下一步行动**：放弃——不再沿"加容量"方向探索；如继续提升，方向应是表示层选择/正则化/数据侧（cluster 加权、方向校正），而非更复杂的头。

## 涉及文件

| 文件 | 操作 |
|------|------|
| `ESM_tf_clf/model.py` | 增量 — 新增 `AttentionPoolClassifier` |
| `ESM_tf_clf/train_adaptive_pool.py` | 新建 — 分批 RAM 提取 + 全层扫描训练 + target-val 评估 + 三方对比 |
| `ESM_tf_clf/output/adaptive_pool_results.json` / `.log` | 新结果 |
| `AGENTS.md` | ESM_tf_clf 条目补新脚本 |

## 踩坑记录

- HF hub 断网：需 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 走本地缓存。
- 容器 cgroup 内存上限 62GB（`/sys/fs/cgroup/memory.max`），`free`/`/proc/meminfo` 显示宿主机值；大批量 RAM 方案必须先查 cgroup。
- 管道 `| grep | tee` 会掩盖被 kill 进程的退出码，长任务用 `set -o pipefail`。
