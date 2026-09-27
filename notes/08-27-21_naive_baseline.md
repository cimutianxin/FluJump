# 实验：朴素生物学基线（人源序列相似度）vs ESM-2 linear probe

- **日期**：08-27-21
- **目的**：RESULT.md §1 待补第一项——检验 probe 的跨亚型迁移能力是否只是"与已知人源分离株全局序列相似度"的复述（go/no-go 检验）。

## 方法

- 新目录 `validation_exp/naive_baseline/`（config.py + run_naive_baseline.py + output/），模仿 validation_exp 现有模式。
- 打分：query 与 **H1+H3 train split 人源参考集**（3,476 条）的最大 pairwise identity（`max_id_human`）、top-5 均值（`topk_id_human`）、人源−非人源差值（`diff_id`，非人源参考 2,457 条）。参考集禁用 H5/H7 人源序列（人源身份是 jump_human 标签定义成分，防泄漏）。
- identity 在 MAFFT 对齐序列（H3 参考株统一坐标，长度 1039）上按双非 gap 位匹配比例计算，numpy 分块向量化（194s）。
- 评估：test / h5_holdout / h7_holdout × label_is_jump × label_is_jump_human，cluster 级 bootstrap（1000 次）CI 与 P(AUC>0.5)。
- 参照值：ESM-2 linear probe H5 0.79（L3）；H7 target-val 选层 0.892/0.854（**原始口径、无符号翻转**——中层 L17/L13 在 H7 上原始方向即为正；09-19 标签修复后重跑为 0.918/0.854，协议审计见 `09-27-01_h7_flip_protocol_audit.md`）。

## 结果

| split | 标签 | max_id_human 原始 AUC | 翻转 AUC | boot CI（原始） | P(>0.5) |
|-------|------|----------------------|----------|----------------|---------|
| test | jump | 0.603 | 0.397 | [0.472, 0.706] | 0.945 |
| test | jump_human | 0.645 | 0.355 | [0.530, 0.737] | 0.992 |
| **h5** | jump | **0.325** | **0.675** | [0.233, 0.417] | 0.000 |
| **h5** | jump_human | **0.336** | **0.664** | [0.242, 0.444] | 0.000 |
| **h7** | jump | **0.238** | **0.763** | [0.142, 0.428] | 0.006 |
| **h7** | jump_human | **0.158** | **0.842** | [0.087, 0.243] | 0.000 |

- topk_id_human 与 max_id_human 几乎一致（H7 jump_human 翻转 0.879）；diff_id 全面更差。
- identity 分布：H5 query 与人源 H1/H3 参考 max identity 中位 0.649；H7 中位 0.521（跨 Group 更低，符合预期）。
- sanity check：test split AUC 0.60/0.65，显著 >0.5 但不算高——全局相似度在同亚型内只有弱排序信号。

## 结论

**GO（主假设成立，但有重要 nuance）：**

1. **H5 上基线失败**：朴素相似度在 H5 holdout 上原始方向 AUC 仅 0.33（方向都错了），即使事后翻转也只有 0.675，明显低于 ESM probe 的 0.79。probe 的 H5 迁移能力**不是**全局序列相似度的复述 → 卖点成立（趋同适应是组合信号，全局相似度捕捉不到）。
2. **H7 nuance 需诚实对待**：基线在 H7 上同样方向反转（原始 0.12–0.24），翻转后 0.76–0.88，与 probe 翻转后（0.85–0.89）**相当**。说明 H7 的反转信号相当程度可由全局相似度解释——"Group 2 边界"的几何根因（亚型身份主导 embedding）本身就与序列分歧同源。这支持 §3 的"Group 边界 = 系统发育边界"叙事，但意味着 H7 结果不能单独作为 probe 优越性的证据；**H5（同 Group 1 内迁移、方向无需翻转）才是 probe 价值的核心证据**。
3. 论文表述建议：§1 主打"H5 上 probe > 朴素基线"；§3 可引用"H7 反转连朴素相似度都复现"作为 Group 边界是深层生物结构的佐证而非数值伪影。

- **下一步行动**：继续。§1 此项可标记完成（结果确定性计算、bootstrap 已过、日志已写）；建议下一步做 §2 位点归因或 §3 H10 扩展。

## 涉及文件

- 新建：`validation_exp/naive_baseline/config.py`、`run_naive_baseline.py`、`output/naive_baseline_results.json`、`output/run.log`
