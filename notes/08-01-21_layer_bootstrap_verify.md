# 实验：关键层 cluster bootstrap 验证

- **日期**：08-01-21
- **目的**：层扫描在 31 层 × 2 标签中挑最优存在多重比较风险，且结果对 C 敏感（L26 固定 C=1.0 时 H7 从 0.705 掉到 0.455）；用 cluster bootstrap 验证 headline 层的稳健性

## 方法

- `verify_layers_bootstrap.py`：训练集按 cluster 有放回重采样（B=20），与层扫描相同的 GridSearchCV 流程重训，报 H5/H7 AUC 分布、90% 区间与 P(AUC>0.5)。
- 候选：jump L26 / L17、jump_human L13（均为 H1+H3 训练）。
- 输出：`output/verify_layers_bootstrap.json`。

## 结果

| 配置 | 原始 H7 | bootstrap H7 mean±std | 90% 区间 | P(>0.5) | 判决 |
|---|---|---|---|---|---|
| jump L26 | 0.705 | 0.481 ± 0.181 | [0.25, 0.73] | 0.40 | **淘汰**（原始值是运气） |
| jump L17 | 0.893 | 0.666 ± 0.197 | [0.31, 0.88] | 0.80 | 倾向正向，不够稳 |
| jump_human L13 | 0.863 | **0.798 ± 0.094** | [0.66, 0.91] | **1.00** | **稳健** |

附带：三者的 H5 bootstrap 都差（0.38–0.62，P(>0.5) ≤ 0.75）——H5↔H7 trade-off 在 bootstrap 下依然存在。

## 结论

1. **jump_human 的第一个稳健解**：layer 13（中层）embedding + Ridge LR，H7 原始方向 AUC ≈ 0.80，20/20 bootstrap 全为正——这是目前唯一不需要手动翻转、统计上站得住的 H7 jump_human 预测器。代价：H5 迁移失效（0.38），两个 holdout 不能用同一层。
2. jump 任务尚无稳健层解（L26 淘汰，L17 边缘）。
3. 下一步行动：**继续**——等 LoRA 微调结果，看能否用单一模型同时保住 H5 和 H7。
