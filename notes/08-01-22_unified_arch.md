# 实验：统一架构探索（层选择不碰测试标签）+ avg_mid bootstrap

- **日期**：08-01-22
- **目的**：回答"分亚型选层（H5→L28/H7→L13）写进论文会不会被质疑"——会（用测试标签选超参）。寻找层使用方式完全由训练数据决定、且 H7 方向正确的统一架构

## 方法

- `unified_arch.py`，4 个变体（H1+H3 训练，test/H5/H7 一次性评估）：
  1. `avg_all`：31 层全平均 → probe
  2. `avg_mid`：中层 10–22 平均（事前指定的范围）→ probe
  3. `concat_l1`：31 层 concat（19840 维）+ L1 probe（训练自己挑层）
  4. `elmo_mix`：ELMo 式可学习 softmax 层权重 + 线性头（torch/Adam，只优化训练 BCE）
- `verify_avg_mid_bootstrap.py`：对唯一方向正确的变体做 cluster bootstrap（B=20）。
- 输出：`output/unified_arch.json`、`output/verify_avg_mid_bootstrap.json`。

## 结果

### 四变体（H7 原始方向 AUC）

| 变体 | jump test/H5/H7 | jump_human test/H5/H7 |
|---|---|---|
| avg_all | 0.983 / 0.404 / 0.422 | 0.992 / 0.386 / 0.298 |
| **avg_mid** | 0.984 / 0.338 / **0.718** | 0.990 / 0.232 / **0.732** |
| concat_l1 | 0.990 / 0.517 / 0.344 | 0.990 / 0.357 / 0.247 |
| elmo_mix | 0.975 / 0.367 / 0.329（top 层 8/16/9） | 0.982 / 0.235 / 0.378（top 层 8/6/14） |

### avg_mid 的 cluster bootstrap（B=20）

| 标签 | H7 mean±std | 90% 区间 | P(>0.5) | 判决 |
|---|---|---|---|---|
| jump | 0.497 ± 0.109 | [0.36, 0.68] | 0.40 | 不过 |
| jump_human | 0.642 ± 0.151 | [0.45, 0.88] | 0.75 | 边缘 |

## 结论

1. **没有 test-label-free 的统一架构能稳健修复 H7**：唯一方向正确的 avg_mid 在 bootstrap 下 jump 不过、jump_human 仅边缘。headline 数字（0.72/0.73）同样有运气成分。
2. **ELMo/concat_l1 的失败有信息量**：让训练过程自己定层（无论学权重还是 L1 选择）只会收敛到对训练集有用的层（8/16/9），训练分布内信号推不出跨亚型方向——佐证 H7 反转不是优化问题。
3. 结论：统一架构路线**放弃**；论文写法转向"诚实的统一模型 + L13 作为表征分析发现"（L13 jump_human bootstrap P=1.00 依然成立，作为 finding 不需要它是统一方案）。
4. 下一步行动：**放弃**统一高分路线；按写法 C 组织论文（主模型 + 深度方向翻转分析）。
