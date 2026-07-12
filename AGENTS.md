## 笔记规则

当用户说"做笔记"或"记笔记"时，按以下步骤执行：

1. 如果用户没有指定主题名，从对话中推断（取核心任务的关键词），并向用户确认。
2. 在 `expnotes/` 下创建 `MM-DD-HH-主题.md`（例：`07-10-15-蛋白质结构预测.md`），小时用 24 小时制。
3. 文件内容按以下模板输出（缺失的章节写"无"）：

| 章节 | 内容 |
|------|------|
| 背景与目标 | 本次要解决的问题或实验目的 |
| 关键步骤 | 按时间顺序列出命令、代码修改、工具调用 |
| 结果与发现 | 重要输出、数据、报错、解决方式 |
| 涉及文件 | 本次修改/创建的文件路径列表 |
| 待办 | 下一步需要做的事 |

4. 只提取对话中的实质信息，不要复制闲聊或无意义轮次。
5. 创建后告知用户文件路径，并询问是否需要修改内容。

---

## 实验日志

**每次实验完成后必须写实验日志。** 日志存放于 `experiments/` 目录，文件名格式：

```
experiments/MM-DD-HH_<简短描述>.md
```

### 日志内容模板

```
# 实验：<简短描述>

- **日期**：MM-DD-HH
- **目的**：本次实验要验证什么

## 方法
- 模型/参数/数据配置的变更要点

## 结果
- 关键指标（必须有数值）
- 与 baseline 对比（如有）

## 结论
- 是否达到预期
- 下一步行动（继续 / 调整 / 放弃）
```

### 日志要求

| 要求 | 说明 |
|------|------|
| 每次实验必写 | 无论结果好坏，哪怕只跑了几行代码 |
| 文件名带日期 | 便于按时间排序和回顾 |
| 关键指标必须有数值 | 不能只写"效果好" |
| 结论明确标注下一步行动 | 继续 / 调整 / 放弃 |

---

## 项目结构

```
FluJump/
├── AGENTS.md                       # 本文件（AI Agent 行为规范）
├── README.md                       # 项目说明（Agent 只读，禁止修改）
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据（禁止修改）
│   ├── processed/                  # 清洗后数据
│   ├── processed_isolate/          # isolate 处理数据
│   ├── processed_isolate_MAFFT/    # MAFFT 对齐数据
│   ├── splits/                     # 训练/测试集划分
│   ├── dataset_borkenhagen/        # Borkenhagen 数据集
│   └── dataset_borkenhagen_raw/    # Borkenhagen 原始数据
├── datascripts/                    # 数据处理脚本
├── mainpipeline/                   # 主流程代码（性能优化时先备份旧版本）
├── expnotes/                       # 对话笔记（MM-DD-HH-主题.md）
├── experiments/                    # 实验日志（MM-DD-HH_描述.md）
├── Alberts/                        # Alberts 2024 复现
├── Borkenhagen/                    # Borkenhagen 2024 复现（CNN baseline）
├── Chen/                           # Chen 相关
└── ESM_clf/                        # ESM 分类器实验
```

---

## 设计决策

### Split 策略：不使用 Cluster-stratified Split

**决策**：当前 `isolate_split.csv` 按 **isolate 级别** 随机划分 train/val/test，**不按 cluster 隔离**。同一 cluster 的多个 isolate 可能分散在不同 split 中。

**理由**：本项目的使用场景是 **"给定一条新序列，预测其 interval/jump"** —— 输入是单条 HA 序列，输出是该序列对应的预测值。在这种场景下：

1. **每个 isolate 是一个独立样本**。虽然同一 cluster 的 isolate 共享 CD-HIT 聚类标签，但它们的序列不同（identity ≥ 阈值但非相同），且 `iso_interval_days` 是 isolate 级别独立计算的。
2. **Cluster 级 split 会破坏数据分布**。37 个 cluster 大小差异极大（最大 465 条，最小 1 条），按 cluster split 会导致 train/val/test 的类别分布极不均衡，小样本类别（如 5yr+ 仅 7 条）可能完全集中在某个 split。
3. **实践中的评估场景**：给定一条新序列，模型不应知道它属于哪个 cluster（实际部署时没有 cluster 信息），因此 isolate 级别的随机 split 更贴合真实使用方式。

**已知代价**：同一 cluster 的 isolate 序列高度相似，split 间存在一定信息泄漏。这会导致评估指标**略微偏高**，但考虑到任务难度（R²≈0.1），当前指标已足够保守，泄漏影响有限。如果未来指标显著提升（如 R²>0.3），需重新审视此决策。

---

## 其他规则

- Agent 可以阅读 `README.md`，但不能修改 `README.md`。
- `mainpipeline/` 中存储主流程代码，如果有更好性能的方法，则备份之前的主流程到 `mainpipeline/backup/` 后再替换。
