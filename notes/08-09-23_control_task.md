# 实验：control task（打乱标签对照）— linear vs transformer 的容量归因

- **日期**：08-09-23
- **目的**：Hewitt–Liang 式对照，检验"linear 的迁移 AUC 来自表示、transformer 的同分布高分来自 probe 容量"的论断。

## 方法

- `reviewer_exp/control_task/`：`run_control_linear.py`（L13/17/28 × 2 标签 × 5 shuffle seeds，仅置换 H1+H3 train 标签，GridSearchCV 同正式流程）；`run_control_tf.py`（L17/28 × 2 标签 × 5 shuffle seeds，固定 30 epoch 不早停测容量，同时跟踪 best-val 状态近似 early-stop 协议）。
- 读出：fit_train_auc（对训练所用标签的拟合度=容量）+ 各处真实标签 AUC（应≈0.5）。
- 输出：`output/control_linear.json`、`output/control_tf.json`。

## 结果

### linear（real vs shuffled 5 seeds 范围）

| 配置 | fit_train（容量） | real_test | real_h7 |
|---|---|---|---|
| real | 0.995–0.998 | 0.984–0.992 | 0.21–0.89（层间差异，同正式实验） |
| shuffled | **0.59–0.83**（均值 ~0.73） | 0.38–0.73 | **0.20–0.92（散布极大）** |

### transformer（固定 30 epoch）

| 配置 | fit_train（容量） | bestval real_test | bestval real_h7 |
|---|---|---|---|
| real | 0.985–0.991 | 0.974–0.986 | 0.19–0.56 |
| shuffled | **0.53–0.56** | 0.67–0.82 | 0.19–0.77 |

## 结论

1. **两种 probe 的 selectivity 都为正**：real fit（0.98–0.99）远超 noise fit（linear ≤0.83，tf ≤0.56）——两者的同分布高分都不是纯容量产物。
2. **"transformer 记忆噪声"假设不成立**：30 epoch 预算内 tf 对噪声的拟合（0.53–0.56）甚至低于 linear（0.59–0.83）。此前"tf 测的是 probe 容量"的表述需修正：tf 迁移崩塌的原因是**拟合了训练分布内真实但不可迁移的结构**，不是裸容量记忆。论文措辞应改为"head 复杂度拟合分布内信号"。
3. **意外发现 1（重要）**：shuffled 训练的 probe 在 H7 真实标签上的 AUC 散布 0.20–0.92——cluster 结构数据上的 null 方差远大于 iid 假设。这独立证明了 cluster bootstrap / 多 seed 协议的必要性，可作论文协议一节的实证动机。
4. **意外发现 2**：shuffled 下 best-val 早停能从噪声中"选出" 0.67–0.82 的 test AUC——自适应模型选择本身会制造表观信号，对任何基于 val 选择的协议都是警示。
5. 下一步行动：**继续**——control task 按上述修正后的措辞写入论文；null 散布大这一发现并入评估协议论证。

## 涉及文件

- 新建 `reviewer_exp/control_task/`（config / run_control_linear / run_control_tf）
- 输出 `reviewer_exp/control_task/output/control_{linear,tf}.{json,log}`
