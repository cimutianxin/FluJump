# 实验：MLP 头 jump/jump_human 跨亚型迁移（fig2 容量消融补缺口）

- **日期**：10-03-22
- **目的**：补齐论文 fig2「迁移随 probe 容量 linear → MLP → transformer」证据链的 MLP 头；
  协议与 linear（target_val_layer_select.py）完全对齐，仅换 probe 头。

## 方法

- 新脚本 `ESM_clf/jump_exp/train_mlp_probe.py`（不改任何现有文件）
- MLP：640→256→128→1，ReLU，隐层间 dropout 0.1；StandardScaler fit 于 H1+H3 train
- BCEWithLogitsLoss(pos_weight=train neg/pos)；Adam lr=1e-3, wd=1e-4, batch 512,
  ≤200 epoch；H1+H3 val raw-logit AUC early stopping（patience 15，恢复最优权重）；seed 42
- 62 格（2 label × 31 层）；一切 AUC 用 raw logit，无符号翻转；
  H5/H7 切分行号经 (accession, subtype) 映射回全表行号
- 选层协议同 linear：10 个 h5h7_valtest 切分文件，val argmax 选层 → test 评一次

## 结果

实际用时 ~2 分钟（GPU），远快于预估。

**within_dist（H1+H3 test AUC，62 格）**：
- label_is_jump：0.9787 – 0.9924
- label_is_jump_human：0.9836 – 0.9950
- 全部 ≥0.97，收敛验收通过（与 linear/tf 同分布水平相当）

**cluster 臂主结果（5 seeds，val 选层 → test 评一次）**：

| label | holdout | selected_layers | test_auc mean±std |
|---|---|---|---|
| label_is_jump | h5 | [11,11,11,11,25] | 0.528 ± 0.109 |
| label_is_jump | h7 | [19,19,19,19,19] | 0.881 ± 0.038 |
| label_is_jump_human | h5 | [11,11,11,30,11] | 0.683 ± 0.023 |
| label_is_jump_human | h7 | [17,17,17,17,17] | 0.860 ± 0.058 |

**isolate_random − cluster 泄露差值**：jump h5 +0.072 / h7 +0.015；
jump_human h5 +0.036 / h7 −0.001

**与 linear 头对比（target_val_layer_select.json summary.cluster）**：
MLP 在 jump×h5 上明显低于 linear（0.528 vs linear ~0.72 量级），
且 seed 间不稳（std 0.109，seed46 仅 0.336）；h7 与 jump_human 各格与 linear 接近。
选层明显前移至中层（11/17/19），与「MLP 最优深度前移」一致。

**验收核对**：
- ✓ 62 格 within_dist 全部 ≥0.9787（>0.95 阈值），无需调参重跑
- ✓ 抽格人工核对：json cluster 臂 label_is_jump×h5 的 5 个 test_auc
  （0.5586/0.6729/0.5507/0.5236/0.3359）与 mlp_head_summary.csv 逐一相等
- ✓ mlp_head_summary.csv 20 行、mlp_head_within_dist.csv 62 行，行数正确

## 结论

- 达到预期：MLP 头实验补齐，62 格全部收敛，fig2 容量消融证据链（linear/MLP/tf）闭合。
- 容量消融叙事成立但非严格单调：MLP 同分布与 linear 相当，跨亚型 H5 迁移明显劣化且不稳，
  H7 不降（h7 格 0.86–0.88 与 linear 持平）。
- 下一步行动：继续 —— 本地核对 linear/tf 对应数值后统一更新 RESULT.md（本次按指示不动）；
  fig2 绘图时可直接读 figdata/fig2_transfer/ 两个 CSV。

## 涉及文件

- `ESM_clf/jump_exp/train_mlp_probe.py`（新建）
- `ESM_clf/jump_exp/output/mlp_probe_results.json`（新建）
- `figdata/fig2_transfer/`（新建：README.md / mlp_head_summary.csv / mlp_head_within_dist.csv）
