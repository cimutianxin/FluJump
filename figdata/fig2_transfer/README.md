# fig2_transfer — probe 容量消融（linear → MLP → transformer）跨亚型迁移数据

本目录存放论文 fig2「迁移随 probe 容量单调变差」证据链中 **MLP 头** 的机器可读数据。
linear 头见 `ESM_clf/jump_exp/output/target_val_layer_select.json`，
transformer 头见 `ESM_tf_clf/output/tf_probe_results.json`，本目录为 MLP 头补充。

生成脚本：`ESM_clf/jump_exp/train_mlp_probe.py`（协议与 linear 完全对齐，仅换 probe 头）。

## MLP 结构与训练超参

- 结构：640 → 256 → 128 → 1，ReLU，隐层间 dropout 0.1（输入为 ESM-2 150M
  各层 mean-pooled embedding，StandardScaler fit 于 H1+H3 train）
- 损失：BCEWithLogitsLoss(pos_weight = train neg/pos)
- 优化：Adam lr=1e-3，weight_decay=1e-4，batch 512，最多 200 epoch
- early stopping：isolate_split 的 H1+H3 val split，raw-logit AUC，patience 15，
  恢复最优权重；torch.manual_seed(42)

## 文件说明

### mlp_head_summary.csv（20 行，cluster 臂长表）

| 列 | 含义 |
|---|---|
| label | `label_is_jump` / `label_is_jump_human` |
| holdout | `h5` / `h7`（holdout 亚型） |
| seed | 42–46（`data/splits/h5h7_valtest/cluster_seed*.csv`） |
| selected_layer | 该 seed 下 val AUC argmax 选中的层（0–30） |
| val_auc | 选中层在该 seed H5/H7 内部 val 上的 AUC（raw logit 口径） |
| test_auc | 选中层在对应 test 上评一次的 AUC（选择不接触 test 标签） |

### mlp_head_within_dist.csv（62 行）

| 列 | 含义 |
|---|---|
| label | 同上 |
| layer | 0–30 |
| h13_test_auc | 该层 MLP 在 H1+H3 test split 上的 AUC（同分布口径，raw logit） |

## 口径（与全项目一致）

- 一切 AUC 用 **raw logit**（sigmoid 之前），禁止概率、禁止任何符号翻转
- cluster target-val 协议：H5/H7 内部按 cluster 整体切 val/test（5 seeds），
  val argmax 选层 → test 只评一次；isolate_random 臂仅用于泄露量化（见 JSON）
- H5/H7 切分行号一律经 (accession, subtype) 映射回全表行号
