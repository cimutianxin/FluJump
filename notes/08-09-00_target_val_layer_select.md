# 实验：H5/H7 target-val 层选择（val 选层、test 只评一次）+ cluster/isolate 泄露对比

- **日期**：08-09-00
- **目的**：替代"全量 H5/H7 看 AUC 选层"的做法——把 H5/H7 内部 5:5 切出 val（选层）/ test（只评一次），使分亚型层选择合规；并用 cluster 臂 vs isolate 随机臂量化近重复序列横跨 val/test 的泄露量。

## 方法

- 新切分体系 `data/splits/h5h7_valtest/`（`datascripts/build_h5h7_valtest_splits.py`）：
  - cluster 臂（主）：亚型内 cluster 整体归入 val/test，stratify by (jump, jump_human) 组合，seeds 42–46
  - isolate_random 臂（对比）：按行随机 5:5，同 seeds
  - H5: 1,747 isolate / 441 cluster；H7: 838 / 248（aligned 表口径，与 isolate_split 一致）
- 实验脚本 `ESM_clf/jump_exp/target_val_layer_select.py`：probe 只依赖 (label, layer) 与 H1+H3 train，故每 (label, layer) 仅训 1 次 GridSearchCV（62 次），在 10 个切分文件 × 4 mask 上评 AUC；每 (文件, 标签, 亚型) val argmax 选层 → test 评一次；cluster_seed42 选中配置做 test 侧 cluster bootstrap（B=1000）。
- 输出：`output/target_val_layer_select.json` + `.log`。

## 结果

### 主表：val 选层 → test（5 seeds）

| 臂 | 标签 | 亚型 | 选层频率 | test AUC |
|---|---|---|---|---|
| cluster | jump | H5 | L28 ×4, L11 ×1 | 0.786 ± 0.040 |
| cluster | jump | H7 | **L17 ×5** | 0.892 ± 0.027 |
| cluster | jump_human | H5 | L11 ×3, L1/L28 各1（不稳） | 0.688 ± 0.103 |
| cluster | jump_human | H7 | **L13 ×5** | 0.854 ± 0.060 |
| isolate_random | jump | H5 | L28 ×5 | 0.802 ± 0.006 |
| isolate_random | jump | H7 | L17 ×5 | 0.893 ± 0.014 |
| isolate_random | jump_human | H5 | L28 ×5 | 0.799 ± 0.013 |
| isolate_random | jump_human | H7 | L13 ×5 | 0.865 ± 0.007 |

### 泄露量化（isolate_random − cluster test AUC 均值差）

jump H5 +0.016 / jump H7 +0.001 / **jump_human H5 +0.112** / jump_human H7 +0.011

### cluster_seed42 test cluster bootstrap（B=1000）

| 标签 | 亚型 | 层 | AUC | 90% CI | P(>0.5) |
|---|---|---|---|---|---|
| jump | H5 | L28 | 0.820 | [0.626, 0.892] | 0.98 |
| jump | H7 | L17 | 0.880 | [0.585, 0.939] | 0.97 |
| jump_human | H5 | L1 | 0.483 | [0.294, 0.724] | 0.42 |
| jump_human | H7 | L13 | 0.893 | [0.814, 0.961] | **1.00** |

## 结论

1. **val 选层协议收敛到与历史 headline 相同的层**（H5 jump→L28、H7 jump→L17、H7 jump_human→L13），但现在选择过程完全合规（不碰 test 标签），可直接写进论文方法。H7 两个标签的选择 5/5 seed 稳定。
2. **泄露主要集中在信号最弱处**：jump_human H5 泄露 +0.112，且 cluster 臂该格选择不稳（L11/L1/L28 漂移）——近重复对子让 isolate 臂误以为 L28 稳选。其余三格泄露 ≤0.016，说明此前对这三格的结论基本不受切分方式影响。
3. **H5 jump_human 是真弱**，不是切分假象：cluster 臂 seed42 选出 L1 后 test AUC 仅 0.483（P(>0.5)=0.42），5-seed 均值 0.688±0.103 高度依赖选层运气。论文应对 H5 jump_human 降级表述或转表征分析。
4. H7 jump_human L13 bootstrap P(>0.5)=1.00，作为最稳的 finding。
5. 下一步行动：**继续**——论文按 target-val 协议写层选择；H5 jump_human 单独讨论其不稳定性。

## 涉及文件

- 新建 `datascripts/build_h5h7_valtest_splits.py`、`ESM_clf/jump_exp/target_val_layer_select.py`
- 新建 `data/splits/h5h7_valtest/`（10 个切分 CSV）
- 修改 `ESM_clf/jump_exp/config.py`（+H5H7_SPLIT_DIR）、`data/README.md`、`datascripts/README.md`
- 输出 `ESM_clf/jump_exp/output/target_val_layer_select.{json,log}`

## 追加（09-27）：09-19 标签修复后重跑版本

09-19 host 标签修复（7 簇 jump 1→0、jump_human 0 翻转，见 `09-19-16_feline_host_audit.md`）后本脚本被重跑（JSON mtime 09-19 13:49），磁盘 `target_val_layer_select.json` 为修复后版本，本 .md/.log 仍为 08-09 原版：jump H7 **0.918±0.023**（L17×5；seed42 bootstrap 0.9205 [0.834, 0.957] P(>0.5)=1.00）、jump H5 **0.836±0.018**（选层稳定为 L28×5）；jump_human 两格数值不变（0.854±0.060、P=1.00 等）。RESULT.md §1/§3/§4 已于 09-27 统一刷新为修复后数值。协议审计（确认原始口径、全程无符号翻转）：`09-27-01_h7_flip_protocol_audit.md`。
