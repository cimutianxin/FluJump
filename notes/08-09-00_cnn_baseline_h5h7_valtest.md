# 实验：CNN baseline 在 h5h7_valtest 新切分下的重训与重评估

- **日期**：08-09-00
- **目的**：把 Borkenhagen CNN baseline 纳入 target-val 新切分体系，与 ESM probe（`notes/08-09-00_target_val_layer_select.md`）在完全相同的 h5_test/h7_test 上对比。

## 方法

- **前置修复**：08-02 对齐重建后 `aligned_ha_seq` 全量等长 1039（CNN 旧 checkpoint 按旧对齐 581 训练，已不兼容）。`config.py` ALIGNED_LENGTH 581→1039；`model.py` input_length 改用配置常量（原硬编码 581）；旧 checkpoint 备份至 `output/models/backup_aligned581/`。
- 重训 stage1（宿主预训练，early stop epoch 57，best val AUC 0.9942）+ stage2（两标签微调；选择全部在 H1+H3 val，未碰 H5/H7，新协议下天然合规）。H1+H3 test：jump AUC 0.994 / jump_human 0.996。
- 新脚本 `eval_h5h7_valtest.py`：与 ESM 实验同构——10 切分文件 × 4 mask、两臂 5-seed mean±std、泄露差值、cluster_seed42 test cluster bootstrap（B=1000）、全量 holdout 旧口径参照。
- 输出：`output/eval_h5h7_valtest.{json,log}`，重训日志 `stage1_retrain.log` / `stage2_retrain.log`。

## 结果

### CNN 在新切分下的 test AUC（5 seeds，cluster 臂为主）

| 标签 | 亚型 | cluster 臂 | isolate_random 臂 | bootstrap P(>0.5)（seed42） |
|---|---|---|---|---|
| jump | H5 | 0.458 ± 0.126 | 0.462 ± 0.012 | 0.28 |
| jump | H7 | 0.399 ± 0.066 | 0.415 ± 0.014 | 0.41 |
| jump_human | H5 | 0.597 ± 0.120 | 0.550 ± 0.011 | 0.61 |
| jump_human | H7 | 0.410 ± 0.045 | 0.410 ± 0.004 | **0.00** |

全量 holdout（旧口径）：jump H5 0.459 / H7 0.416；jump_human H5 0.549 / H7 0.412。
泄露差值（isolate − cluster）：均 ≤0.015，jump_human H5 为 −0.046（反向）。

### 与 ESM probe 对比（同一切分、cluster 臂 5 seeds）

| 标签 | 亚型 | CNN | ESM（val 选层） |
|---|---|---|---|
| jump | H5 | 0.458 ± 0.126 | **0.786 ± 0.040** |
| jump | H7 | 0.399 ± 0.066 | **0.892 ± 0.027** |
| jump_human | H5 | 0.597 ± 0.120 | 0.688 ± 0.103 |
| jump_human | H7 | 0.410 ± 0.045 | **0.854 ± 0.060** |

## 结论

1. **CNN 跨亚型迁移全面低于随机水平附近**（四格 P(>0.5) ≤ 0.61，H7 jump_human P=0.00），在同一切分上与 ESM 差距 0.25–0.49 AUC——ESM 的优势在新协议下不但保持，且比旧口径（CNN 0.54/0.66）更大。
2. CNN 两臂泄露差值 ≈0，符合预期：模型本身没学到可迁移信号时，切分方式无从泄露。反衬 ESM 表中 jump_human H5 的 +0.112 是真实存在的泄露信号。
3. 注意旧 eval_results.json 的 0.54/0.66 是旧对齐（581）上的数，与当前数据不可比；新口径以 `eval_h5h7_valtest.json` 为准。
4. 下一步行动：**继续**——论文 baseline 对比表按本日志的 cluster 臂数字写；如需旧口径对照可重跑 `evaluate.py`（当前 checkpoint 已兼容新数据）。

## 涉及文件

- 修改 `Borkenhagen/cnn_baseline/config.py`（ALIGNED_LENGTH 1039）、`model.py`（input_length 用配置）
- 新建 `Borkenhagen/cnn_baseline/eval_h5h7_valtest.py`
- `output/models/backup_aligned581/`（旧 checkpoint 备份）；新 checkpoint `stage1_best.pt` / `stage2_*_best.pt`
- 输出 `output/eval_h5h7_valtest.{json,log}`、`stage1_retrain.log`、`stage2_retrain.log`
