# 实验：CNN 选择预算对齐（+ 发现并修复 sigmoid 饱和低估）

- **日期**：09-12-12
- **目的**：RESULT.md §1 最后待补项。ESM 侧享受 target-val 选层自由度（31 层选 1），CNN 此前只用单一 checkpoint 最终 logit，存在"优势来自选择自由度而非表示质量"的替代解释。给 CNN 同等预算：8 个候选表示 probing + 同一 target-val 选择协议。

## 方法

- 候选表示（`extract_intermediate_features.py`，borkenhagen env）：conv1–5 block 输出 mean-pool（32/64/128/256/512 维；stage2 freeze_conv → 三 checkpoint 共享，只提一次）+ dense_stage1(128) + dense_stage2_<label>(128) + logit_stage2_<label>，共 8 个/标签。行序 = isolate_split.csv（已验证 accession/subtype 逐行一致）
- 协议（`probe_target_val.py`）：与 `target_val_layer_select.py` 逐条对齐——同一 10 个切分文件、同一 Ridge LR probe（StandardScaler + GridSearchCV，C∈[1e-3,100]、5-fold、seed42）、val argmax 选表示 → test 评一次、cluster_seed42 bootstrap B=1000
- 不做 epoch/checkpoint 级选择：ESM 侧自由度是"冻结权重下的表示选择"，训练动态轴不对等

## 意外发现：float32 sigmoid 饱和低估了历史 CNN AUC

行序校验时发现 stage2 jump_human logit 复算全量 H5 AUC=0.702，与 `eval_h5h7_valtest.json` 的 0.549 不符。根因：CNN logit 幅度达 ±170，float32 sigmoid 在 |logit|>88 饱和为 0/1，并列秩把 AUC 拉向 0.5（与 AGENTS.md 既有"H7 翻转必须用 -logits 不能用 1-p"同类）。已修复 `evaluate.py` 与 `eval_h5h7_valtest.py`（AUC 用 raw logits，阈值指标用 logit≥0）并重跑：

| 口径 | jump H5 | jump H7 | jump_human H5 | jump_human H7 |
|---|---|---|---|---|
| sigmoid（旧，错误） | 0.458 | 0.399 | 0.597±0.120（P=0.61） | 0.410（P=0.00） |
| **logit（修正）** | 0.458±0.126 | 0.399±0.066 | **0.723±0.080（P=1.00）** | **0.517±0.103（P=0.47）** |

jump 两格不受影响（logit 幅度小未饱和）；§5 的 CNN 2026 前向对比（run_2026_forward_cnn.py 用 raw logits）不受影响。

## 结果：同等选择预算下 CNN 仍全面低于 ESM

cluster 臂 test AUC（5 seeds，mean±std）：

| 格 | CNN logit 头（无选择） | CNN + 同等预算（val 选择） | ESM probe（val 选层） |
|---|---|---|---|
| jump H5 | 0.458±0.126 | 0.376±0.082 | **0.786±0.040** |
| jump H7 | 0.399±0.066 | 0.637±0.116 | **0.892±0.027** |
| jump_human H5 | **0.723±0.080** | 0.609±0.186 | 0.688±0.103 |
| jump_human H7 | 0.517±0.103 | 0.721±0.057 | **0.854±0.060** |

- **替代解释被排除**：给 CNN 同等选择自由度后，4 格中没有一格被拉到 ESM 水平；ESM 优势是表示质量而非选层自由度
- CNN 的 val 选择不稳定（各 seed 选中 conv2/conv3/dense_stage1/conv5 不一），选择过程反而拉低 test（jump_human H5 从 0.723 → 0.609）——无可迁移信号时选择自由度无红利，这本身即证据
- 附带发现：H7 两格 CNN 最常选中 **dense_stage1**（宿主预训练表示，未接触 jump 标签；jump_human H7 bootstrap AUC 0.770 P(>0.5)=1.00）——jump 微调反而破坏 H7 可迁移信号
- **诚实记录**：jump_human H5 格 CNN logit 头（0.723）> ESM（0.688），是唯一 CNN 不落下风的格；§1 叙事重心在 jump 标签（0.46 vs 0.79）与 H7 迁移，此格需在论文中如实并置

## 涉及文件

- 新增：`Borkenhagen/cnn_baseline/extract_intermediate_features.py`、`probe_target_val.py`、`output/interm_feats/*.npy`、`output/probe_target_val.json`
- 修改：`evaluate.py`、`eval_h5h7_valtest.py`（logit 口径修复）+ 重跑后的 `output/eval_results.json`、`output/eval_h5h7_valtest.json`

## 结论

- §1 待补项完成：CNN 在同等选择预算下仍全面（4 格中 3 格显著）低于 ESM probe，"选层自由度"替代解释排除
- 下一步：更新 RESULT.md §1（勾掉待补、修正 CNN 数值、如实记录 jump_human H5 格）与 AGENTS.md 踩坑（sigmoid 饱和条目扩展到 CNN 侧）

## 追加（09-27）：09-19 标签修复后重跑版本

09-19 host 标签修复后 `probe_target_val.py` 被重跑（JSON mtime 09-19 13:30）。修复后 cluster 臂（CNN 选择后 vs ESM 同协议）：jump H5 0.346±0.076 vs 0.836±0.018、jump H7 0.645±0.119 vs 0.918±0.023；jump_human 两格不变（0.609±0.186 vs 0.688±0.103；0.721±0.057 vs 0.854±0.060）。选中频率：H7 两格仍以 dense_stage1 为主（jump 4/5、jump_human 3/5），"jump 微调破坏 H7 可迁移信号"等原结论不变。RESULT.md §1 已于 09-27 刷新为修复后数值（协议审计 `09-27-01_h7_flip_protocol_audit.md`）。
