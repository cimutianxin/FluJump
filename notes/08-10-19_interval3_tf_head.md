# 实验：interval 3 类预测（合并 human_first+<1yr）— Transformer 头

- **日期**：2026-08-10-19
- **目的**：把 4 类 interval 标签的 human_first(raw<0) 与 <1yr 合并为一类 <1yr（3 类任务），用 ESM_tf_clf 的冻结 per-residue embedding + 可训练 Transformer 头预测，数据/输出全部新建文件（4 类资产不动）

## 方法

- 标签：`y_cat3` = 0 `<1yr`(raw<365) / 1 `1-3yr` / 2 `3yr+`，由共享 loader 的未 clamp raw days 派生；分布 549/50/66（82.6%/7.5%/9.9%，高度不平衡），train 380/38/48，val 87/6/6，test 82/6/12。
- 数据：`interval_data.load_interval_data()` 新增 `y_cat3` 与 `emb_rows` 返回键；按 emb_rows 切片 `ESM_tf_clf/output/residue_emb_L{13,17,28}.npy`（665 条，fp16 上 GPU）。
- 模型：`ResidueTransformerClassifier(out_dim=3)`（d_model=256, 2 层, attention pooling）；CE loss + balanced class_weight（[0.409, 4.088, 3.236]）。
- 协议：每 (层 × seed 42/123/456) 训练，early stop on val balanced_acc（patience 10）→ 3 seed softmax 概率集成 → val 按 bal_acc 选唯一 winner 层 → test 只评一次 + bootstrap 95% CI（B=1000）+ majority / stratified random 基线。

## 结果

val（3 seed 集成）：

| 层 | val bal_acc | val acc | val macro_f1 |
|----|------------|---------|--------------|
| L13 | 0.642 | 0.737 | 0.533 |
| L17 ★ | 0.847 | 0.869 | 0.688 |
| L28 | 0.812 | 0.778 | 0.591 |

**TEST（L17，仅一次评估，n=100）**：
- acc=0.840 [0.770, 0.910]，**bal_acc=0.757 [0.593, 0.924]**，macro_f1=0.670 [0.521, 0.806]
- 基线 bal_acc：majority 0.333 / random 0.332
- per-class recall：<1yr 0.854 / 1-3yr 0.500 / 3yr+ 0.917
- 单 seed val bal_acc 波动较大（L17: 0.695/0.613/0.839），集成后稳定；test CI 宽（小类仅 6/12 条）。

## 结论

- 3 类合并后 Transformer 头信号强：test bal_acc 0.757 远超随机基线 0.333；与 4 类 SVM winner（bal_acc 0.697，任务不同不可直接比）相比，3 类任务本身更容易（<1yr 占 82.6%）。
- L17 胜出与 jump 任务 L13/17 优于 L28 的层规律一致。
- 小类 1-3yr recall 仅 0.500（n=6），是主要短板；<1yr 与 3yr+ 区分良好。
- **下一步行动**：继续 — 可在此 3 类设定上补 mean-pooled 线性/SVM 对照（同协议）以量化 transformer 头的增益；或对 1-3yr 类做过采样/阈值调优。

## 涉及文件

| 文件 | 操作 |
|------|------|
| `ESM_clf/interval_exp/interval_data.py` | 增量修改 — 新增 `y_cat3` / `emb_rows` 返回键 + docstring |
| `ESM_clf/interval_exp/prepare_interval_data_3cls.py` | 新建 — 3 类数据薄封装 |
| `ESM_clf/interval_exp/output/labels_interval_cat3.npy`、`interval_data_info_3cls.json` | 新数据文件（4 类资产未动） |
| `ESM_tf_clf/model.py` | 增量修改 — `out_dim` 参数（默认 1，向后兼容） |
| `ESM_tf_clf/train_tf_interval.py` | 新建 — 3 类 transformer 头训练 + 规范化评估 |
| `ESM_tf_clf/output/tf_interval3_results.json` / `.log` | 新结果 |
| `AGENTS.md` | ESM_tf_clf 条目补新脚本说明 |
