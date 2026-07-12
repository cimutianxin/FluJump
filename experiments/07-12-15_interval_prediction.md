# 实验：ESM-2 + Linear — Interval 预测（H1+H3 jump=1）

- **日期**：2026-07-12
- **目的**：在 H1+H3 中 label_is_jump=1 的 isolate 上，用 ESM-2 150M embedding + linear probe 预测 jump interval，对比回归（iso_interval_days）和分类（jump_interval_cat）

## 🔧 前置修复：jump_interval_cat 全为空

### 根因
`datascripts/build_jump_labels.py` 中 `jump_interval_cat` 仅计算 `animal→human` 正值：
```python
d = days_between(animals[0][1], host_ym["human"])
if d is not None and d > 0:  # ← 只存正值
    jump_interval_cat = cat_interval(d)
```
H1+H3 36 个 jump cluster 全为 `human|swine`（人先于猪出现），`days_between(swine, human) < 0` 永不满足 → 全部为空。

### 修复
改为计算 **所有宿主中最早→最晚的时间跨度**（绝对值，非方向性）：
```python
if label_is_jump:
    all_yms = [host_ym[h] for h in non_unk if host_ym[h]]
    if len(all_yms) >= 2:
        all_yms.sort()
        d = days_between(all_yms[0], all_yms[-1])
        if d is not None and d >= 0:
            jump_interval_cat = cat_interval(d)
```
结果：36/36 → `<1yr`:20, `1-3yr`:9, `3-5yr`:5, `5yr+`:2。

### 数据更新步骤
1. 修改 `datascripts/build_jump_labels.py`
2. 运行 `build_jump_labels.py` → 重新生成 `all_subtypes_simplified.csv`
3. 运行 `build_isolate_dataset.py` → 生成 `all_isolates.csv` 和 `all_isolates_clean.csv`
4. 从 git 恢复 `all_isolates_aligned.csv`（MAFFT 对齐不变），合并 `iso_interval_days` 列
5. 用新 cluster 标签直接更新 `jump_interval_cat` 列（更新 1409 行）

## 方法

- **输入**：H1+H3 isolate，仅 `label_is_jump=1` 且 `jump_interval_cat` 非空（724 条）
- **特征**：复用已有 ESM-2 150M embedding（L1/L3/L1L3 mean-pooled from raw ha_sequence）
- **Split**：沿用 isolate_split.csv（485/112/97 train/val/test），34/37 cluster 跨 split
- **目标**：
  - 分类：`jump_interval_cat`（cluster 级，<1yr 59% / 1-3yr 24% / 3-5yr 15% / 5yr+ 1%）
  - 回归：`iso_interval_days`（isolate 级，661 条有效，mean=-57, std=844）
- **模型**：StandardScaler + Ridge/Lasso + GridSearchCV

## 结果

### 分类（4 类：jump_interval_cat，cluster 级标签）

| Embedding | Test Acc | Balanced Acc | Macro F1 | Best C |
|-----------|----------|-------------|----------|--------|
| L3        | **0.691** | **0.584** | **0.454** | 1.0 |
| L1        | 0.670 | 0.561 | 0.436 | 1.0 |
| L1L3      | 0.670 | 0.553 | 0.430 | 1.0 |
| random    | 0.25  | 0.25  | 0.25  | — |
| majority  | 0.59  | 0.25  | —    | — |

- 较旧版（binned iso_interval_days, acc=0.77）下降 ~8 个百分点
- 原因：新标签是 cluster 级（同一 cluster 共享），泄漏影响减弱
- 5yr+ 类 7 条（1%）极难学习

### 回归（iso_interval_days 连续值，661 条有效）

| Model | R² | Pearson r | MAE | RMSE | Best α |
|-------|-----|-----------|-----|------|--------|
| L1L3 + Ridge | **0.103** | **0.321** | 560 | 800 | 1000 |
| L1 + Ridge | 0.095 | 0.309 | 560 | 803 | 1000 |
| L3 + Ridge | 0.089 | 0.301 | 549 | 806 | 1000 |
| Lasso (all) | ≈0 | NaN | 523 | 844 | 1000 |
| baseline (mean) | -0.000 | — | 523 | 844 | — |

- R² 较旧版（0.37）大幅下降至 0.10
- Pearson r=0.32 (p<0.003)，仍有统计显著性但效果很弱
- Lasso 完全坍缩为常数预测
- MAE ≈ RMSE ≈ baseline → 模型预测几乎等同于猜均值

### 回归→分箱 vs 分类

未重跑（compare_interval.py 需适配新 filter，次要分析）

## 结论

- **分类任务**：acc=0.69（vs majority 0.59），仍有 ~10pp 提升，但较之前下降。使用 cluster 级 `jump_interval_cat` 减少了泄漏导致的虚高。
- **回归任务显著变弱**：R²=0.10（vs 之前 0.37），Pearson r=0.32。`iso_interval_days` 的预测几乎不 work。
- **分类优于回归**：cluster 级分类标签更可靠，isolate 级连续回归受噪声影响大。
- **数据瓶颈依然**：
  - 仅 724 条（661 回归有效），97 test
  - 34/37 cluster 跨 split 泄漏（用户暂不处理）
  - 5yr+ 类仅 7 条无法学习
- **下一步行动**：调整（考虑 ordinal regression、结合 cluster 特征、或接受仅做分类）

## 涉及文件

- `datascripts/build_jump_labels.py` — **修复** jump_interval_cat 计算逻辑
- `ESM_clf/interval_exp/config.py` — 配置
- `ESM_clf/interval_exp/prepare_interval_data.py` — 数据准备（适配新 filter）
- `ESM_clf/interval_exp/train_interval_clf.py` — 分类实验
- `ESM_clf/interval_exp/train_interval_reg.py` — 回归实验（含 NaN 过滤）
- `ESM_clf/interval_exp/compare_interval.py` — 回归 vs 分类对比
- `ESM_clf/interval_exp/output/` — 输出
