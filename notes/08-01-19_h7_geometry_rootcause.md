# 实验：H7 方向反转的几何根因分析

- **日期**：08-01-19
- **目的**：弄清 H7 方向反转是 embedding 空间域偏移伪影还是谱系构成问题，为架构修复选路

## 方法

- 新脚本 `ESM_clf/jump_exp/analyze_geometry.py`：
  1. H1+H3 train 上训 Ridge LR probe，取权重方向 w，计算各亚型质心偏移方向 d 与 w 的余弦，及亚型内正/负样本在 w 上的投影；
  2. H7 逐 cluster 的 w 投影 vs 标签；
  3. PCA 可视化（`output/figs/pca_L3_*.png`）；
  4. H7 阳性 cluster 的 strain/年份构成。
- subtype/cluster_id 一律用 `isolate_split.csv`（11060 行，与 embedding/labels 行序一致），不用 07-12 重建后漂移的 aligned CSV。

## 结果

### cos(w, d) 与投影（L3, label_is_jump_human, best_C=1.0）

| 亚型 | cos(w,d) | proj(+) | proj(−) |
|---|---|---|---|
| H1 | +0.013 | +1.354 | +0.027 |
| H3 | −0.014 | +1.355 | −0.246 |
| H5 | +0.012 | +1.015 | +0.263 |
| H7 | −0.122 | **−5.099** | **−3.280** |

- H1/H3/H5：正样本投影为正、负样本在 0 附近——正常。
- **H7 整个亚型（无论正负）落在 w 负侧深处**；且 H7 阳性（−5.1）比阴性（−3.3）更负。
- `label_is_jump` 同规律（H7: proj(+) −2.409 / proj(−) −1.494）；L1/L1L3 同规律。
- H7 阳性 cluster 中 93%（jump）/ 100%（jump_human）的投影 < 0。

### H7 阳性 cluster 构成（jump_human，7 个）

cluster 109 (n=100, H7N9, 2013–2015)、154 (n=13, H7N9, 2014)、155 (n=23, H7N9, 2013–2014)、150/151/149/198（2014–2015, H7/mixed）——**全部是 2013–2015 H7N9 暴发事件及其同期谱系**。

### PCA

PC1（46.2% 方差）几乎按亚型分离：H3 左、H5 左下、H7 中、H1 右——**亚型身份主导 embedding 几何**。

## 结论

- 根因判定：**宿主/亚型特征混淆**。probe 在 H1+H3 上学到的 w 本质近似"人/猪适应方向"；H7 是 Group 2 禽源亚型，整体深度落在负半空间，且 H7N9 鸡源分离株比其它 H7 更"禽"、更负——所以 H7 内部排序也反。不是数值伪影，是生物学混淆。
- 决策门：走"域不变表示 + 谱系加权"路线（erasure / cluster 加权），同时看 LOSO 矩阵验证 Group 结构假设。
- 下一步行动：**继续**（`train_probe_interventions.py`：亚型方向擦除 × cluster 加权；LOSO 矩阵运行中）

## 附：数据漂移警示

`data/processed_isolate/all_isolates_clean.csv` 与 `processed_isolate_MAFFT/all_isolates_aligned.csv` 于 07-12 17:26 重建（11060→11261 行，267 条 isolate 亚型判定改变，如 110 条 H5→H7、143 条 H1→H3），而 split/embedding/labels 均为 07-10 旧版。**旧脚本（train_probe.py 等）现在直接读 aligned CSV 会静默行错位。** 需要决定：以哪版亚型判定为准，并重跑 build_splits + extract_embeddings。
