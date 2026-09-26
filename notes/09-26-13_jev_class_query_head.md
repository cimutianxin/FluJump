# 实验：JEV 类别查询+原型打分类头（target-val 协议首轮评估）

- **日期**：09-26-13
- **目的**：检验 JEV 启发的"类别查询 cross-attention + 原型 cos 打分"分类头（`jev_exp/model.py: ClassQueryPrototypeHead`）在 jump / jump_human 跨亚型迁移上是否优于现有读出方式（mean-pool linear / attention pooling / transformer 头）

## 方法

- 新建自包含目录 `jev_exp/`（不改现有 workflow 任何代码），只读复用 `ESM_tf_clf/output/residue_emb_L{13,17,28}.npy` per-residue embedding（冻结 ESM-2 150M）。
- 头结构：Linear(640→256) → 2 可学习类别查询 cross-attn（nhead=8）→ 2 可学习原型 cos 打分 / 可学习 τ（init 0.1）→ 2 路 softmax CE。无 encoder、无 FFN，容量对齐 attention pooling 档。
- 协议逐项复刻 `ESM_tf_clf/train_tf_probe.py`：H1+H3 train/val early stop、3 种子 logit 集成、10 个 h5h7 切分文件 × 4 mask、val AUC argmax 选层 → test 评一次、cluster_seed42 cluster bootstrap B=1000。AUC 用 logit（s₁−s₀）口径，增报 AUPRC、MCC（val Youden 阈值）。
- 温度缩放校准：T 拟合于 H1+H3 val，评估 NLL/Brier/ECE(15 bin)。

## 结果

**cluster 臂 test AUC（mean±std，5 seeds）——四臂并排：**

| 标签 | 亚型 | linear | attnpool | tf | **jev** |
|---|---|---|---|---|---|
| jump | h5 | **0.836**±0.018 | 0.612±0.063 | 0.598±0.154 | 0.530±0.096 |
| jump | h7 | **0.918**±0.023 | 0.776±0.018 | 0.768±0.011 | 0.671±0.054 |
| jump_human | h5 | **0.688**±0.103 | 0.635±0.078 | 0.617±0.073 | 0.622±0.113 |
| jump_human | h7 | **0.854**±0.060 | 0.790±0.024 | 0.439±0.094 | 0.645±0.070 |

- H1+H3 test（分布内 sanity）：jev 0.961–0.980，与 attnpool 相当、略低于 tf（0.983–0.991）——头本身训练正常，问题出在跨亚型迁移。
- jev 四格 cluster 臂全部低于 attnpool，更远低于 linear；isolate_random 臂同样（jump h5: 0.649 vs attnpool 0.646 持平、linear 0.829）。
- bootstrap（cluster_seed42）：四格 P(>0.5)=0.58–0.90，jump_human h5 格接近随机。
- MCC 四格均为轻微负值（-0.03~-0.05）：val 上选的 Youden 阈值在 H5/H7 分布漂移下完全失效。
- 校准：h13 test 温度缩放后 NLL/Brier/ECE 全面改善（如 jump L28：NLL 0.112→0.106，ECE 0.022→0.019）；H5/H7 上 NLL 高达 1.0–2.4（分布漂移 + H7 方向反转），温度缩放仅部分缓解。
- 训练成本：18 配置共 ~27 min（60–111 s/配置），约为 tf 头的 1/4。

## 结论

- **未达预期**：查询式读出 + 原型打分在跨亚型迁移上没有任何收益，四格全部不及 attention pooling，与"容量/参数化超出 mean-pool 即伤迁移"的项目既有证据一致（tf 头同样不及 linear）。
- 分布内高性能（0.96–0.98）+ 迁移差 → 头学会了 H1+H3 特有的表面模式，符合文档审阅时的预警（类别查询不自动带来泛化）。
- **下一步行动**：放弃该方向的当前形态。若继续，候选调整：① 消融（enc+query / meanpool+proto）定位瓶颈；② 更强正则（dropout↑、权重衰减↑、早停更严）；③ 直接在 mean-pooled embedding 上做原型打分（去掉序列维度的自由度）。是否投入待定。
- 结果未达论文级标准，不进 RESULT.md。

**产物**：`jev_exp/output/jev_results.json`、`jev_calibration.json`、`compare_summary.log`、`jev_train.log`、`scores/`（集成 logit）、`models/`（18 个 checkpoint）。
