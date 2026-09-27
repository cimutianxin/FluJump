# 实验：VirHostPRED 外部基线全量打分与正式评估

- **日期**：09-28-02
- **目的**：对 VirHostPRED（Beltrán et al. 2026，web server 形态，见 notes/09-28-00）
  做全量打分（~15k 条）与和既有 workflow 完全对称的正式评估（H5/H7 四格 +
  前向三年 + ΔAUC + 风险分层），主分数 p_human 预注册、无翻转无择优。

## 方法

- **打分**：`score_full.py`（batch=50，checkpoint 断点续跑，非 200 统一指数退避
  [5,15,45,90,180]s，429 加罚 60s，连续 3 批失败熔断；fasta 头用
  `{accession}|{subtype}` 复合 ID 防双亚型撞键）。池构建 `prepare_score_pool.py`。
- **probe 对照**：`compute_probe_scores.py`——H5/H7 用 target-val 选中层
  （cluster 臂 5 seeds：jump/H5=L28×5、jump/H7=L17×5、jump_human/H5=
  {L1,L11,L11,L28,L11}、jump_human/H7=L13×5）CPU 重训 ridge probe 打全量
  holdout；前向三年复刻 multiyear 协议（2024/2025 因果 ≤cutoff 标签、2026
  全量 train headline 标签，均 L3）。三重对账通过：labels npy==当前 csv、
  all_layers[28]==MEAN_EMB_L3、前向 jump_human logits 与 ranking CSV
  max|Δ|=1e-14。
- **评估**：`evaluate_baseline.py`，cluster bootstrap / permutation 逐行复刻
  run_multiyear_forward.py（B=10000，seed 42，弃退化重采样，p=#(≥obs)/B）；
  簇聚合 score=mean、label=max，键 (subtype, cluster_id)；配对 ΔAUC 用同一批
  重采样簇（H5/H7 格 probe 取 seed42 层，5 种子均值另报）。

## 打分运行记录

- 池 11,261 条唯一 (accession, subtype)；非标准残基跳过 696 条（6.18%，各池
  3.7–8.1%，<10% 停止线），实打 **10,565 条**，211 批全 200、**35.8 min 墙钟**
  （均值 10.1 s/批，全程仅 2 次 503 瞬态重试自愈，无熔断、无 400 隔离）。
- 抽查：`--verify 5` 重打 5 条 p_human 全一致（服务端确定性）；随机 5 条与
  all_isolates_clean.csv 按 (accession, subtype) join 5/5 匹配。
- p_human 分布：mean 0.860，std 0.043，IQR [0.849, 0.881]——整体饱和高位。
- 坑（本任务新增）：①score_full 初版 tuple 解包 bug 把 subtype 当序列 POST，
  制造了此前误判为"服务端限流/内容拒绝"的全部 400（含上轮 AAA16879 假阳性
  隔离）；②裸 accession 头在双亚型行上撞键丢行（parse 48/50），改复合头解决。

## 结果

### (a) 描述性口径：p_human vs 分离株 host_category==human（isolate AUC）

| 池 | n_scored/n_total | AUC |
|---|---|---|
| H5 holdout | 1643/1747 | 0.498 |
| H7 holdout | 807/838 | 0.412 |
| 2024 | 2573/2677 | 0.470 |
| 2025 | 962/1020 | 0.277 |
| 2026 | 191/201 | **0.0585** |

工具不排序分离株宿主（种级语义饱和）；前向池上反向（野鸟暴发株得分最高）。

### (b) H5/H7 四格（cluster AUC，基线 vs probe target-val 层同池）

| 格 | 基线 cluster AUC [95% CI] (perm p) | probe cluster AUC | Δ 95% CI（配对 bootstrap） |
|---|---|---|---|
| jump × H5 | 0.543 [0.437, 0.643] (p=0.252) | 0.679（5 seeds 同层 L28） | [-0.296, +0.026] |
| jump × H7 | 0.370 [0.250, 0.491] (p=0.932) | 0.907 | **[-0.671, -0.408]** |
| jump_human × H5 | 0.533 [0.409, 0.651] (p=0.343) | 0.750±0.098 | [-0.412, +0.004] |
| jump_human × H7 | 0.354 [0.233, 0.474] (p=0.901) | 0.868 | **[-0.650, -0.384]** |

isolate AUC：0.546/0.326/0.551/0.321。四格 perm p 全不显著；H7 两格方向
低于 0.5。基线无任何选择自由度、无 target-val——它没有任何预算可用，这一
不对称使对比对基线偏宽（其数字仍为泄漏方向上限，见下）。

### (c) 前向三年（headline 簇继承标签；2024/2025 阳性簇集两标签相同，AUC 相同为正确结果，已与 multiyear n_pos_clusters 核对）

| 年 | isolate AUC | cluster AUC [95% CI] (perm p) | 因果口径 | 阳性名次 |
|---|---|---|---|---|
| 2024 | 0.562 | 0.434 [0.207, 0.644] (p=0.720) | 0.078 | 7 阳性簇簇级名次 32–122/136 |
| 2025 | 0.242 | 0.253 [0.111, 0.398] (p=0.991) | 0.208 | 8 阳性簇 91–182/201 |
| 2026 | 0.546 | 0.207 [0.085, 0.356] (p=0.906) | 0.207 | 9 条 scored 阳性 isolate 名次 65–95/191；阳性簇 (H5,70) 52/60、(H1,15) 43/60 |

风险分层（阈值仅 train ≤cutoff 基线分数定）：**三年两标签全部阈值退化**
（高=train 阳性中位 ~0.884 < 中=train 阴性 P95 ~0.915，p_human 饱和所致）。
2026 中+ 档 60 条全阴性、0/9 命中、FPR 0.33；2025 中+ 199–208 条、
0–1/49 命中、FPR 0.22。完全无分层价值。

### (d) ΔAUC（基线 − probe L3，配对 cluster bootstrap 95% CI）

- 2024 jump [−0.734, −0.232] / jump_human [−0.453, −0.150]
- 2025 jump [−0.881, −0.562] / jump_human [−0.852, −0.555]
- 2026 jump [−0.915, −0.626] / jump_human [−0.930, −0.644]

前向六格 ΔCI 全部排除 0（基线显著劣于 probe）。

## 结论

- **预期方向成立**：种级语义导致 p_human 在流感上整体饱和（mean 0.86），
  分离株宿主判别近随机（H5 0.498），cluster 级 jump 区分四格不显著、
  前向三年全败（2026 cluster 0.207，对偶于 probe 0.992 / CNN jump 0.865 /
  naive 0.046——四方法里仅高于完全反向的 naive）。
- **泄漏声明**：VirHostPRED 训练集为 NCBI Virus RefSeq 人源病毒蛋白
  （构成见 notes/09-28-00），几乎必然含人流感 HA；其训练序列未公开、无法
  accession 级重叠核查，故上述数字为**泄漏方向的上界**——即便如此仍全败。
- isolate 级泄漏虚高未观察到（isolate AUC 也仅 0.24–0.56）。
- **下一步行动**：结果进 RESULT.md（§1 四格一行 + §5 四方对比）；该基线
   storyline 关闭，不再迭代。

## 涉及文件

- `validation_exp/hostpred_baseline/prepare_score_pool.py`、`score_full.py`、
  `compute_probe_scores.py`、`evaluate_baseline.py`
- output/：score_pool.csv、skipped_nonstd.csv（696 条）、scores_all.csv
  （10,565 条）、score_timing.csv、score_full_run.log、probe_scores_h5h7.csv、
  probe_scores_forward.csv、hostpred_baseline_h5h7.json、
  hostpred_baseline_forward.json
