# RESULT.md — 论文骨架与确定结果记录

> 本文件记录**确定会写进论文的结果**及其出处。每产出一个论文级结果（实验完成、数值稳定、多 seed/bootstrap 验证通过），必须更新本文件对应小节；结果未达"确定"标准前写入对应实验日志（notes/），不进入本文件。
>
> 论文定位：分析型论文（probing 方法学 × 病毒宿主跳跃）。

## 核心叙事

冻结的 ESM-2 表示空间中存在可被线性读出的"宿主适应方向"（host-adaptation axis）。linear probe 是测量该方向的仪器：它测到什么、在哪里失效，本身就是生物学发现——方向在 Group 1 亚型（H1/H3/H5）内跨亚型可迁移（趋同适应的线性可读性），在 H7/H10 亚支（Group 2 内的 H7 姐妹群）系统性反转（失效边界 = H7/H10 亚支边界，经 H10 复现 + H4 对照钉死），且方向随模型深度翻转（中层与末层编码的 jump 方向相反，跨三个规模复现）。

一句话 pitch：*A frozen protein language model linearly encodes influenza host adaptation; reading that direction out recovers known biology, and its failure boundary recovers HA phylogenetic group structure.*

---

## §1 线性读出存在：Group 1 内跨亚型迁移

**论点**：H1+H3 训练的 linear probe 在 H5 holdout 上迁移良好（该优势在 jump 标签与 H7 迁移上确立；jump_human H5 为双线最弱格，CNN 不落下风，需单独并置表述），且该信号来自表示本身而非 probe 容量。

**确定结果**：
- ESM-2 L3（中层）embedding + Ridge LR：**jump 标签** H5 迁移 AUC ~0.79（修复前标签口径；09-19 host 标签修复后复现 0.8255，target-val 口径 0.836±0.018），优于 CNN baseline（logit 口径 jump H5 0.459 全量 holdout、修复前标签；详见下条选择预算对齐）；**jump_human H5 格降级表述，不声称优势**——ESM 0.688±0.103（bootstrap P(>0.5)=0.42、选层不稳）不优于 CNN logit 头（全量 holdout 0.702 / cluster 臂 0.723±0.080，P=1.00），为双线唯一 CNN 不落下风的格，论文需并置（`ESM_clf/jump_exp`，实验日志 `notes/07-12-ESM2-jump实验与H7方向反转.md`）
- **CNN 选择预算对齐（09-12 建立、09-19 标签修复后重跑，本小节对比的最终口径）**：给 CNN 同等表示选择预算（conv1–5 mean-pool + stage1/stage2 dense + stage2 logit 共 8 候选 probing + 同一 target-val 选择协议；**两侧均为原始 logit AUC、val argmax（max）选择、全程无符号翻转**），cluster 臂 test AUC（CNN 选择后 vs ESM）：jump H5 0.346 vs **0.836**、jump H7 0.645 vs **0.918**、jump_human H5 0.609 vs 0.688、jump_human H7 0.721 vs **0.854**（09-19 修复后重跑值；修复前为 0.376/0.637/0.609/0.721 vs 0.786/0.892/0.688/0.854——jump_human 两格不受标签修复影响，结论不变）——4 格无一被选择自由度拉到 ESM 水平，且 CNN 的 val 选择不稳定反而拉低 test（无信号时选择无红利），"优势来自选层自由度"替代解释排除。诚实记录：**jump_human H5 格 CNN stage2 logit 头（0.723±0.080，P(>0.5)=1.00）不落下风**（ESM 0.688±0.103，且 ESM 该格选层不稳——10 切分选层在 L1/L11/L28 间摇摆、bootstrap P(>0.5)=0.42）——双线最弱格，论文需并置、不声称优势；附带发现 H7 上 CNN 最常选中 stage1 宿主预训练 dense（jump 微调破坏 H7 可迁移信号）（`Borkenhagen/cnn_baseline/probe_target_val.py`，日志 `notes/09-12-12_cnn_selection_budget.md`，协议审计 `notes/09-27-01_h7_flip_protocol_audit.md`）
- **CNN 历史数值修正（09-12）**：旧 eval 用 float32 sigmoid 概率算 AUC，|logit|>88 饱和退化——jump_human H5 曾记 0.597（P=0.61），logit 真值 0.702/0.723；jump 两格不受影响；§5 CNN 2026 前向对比用 raw logits 不受影响（`evaluate.py`/`eval_h5h7_valtest.py` 已修复重跑，日志同上）
- probe 复杂度消融：linear → MLP → transformer 头同分布相当（test 0.98–0.99），跨亚型迁移单调变差（transformer 头 H7 jump_human 0.44）（`ESM_tf_clf`，日志 `notes/08-09-19_esm_tf_probe.md`）
- Hewitt–Liang control task：两种 probe 的 selectivity 均为正（real fit 0.98–0.99 vs shuffled fit linear ≤0.83 / tf ≤0.56），同分布高分非纯容量产物；注意措辞修正——transformer 迁移崩塌的原因是"拟合了训练分布内真实但不可迁移的结构"，不是裸容量记忆（`validation_exp/control_task`，日志 `notes/08-09-23_control_task.md`）
- 评估协议结果：cluster 级切分 vs isolate 随机切分量化出 +0.11 AUC 近重复泄露；shuffled probe 在 H7 真实标签上 AUC 散布 0.20–0.92，证明 cluster bootstrap/多 seed 协议必要性；shuffled 下 best-val 选择可制造 0.67–0.82 表观信号（同上）
- 朴素生物学基线（与 H1+H3 train 人源株的全局序列相似度排序，MAFFT 对齐 Hamming identity）：**H5 上失败**——原始方向 AUC 0.33（方向错误），事后翻转也仅 0.68，明显低于 probe（0.79，09-19 修复后 0.83），probe 的 H5 迁移不是全局相似度的复述（卖点成立）；**H7 nuance**：基线同样方向反转，翻转后 0.76–0.88 与 probe 相当，H7 反转信号部分可由全局相似度解释——H5 才是 probe 价值的核心证据，H7 反转的普适性反成 Group 边界深度的佐证（`validation_exp/naive_baseline`，日志 `notes/08-27-21_naive_baseline.md`）

**待补**：
- [x] CNN baseline 选层/选择预算对齐：**已完成（09-12）**，结果见上"确定结果"第 2 条——同等预算下 CNN 四格无一达到 ESM 水平，替代解释排除；注意 jump_human H5 格 CNN logit 头（修正后 0.723）不落下风，措辞需精确（优势主要在 jump 标签与 H7 迁移）

## §2 方向的生物学锚定

**论点**：监测标签训出的方向 w 对应可测量的分子表型，而非纯代理标签伪影。

**确定结果**：
- 外部锚定：jump/jump_human probe（L13/17/28）对 Borkenhagen 402 条实验 binding 数据零样本打分，AUC 0.646–0.79 全部 >0.5（下限为 jump L28 auc_all 口径）；jump_human > jump（L17 test 0.785），与"jump_human 更接近人型适应"的预期一致；binding 表型本身层间无特异性（全层 0.94–0.97），jump 方向的层特异性不是 binding 副产品（`validation_exp/external_binding`，日志 `notes/08-09-23_external_anchor.md`）
- 位点归因（部分验证）：logit 逐位点分解显示标记位点富集边缘/阴性（jump Fisher p=0.059，jump_human p=1.0），w 不集中于 RBS 位点（与多位点组合信号自洽）；ISM 饱和突变（59k 条）显示已知人适应替换方向性显著但异质——G228S（Δlogit 中位 +0.97，frac+ 0.92）、D190N（+0.81，0.88）显著推向 jump_human+，Q226L/D190E/D225G 显著推向负方向；标记组 vs 对照组 |Δlogit| 无差异（0.80 vs 1.01）（`validation_exp/site_attribution`，日志 `notes/08-27-22_site_attribution.md`）
- 坐标基础工作：canonical H3 编号 = 原始位置 −16（信号肽不入编号），经 PDB 1HGG/X-31 验证（canonical 226=L、228=S）；映射固化于 `validation_exp/site_attribution/output/col_to_h3.json`
- 论文表述口径："部分验证"、"锚定的是方向而非数值"；监测标签对应"2000 年后溢出事件"而非"1968 式大流行适应"，Q226L 负方向与此差异一致

**待补**：无（位点归因已完成，结论为部分验证）

## §3 方向的失效边界：H7/H10 亚支反转

**论点**：probe 的失败模式本身复现生物学分类结构——H7 反转不是数值伪影，是 H7/H10 亚支（Group 2 内）的生物学混淆；H10 零样本复现 + H4（Group 2 / H3 亚支）阴性对照钉死边界位置。

**确定结果**：
- 现象（历史全量 holdout 诊断口径）：晚层（末层/倒数第三层）probe 在全量 H7 holdout 上原始 AUC 0.1–0.28（方向相反），`-logits` 翻转后 0.72–0.90——**方向经全量 holdout 事后判定**，仅作现象描述，非 headline 口径（`ESM_clf/jump_exp/train_probe_compare.py`，`output/compare_raw_vs_aligned_*.json`）
- **Headline（target-val 合规协议：原始口径、全程无符号翻转，符号不是协议自由度）**：H7 内部 cluster 5:5 切分（5 seeds），val 原始 AUC argmax 选层 → test 只评一次——jump→L17 **0.918±0.023**、jump_human→L13 **0.854±0.060**（cluster 臂 mean±std）；选中层在 H7 上原始方向即为正（val AUC 0.80–0.95），反转仅见于晚层参照层（L26/L28 原始 0.20–0.49）；cluster_seed42 test bootstrap P(>0.5) **1.00/1.00**（90% CI [0.834, 0.957] / [0.814, 0.961]）。数值为 09-19 host 标签修复后重跑版（修复前 0.892±0.027/P=0.97 与 0.854±0.060/P=1.00；jump_human 不受修复影响）（`ESM_clf/jump_exp/target_val_layer_select.py`，日志 `notes/08-09-00_target_val_layer_select.md`，协议与版本审计 `notes/09-27-01_h7_flip_protocol_audit.md`）
- 几何根因：H7 整个亚型（无论正负）落在 w 负半空间深处（proj(+) −5.1 / proj(−) −3.3），且 H7 阳性比阴性更负；PC1（46.2% 方差）按亚型分离——亚型身份主导 embedding 几何（`analyze_geometry.py`，日志 `notes/08-01-19_h7_geometry_rootcause.md`）
- H7 阳性 cluster 构成：7 个阳性 cluster 全部为 2013–2015 H7N9 暴发事件及同期谱系（H7N9 案例研究素材）
- 翻转 AUC 必须用 decision_function 的 `-logits`，不能用 `1-p`（H7 概率极端集中，数值不稳定）；**该翻转口径仅适用于诊断/历史全量评估——target-val headline 无任何翻转步骤**（见上条与审计笔记）

**待补**：
- [x] **反转的位点级机制重叠（B6，升档关键①）**：已完成（09-19）。逐位贡献分解（logit = Σ scaler(h_i)·w）定义亚型特异"推负"剖面 d_pos_ST[col] = mean(ST,+) − mean(H1H3,+)，H7 vs H10 在 6/6 格（L28/L17/L13 × 双标签）一致强相关：Spearman ρ 0.632–0.718（p ≤ 4e-69），显著高于 H4 对照（0.369–0.421，jump_human H4 无阳性）与 H5 对照（0.163–0.320）；top-20 最负驱动列 Jaccard 重叠 0.176–0.379（H4 仅 0.081–0.143）。结构映射：共享驱动位点 **0 列落经典 RBS**（130-loop/190-helix/220-loop），分布在外圈抗原位点（Ag-A 124/129、Ag-B 157/196 等）与 HA2 stem——机制重叠在非经典分布式位点，与 §2 "w 不富集 RBS" 自洽。G1 另示 H7 推负为弥漫性（57–63% 列为负，H5 对照无偏）。H10 功效限制同 group_boundary（方向性证据）（`validation_exp/reversal_sites`，日志 `notes/09-19-18_reversal_sites.md`）。**正交判定（09-19 G4，预注册规则）**：重叠**非 clade 指纹**——共享列 JS 分歧百分位中位仅 52.8–85.4（<95 阈值，L28 甚至不显著），亚型身份对照 probe 剖面与 jump probe 推负谱几乎无关（ρ 0.11–0.15 vs 阈值 0.7，CV AUC 0.995 sanity 通过）；但功能机制未锁定（1HGG 空间聚集阴性、糖基化信号仅 L17 一致显著）——解读为"超出亚型身份的分布式信号"，机制开放（`g4_orthogonal_evidence.py`，日志 `notes/09-19-18_orthogonal_ab.md`）
- [x] 缓解 n=7 功效质疑（最关键）：**已完成（H10+H4 零样本方向检验）**。H10（H7 姐妹亚支）复现反转**方向**：cluster 级 24/31 层系统性 <0.5（符号检验 p=0.003），几何复现 H7 模式（整个亚型落入 w 负半空间、阳性更深；H10N8 人源簇 jump_human@L13 raw 名次 313/321、翻转后 9/321），但**幅度远弱于 H7**（预注册层 cluster AUC 0.464/0.468，bootstrap P(>0.5)=0.29/0.37 不显著）——定性复现、定量打折。H4（Group 2 但 H3 亚支）**不反转**（3/31 层 <0.5，p=4.6e-06 正向；几何正常）→ 失效边界精确到 **H7/H10 亚支**而非整个 Group 2（与 H3 属 Group 2 却可训练一致）。注意：H10 阳性簇 10/13 为 avian|environmental 监测共检出，哺乳动物跳跃仅 4 簇，幅度打折可能与此有关；jump_human 仅 2 簇功效不足；另 1 簇（H10_143）为已识别的 `/cat/` 子串误判假阳性（主数据同源 bug 已于 09-19 修复，H10 的 24/31 层方向结论不依赖该簇）（`validation_exp/group_boundary`，日志 `notes/09-11-21_h10_h4_group_boundary.md`）
- [x] 时间验证（兼 §5）：pre-2013 训练 → H7N9 排序——已完成，**阴性结果**：pre-2013 probe 无法把 H7N9 簇排前（cluster AUC 0.542 近随机，反转现象在 pre-2013 训练下消失），对照证明失败特异于 H7/Group 2（train CV 0.965、同亚型跨时间 0.787），反向支撑"Group 边界"叙事（`validation_exp/temporal_validation`，日志 `notes/08-28-18_temporal_validation.md`）

## §4 深度与规模

**论点**："末层方向反转"是跨规模普适现象；最优深度随规模前移；150M 是实证最优规模（非算力限制）。

**确定结果**（`validation_exp/scale_replication`，日志 `notes/08-09-23_scale_replication.md`；表中为 09-19 标签修复后重跑值，协议审计 `notes/09-27-01_h7_flip_protocol_audit.md`）：

| 规模 | 标签 | 选层（相对深度） | test AUC | 末层 AUC |
|---|---|---|---|---|
| 150M | jump H7 | L17（0.55） | 0.918±0.023 | 0.211 |
| 650M | jump H7 | L5（0.15） | 0.548±0.091 | 0.149 |
| 3B | jump H7 | L2（0.06） | 0.624±0.056 | 0.294 |
| 150M | jump_human H7 | L13（0.42） | 0.854±0.060 | 0.119 |
| 650M | jump_human H7 | L3（0.09） | 0.645±0.039 | 0.212 |
| 3B | jump_human H7 | L1/L2（0.06） | 0.629±0.028 | 0.275 |

> 表注：所有 AUC 均为**原始（未翻转）logit 口径**——test AUC 来自 target-val 协议（val 原始 AUC argmax 选层、test 评一次，无符号自由度）；末层 AUC 为全量 holdout 原始值，<0.5 即方向反转的直接呈现。数值为 09-19 host 标签修复后重跑版（150M：`ESM_clf/jump_exp/output/target_val_layer_select.json` + 末层 `eval_results.json`；650M/3B：`output/scale_*_{select,sweep}.json`）；修复前版本见 `notes/08-09-23_scale_replication.md`，jump_human 各行不受标签修复影响。

- 三规模均存在"某非末层方向正确 + 末层系统性反转（0.1–0.3）"结构
- 新规律：最优相对深度随规模前移（0.55 → 0.15 → 0.06）
- 规模放大无迁移收益（target-val 峰值：150M jump H7 0.918 → 650M 0.548 / 3B 0.624；jump_human H7 0.854 → 0.645/0.629），150M 为最佳规模

**机制解释（C9，09-26 完成，`validation_exp/depth_reversal` G1–G6，日志 `notes/09-26-16_depth_reversal_mechanism.md`）**：

- 翻转的载体是**亚型内 ± gap** w·(μ_ST+−μ_ST−)，而非亚型整体 offset（AUC 平移不变 ⇒ offset 不驱动排序）：gap 随深度变号（150M mid(13–22) 正 → late(28–30) 负）且与 H7 AUC 逐层强相关——ρ(gap_H7, AUC_H7) = 0.896/0.837（150M，p≤4e-9）、0.634/0.652（650M）、0.680/0.791（3B），三规模同构
- **"最优深度前移"的机制**：gap 变号边界随规模前移（150M 21–26 → 650M 3–6 → 3B 1–5），target-val 最优层 = 变号前正 gap 窗口内的最深点（650M/3B 精确重合 L5/L3/L2；150M L17/L13 在变号点之前的正窗口）
- train 解随深度**近正交旋转**：cos(w_best, w_last) ≈ 0（三规模 |cos|≤0.018；150M 5-seed 稳定 cos=1.0000）——非"同一方向反号"而是"解空间旋转"
- H5↔H7 反相关的几何本质：两亚型 gap 剖面逐层反相，ρ(gap_H5, gap_H7) = −0.70/−0.77（150M）——深度翻转与 H5/H7 trade-off 是同一几何事实的两个侧面
- 残差流分解（G2）：末层 gap 负贡献由中层段（13–22）写入并携带（jump −11.9/总−12.1；jump_human −20.0/总−21.6）
- 位点（G3）与注意力（G4）层面均无随深度的重分配（置换 p≥0.18）——信号保持弥漫分布式，翻转是"同一批分布式特征的读出方向旋转"，不是"换了读哪些位点"；交叉读出（G6）显示表示旋转与读出旋转同量级复合
- **几何对应（V1/V2/A1，09-26 同日追加）**：①可迁移窗口中点 = 表示最大压缩点——participation ratio L17 最低（16.5）、AA 解码 margin L20 最低，ρ(margin/PR, gap_H7) = −0.70/−0.57（p≤8.9e-4；AA 身份 acc 全层饱和 1.0，身份内容从未丢失，变化的是身份相关方向的占比），"最优层 ≈ 压缩最低点"给出 label-free 判据；②亚型内 logit 与"阴性 consensus 典型性"的耦合随深度翻号且 H5/H7 镜像（mid +0.33/−0.40 → late −0.15/+0.48），± 在典型性上无分离（0.891 vs 0.896）——典型性轴是 train 标签未约束的自由度，其翻号是 w 旋转的序列层面足迹；③**谱系决定最优深度（6/6 格跨规模）**：同 Group 的 H5 越深越好（best rd 0.82–1.00，ρ(rd,AUC)=+0.47~+0.67）、反转支 H7 越浅越好（rd 0.06–0.15，ρ=−0.30~−0.59）——不存在对两谱系同时最优的单层

**待补**：无（本节完整）

## §5 应用：监测株排序（ranking protocol）

**论点**：方向 w 上的投影可作"距离人适应还差多少"的连续读数，用于监测株优先级排序（Mollentze 式 ranking 而非判定）。

**确定结果**：
- JumpScorer ranking 协议现行结果 Spearman ρ=0.724±0.011（L3_h256x128，3 seeds，08-03 重算，isolate 级划分偏乐观口径；`ESM_clf/interval_exp/output/eval_jumpscore.json`；旧笔记 `notes/07-12-16_jumpscore_ranking.md` 的 ρ=0.29 已过时）
- interval 预测提供时间维度补充：3 类变体 `y_cat3`，H1357 cluster 级无泄漏划分 bal_acc 0.42（`ESM_clf/interval_exp`，日志 `notes/08-10-19_interval3_h1357_cluster_split.md`）
- **2026 前向时间外推（排序验证，GO）**：H1+H3 train 重训 probe（H5 复现 0.798/0.794 通过断言）对 201 条 2026 分离株排序，jump/jump_human AUC 均 **0.9995**（cluster 级 0.992/1.000），10 条阳性（9 条 2026 年 1–2 月美国野鸟 H5N1 暴发簇 + 1 条猪源 H1）名次 [1–9, 11] 全部进 top 11/201；朴素基线（max identity to 人源参考）AUC **0.046**——方向完全反（2026 主体为季节性人源株，与人源参考 identity 中位 0.995 被排到最上面）；**CNN baseline（Borkenhagen stage2 checkpoint）三方对比**：jump AUC 0.865（阳性名次散布 25–42，弱于 probe）、**jump_human AUC 0.229 方向反转**（9/10 阳性排到 144 名之后）——jump_human 前向排序上 probe 是唯一有效方法。功效限制：阳性仅 10 条/2 簇，表述为排序验证而非分类性能（`validation_exp/temporal_validation`，日志 `notes/08-28-18_temporal_validation.md`）
- **单株风险分层 workflow + 2026 前瞻验证（09-19，TODO-8 完成）**：`mainpipeline/risk_scorer.py` 交付端到端流程（序列 → ESM-2 L3 → logit → train 分位数 → 高/中/低分层；阈值仅用 ≤2025 train 固定：高=train 阳性中位、中=train 阴性 P95；H7/H10 显式判为边界外不适用）。2026 前瞻集（201 条）分层命中：jump_human 中及以上 **8/10 阳性命中、0 误报**（precision 1.0），jump 9/10 命中、1 误报（FPR 0.005）；朴素基线同规则下阈值退化（阴性 P95 封顶 1.0）且完全反向（高层 50–79 条全阴性、0/10 命中）。**诚实披露**：2026 阳性 train 分位数中位仅 0.88–0.93，达不到 train 阳性中位的高层——回测校准的绝对阈值对新暴发株系统性偏严，高层前瞻几乎不触发，中界（阴性 P95）才是有效预警线；与 §5"排序可移植、绝对水平不可移植"自洽（`run_2026_workflow_eval.py`，日志 `notes/09-19-19_risk_workflow.md`）
- **前向加固（09-19，B4 完成）**：cluster 级统计推断——L3 bootstrap 95% CI jump [0.953,1.0] / jump_human [1.0,1.0]，cluster permutation p=0.0008/0.0002（B=10000）；**≤2025 标签口径 0 变化**（(H5,70) 的 ≤2025 成员已含 human|avian|bovine、(H1,15) 已含 human|swine，循环风险排除；30/65 簇为 2026 新簇、两口径均阴性），新口径 cluster AUC 不变；**L1 末层对照**：jump 仍强（cluster 0.984，p=0.002）但 jump_human 末层失效（isolate 0.314 / cluster 0.730，p=0.149 不显著）——与 §4 末层反转自洽，jump_human 前向结论依赖中层选择，论文需如实披露（`run_2026_forward_hardening.py`，日志 `notes/09-19-15_2026_forward_hardening.md`；注意：簇标签/聚合一律按 (subtype, cluster_id) 键，cluster_id 为亚型内编号）
- **多年份前向复现（09-26，TODO-B7 完成）**：前向排序从"2026 单年 2 簇"扩展为**三年 17 阳性簇**可复现规律。2024（train ≤2023，3,426 条；eval 2,677 条/136 簇/7 阳性簇）与 2025（train ≤2024，5,262 条；1,020 条/201 簇/8 阳性簇），**训练标签按 ≤cutoff 簇成员重算**防未来信息泄漏（训练阳性变化极小：439→432）：L3 jump cluster AUC **0.907** [0.825,0.973] / **0.972** [0.934,0.996]（perm p<1e-4），jump_human **0.744** [0.585,0.881]（p=0.018，六格最弱、与 §1 最弱格一脉相承）/ **0.955** [0.909,0.990]（p<1e-4），因果（≤cutoff）口径 0.99/0.68/0.99/0.95；朴素基线两年方向均反（0.275/0.170，同 2026 的 0.046）。**首跳簇 anticipation（新增超 2026 的证据）**：2024 六个首跳簇（≤2023 口径尚非阳性）在 jump 标签下簇级名次 3/10/13/20/28/36 全进 top 26%（136 簇），含 (H5,70) 美国奶牛暴发相关 H5N1 簇（含 bovine/human/avian 成员，clean 表 avian 占多数）名次 **3/136**——w 方向对"该年首次跳跃"谱系有前瞻排序能力（jump_human 较弱：(H5,70) 55/136，如实并置）。L1 对照：jump 仍强（0.872/0.941），jump_human 末层失效复现（2024 cluster 0.663 n.s.、因果口径 0.333；2025 isolate 0.419 反向）——与 §4 自洽。分层：2025 jump 高层**首次前瞻触发**（26 条 precision 0.96 recall 0.50 FPR 0.001；2026 高层 jump_human 不触发系分布漂移，jump 高档触发 1 条且命中 P=1.0），2024 jump 中+ 82 条 precision 0.79 recall 0.34 FPR 0.007（`run_multiyear_forward.py`，日志 `notes/09-26-14_multiyear_forward.md`）

**待补**：
- [x] 时间维度验证：**前向 2026 已达成**（见上，AUC 0.9995 vs 朴素基线 0.046）；**回溯 H7N9 为阴性结果**——pre-2013 H1+H3 训练（label_is_jump，1048 行/158 阳性）→ H7 2013+ 排序，probe cluster AUC raw 0.542/翻转 0.458（近随机，且 pre-2013 训练下 H7 反转现象本身消失），7 个 H7N9 阳性簇名次双向均在 33–58/90 中段；对照显示 train CV 0.965、同亚型跨时间（2013+ H1+H3）0.787，失败**特异于 H7/Group 2** 而非时间外推本身——与 §3 Group 边界叙事自洽（`validation_exp/temporal_validation`，日志 `notes/08-28-18_temporal_validation.md`）
- [x] ranking 协议与 §1 朴素基线的正面对比（已由 2026 前向实验完成，三方对比：probe 0.9995 ≫ CNN jump 0.865 / jump_human 0.229 反转 ≫ 朴素基线 0.046 反向）

---

## 项目审阅（2026-09-12）：完整度与创新点评估

> 审阅范围：notes/ 全部 38 篇日志、各实验目录 output/*.json 逐项数值核对、data/ 行数与 README 基准核对、Alberts/Chen/mainpipeline 目录状态。核对结论：**RESULT.md 记录数值与 output 全部一致**（仅 §2 一处下限措辞偏差 0.646 vs 0.65）；**data/README.md 行数基准全部可复现**；无"跑到一半"的实验；mainpipeline/ 为空（与 AGENTS.md 一致）。

### 完整度评估

| 小节 | 状态 | 残余弱点 |
|---|---|---|
| §1 线性读出 | 完整：CNN 选择预算对齐（09-12）排除"选层自由度"替代解释，五节证据链闭合 | jump_human H5 是双线最弱格：ESM 0.688 选层不稳（bootstrap P=0.42），CNN logit 头 0.723 不落下风 → 已降级表述（09-16，§1 已并置、不声称优势） |
| §2 生物学锚定 | 完整，但结论限于"部分验证"（锚定方向而非数值） | w 不富集 RBS 位点；缺机制性新发现，机制深度有限 |
| §3 失效边界 | 完整：H10 定性复现 + H4 阴性对照钉死 H7/H10 亚支 | H10 幅度弱（0.46–0.47，bootstrap 不显著）；阳性簇 10/13 为监测共检出；jump_human 仅 2 簇功效不足 |
| §4 深度与规模 | 完整：三规模复现，规律一致；**机制已解释（09-26 C9）** | "最优深度前移（0.55→0.15→0.06）"机制 = 亚型内 gap 变号边界随规模前移；解近正交旋转（cos≈0）；位点/注意力无重分配。残余：650M/3B 峰值远低于 150M 原因未知 |
| §5 监测排序 | 前向多年复现（2024/2025/2026 三年 17 阳性簇，jump cluster 0.907/0.972/0.992）+ 回溯阴性已与 §3 自洽 | 2024 jump_human 格最弱（0.744，p=0.018 仍显著）；首跳簇 anticipation 仅 jump 标签成立；interval 无泄漏划分仅 bal_acc 0.42（诚实但弱） |

数据/协议层面已知局限（论文局限节需声明）：标签为监测代理（检出≠真实跳跃）；isolate 标签继承 cluster"未来行为"的结构性泄漏已用 cluster 划分 + H5/H7 holdout + 泄露对照臂量化（+0.11）；地理偏差 USA 65%；isolate 级阳性率 12.8%（cluster 级 3.2%）；host 关键词裸子串误判已于 09-19 修复（7 簇 jump 标签 1→0、41 条 isolate 全在 H5/H7 holdout、jump_human 0 翻转；§1/§3/§4 headline 数字已于 09-27 统一刷新为修复后重跑值——jump_human 各格不变，jump 格：ESM H7 0.892→0.918、H5 0.786→0.836，CNN H7 0.637→0.645、H5 0.376→0.346，650M H7 0.579→0.548，3B H7 0.634→0.624；方向性结论不变，审计 `notes/09-27-01_h7_flip_protocol_audit.md`）。

### 创新点（按卖点强度排序）

1. **失效边界即生物学边界**（最独特卖点）：probe 的失败模式复现 HA 分类结构——H7 方向反转非数值伪影，边界经 H10 零样本复现（24/31 层，p=0.003）+ H4 阴性对照精确钉在 H7/H10 亚支。"模型在哪里失败"本身是发现。
2. **宿主适应方向的线性可读性与跨亚型迁移**：冻结 ESM-2 表示中 jump/jump_human 方向可线性读出并在 Group 1 内跨亚型迁移（H5 0.79，09-19 标签修复后 0.83）；朴素全局相似度基线在 H5 失败（raw 0.33/翻转 0.68）→ 信号不是序列相似度的复述。
3. **方向随深度翻转 + 跨规模普适 + 机制解释**：150M/650M/3B 三规模均存在"非末层方向正确 + 末层系统性反转（0.1–0.3）"；最优相对深度随规模前移（0.55→0.15→0.06）；机制已钉死（C9）——载体是亚型内 ± gap 随深度变号（ρ(gap,AUC)=0.63–0.90 三规模同构）、gap 变号边界随规模前移、train 解近正交旋转（cos≈0）；150M 实证最优（规模放大无迁移收益，反直觉）。
4. **监测排序的决定性三方对比**（2026 前向）：probe 0.9995（阳性全进 top 11/201）≫ CNN jump 0.865 / jump_human 0.229 反转 ≫ 朴素基线 0.046 反向——jump_human 前向排序上 probe 是唯一有效方法。
5. **方法学贡献**：target-val 合规选择协议 + Hewitt–Liang control task + CNN 同等选择预算对齐（排除"优势来自选层自由度"替代解释）；shuffled probe + best-val 可制造 0.67–0.82 假信号的警示；float32 sigmoid 饱和致 AUC 并列秩退化、一切排序评估用 raw logits 的口径教训。

### 已完成但未进论文骨架的结果（备查）

- **LoRA 微调（08-01-22，`jump_exp/finetune_lora.json`）**：备查区旧表述"高于冻结 probe、存在张力"系误读——0.917/0.947 来自 **H5 在训练集内**的 H1+H3+H5 配置（非 holdout）。同协议 H1+H3→H5 holdout 对比：LoRA jump H5 **0.466**（冻结 probe 0.79，迁移崩塌）、jump_human H5 0.755（与冻结 0.688 互有胜负）；H7 四格 ≤0.45（H5 联合训练反转加深至 0.125）。结论：微调破坏可迁移方向，**无张力**——并入 §1"容量超出 mean-pool 即伤迁移"证据链（tf 头 / attnpool / jev 头同向阴性）。
- **四条修复/架构路线全阴性**：INLP 擦除 + cluster 加权、LoRA（对 H7）、统一架构（avg_mid/elmo_mix）、attention pooling（容量阶梯 linear > attnpool > tf 头）——"线性头为迁移最优"的完整证据链，§1 目前仅引 tf_probe 一篇，论文可并置。
- **interval isolate 级口径**：4 类分类 bal_acc 0.697 / 回归 Spearman 0.719 / JumpScorer ρ=0.724（均为偏乐观 isolate 划分，§5 只收无泄漏口径 0.42）；interval 路线偏离主线叙事，已决定不进论文（09-15 TODO 清理）。
- **binding_exp 监督 probing**（07-10，AUC 0.95+）：已被 external_binding 零样本锚定取代。

### TODO

**A. 写作前必须修正（口径级）**
1. [x] ~~jump_human H5 格降级表述：并置 CNN 0.723，标注 ESM 该格选层不稳（bootstrap P=0.42）——双线最弱格~~ **已完成（09-16）**：§1 论点句、确定结果第 1/2 条均已落实降级表述（并置 CNN 0.723±0.080 P=1.00、标注 ESM 选层不稳 P=0.42、不声称优势）。
2. [x] ~~§2 "AUC 0.65–0.79" 下限修正：实际最低 0.646（jump L28 auc_all 口径）~~ **已完成（09-16）**：§2 确定结果第 1 条已改为 0.646–0.79 并注明口径。
3. [x] ~~JumpScorer 数值过时：`notes/07-12-16_jumpscore_ranking.md` 记 ρ=0.29，现行 `interval_exp/output/eval_jumpscore.json` 为 ρ=0.724（08-03 重算）——更新笔记或在 §5 引现行数值~~ **已完成（09-16）**：§5 确定结果第 1 条已引现行 ρ=0.724±0.011（L3_h256x128，isolate 级偏乐观口径），旧笔记已加置顶过时标注。

**B. 实验加固（按目标档位分级）**

当前档位（Bioinformatics / PLOS CB）投稿不依赖 B 组任何一项；B6/B7/C9 三项是**升档关键项**——若目标上调至 Nat Commun / Cell Systems 档，它们从"可选"变为"必须"。**进度：①B6（09-19）、②B7（09-26）、③C9（09-26）三项已全部完成——升档关键项齐备。**

4. [x] ~~§5 前向实验加固三小项~~ **已完成（09-19）**：cluster bootstrap CI（L3 jump [0.953,1.0] / jump_human [1.0,1.0]）+ cluster permutation p（0.0008/0.0002，B=10000）；≤2025 标签口径 0 变化（循环风险排除，AUC 不变）；L1 对照层敏感性——jump 稳健（0.984，p=0.002）、jump_human 末层失效（0.730，p=0.149，与 §4 末层反转自洽，披露点）。脚本 `validation_exp/temporal_validation/run_2026_forward_hardening.py`，日志 `notes/09-19-15_2026_forward_hardening.md`（附 cluster_id 亚型内编号踩坑）。
5. [x] ~~主 pipeline feline 误判 bug 排查~~ **已完成（09-19）**：机制 = host 关键词裸子串匹配（`cat`/`pig`/`cow`/`air`/`water` 误伤 furcata/flycatcher/pigeon/Moscow 等）；今晨 ad-hoc 修复（49 行）已脚本化固化进 `clean_data.py`/`download_flu.py`/`supplement_h5.py`（`\b` 词边界 + 垃圾 species 内嵌株名提取 + improve-only 覆盖），重跑 clean 与现行表全列 diff=0。影响面：主数据零 feline 残留；7 簇 `label_is_jump` 1→0（H5×4/H7×3，第二宿主均为误判 environmental 鸟类行），`label_is_jump_human` 0 翻转；41 条 isolate 全在 H5/H7 holdout；关键簇安全（(H5,70) 触及但未翻转）。历史 headline 算自旧标签（41/11,060=0.37% 假阳性已移除），方向性结论不变——§1/§3/§4 headline 数字已于 09-27 统一刷新为修复后重跑值（见上方局限性行与 `notes/09-27-01_h7_flip_protocol_audit.md`）；B4 加固中 H5 jump 复现上移（0.798→0.8255）主要来自此修复。日志 `notes/09-19-16_feline_host_audit.md`。
6. [x] ~~**（升档关键①）H7 vs H10 反转驱动位点对比 + 结构映射**~~ **已完成（09-19）**：见 §3 待补首条——6/6 格 H7~H10 推负剖面 ρ 0.632–0.718（p ≤ 4e-69）> H4/H5 对照，共享驱动位点 0 列落经典 RBS（非经典分布式位点机制重叠）。`validation_exp/reversal_sites`，日志 `notes/09-19-18_reversal_sites.md`。
7. [x] ~~**（升档关键②）多年份前向复现**~~ **已完成（09-26）**：2024/2025 前向复现成功，三年 17 阳性簇可复现规律成立；训练标签按 ≤cutoff 簇成员重算（防未来泄漏）为协议必要新增。jump cluster AUC 0.907/0.972（perm p<1e-4）、jump_human 0.744（p=0.018）/0.955（p<1e-4），朴素基线方向全反；首跳簇 anticipation 成立（2024 六首跳簇全进 top 26%，(H5,70) 名次 3/136）；L1 jump_human 末层失效复现；2025 jump 高层首次前瞻触发（precision 0.96）。详见 §5 确定结果末条与日志 `notes/09-26-14_multiyear_forward.md`。
8. [x] ~~**（应用落地）单株高风险分类 workflow + 2026 前瞻集验证**~~ **已完成（09-19）**：见 §5 确定结果末条——workflow 交付（`mainpipeline/risk_scorer.py`），前瞻验证 jump_human 中界 8/10 命中零误报；jump_human 高层阈值前瞻不触发（分布漂移，阳性 train 分位数中位 0.88–0.93），表述为分层预警+分位数读数。日志 `notes/09-19-19_risk_workflow.md`。

**C. 未来工作（当前档位不进本文）**
9. [x] ~~**（升档关键③）深度方向翻转的机制解释**（残基层级分析 / 注意力归因）~~ **已完成（09-26）**：见 §4"机制解释"小节——翻转载体 = 亚型内 ± gap 随深度变号（ρ(gap,AUC)=0.63–0.90 三规模同构），gap 变号边界随规模前移解释"最优深度前移"，train 解近正交旋转（cos(w_best,w_last)≈0）；残差流：负贡献中层段写入并携带；位点/注意力无重分配（信号弥漫分布式）。"亚型身份全层贯穿、方向随深度翻转"的 08-01-22 开放问题已回答。`validation_exp/depth_reversal`（G1–G6 自包含，未改任何现有 workflow），日志 `notes/09-26-16_depth_reversal_mechanism.md`。
