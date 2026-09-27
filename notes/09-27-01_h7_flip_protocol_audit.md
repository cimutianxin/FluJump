# 审计：H7 headline 数字（0.892/0.854）的符号翻转协议核查

- **日期**：09-27-01
- **性质**：纯代码与已有输出审计（未训练、未用 GPU、未修改任何文件；唯一产出为本文件）
- **动因**：回应"符号翻转是否接触了 test/全量 holdout 标签"的方法学质疑，精确确认 RESULT.md §1/§3/§4 中 H7「翻转后」target-val 数字的实际生成协议

**一句话结论**：H7 headline 数字（jump L17 0.892±0.027 / jump_human L13 0.854±0.060）的生成协议中**不存在任何符号翻转步骤**——它们是原始 logits 的 AUC，层由 H7 内部 val 一半的原始 AUC argmax 选出，test 只评一次。符号翻转只存在于历史全量 holdout 诊断脚本中。「翻转后 0.72–0.90」的表述把两个不同协议的数字混写在一起了。另发现：磁盘上的 JSON 是 09-19 host 标签修复后的重跑版本，与 notes/RESULT.md 引用的 08-09/09-12 数值存在系统性漂移（仅 jump 标签格）。

---

## Q1. `target_val_layer_select.py` 选层口径与 JSON 内容判断

### 代码事实

`ESM_clf/jump_exp/target_val_layer_select.py`（213 行通读，mtime 2026-08-09 00:14，自首次运行前至今未变）：

- L96：`per_mask[mask_name] = float(roc_auc_score(y[rows], sc[p]))` —— AUC 全部用 `decision_function` 的**原始打分**，无取负、无 1-p。
- L110–112：
  ```python
  val_aucs = {li: aucs[label_col][li][fname][val_mask] for li in aucs[label_col]}
  best = max(val_aucs, key=val_aucs.get)
  ```
  确认：对**原始** val AUC 取 **argmax**（max，非 min）。
- L116：`"test_auc": aucs[label_col][best][fname][test_mask]` —— 选中层在 test 上评一次，同为原始口径。
- L168–179（bootstrap）：对选中层的原始 test 分数做 cluster bootstrap，无任何符号处理。
- 全文 grep 不到 `翻转|flip|-logits|1 - `——**零符号处理**。

### 用户前提的证伪

「H7 原始 AUC 全部 <0.5」**不成立**：反转是**晚层现象**，中层在 H7 上原始方向为正。JSON `per_file.cluster_seed42.label_is_jump.h7` 原文摘录：

```json
"h7": {
  "selected_layer": 17,
  "val_auc": 0.9260213822069493,
  "test_auc": 0.9205036301810496,
  "ref_test_auc": {"L26": 0.3677281499862145, "L17": 0.9205036301810496, "L28": 0.2406028857641761}
}
```

同一 seed 下参照层 L26=0.368、L28=0.241（原始 <0.5，反转层），而 L17 val 原始 AUC=0.926（>0.5）。argmax 合法地选中 L17，**不需要也不可能**涉及翻转。（层号约定：该脚本 0-based 从输入层数起，L28≈项目反向计数口径的「L3/倒数第三层」，L30=末层；L17/L13 是中层。）

### per_file：5 个 cluster seed 的 H7 原始数值（JSON 摘录）

`label_is_jump` h7（5/5 选中 L17）：

| seed | selected_layer | val_auc | test_auc |
|---|---|---|---|
| 42 | 17 | 0.9260213822069493 | 0.9205036301810496 |
| 43 | 17 | 0.9323735695828719 | 0.9136632430750078 |
| 44 | 17 | 0.8985764168681171 | 0.9555832192207145 |
| 45 | 17 | 0.9460534113284013 | 0.884870848708487 |
| 46 | 17 | 0.9251483745238728 | 0.9131229995427527 |

`label_is_jump_human` h7（5/5 选中 L13）：

| seed | selected_layer | val_auc | test_auc |
|---|---|---|---|
| 42 | 13 | 0.79576031656303 | 0.8925960014317124 |
| 43 | 13 | 0.8472906403940886 | 0.8537564322469983 |
| 44 | 13 | 0.8470276192123622 | 0.874905422446406 |
| 45 | 13 | 0.8276331870729899 | 0.9083669459096082 |
| 46 | 13 | 0.9257687457561354 | 0.739016439909297 |

### bootstrap_cluster_seed42 的 H7 原始数值（JSON 摘录）

```json
"label_is_jump":   {"h7": {"layer": 17, "auc_full": 0.9205036301810496,
                    "p05": 0.8335540228219948, "p95": 0.9568412253750226, "p_above_0.5": 1.0}}
"label_is_jump_human": {"h7": {"layer": 13, "auc_full": 0.8925960014317124,
                    "p05": 0.8141810789572363, "p95": 0.9606558235594878, "p_above_0.5": 1.0}}
```

### JSON 是原始口径还是翻转口径？——**原始口径**，三条独立证据

1. 生成代码无符号处理（上述 L96/112/116/173/176）。
2. JSON 中原样保留大量 <0.5 的值（ref_test_auc L28 h7 = 0.2406/0.2045/0.2541/0.2942；h5 jump_human seed42 test_auc = 0.4827）——翻转口径的报告不可能出现 <0.5。
3. 同层 val 与 test 同号一致（如 L17 val 0.926 / test 0.921），summary 与 per_file 逐项吻合（jump h7 均值 0.9175487881456024 = 5 个原始 test_auc 的均值）。

### 重要：磁盘 JSON ≠ 论文引用值（数据版本漂移）

- `output/target_val_layer_select.log`（2026-08-09 00:18）与 `notes/08-09-00_target_val_layer_select.md` 一致：jump h7 **0.892±0.027**、jump_human h7 **0.854±0.060**、bootstrap jump h7 AUC=0.880 [0.585, 0.939] P=0.97、h5 jump 选层 {28:4, 11:1} 0.786±0.040。
- 磁盘 JSON（mtime **2026-09-19 13:49**，重跑）：jump h7 **0.9175±0.0226**、bootstrap 0.9205 [0.834, 0.957] P=1.00、h5 jump {28:5} 0.8357±0.0175；jump_human h7 **0.8537±0.0602 与旧值完全一致**。
- 归因：09-19 host_category 裸子串误判修复使 7 簇 `label_is_jump` 1→0（H5×4/H7×3）、`label_is_jump_human` 0 翻转（RESULT.md L121/L150 已披露「历史 headline 算自修复前标签」）。时间链：labels npy 重建 13:20 → CNN 重跑 13:30 → ESM 重跑 13:49 → scale 重跑 13:53/14:11。**jump 格全动、jump_human 格全不动**，与修复影响面精确互证。
- 因此「0.892 必然来自另一个版本或后处理」的猜测：**版本判断正确，后处理判断错误**——0.892 来自 08-09 运行的原始口径（仅存于 .log 与 notes），磁盘 JSON 是 09-19 重跑的原始口径（0.9175）。两个版本都无翻转。

## Q2. 0.892±0.027 的确切计算过程追溯

grep `notes/` 中 0.892/0.854 的全部出现：08-09-00_target_val_layer_select.md（源头）、08-09-00_cnn_baseline_h5h7_valtest.md（对照引用）、08-09-19_esm_tf_probe.md（对照）、08-09-23_scale_replication.md（「150M（旧）」行）、08-11-00_adaptive_pooling.md（对照）、08-27-21_naive_baseline.md（**误标为「H7 翻转后 0.892/0.854」**）、09-12-12_cnn_selection_budget.md（对照）、09-26-13_jev_class_query_head.md（对照）。全部指向同一次 08-09 运行，无任何笔记描述过「翻转后取 min/取 1-AUC」的计算。

确切过程（对每 seed∈{42..46}、每标签）：

1. probe 只在 H1+H3 train 上训练（StandardScaler + GridSearchCV 5-fold，C∈[1e-3,100]，seed42），每层 1 个、共 31 层；
2. H7 holdout 内部按 cluster 5:5 切 val/test（stratified）；**层在 val 一半上定**——31 层原始 val AUC argmax，5/5 seed 收敛 L17（jump）/L13（jump_human）；
3. **test 只在选中层上评一次**（原始 logit AUC），5 seeds 取 mean±std → 0.892±0.027 / 0.854±0.060；
4. **符号：从未被设定或翻转**——协议中不存在符号自由度；
5. 全量 H7 holdout 标签不进入任何选择步骤；唯一接触是 `ref_test_auc` 对照列（L117–118，打印历史参照层 L26/L17/L28 的 test AUC，仅作对照、不影响选择）与 bootstrap（在 test 一半上做，属不确定性量化而非选择）。
6. 诚实声明：参照层集合与「中层在 H7 不反转」的**研究者先验**来自更早的全量 holdout 诊断（脚本 docstring L12–13 与 JSON `ref_layers_note` 已声明「参照层源于全量 H5/H7 观察（历史 headline），仅作对照」）——假设空间的形成看过 holdout，但选择动作本身合规。

## Q3. CNN 基线对称性检查（`Borkenhagen/cnn_baseline/probe_target_val.py`）

- L136–138：与 ESM 逐条相同的 **val argmax（max）选表示 → test 评一次**；`cnn_baseline/` 全目录 grep 不到任何翻转代码。
- L117 原始 `roc_auc_score`；L120–125 `full_holdout` 块注明「描述性参照：全量 holdout（不用于选择）」，同为原始口径。
- 原始口径直接证据：`per_file.cluster_seed42.label_is_jump.h7.val_aucs_all` 保留 conv1=0.2226、conv3=0.2806 等大量 <0.5 原始值；选中 `dense_stage1` val=0.7869（>0.5）。
- `output/probe_target_val.json`（mtime 09-19 13:30，同为修复后重跑）summary.cluster：
  - jump h7：**0.645148652014093 ± 0.11877797747072874**（notes/RESULT 引用 0.637±0.116，09-12 版本）；
  - jump_human h7：**0.7212582253609857 ± 0.056848414199443066**（与引用值 0.721±0.057 一致——jump_human 标签 0 翻转）；
  - bootstrap seed42：jump h7 dense_stage1 auc_full 0.6403 P(>0.5)=0.9095；jump_human h7 dense_stage1 0.7699 P=0.9979。
- **对称性结论：两侧口径完全对称**——同一批 10 个切分文件、同一 Ridge LR probe、同一 max-argmax、两侧均无符号翻转。CNN 的 0.637/0.721（引用值）与 0.645/0.721（现 JSON）都是原始口径。未发现不对称项；唯一差异是 CNN 侧多保留了 `val_aucs_all` 与 `full_holdout_descriptive` 两个纯展示块。

## Q4. 规模复现（`validation_exp/scale_replication/run_scale_select.py` L111 及 select JSON）

- L109–111：与 150M 脚本相同的**原始 val AUC argmax（max）**；目录内 grep 翻转关键词仅命中 docstring（L1/L5 描述「方向翻转曲线」），无翻转代码。
- L83–88 的 sweep（全层全量 holdout AUC 曲线）也是原始口径：`scale_650m_sweep.json` 末层 L33 h7_full = **0.1494 / 0.2117**（<0.5 原样保留），选中层 L5/L3 h7_full = 0.5938 / 0.6643（>0.5）。
- select JSON 原始口径抽样：seed42 650M jump h7 L5 val=0.6007 test=0.5977；3B jump h7 L2 val=0.6830 test=0.7115（均原始 >0.5）。
- 表 S1 数字口径判定：**原始口径、max 选层**。引用值（08-09 运行，notes/08-09-23、RESULT.md §4 表）vs 现 JSON（09-19 重跑）：

| 格 | 引用值（论文/notes） | 现 JSON summary.cluster | bootstrap P(>0.5) 引用→现 |
|---|---|---|---|
| 650M jump H7 | 0.579±0.080（L5×5） | 0.5480±0.0910（L5×5） | 0.93 → 0.823 |
| 650M jump_human H7 | 0.645±0.039（L3×5） | 0.6448±0.0390（L3×5）✓ | 0.91 → 0.907 |
| 3B jump H7 | 0.634±0.057（L2×4） | 0.6243±0.0561（{2:4,1:1}） | 0.98 → 0.956 |
| 3B jump_human H7 | 0.629±0.028（L1/L2） | 0.6291±0.0279（{1:2,2:3}）✓ | 0.90 → 0.904 |

同样呈现「jump 格漂移、jump_human 格不变」的 09-19 数据版本签名。选层本身两版本一致（L5/L3/L2 稳定），仅 AUC 数值漂移。

## Q5. 全局翻转代码清单（grep 三目录）

命令等价于 `grep -rn "翻转\|flip\|-logits\|1 - "`（ESM_clf/jump_exp/、validation_exp/scale_replication/、Borkenhagen/cnn_baseline/，*.py）：

**(a) 仅诊断/展示用**：

| 文件：行号 | 内容 | 性质 |
|---|---|---|
| `ESM_clf/jump_exp/train_probe_compare.py:65–70, 83–94` | 全量 h7_holdout 上并排算 `h7_auc_raw` 与 `h7_auc_flip = roc_auc_score(y7, -logits7)` | 07-12 历史对比表；其 flip 区间（jump 0.72–0.79 / jump_human 0.79–0.90，见 `output/compare_raw_vs_aligned_*.json`）正是 RESULT.md §3「翻转后 0.72–0.90」的来源——**方向判定看了全量 holdout，属非合规历史协议，但不是 headline 点估计的路径** |
| `ESM_clf/jump_exp/diagnose_direction.py:133, 148` | `roc_auc_score(y7, -logits7)` 打印 | 诊断 |
| `ESM_clf/jump_exp/diagnose_h7.py:54–58` | `1 - p` 翻转 | 诊断（1-p 口径已弃用） |
| `ESM_clf/jump_exp/verify_h7_auc.py:47–69` | `-logits` 翻转精度验证 | 诊断 |
| `ESM_clf/jump_exp/loso_matrix.py:12` | 注释「原始方向，不翻转」 | 明确无翻转 |
| `validation_exp/scale_replication/run_scale_select.py:83–88` | sweep 全层原始 AUC 曲线（<0.5 原样落盘） | 诊断曲线 |
| `Borkenhagen/cnn_baseline/` | grep 零命中 | 无翻转代码 |
| `ESM_tf_clf/`、`mainpipeline/`（超范围补查） | grep 零命中 | 无翻转代码 |

**(b) 进入论文 headline 数字的路径**：

- `ESM_clf/jump_exp/target_val_layer_select.py`（§1/§3 的 0.892/0.854）、`Borkenhagen/cnn_baseline/probe_target_val.py`（§1 CNN 对照）、`validation_exp/scale_replication/run_scale_select.py`（§4 表 S1）——**三个 headline 脚本均无符号处理**。结论：没有任何 headline 数字经过符号翻转；翻转只存在于 (a) 类诊断与历史全量评估。

---

## 协议事实陈述（可直接改写进论文 Methods）

> Probe 训练仅使用 H1+H3 训练集（Ridge 逻辑回归，StandardScaler + 5-fold 分层 GridSearchCV，C∈[10⁻³,10²]，seed=42），每个冻结层（共 31 层）训练一个 probe。对 H5/H7 _holdout_ 亚型，我们在其内部按 CD-HIT cluster 做 5:5 分层切分（5 个种子），仅在 val 一半上对 31 层的原始 decision-function AUC 取 argmax 选择层，随后在对应 test 一半上以原始 logits 评估一次 AUC；**全过程不存在符号翻转步骤——符号不是协议的自由度**，test 标签仅在选中层上评估一次，全量 holdout 标签不进入任何选择或符号判定步骤。H7 的 headline 结果（jump：L17，0.892±0.027；jump_human：L13，0.854±0.060；cluster 臂 5 种子 mean±std，cluster bootstrap P(>0.5)=0.97/1.00）因此在选中层上是**原始方向**的 AUC——这些中层在 H7 上本就不反转（val AUC 0.80–0.95）；方向反转是晚层现象（L26/L28 原始 AUC 0.20–0.49），属另一组全层诊断结果。历史上「原始 0.1–0.28、−logits 翻转后 0.72–0.90」的区间来自早期在全量 H7 holdout 上对末层/倒数第三层的评估（方向经事后判定），仅用于现象描述，与 headline 点估计的生成路径无关。

## 论文修改建议清单（仅建议，未改稿）

1. **拆分 RESULT.md §3（L48）的混写句**：「`-logits` 翻转后 0.72–0.90（target-val 选层：jump→L17 0.892±0.027…）」把历史全量翻转区间与 target-val 原始口径点估计写进同一括号，直接招致「翻转接触 holdout 标签」的质疑。建议拆为两句并各自标注口径：现象句（晚层全量 H7 原始 0.1–0.28）+ headline 句（target-val 原始口径、无翻转）。论文正文/Methods 中所有「flipped H7 AUC = 0.892/0.854」类表述应改为「raw AUC at the val-selected layer」。
2. **同步修正二手引用**：`notes/08-27-21_naive_baseline.md` L12「H7 翻转后 0.892/0.854（target-val 选层）」同样误标；若论文引用 naive baseline 对比，措辞需一并改为原始口径。
3. **统一数据版本**：notes/RESULT.md/表 S1 引用的是 09-19 host 标签修复**前**的 08-09/09-12 运行；磁盘 JSON 为修复后重跑，jump 格全部漂移（ESM jump H7 0.892→0.9175±0.0226 且 bootstrap P 0.97→1.00；CNN jump H7 0.637→0.645、jump H5 0.376→0.346；650M jump H7 0.579→0.548±0.091 且 P 0.93→0.82；3B jump H7 0.634→0.624±0.056）。RESULT.md L121/L150 已披露「历史 headline 算自修复前标签」，但正文表格未逐格标注版本。建议：统一改用修复后（现 JSON）数字并在 Methods 声明标签版本；或维持旧数字但在每个表注标明标签版本。当前混用状态存在被审稿人对出内部不一致的风险（如 §1 CNN 对照表用旧 ESM 0.892 对比旧 CNN 0.637，而磁盘资产已是 0.9175 vs 0.645）。
4. **表 S1/§4 加表注**：「末层 AUC 0.251/0.163/0.305/0.121/0.212/0.275」等列为原始口径，建议表注写明「all AUCs are raw (unflipped); sign reversal is visible as AUC<0.5」。
5. **Methods 增加一句预防性声明**：「符号从不作为选择自由度：target-val 协议仅选择层，不选择读出符号」，直接封堵该类质疑。
6. **参照层透明声明**：target-val JSON 的 `ref_test_auc`（历史参照层的 test AUC 对照列）与「参照层源于全量观察」的 docstring 声明可保留在 Methods 或附录，表明假设来源与选择动作的分离。
7. **（可选）附上协议自证材料**：将 `target_val_layer_select.json` 的 per_file val/test 原始值（含 <0.5 的参照层值）作为补充表，用「存在 <0.5 原始值」本身证明无翻转后处理。

## 涉及文件（本次审计只读）

- 代码：`ESM_clf/jump_exp/target_val_layer_select.py`、`train_probe_compare.py`、`diagnose_direction.py`、`diagnose_h7.py`、`verify_h7_auc.py`、`loso_matrix.py`；`Borkenhagen/cnn_baseline/probe_target_val.py`；`validation_exp/scale_replication/run_scale_select.py`
- 输出：`ESM_clf/jump_exp/output/target_val_layer_select.{json,log}`、`compare_raw_vs_aligned_*.json`；`Borkenhagen/cnn_baseline/output/probe_target_val.json`；`validation_exp/scale_replication/output/scale_{650m,3b}_{select,sweep}.json`
- 笔记/文档：`notes/08-09-00_target_val_layer_select.md`、`08-09-00_cnn_baseline_h5h7_valtest.md`、`08-09-23_scale_replication.md`、`09-12-12_cnn_selection_budget.md`、`08-27-21_naive_baseline.md` 等 8 处引用点；`RESULT.md` L21/L48/L64–71/L121/L150

---

## 已执行修改（09-27）

依据上方"论文修改建议清单"第 1–5 条（第 6 条既有声明保留、第 7 条可选项略）：

1. `RESULT.md` §3：混写句拆为"历史全量诊断现象"与"target-val headline（原始口径、全程无符号翻转）"两条；headline 刷新为 09-19 重跑值（0.918±0.023 / 0.854±0.060，bootstrap P=1.00/1.00）；`-logits` 踩坑条注明仅诊断口径。
2. `RESULT.md` §1：CNN 选择预算对齐四格刷新为修复后重跑值（CNN 0.346/0.645/0.609/0.721 vs ESM 0.836/0.918/0.688/0.854），注明两侧原始 logit 口径、val argmax、无翻转；首条 H5 ~0.79 补注修复后数值（0.8255 / target-val 0.836±0.018）。
3. `RESULT.md` §4：表 S1 jump 行刷新（test 0.918/0.548/0.624，末层 0.211/0.149/0.294；jump_human 行不变），新增"全部原始口径、无符号自由度"表注；"规模放大无迁移收益"行数值同步。
4. `RESULT.md` 局限性行与 TODO-5：「历史 headline 算自修复前标签」改为「headline 已于 09-27 统一刷新为修复后重跑值」并附漂移清单；创新点 2 的 H5 0.79 补注修复后 0.83。
5. `notes/08-27-21_naive_baseline.md` L12「H7 翻转后 0.892/0.854」改为原始口径表述 + 修复后数值。
6. `AGENTS.md` 核心结论：H7 反转条注明 0.72–0.90 为历史全量诊断口径、target-val headline 无符号翻转。
7. 三篇原始日志（`08-09-00_target_val_layer_select.md`、`08-09-23_scale_replication.md`、`09-12-12_cnn_selection_budget.md`）各追加"09-19 重跑版本"节，保持 RESULT.md 数值与日志可互查。

未改：任何代码、数据、output JSON；RESULT.md §2/§5 及其余条目（不涉及 H7 target-val headline 口径，或数值本身已产出于 09-19 修复后）。
