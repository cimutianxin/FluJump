# 实验：ESM-2 层编号两套约定（A/B）权威映射审计

- **日期**：09-27-22
- **目的**：纯代码审计（不跑模型、不改代码/数据/RESULT.md），把代码库中并存的两套 ESM-2 层编号约定对齐成一份权威映射表，逐条核对论文表述所属约定，标出混用处，并核查相对深度分母一致性。

## 方法

通读两套约定的定义性脚本与全部消费方（grep `hidden_states[...]`、`esm_emb_150M_{L1,L3,L1L3}`、`all_layers.npy`、`residue_emb_L`、`CANDIDATE_LAYERS`、`相对深度/rel_depth`），并用 `target_val_layer_select.json` 等输出文件核实选层记录方式。相对深度分母用各候选分母逐一试算论文数值（0.55/0.42/0.90/0.15/0.09/0.06/0.82/1.00/0.47/0.39）反推实际口径。

---

## (a) 两套约定的定义、使用范围与换算表

### 约定 (A)：自顶向下，"Lk = 倒数第 k 层"

**定义**（`ESM_clf/jump_exp/extract_embeddings.py:50-51`）：

- `L1 = out.hidden_states[-1]` —— HF hidden_states 元组最后一项 = **最后一个 transformer 层的输出**；
- `L3 = out.hidden_states[-3]` —— **倒数第三个 transformer 层的输出**；
- `L1L3` = 两者 concat（1280 维）。

150M（esm2_t30，30 个 transformer 层，hidden_states 共 31 项）下：L1 ↔ 索引 30，L3 ↔ 索引 28。aligned 变体 `extract_embeddings_aligned.py:59-63` 与 `ESM_clf/binding_exp/extract_embeddings.py:35-36` 同一定义。

**使用范围（文件级清单）**：

| 目录 | 文件 | 用法 |
|---|---|---|
| ESM_clf/jump_exp | extract_embeddings.py / extract_embeddings_aligned.py | 定义处，产 `esm_emb_150M_{L1,L3,L1L3}.npy` |
| ESM_clf/jump_exp | train_probe.py、train_probe_compare.py、train_probe_interventions.py、loso_matrix.py、analyze_geometry.py、diagnose_h7.py、verify_h7_auc.py | 消费 L1/L3/L1L3 npy（`EMB_TYPES=["L1","L3","L1L3"]`） |
| ESM_clf/binding_exp | extract_embeddings.py + probe 脚本（probe_final.json 键 L1/L3/L1L3） | 同约定 |
| ESM_clf/interval_exp | config.py:19 `EMB_TYPES=["L1","L3","L1L3"]` + legacy/* + 最终脚本 | 同上 |
| mainpipeline | risk_scorer.py:30 `MEAN_EMB_L3`；:136 `hidden_states[-3]  # L28 = 主线 L3` | A 命名，自带桥接注释 |
| validation_exp/temporal_validation | config.py:16-17（MEAN_EMB_L3/L1）、extract_2026_embeddings.py:39（`hidden_states[-3]`→`emb_2026_L3.npy`）、run_2026_forward.py、run_2026_forward_hardening.py:38、run_multiyear_forward.py:41（`LAYERS={"L3":…,"L1":…}  # L1=末层（对照），L3=主线层`）、run_h7n9_retrospective.py、run_retro_controls.py、run_2026_workflow_eval.py | 全套前向/回溯监测均 A 命名 |
| validation_exp/site_attribution | config.py:17 `MEAN_EMB_L3`、run_logit_decomposition.py:52、run_ism.py:153（`hidden_states[-3]  # L28`） | A 文件名 + B 数字注释并存 |
| validation_exp/reversal_sites | config.py:12 `MEAN_EMB_L3` | 桥接引用 |

文档层面：AGENTS.md §1/§4.1 亦按 (A) 记述（"L1（最后一层）/L3（倒数第三层）"）。

### 约定 (B)：0-based 自底向上，索引 = HF hidden_states 元组下标

**定义**（`ESM_clf/jump_exp/extract_all_layers.py:42,54-58`）：`n_layers = num_hidden_layers + 1`，`arr[li] = out.hidden_states[li]`，产 `esm_emb_150M_all_layers.npy` **(31, 11060, 640)**，注释明确"含 embedding 层 0"。即：

- 索引 **0 = embedding 层输出**（未过任何 transformer block）；
- 索引 **k（1≤k≤30）= 第 k 个 transformer 层（自底向上）的输出**；
- 索引 **30 = 最后一层** = hidden_states[-1]。

`ESM_tf_clf/extract_residue_embeddings.py:58-60` 直接 `out.hidden_states[li]` 取 `CANDIDATE_LAYERS=[13,17,28]`（config.py:19），且 :70-75 抽查与 all_layers 同层余弦 ≈1 —— `residue_emb_L{13,17,28}.npy` 即 B 索引 13/17/28。

**规模扩展**（`validation_exp/scale_replication/`）：

| 规模 | 模型 | transformer 层数 | hidden_states 条目数 | 数组形状 | 索引范围 |
|---|---|---|---|---|---|
| 150M | esm2_t30_150M | 30 | 31 | (31, 11060, 640) | 0–30 |
| 650M | esm2_t33_650M | 33 | 34 | (34, 11060, 1280) fp16 | 0–33 |
| 3B | esm2_t36_3B | 36 | 37 | (37, 11060, 2560) fp16 | 0–36 |

索引语义三规模一致（0=embedding，末索引=最后一层；extract_all_layers.py:65-67 抽查 `hidden_states[-1] == arr[-1]`）。`depth_reversal`（config.py:34 `N_LAYERS=31 含 embedding 层 0`；:66-67 `MID_LAYERS=range(13,23)`、`LATE_LAYERS=[28,29,30]`）与 `group_boundary`（extract_embeddings.py:38 同注释）全部沿用 B。

**使用范围（文件级清单）**：

| 目录 | 文件 |
|---|---|
| ESM_clf/jump_exp | extract_all_layers.py（定义）、layer_sweep.py、layer_probing.py、target_val_layer_select.py（:54 载 all_layers；JSON `selected_layer` 存整数索引，如 jump H7 选 `17`×5、jump_human H5 格选层含 `1/11/28`）、verify_layers.py、verify_layers_bootstrap.py、verify_avg_mid_bootstrap.py、unified_arch.py |
| ESM_tf_clf | extract_residue_embeddings.py、train_tf_probe.py、train_tf_interval.py、train_adaptive_pool.py（全层扫描） |
| validation_exp/scale_replication | extract_all_layers.py、run_scale_select.py（sweep 键 `f"L{li}"`，li∈0..33/36） |
| validation_exp/depth_reversal | config.py + g1–g6、v1、v2 全部（`13–22`/`28–30` 层段、REPRO_REF 键 13/17/28） |
| validation_exp/reversal_sites | config.py（ALL_LAYERS_NPY、RESIDUE_EMB_L28）、g2/g3/g4（如 g4:324 `ALL_LAYERS_NPY[28]`） |
| validation_exp/group_boundary | extract_embeddings.py（h10_h4_emb_all_layers.npy，同 31 层约定）、run_direction_test.py、supplement_analysis.py（"24/31 层" 分母 31=条目数） |
| validation_exp/external_binding | extract_all_layers_binding.py（(31,402,640)）、run_anchor.py（`ANCHOR_LAYERS=[13,17,28]`，config.py:16 注释明写来源） |
| validation_exp/control_task | config.py:16 `CANDIDATE_LAYERS=[13,17,28]`、run_control_linear.py、run_control_tf.py |
| jev_exp | config.py `from ESM_tf_clf.config import *`、train_jev.py、calibrate.py（继承 CANDIDATE_LAYERS） |

### 精确换算表

一般式（三规模通用）：**约定 (A) 的 Lk ↔ 约定 (B) 的索引 (E − k)**，其中 E = hidden_states 条目数（31/34/37）。

| 说法 | hidden_states 下标 | (B) 索引 150M | (B) 索引 650M | (B) 索引 3B | 含义 |
|---|---|---|---|---|---|
| (A) L1 | [-1] | **30** | **33** | **36** | 最后一个 transformer 层输出 |
| (A) L3 | [-3] | **28** | **31** | **34** | 倒数第三个 transformer 层输出 |
| (A) L1L3 | [-1]⊕[-3] | 30⊕28 | 33⊕31 | 36⊕34 | concat 特征（非层索引） |
| (B) L13 | [13] | 13（第 13 层，倒数第 18 条） | 13 | 13 | target-val/锚定候选层 |
| (B) L17 | [17] | 17（第 17 层，倒数第 14 条） | 17 | 17 | 同上 |
| (B) L28 | [28] | 28 ≡ (A) L3 | 28 | 28 | 150M 下两约定交点 |
| (B) L0 | [0] | 0 | 0 | 0 | embedding 层输出，(A) 无对应 |

注意：**只有 150M 的索引 28 在两约定下碰巧同名同物**（(A) "L3" = (B) 28，risk_scorer.py:136 注释 "L28 = 主线 L3" 即此桥接）；其余数字跨约定一律不同物（详见 (b)）。

---

## (b) 论文表述逐条对照（论文写法 → 代码索引 → 约定一致性）

| # | 论文表述 | 代码出处 | 代码索引（B 口径） | 所属约定 | 混用标记 |
|---|---|---|---|---|---|
| 1 | "jump L17 / jump_human L13（相对深度 0.55/0.42）" | `ESM_clf/jump_exp/target_val_layer_select.py`（REF_LAYERS [26,17,28]/[13,28]；JSON selected_layer = 17×5 / 13×5） | all_layers[17] / [13] = 第 17/13 个 transformer 层 | **B** ✓（与深度分析段一致） | 相对深度分母为 31，与 g5 代码 (/30) 不一致——见 (c)；同实验 jump_human H5 格选层 "L1/L11/L28 摇摆"（RESULT.md §1）中的 **"L1" = B 索引 1（第一个 transformer 层），不是末层** |
| 2 | Table S1："650M L5/L3、3B L2/L1L2（0.15/0.09/0.06）" | `scale_replication/run_scale_select.py` sweep 键 `f"L{li}"`；select JSON（650M jump 选 5×5、jump_human 选 3×5；3B 选 2×4+1×1） | 650M 索引 5/3；3B 索引 2/1（= 自底第 5/3/2/1 个 transformer 层） | **B** ✓ | **同名冲突高危**："L3"(650M 行) = 浅层第 3 层，主线 (A) "L3" = 150M 倒数第三层；"L1/L2"(3B 行) 的 "L1" = 自底第 1 层，(A) "L1" = 末层——同一符号在两行语义相反；"L1/L2" 排版还易被误读为 (A) 的 concat 特征 "L1L3" 同类 |
| 3 | 深度机制段 "layers 13–22 正 / 28–30 负" | `depth_reversal/config.py:66-67`（MID=range(13,23)、LATE=[28,29,30]）；源头 layer_sweep.py/layer_probing.py 同 B | 索引 13–22 / 28–30（28–30 = 倒数第 3/2/1 层） | **B** ✓ 内部一致 | 无；但 28 ≡ (A) L3、30 ≡ (A) L1，与 #4 的写法指同一批层 |
| 4 | 前向监测 "图 6：L3 probe" 与正文 "the final layer (L1 in our indexing)" | `temporal_validation/run_2026_forward_hardening.py:38`、`run_multiyear_forward.py:41`（`LAYERS={"L3": esm_emb_150M_L3.npy, "L1": esm_emb_150M_L1.npy}`）、risk_scorer.py:136 | "L3" = hidden_states[-3] = **B 30？否——B 28**；"L1" = hidden_states[-1] = **B 30** | **A** ✗ | **论文级主混用处**：§5/图 6 用 (A) 命名，而深度分析段与 §4 表用 (B) 命名；同一层在稿件中既叫 "L3" 又叫 "28"；"final layer (L1)" 的 "L1" 与 target-val 表及 3B 行中的 "L1"（= B 索引 1）直接冲突 |
| 5 | anchoring/位点归因的 "L13/L17/L28" | `external_binding/config.py:17`（ANCHOR_LAYERS=[13,17,28]，注释明写"target-val 选出的层"）；site_attribution 实体层为 L28（RESIDUE_EMB=`residue_emb_L28.npy`（B 文件）+ MEAN_EMB_L3（A 文件）+ run_ism.py:153 `hidden_states[-3] # L28`） | 索引 13/17/28 | **B** ✓ | 表述层一致 B；代码内部 A/B 文件并存但已有显式桥接注释，无误读风险 |

补充：输出文件层面，`external_binding/output/external_anchor_results.json` 的 sweep 键 "L1"/"L3" 是 B 索引 1/3，而 `ESM_clf/binding_exp/output/probe_final.json` 的 "L1"/"L3" 是 A 约定（末层/倒数第三层）——两个 binding 相关目录对同一符号口径相反，读者自行查 JSON 时会踩坑。

---

## (c) 相对深度分母核查 + 统一建议

### 分母核查（用论文数值反推）

对候选分母逐一试算（N_tr = transformer 层数 30/33/36；N_ent = 条目数 31/34/37）：

| 数值 | 层 | /N_tr | /N_ent | 判定 |
|---|---|---|---|---|
| 150M 0.55 | L17 | 17/30=0.567→0.57 | **17/31=0.548→0.55 ✓** | 分母 31 |
| 150M 0.42 | L13 | 13/30=0.433→0.43 | **13/31=0.419→0.42 ✓** | 分母 31 |
| 150M 0.90（A1 表 L28） | L28 | 28/30=0.933→0.93 | **28/31=0.903→0.90 ✓** | 分母 31 |
| 650M 0.15 | L5 | 5/33=0.152→0.15 ✓ | 5/34=0.147→0.15 ✓ | 两可 |
| 650M 0.09 | L3 | 3/33=0.091→0.09 ✓ | 3/34=0.088→0.09 ✓ | 两可 |
| 650M 0.82/1.00（A1 表 L27/L33） | L27/L33 | **27/33=0.818 ✓ / 33/33=1.00 ✓** | 27/34=0.794→0.79 ✗ / 33/34=0.97 ✗ | 分母 33 |
| 3B 0.06 | L2 | **2/36=0.056→0.06 ✓** | 2/37=0.054→0.05 ✗ | 分母 36 |
| 3B 0.47/0.39（A1 表 L17/L14） | L17/L14 | **17/36=0.472 ✓ / 14/36=0.389 ✓** | 17/37=0.459→0.46 ✗ / 14/37=0.378→0.38 ✗ | 分母 36 |

结论：**分母不一致**——论文/RESULT.md/notes 中 150M 的相对深度按分母 31（条目数，embedding 计入）计算，650M/3B 按分母 33/36（transformer 层数，embedding 不计）计算；同一篇笔记（09-26-16 A1 表）内两行口径即不相同。而 `g5_scale_geometry.py` 代码本身三规模统一 `li/(n_layers−1)`（:110，650M/3B）+ 150M 硬编码 `/30`（:165,170,186），即代码口径 = **li/N_tr（30/33/36）**，是内部一致的——与论文 150M 数字（0.55/0.42/0.90）不吻合（代码口径下应为 0.57/0.43/0.93）。

### 建议的统一写法

1. **全文统一采用约定 (B)**：0-based 自底向上，在方法节首次出现时显式声明——"layer index counts the embedding output as index 0; index k (1 ≤ k ≤ N) is the output of the k-th transformer block; N = 30/33/36 for the 150M/650M/3B models; the final layer is index N"。禁止再出现无修饰的 "L1"/"L3"（除非定义后专指 (A)，但建议彻底废弃 (A) 写法）。
2. **或等价地用"倒数第 k 层"表述**（"third-from-last layer"），跨规模无歧义，但不利于与索引精确对应，建议仅作辅助。
3. **相对深度统一分母 = N_tr（30/33/36，即最大索引）**：embedding=0.0、末层=1.0，与 g5 代码一致，改动最小——150M 三个数 0.55→**0.57**、0.42→**0.43**、0.90→**0.93**，"最优相对深度随规模前移 0.55→0.15→0.06" 变为 "0.57→0.15→0.06"，叙事与结论不变。（若坚持保留 0.55/0.42，则须统一改分母为条目数 31/34/37，代价是 3B 0.06→0.05、A1 表 650M/3B 四数全变，不推荐。）

### 需要改写的稿件位置清单（仅建议，不改稿）

1. **方法节**（embedding 提取描述）：首次声明索引约定（按建议 1）。
2. **正文深度分析段**（"layers 28–30 为末层"、"L17 相对深度 0.55"）：约定已是 B，仅需把相对深度改为统一分母后的 0.57（L13 处 0.43）。
3. **§4 规模表 / Table S1**（650M L5/L3、3B L2/L1L2 行）：表注声明"层号为 0-based 索引、分母为 N_tr"；3B 行 "L1/L2" 建议改写为 "layers 1 and 2" 或 "L02（4/5 seed；L01 1/5）"，避免与末层/concat 特征混淆。
4. **§5 前向监测正文**（"the final layer (L1 in our indexing)"）：改为 "the final layer (index 30)"；"L1 对照层" 全部改 "layer-30 (final-layer) control"。
5. **图 6 说明**（"L3 probe"）：改为 "the layer-28 probe (third-from-last transformer layer)"。
6. **RESULT.md §1 括号注**（"选层在 L1/L11/L28 间摇摆"）：该 "L1" 系 B 索引 1，按统一写法应为 "L01/L11/L28"——仅记录，按任务要求不改动 RESULT.md。
7. 论文任何引用 binding 全层扫描的附图/附表：注明 external_anchor_results.json 的 "L1"/"L3" 为 B 索引（与 binding_exp/probe_final.json 的 A 键同名不同物）。

## 结果

- 两套约定的定义、各自文件级使用范围、三规模精确换算表已固化（见 (a)）；一般式 (A) Lk ↔ (B) 索引 (E−k)，E=31/34/37。
- 论文五条表述逐条定位：#1/#2/#3/#5 属约定 (B)（内部一致），#4（前向监测 "L3 probe" + "final layer (L1)"）属约定 (A)，是论文级主混用处；另标出四处同名冲突（M1–M4 见 (b) 表）与 binding 两目录 JSON 键口径相反。
- 相对深度分母经反推证实不一致：150M 按 /31、650M/3B 按 /33、/36；g5 代码统一 /N_tr（30/33/36）。建议统一为 li/N_tr，150M 数值相应 0.55→0.57、0.42→0.43、0.90→0.93。

## 结论

达到预期：权威映射表已建立，混用处全部定位，统一方案（约定 B + 分母 N_tr + 稿件改写清单）已给出。下一步行动：**继续**——由用户在写稿时按 (c) 清单落实统一写法；本审计未修改任何代码、数据与 RESULT.md。

## 涉及文件

- 新建：`notes/09-27-22_layer_index_audit.md`（本文件）
- 只读审计（未修改）：ESM_clf/jump_exp/{extract_embeddings,extract_embeddings_aligned,extract_all_layers,train_probe,target_val_layer_select,layer_sweep}.py、ESM_clf/jump_exp/config.py、ESM_tf_clf/{config,extract_residue_embeddings}.py、validation_exp/{scale_replication,depth_reversal,temporal_validation,external_binding,site_attribution,reversal_sites,group_boundary,control_task}/ 相关脚本、mainpipeline/risk_scorer.py、RESULT.md、notes/08-09-00_target_val_layer_select.md、notes/08-09-23_scale_replication.md、notes/09-26-16_depth_reversal_mechanism.md
