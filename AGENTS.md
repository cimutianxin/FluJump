# AGENTS.md — FluJump 项目指南（供 AI 编码 Agent 阅读）

> 本文件面向对项目一无所知的 Agent。读完本文件即可了解项目目标、结构、环境、运行方式与约定。

---

## 1. 项目概述

FluJump 是一个科研项目：基于流感病毒 HA（血凝素）蛋白序列 + 元数据，预测病毒是否会发生**跨物种传播（jump）**，尤其是从动物跳跃到人类，并预测跳跃的**时间间隔（interval）**。

- **任务类型**：（seq + feature）→ 二分类 / 多分类预测
  - `label_is_jump`：是否跨物种（任意 ≥2 宿主）
  - `label_is_jump_human`：是否跨物种到人
  - interval 预测：动物 → 人的时间间隔（分箱 `<1yr`/`1-3yr`/`3yr+` 或连续天数）
- **数据**：NCBI Protein 下载的 2000 年后 H1/H3/H5/H7 四种亚型 HA 序列
- **主线方法**：ESM-2 150M（`facebook/esm2_t30_150M_UR50D`）冻结 embedding + Ridge Logistic Regression probe
- **Baseline**：Borkenhagen 2024 CNN（one-hot 序列 + CNN，两阶段训练）
- **评估指标**：AUC、ACC（还有 F1 等）；核心设定是 **H1+H3 训练 → H5/H7 holdout 亚型外推迁移**

**核心结论（截至 2026-09）**：ESM-2 L3 层 embedding 在 H5 迁移上优于 CNN（jump AUC ~0.79 vs 0.46；jump_human H5 格 CNN 修正后 0.72 不落下风，需并置）；同等选择预算对齐后 CNN 四格无一达到 ESM 水平（09-12）；H7 存在方向反转（晚层原始 AUC < 0.5；`-logits` 翻转后 0.72–0.90 为历史全量诊断口径——target-val headline 数字为原始口径、全程无符号翻转，中层 L17/L13 在 H7 本不反转，协议审计见 `notes/09-27-01_h7_flip_protocol_audit.md`），失效边界 = H7/H10 亚支（09-11 H10 复现 + H4 对照）；raw 序列是主方案，MAFFT 对齐序列为补充。

---

## 2. 环境与运行

### 2.1 Conda 环境

主项目使用 conda 管理环境（**不是** `.venv/`，该目录基本为空，勿用）：

| 环境 | 路径 | 用途 | 关键依赖 |
|------|------|------|----------|
| `env1` | `/root/miniconda3/envs/env1` | **主环境**：数据处理 + ESM-2 实验 | Python 3.10, torch 2.11, transformers 5.13, scikit-learn 1.7, pandas, biopython, mord 0.7, xgboost 3.2 |
| `borkenhagen` | `/root/miniconda3/envs/borkenhagen` | CNN baseline 复现 | Python 3.10, torch 2.5.1+cu121 |

- base 环境缺少 transformers，跑 ESM 相关代码必须用 env1 的 Python。
- 系统工具：`cd-hit` 在 `/root/miniconda3/bin/`（base 环境），`mafft` 在 `/usr/bin/`。
- GPU：NVIDIA RTX 4080 SUPER（32 GB），ESM-2 embedding 提取和 CNN 训练都用 CUDA。

### 2.2 常用命令

所有脚本都从**项目根目录** `/root/autodl-tmp/FluJump` 运行（脚本内用 `sys.path.insert(0, ".")` 或相对路径）。

```bash
cd /root/autodl-tmp/FluJump

# ── 数据处理 pipeline（env1）──
python datascripts/run_pipeline.py                    # Phase 1-3 一键：清洗→CD-HIT 聚类→Jump 标签
python datascripts/build_isolate_dataset.py           # cluster 级 → isolate 级全量数据集
python datascripts/rebuild_isolate_tables.py          # clean/2026/aligned 三表（行序对齐 split；aligned 查 merged faa，不重跑 MAFFT）
python datascripts/build_splits.py                    # 生成 train/val/test/H5/H7 holdout 划分

# ── ESM-2 实验（env1）──
/root/miniconda3/envs/env1/bin/python ESM_clf/jump_exp/extract_embeddings.py   # 提取 embedding（~3 min on GPU）
/root/miniconda3/envs/env1/bin/python ESM_clf/jump_exp/train_probe.py          # Ridge LR 训练 + H5/H7 迁移评估

# ── ESM_tf_clf：transformer 头实验（env1）──
/root/miniconda3/envs/env1/bin/python ESM_tf_clf/extract_residue_embeddings.py  # per-residue embedding（L13/17/28，~10 min）
/root/miniconda3/envs/env1/bin/python ESM_tf_clf/train_tf_probe.py              # transformer 头训练 + target-val 协议评估

# ── CNN baseline（borkenhagen 环境）──
cd Borkenhagen/cnn_baseline
/root/miniconda3/envs/borkenhagen/bin/python train_stage1.py    # 阶段1：宿主预训练
/root/miniconda3/envs/borkenhagen/bin/python train_stage2.py    # 阶段2：jump 分类微调
```

### 2.3 测试策略

本项目**没有单元测试框架**（无 pytest、无 CI）。验证方式是：

- 实验脚本输出的数值指标（AUC/ACC，写入各 `output/*.json`）；
- 多 seed（5 个）重复评估确认稳定性；
- 诊断脚本（如 `ESM_clf/jump_exp/diagnose_h7.py`、`verify_h7_auc.py`）核查异常结果；
- 数据处理 pipeline 各阶段的行数/分布检查（见 `data/README.md` 中的行数表，可作为回归校验基准）。

修改实验代码后，至少重跑对应脚本并对比 `output/` 中 JSON 的关键指标；**每次实验必须写日志**（见第 5 节）。

---

## 3. 项目结构

```
FluJump/
├── AGENTS.md                       # 本文件（AI Agent 行为规范）
├── RESULT.md                       # 论文骨架 + 确定写进论文的结果记录（见第 6.1 节，产出论文级结果后必须更新）
├── README.md                       # 项目说明（Agent 只读，禁止修改）
├── data/                           # 数据目录（详见 data/README.md）
│   ├── raw/                        # NCBI 原始下载（禁止修改）
│   ├── processed/                  # 清洗 + CD-HIT 聚类 + cluster 级标签
│   ├── processed_isolate/          # isolate 级数据集（训练主数据 all_isolates_clean.csv, 11,060 行）
│   ├── processed_isolate_MAFFT/    # MAFFT 对齐数据集（all_isolates_aligned.csv）
│   ├── splits/                     # isolate_split.csv / cluster_split.json
│   ├── dataset_borkenhagen/        # Borkenhagen 2024 外部验证集（402 条 binding 数据）
│   └── dataset_borkenhagen_raw/    # Borkenhagen 原始数据
├── datascripts/                    # 数据处理脚本（详见 datascripts/README.md）
│   └── borkenhagen_data/           # Borkenhagen 数据解析脚本
├── mainpipeline/                   # 主流程代码（risk_scorer.py：单株风险分层 workflow，加载即重训 probe；性能优化时先备份旧版本到 backup/）
├── notes/                          # 统一笔记目录：对话笔记（MM-DD-HH-主题.md，连字符）+ 实验日志（MM-DD-HH_描述.md，下划线）
├── Alberts/                        # Alberts 2024 复现（仅文档）
├── Borkenhagen/                    # Borkenhagen 2024 复现
│   └── cnn_baseline/               # CNN baseline：config/data_prep/model/train_stage1/train_stage2/evaluate（+extract_intermediate_features/probe_target_val：选择预算对齐，AUC 一律 logit 口径）
├── Chen/                           # Chen 相关（目前为空）
├── ESM_clf/                        # ESM-2 分类器实验（主线）
│   ├── binding_exp/                # 实验1：Borkenhagen binding 数据集 probing
│   ├── jump_exp/                   # 实验2：jump / jump_human 二分类 + H5/H7 迁移
│   └── interval_exp/               # 实验3：interval 预测（4 类分类 + 回归，interval_data.py 共享 loader）
├── ESM_tf_clf/                     # ESM-2 冻结 per-residue embedding + 可训练头（L13/17/28，target-val 协议；train_tf_interval.py 为 interval 3 类变体；train_adaptive_pool.py 为 attention pooling 全层扫描，hidden states 常驻 RAM 不落盘）
├── validation_exp/                 # 严谨性与完整性验证实验（自包含，只读复用其他目录资产）
    ├── control_task/               # 表示 selectivity 对照（Hewitt–Liang 式打乱标签，linear + transformer）
    ├── external_binding/           # 实验 binding 数据外部锚定（Borkenhagen 402 条，方向的生物学对应验证）
    ├── scale_replication/          # ESM-2 650M/3B 多规模复现（层方向翻转的普适性 + target-val 选择）
    ├── naive_baseline/             # 朴素生物学基线（与 H1+H3 train 人源株的全局序列相似度排序，H5/H7 同协议对比 probe）
    ├── site_attribution/           # 位点归因（列↔canonical H3 编号映射经 1HGG 验证：canonical=原始−16；logit 逐位点分解 + ISM 饱和突变）
    ├── group_boundary/             # H7/H10 亚支边界零样本检验（H10 复现反转方向 24/31 层 p=0.003、H4 对照不反转；自包含 H10/H4 下载+聚类+全层 probe 打分）
    ├── reversal_sites/             # H7 vs H10 反转驱动位点对比（d_pos 推负剖面 6/6 格 ρ 0.63–0.72 > H4/H5 对照；共享位点不落 RBS；g1 主数据剖面 / g2 H10H4 贡献提取 / g3 重叠+结构映射 / g4 正交检验：否定 clade 指纹解读，机制开放）
    ├── depth_reversal/             # 深度方向翻转机制解释（C9：翻转载体=亚型内 ± gap 随深度变号，ρ(gap,AUC)=0.63–0.90 三规模同构；gap 变号边界随规模前移=最优深度前移机制；解近正交旋转 cos(w_best,w_last)≈0；G1 层几何 / G2 残差流（负贡献中层段写入）/ G3 残基×层（null）/ G4 注意力（null）/ G5 规模普适 / G6 交叉读出双旋转分解；V1 AA 身份解码（margin/PR U 形，最优层≈压缩最低点）/ V2 典型性耦合翻号（H5/H7 镜像）/ A1 谱系决定最优深度 6/6 格）
    └── temporal_validation/        # 时间维度验证（2026 前向外推：probe AUC 0.9995、阳性全进 top 11/201、朴素基线 0.046 反向；H7N9 回溯阴性：失败特异于 H7/Group 2，同亚型跨时间对照 0.787；前向加固 run_2026_forward_hardening.py：cluster bootstrap CI/permutation p/≤2025 标签口径/L1 对照层；run_2026_workflow_eval.py：单株分层前瞻验证，中界 8/10 命中零误报、高层因分布漂移不触发；run_multiyear_forward.py：多年份前向复现 2024/2025，训练标签按 ≤cutoff 簇成员重算防未来信息泄漏）
└── jev_exp/                        # JEV 判别器实验（新建 2026-09-26：自包含、不改现有 workflow 任何代码，用 jev 作分类器与主线 probe 对比；算法细节待定）
```

---

## 4. 代码组织与约定

### 4.1 模块划分

- **`datascripts/`**：线性数据处理 pipeline，每个脚本对应一个 Phase，输入输出都是 `data/` 下的 CSV/FASTA。脚本职责见 `datascripts/README.md` 的表格。
- **`ESM_clf/<实验>/`**：每个实验目录自包含，统一模式：
  - `config.py`：所有路径、超参数集中定义（`from ESM_clf.xxx.config import *` 导入）
  - `extract_embeddings.py`：用 transformers `EsmModel`（无 LM head）提取 L1/L3/L1L3 mean-pooled embedding，存 `.npy`
  - `train_*.py` / `compare_*.py`：训练与对比脚本，结果写 `output/*.json`
  - embedding 按 CSV 行顺序存储，直接按行与数据对齐
- **`Borkenhagen/cnn_baseline/`**：复现代码，`config.py` + `data_prep.py` + `model.py` + 两阶段训练脚本，模型权重存 `output/models/*.pt`。
- **`mainpipeline/`**：主流程代码目录。现有 `risk_scorer.py`（单株风险分层 workflow：序列 → ESM-2 L3 → logit → train 分位数 → 高/中/低分层，阈值仅用 train 固定，H7/H10 判边界外）。

### 4.2 代码风格

- **注释与文档字符串一律用中文**（项目现有惯例），代码标识符用英文。
- 配置集中放 `config.py`，用模块级常量（全大写），脚本开头 `sys.path.insert(0, ".")` 后 `from xxx.config import *`。
- 随机种子统一 `RANDOM_SEED = 42`；评估统一 StandardScaler + GridSearchCV（5-fold stratified）。
- 结果保存为 JSON 到对应实验的 `output/` 目录。
- 数据文件列名、划分策略等事实以 `data/README.md` 为准，修改数据后需同步更新该文件。

### 4.3 重要注意事项（踩坑记录）

- **H7 方向反转**：H7 holdout 上模型预测方向与训练方向相反（原始 AUC ~0.1–0.28）。翻转 AUC 必须用 `decision_function` 的 `-logits` 计算，不能用 `1-p`（H7 预测概率极端集中在 0 附近，`1-p` 数值不稳定）。**同类坑（09-12 扩展）**：一切 AUC/排序评估都必须用 raw logits——CNN 侧 `evaluate.py`/`eval_h5h7_valtest.py` 曾用 float32 sigmoid 概率算 AUC，|logit|>88 时饱和退化为并列秩（jump_human H5 被低估为 0.549/0.597，logit 真值 0.702/0.723）；两脚本已于 09-12 改为 logit 口径并重跑。
- **ESM-2 用 raw 序列**（无 gap）提取 embedding 与预训练分布一致，是主方案；对齐序列中的 gap token（id=30）需在 mean pooling 时排除，且会拖累 H5 迁移。
- **L3 层（倒数第三层）** embedding 在跨亚型迁移上一致优于 L1（最后一层）。
- interval 标签：2026-08-01 起固定为 4 类（`human_first` / `<1yr` / `1-3yr` / `3yr+`），由 interval_exp/interval_data.py 从 clean CSV 重算**未 clamp** 的有符号天数派生（`iso_interval_days` 列本身已在 build_isolate_dataset.py 中被 clamp，负值信息需重算恢复）。2026-08-10 新增 3 类变体 `y_cat3`（human_first 并入 `<1yr`）。
- interval 3 类任务有两套数据设定：**H1+H3 isolate 级划分**（默认，沿用 isolate_split.csv）与 **H1357 cluster 级 70/15/15 划分**（`data/splits/interval_h1357_cluster_split.csv`，无泄漏；经 `load_interval_data(subtypes=..., split_csv=...)` 使用）。注意 cluster 级划分下指标远低于 isolate 级（bal_acc 0.42 vs 0.76），旧 isolate 级数值偏乐观。
- **跨文件 join 必须按 (accession, subtype)**：同一 accession 可在多个亚型下重复出现（260+ 个），且 `all_isolates_clean.csv`/`all_isolates_aligned.csv` 与 `isolate_split.csv`/embedding 行序不一致。embedding 行序与 `isolate_split.csv` 行序逐行一致，严禁按 CSV 行号位置对齐（interval_exp 一律走 `interval_data.load_interval_data()`）。
- **canonical H3 编号 ≠ 原始序列位置**：差 16（信号肽不入编号；HA2 从原始 346 起单独编号）。对齐列 ↔ canonical 编号映射已固化在 `validation_exp/site_attribution/output/col_to_h3.json`（经 PDB 1HGG/X-31 验证：canonical 226 = 原始 242 = L）。
- **跑 ESM 相关脚本须加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`**：模型在本地缓存，但 transformers 5.x 默认先连 HF Hub，网络不通时无报错挂起（进程 poll 在 443 端口）。
- **host_category 推断必须词边界匹配**（09-19 审计）：裸子串 `cat`/`pig`/`cow`/`air`/`water` 曾误伤 furcata/flycatcher/pigeon/Moscow/Cairo/shearwater（H10_143 假阳性簇来源）；`clean_data.py`/`download_flu.py`/`supplement_h5.py` 现为 `\b` 词边界 + 垃圾 species（"Influenza A virus (A/...)" 串位）内嵌株名提取 + improve-only 覆盖。簇键一律 (subtype, cluster_id)——cluster_id 是亚型内编号，跨亚型复用率 76%。

---

## 5. 笔记规则

当用户说"做笔记"或"记笔记"时，按以下步骤执行：

1. 如果用户没有指定主题名，从对话中推断（取核心任务的关键词），并向用户确认。
2. 在 `notes/` 下创建 `MM-DD-HH-主题.md`（例：`07-10-15-蛋白质结构预测.md`），小时用 24 小时制。
3. 文件内容按以下模板输出（缺失的章节写"无"）：

| 章节 | 内容 |
|------|------|
| 背景与目标 | 本次要解决的问题或实验目的 |
| 关键步骤 | 按时间顺序列出命令、代码修改、工具调用 |
| 结果与发现 | 重要输出、数据、报错、解决方式 |
| 涉及文件 | 本次修改/创建的文件路径列表 |
| 待办 | 下一步需要做的事 |

4. 只提取对话中的实质信息，不要复制闲聊或无意义轮次。
5. 创建后告知用户文件路径，并询问是否需要修改内容。

---

## 6. 实验日志

**每次实验完成后必须写实验日志。** 日志存放于 `notes/` 目录（与对话笔记同目录，靠文件名区分），文件名格式：

```
notes/MM-DD-HH_<简短描述>.md
```

### 日志内容模板

```
# 实验：<简短描述>

- **日期**：MM-DD-HH
- **目的**：本次实验要验证什么

## 方法
- 模型/参数/数据配置的变更要点

## 结果
- 关键指标（必须有数值）
- 与 baseline 对比（如有）

## 结论
- 是否达到预期
- 下一步行动（继续 / 调整 / 放弃）
```

### 日志要求

| 要求 | 说明 |
|------|------|
| 每次实验必写 | 无论结果好坏，哪怕只跑了几行代码 |
| 文件名带日期 | 便于按时间排序和回顾 |
| 关键指标必须有数值 | 不能只写"效果好" |
| 结论明确标注下一步行动 | 继续 / 调整 / 放弃 |

---

## 6.1 论文结果记录（RESULT.md）

`RESULT.md`（项目根目录）记录**确定会写进论文的结果**，按论文骨架的五个小节（§1 线性读出存在 / §2 生物学锚定 / §3 失效边界 / §4 深度与规模 / §5 监测排序应用）组织。

规则：

- **每产出一个论文级结果必须更新 `RESULT.md` 对应小节**（在"确定结果"列表中追加数值 + 出处，或勾掉"待补"项）。
- 进入 `RESULT.md` 的标准：实验完成、数值稳定（多 seed / bootstrap 验证通过）、已写实验日志（notes/）。未达标准的结果只进实验日志，不进 `RESULT.md`。
- `RESULT.md` 中的数值必须与实验日志一致；更新时注明代码目录与日志文件出处。
- 论文叙事、风险项发生变化时，同步更新文件开头的"核心叙事"；完整度/创新点/TODO 记录在末尾的"项目审阅"节。

---

## 7. 安全与数据保护

- `data/raw/` 是原始下载数据，**禁止修改**；数据处理一律输出到新文件。
- 不要向外部服务发送数据文件或序列数据；ESM-2 模型从 HuggingFace 本地缓存加载，推理在本地 GPU 完成。
- 不要修改 `README.md`（只读）。
- `mainpipeline/` 中存储主流程代码，如果有更好性能的方法，则备份之前的主流程到 `mainpipeline/backup/` 后再替换。
- 没有密钥/凭证管理需求；仓库无 `.env` 类敏感文件，也不要新增。

---

## 8. 其他规则

- Agent 可以阅读 `README.md`，但不能修改 `README.md`。
- 各子目录的 README（`data/README.md`、`datascripts/README.md` 等）记录该目录的最新状态，修改对应内容后需同步更新。
