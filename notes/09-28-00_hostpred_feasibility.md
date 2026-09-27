# 实验：VirHostPRED（Beltrán et al. 2026）外部基线可行性与冒烟测试

- **日期**：09-28-00
- **目的**：验证已发表病毒宿主预测工具 VirHostPRED 能否作为 FluJump 的外部基线——定位官方代码/权重、跑通最小输入、对 20 条 FluJump 序列冒烟、估算全量打分成本。**不做**全量打分与正式评估，PLL 兜底分支未触发（官方工具可用）。

## 工具定位（步骤 1）

- **论文**：Beltrán JF, et al. "Protein language models enable accurate viral host range prediction." Sci Rep 16:7606 (2026). DOI 10.1038/s41598-026-37765-8（投稿 2025-11-09，接收 2026-01-24）。reference.bib 键 virhostpred2026。
- **官方仓库**：**无公开代码仓库**。GitHub 搜索 `VirHostPred` 零命中；论文 Data availability 称数据集见补充材料、"Additional metadata and scripts ... available from the corresponding author upon reasonable request"。
- **权重**：**不可得**。训练好的 SVM-RBF + StandardScaler 未公开，唯一可执行形态是官方 web server：https://www.biochemintelli.com/virhostpred/ （Django 前端 + Google Cloud GPU 后端，同步 POST 返回 HTML 结果表）。无版本号/tag；以访问日期 2026-09-28 与页面构建戳（site-ui v=pagespeed-20260814a, studio-runtime v=20260610）为版本锚点。
- **模型**：ESM2-t48-15B 末层 mean-pool embedding（>1022 aa 滑窗 900/150）+ SVM-RBF；holdout AUC 0.914 / ACC 0.852。
- **训练集构成（泄漏声明用，摘自正文方法与 SI Table S1/S2/S5）**：NCBI Virus **RefSeq** 蛋白，complete nucleotide + complete assembly；正类 Host=Homo sapiens (taxid 9606)，负类 Host≠9606；CD-HIT 70%（-c 0.7 -n 5 -d 0）去冗余后 2,127 人源 + 10,635 非人源，负类随机下采样（seed 42）至 2,127，平衡集 4,254 条；80/20 分层划分（3,403/851）。负类 73.2% 为噬菌体。**注意：标签是"病毒（种）是否感染人"，非"该分离株采自人"**。论文/SI **未给数据检索日期**；由投稿日推断截止 ≤2025-11。SI 仅为 docx 统计表，训练序列本身未沉积（Europe PMC 与 Nature SI 均只有 MOESM1-8 表格/图）。
- **输入要求**：蛋白 FASTA（非核苷酸）；单次 ≤100 条；最短 10 aa；**只接受标准 20 氨基酸字母，含 X 等非标准字符直接 HTTP 400**（实测，与论文"替换为 X"的自建流程不一致）；HA ~560 aa 远低于 1022 上限。fasta 头取首字段作 Sequence ID。
- **输出字段**：每条一行 `HUMAN VIRUS: xx.x%` 或 `NON-HUMAN VIRUS: xx.x%`（预测类别 + 其概率）。可解释为 P(human)：HUMAN 时 p=xx.x%，NON-HUMAN 时 p=1−xx.x%（脚本已按此折算，列 `p_human`；原文保留 `raw_prediction`）。无多类宿主概率。

## 安装与示例（步骤 2）

无需安装（web-only；未新建 conda 环境，未装新包；用 env1 的 requests 2.34.2 / pandas 2.3.3）。最小示例 = 页面占位序列 1 条 POST：

```
GET  /virhostpred/                     -> 取 csrftoken cookie + csrfmiddlewaretoken
POST /virhostpred/ (Referer+Origin 头必须, 否则 403 CSRF fail)
     fields: csrfmiddlewaretoken, fasta_sequence=">example_seq_placeholder\nMSMRTRNWCVTSW..."
-> 200, 11.6 s, "NON-HUMAN VIRUS: 75.6%"
```

## 冒烟测试（步骤 3，20 条）

抽样：`prepare_smoke_fasta.py`，seed 42；**抽样池预先过滤非标准残基序列**（服务端拒绝 X，见上）。join 一律按 (accession, subtype)。全 20 条一批 POST：HTTP 200，**14.5 s**，20/20 解析成功。

| accession | subtype | cluster_id | host_category（我方标签） | 组 | jump_human | VirHostPRED 原始输出 | p_human |
|---|---|---|---|---|---|---|---|
| WZL43931 | H3 | 123 | human | human_H3 | 0 | HUMAN VIRUS: 85.6% | 0.856 |
| XFD28191 | H3 | 123 | human | human_H3 | 0 | HUMAN VIRUS: 85.3% | 0.853 |
| WBR44974 | H3 | 126 | human | human_H3 | 0 | HUMAN VIRUS: 86.7% | 0.867 |
| WZL42111 | H3 | 123 | human | human_H3 | 0 | HUMAN VIRUS: 85.3% | 0.853 |
| WZL48519 | H3 | 123 | human | human_H3 | 0 | HUMAN VIRUS: 85.3% | 0.853 |
| ADM73543 | H1 | 903 | human | human_H1 | 0 | NON-HUMAN VIRUS: 54.0% | 0.460 |
| YCB21064 | H1 | 6 | human | human_H1 | 0 | HUMAN VIRUS: 87.8% | 0.878 |
| XFC83801 | H1 | 32 | human | human_H1 | 0 | HUMAN VIRUS: 87.2% | 0.872 |
| XFC81170 | H1 | 32 | human | human_H1 | 0 | HUMAN VIRUS: 87.2% | 0.872 |
| ADM73533 | H1 | 903 | human | human_H1 | 0 | NON-HUMAN VIRUS: 54.0% | 0.460 |
| ALP30744 | H5 | 397 | avian (chicken) | avian_H5 | 0 | HUMAN VIRUS: 84.8% | 0.848 |
| BAJ23238 | H5 | 344 | avian (duck) | avian_H5 | 0 | HUMAN VIRUS: 85.8% | 0.858 |
| AAN17266 | H5 | 315 | avian (mallard, H5N2 1984) | avian_H5 | 0 | HUMAN VIRUS: 81.6% | 0.816 |
| QPM32908 | H5 | 94 | avian (chicken) | avian_H5 | 0 | HUMAN VIRUS: 85.5% | 0.855 |
| ALB07520 | H5 | 94 | avian (goose) | avian_H5 | 0 | HUMAN VIRUS: 85.4% | 0.854 |
| YEV12912 | H5 | 70 | avian (Branta canadensis, 2026) | avian_H5_2026_c70 | 1 | HUMAN VIRUS: 86.5% | 0.865 |
| YEO13386 | H5 | 70 | avian (Branta canadensis, 2026) | avian_H5_2026_c70 | 1 | HUMAN VIRUS: 86.7% | 0.867 |
| YEO13382 | H5 | 70 | avian (Anser caerulescens, 2026) | avian_H5_2026_c70 | 1 | HUMAN VIRUS: 86.6% | 0.866 |
| YEO13359 | H5 | 70 | avian (Branta canadensis, 2026) | avian_H5_2026_c70 | 1 | HUMAN VIRUS: 86.4% | 0.864 |
| YEO13314 | H5 | 70 | avian (Branta canadensis, 2026) | avian_H5_2026_c70 | 1 | HUMAN VIRUS: 86.7% | 0.867 |

组均值 p_human：human_H3 0.856 / human_H1 0.708 / avian_H5 0.846 / avian_H5_2026_c70 0.866。

**结果解读（冒烟级，非评估）**：工具语义是种级"该病毒能否感染人"——禽流感 H5（含 2026 野鸟 c70 簇）全部判 HUMAN ~0.82–0.87，与我方分离株级 host_category=avian 标签系统性不一致；这不是工具错误，而是**任务定义差异**（甲流作为种确实感染人）。隔离级动态范围仍存在：两条 2009 季节性 H1（ADM73543/ADM73533，cluster 903）被判 NON-HUMAN 54%，说明分数对株系差异敏感。作为 jump/宿主基线的可用性需下轮正式评估（AUC vs host_category、vs label_is_jump_human）。

## 计时与成本估算（步骤 4）

实测：单条 POST 11.6 s（含 GET token + 服务端排队）；20 条一批 14.5 s（≈0.73 s/条，批量摊薄后单条边际成本极低）。全量（按任务给定的 21,104 条目口径 = H5/H7 holdout 2,585 + 前向池 2,677/1,020/201 + 训练集 3,426/5,262/5,933；去重后 ~15,000）：

- 100 条/批 → ~212 批；按 0.73 s/条 + 批间 2 s 礼貌间隔 ≈ **4.5–5 h 上限**；若服务端批处理亚线性（20 条仅 14.5 s），可能 1–2 h。
- **远低于 24 h 停止线**；无需本地 GPU；需要断点续跑（每批落盘 checkpoint）与 400/5xx 重试退避。
- 风险：服务端未知限流/排队；含 X 序列需跳过（我方全量中此类序列占比需在正式跑前统计）。

## 遇到的 blocker

1. 无公开仓库与权重（仅 web server；脚本/元数据需邮件索取）。
2. POST 403：Django CSRF 要求 Referer/Origin 头（已解决）。
3. 20 条首跑 HTTP 400：WZL46639 含 X，服务端拒绝非标准残基（改为抽样池预过滤；不重写序列）。
4. PMC 补充材料下载有 JS cookie 拦截 → 改用 Europe PMC `supplementaryFiles` 端点；确认 SI 只有统计表，训练序列未公开。
5. Nature 正文页反爬 → 方法与数据可用性信息取自 PMC 全文（PMC12936077）。

## 结论与下一轮建议

- **可行性成立**：官方 web server 可程序化批量打分，输出可直接折算 P(human)。不触发任何停止条件。
- **下一轮主分数**：`p_human`（由 "HUMAN/NON-HUMAN VIRUS: xx.x%" 折算）。建议全量跑批脚本加 batch=100 + checkpoint + 重试；预计 1–5 h。
- **必须先行**：泄漏审计——VirHostPRED 训练集含 RefSeq 人源病毒蛋白（几乎必然含人流感 HA），与 FluJump 训练/评估序列可能重叠；其序列集未公开，需按 accession 反查或邮件索取后再下结论。
- **解释框架**：该工具预测种级宿主范围而非分离株宿主，avian H5 全判 HUMAN 是预期行为；基线价值在于分数的株级变异（0.46–0.88），正式评估应对 host_category 与 jump 标签分别算 AUC。
- **下一步行动**：继续（下一轮做全量打分 + 泄漏审计 + 正式评估）。

## 涉及文件

- `validation_exp/hostpred_baseline/prepare_smoke_fasta.py`（抽样/写 fasta+meta）
- `validation_exp/hostpred_baseline/run_virhostpred.py`（POST/解析/折算 p_human/join）
- `validation_exp/hostpred_baseline/output/smoke20.fasta`、`smoke20_meta.csv`（输入快照）
- `validation_exp/hostpred_baseline/output/smoke_scores.csv`（20 条结果）、`smoke_timing.csv`（计时）
