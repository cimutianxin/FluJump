# FluJump 数据目录说明

> 最后更新: 2026-08-02

---

## 目录结构总览

```
data/
├── raw/                          # NCBI 原始下载
├── processed/                    # 清洗 + CD-HIT 聚类 + Jump 标签
├── processed_isolate/            # Isolate 级数据集（训练用）
├── processed_isolate_MAFFT/      # MAFFT 对齐后的 isolate 数据集
├── splits/                       # 训练/验证/测试划分
├── dataset_borkenhagen_raw/      # Borkenhagen 2024 原始外部数据
└── dataset_borkenhagen/          # Borkenhagen 处理后的外部验证集
```

---

## 1. raw/ — NCBI 原始下载

NCBI Protein 数据库查询 `Influenza A virus hemagglutinin {H1/H3/H5/H7}`。

| 文件 | 行数 | 列数 | 说明 |
|------|:---:|:---:|------|
| `H1_raw.csv` | 7,170 | 22 | H1 原始数据 |
| `H3_raw.csv` | 6,673 | 22 | H3 原始数据 |
| `H5_raw.csv` | 3,337 | 22 | H5 原始数据（含补充下载） |
| `H7_raw.csv` | 1,500 | 22 | H7 原始数据（含补充下载） |

**22 列**: accession, strain_name, ha_sequence, host_species, collection_date, serotype, geo_location, country, isolation_source 等

**生成脚本**: `datascripts/download_flu.py`, `datascripts/supplement_h5.py`

---

## 2. processed/ — 清洗 + 聚类 + 标签

Pipeline: 清洗 → CD-HIT 99% 聚类 → Jump 标签

### 2.1 清洗数据

| 文件 | 行数 | 说明 |
|------|:---:|------|
| `H1_clean.csv` | 4,113 | 去 PDB/synthetic/长度异常 |
| `H3_clean.csv` | 4,682 | 同上 |
| `H5_clean.csv` | 2,020 | 同上 |
| `H7_clean.csv` | 899 | 同上 |

过滤: PDB 条目 3,384 / seq_length <400 或 >600 共 3,580 / synthetic construct 2  
保留无日期的记录（year=-1, 584 条）

### 2.2 聚类结果

CD-HIT 参数: 99% identity, `-s 0.8 -g 1 -n 5`

| 文件 | 行数 | 说明 |
|------|:---:|------|
| `H1_clusters.tsv` | 4,113 | `accession → cluster_id` 映射 |
| `H3_clusters.tsv` | 4,682 | 同上 |
| `H5_clusters.tsv` | 2,020 | 同上 |
| `H7_clusters.tsv` | 899 | 同上 |
| `H1_consensus.fasta` | 908 seqs | 908 个 cluster 的代表序列 |
| `H3_consensus.fasta` | 693 seqs | 693 个 cluster 的代表序列 |
| `H5_consensus.fasta` | 499 seqs | 499 个 cluster 的代表序列 |
| `H7_consensus.fasta` | 262 seqs | 262 个 cluster 的代表序列 |

### 2.3 标签数据集（Cluster 级）

| 文件 | 行数 | 列数 | 说明 |
|------|:---:|:---:|------|
| `all_subtypes_simplified.csv` | 2,362 | 20 | **Cluster 级完整标签数据集** |

**20 列**:

| 列 | 含义 |
|------|------|
| `cluster_id` | CD-HIT cluster ID |
| `subtype` | H1/H3/H5/H7 |
| `cluster_size` | cluster 内 isolate 数 |
| `consensus_ha_seq` | cluster 代表 HA 序列 |
| `serotype` | H1N1/H1N2/... |
| `first_{host}_ym` | 各宿主最早检出时间 (YYYY-MM) |
| `host_categories` | cluster 内出现的宿主列表 |
| `label_is_jump` | 任意 ≥2 宿主 (75, 3.2%) |
| `jump_interval_cat` | 所有宿主最早→最晚时间跨度（07-12 修复，非方向性）: `<1yr`(47)/`1-3yr`(20)/`3-5yr`(6)/`5yr+`(2) |
| `label_is_jump_human` | 含 human + ≥1 动物 (48, 2.0%) |
| `jump_interval_human_cat` | 同上，允许 negative |
| `label_jump_source` | 来源宿主: avian/swine/bovine |

**生成脚本**: `datascripts/build_jump_labels.py`

---

## 3. processed_isolate/ — Isolate 级数据集

每个 isolate 继承其 CD-HIT 99% cluster 的标签，并拥有独立的 isolate 级 interval。

| 文件 | 行数 | 列数 | 说明 |
|------|:---:|:---:|------|
| `all_isolates.csv` | 11,714 | 27 | **全量**（含 `host_category=unknown` 与 2026 年） |
| `all_isolates_clean.csv` | 11,060 | 27 | **训练主数据集**，去掉 unknown 宿主 + 排除 2026 年；行序 == `isolate_split.csv` == embedding 行序 |
| `2026_isolates_clean.csv` | 201 | 27 | 2026 年独有数据（外部验证/前瞻测试） |

**27 列**:

| 类别 | 列名 | 说明 |
|------|------|------|
| 标识 | `accession`, `subtype`, `cluster_id`, `cluster_size`, `strain_name` | — |
| 序列 | `ha_sequence`, `serotype` | HA 蛋白序列 |
| 宿主 | `host_category`, `host_species` | isolate 自身宿主 |
| 时间 | `collection_year`, `collection_month`, `collection_date` | YYYY-MM |
| 地理 | `geo_location`, `country`, `isolation_source` | — |
| Cluster 标签 | `cluster_hosts` | cluster 内宿主列表 |
| | `label_is_jump` | 任意跨物种 (1,382 条) |
| | `jump_interval_cat` | cluster 级年分箱 |
| | `label_is_jump_human` | 跨物种到人 (1,209 条) |
| | `jump_interval_human_cat` | cluster 级年分箱 |
| | `label_jump_source` | 来源宿主 |
| **Isolate 级 interval** | **`iso_interval_days`** | 🆕 每个 isolate 独立计算的连续天数 |
| | **`iso_interval_cat`** | 🆕 每个 isolate 独立计算的年分箱 |
| Cluster 时间 | `first_human_ym`, `first_avian_ym`, `first_swine_ym`, `first_bovine_ym` | 各宿主最早检出 |

**iso_interval_cat 分布**（`label_is_jump_human=1` 的 1,209 条）:

| 分类 | 数量 | 含义 |
|------|:---:|------|
| `<1yr` | 1,033 | 1 年内跨种（含负值 clamp→0：人检出早于动物的 764 条，见下） |
| `1-3yr` | 91 | 1–3 年 |
| `3-5yr` | 60 | 3–5 年 |
| `5yr+` | 7 | 5 年以上 |
| (空) | 18 | 日期缺失 |

> 注意：`iso_interval_days` 在 `build_isolate_dataset.py` 中将负值 clamp 为 0
> （语义：人先于动物检出 = 跨物种能力已激活 = 最高危），因此 `iso_interval_cat`
> 不再有 `negative` 类。未 clamp 的有符号天数可由
> `ESM_clf/interval_exp/interval_data.py` 从现有列重算恢复。

**生成脚本**: `datascripts/build_isolate_dataset.py`（全量）+ `datascripts/rebuild_isolate_tables.py`（clean/2026 分离 + 行序对齐 split）

---

## 4. processed_isolate_MAFFT/ — MAFFT 对齐数据集

以 H3 reference (A/Aichi/2/1968) 为基准的多序列对齐，所有序列统一长度。

| 文件 | 行数 | 列数 | 说明 |
|------|:---:|:---:|------|
| `all_isolates_aligned.csv` | 11,060 | 29 | **MAFFT 对齐主数据集**，不含 2026；行序 == `isolate_split.csv` |
| `2026_isolates_aligned.csv` | 201 | 29 | 2026 年对齐数据 |
| `H1_aligned.faa` | 4,069 seqs | — | H1 亚型内对齐 |
| `H3_aligned.faa` | 4,597 seqs | — | H3 亚型内对齐 |
| `H5_aligned.faa` | 1,757 seqs | — | H5 亚型内对齐 |
| `H7_aligned.faa` | 838 seqs | — | H7 亚型内对齐 |
| `all_aligned_merged.faa` | 11,261 seqs | — | 合并对齐 FASTA（含 2026，header 为裸 accession） |

**29 列** = 27 列（同 `all_isolates_clean.csv`）+ 对齐专用列:

| 新增列 | 说明 |
|------|------|
| `aligned_ha_seq` | 对齐后序列（含 gap，1070 列） |
| `aligned_length` | 对齐后长度 |
| `gap_count` | gap 数量 |
| `gap_ratio` | gap 比例 |

**生成脚本**: `datascripts/rebuild_isolate_tables.py`（或修复后的 `datascripts/build_aligned_dataset.py`，读 `all_aligned_merged.faa`）

---

## 5. splits/ — 数据划分

| 文件 | 说明 |
|------|------|
| `isolate_split.csv` | Isolate 级划分 (11,060 行): train=5,933 / val=1,267 / test=1,275 / h5_holdout=1,747 / h7_holdout=838；行序 == clean/aligned/embedding |
| `cluster_split.json` | ⚠️ **已过时**（07-10 旧版，与 isolate_split.csv 不一致，且 cluster_id 各亚型内从 0 重启、跨亚型有歧义）。请使用 `isolate_split.csv` |
| `h5h7_valtest/` | H5/H7 内部 5:5 val/test 切分（target-val 层选择协议）：`cluster_seed{42..46}.csv`（cluster 整体归入 val/test，主切分）+ `isolate_random_seed{42..46}.csv`（按行随机，泄露对比臂）。列：accession, subtype, cluster_id, split ∈ {h5_val, h5_test, h7_val, h7_test}；每文件 2,585 行（仅 H5/H7） |
| `interval_h1357_cluster_split.csv` | interval 3 类任务专用（1,191 行）：H1357 全体 jump=1 且 interval 有效的 isolate，**cluster 级 70/15/15**（无 cluster 跨 split；随机搜索均衡划分，seed=42）。train=834 / val=172 / test=185。配套摘要 `interval_h1357_cluster_split.json` |

**划分策略**: H5/H7 整体 holdout（亚型外推测试），H1/H3 按 cluster 7:1.5:1.5 划分。`h5h7_valtest/` 在 holdout 内部再切 val（选层）/ test（只评一次），使用时经 isolate_split.csv 的 (accession, subtype) 映射回 embedding 行号。

**生成脚本**: `datascripts/build_splits.py`；`datascripts/build_h5h7_valtest_splits.py`（h5h7_valtest/）；`datascripts/build_interval_h1357_cluster_split.py`（interval_h1357_cluster_split.csv）

---

## 6. dataset_borkenhagen/ — Borkenhagen 2024 外部验证集

Borkenhagen et al. 2024 的流感 HA 结合 assay 数据，用于外部验证。

### 6.1 原始数据 (dataset_borkenhagen_raw/)

| 文件 | 说明 |
|------|------|
| `DataS1_sequences.fasta` | 406 条 HA 序列 |
| `irv70044-sup-0003-supplementary_file2.csv` | 406 条 binding label (ID, Label) |

### 6.2 处理后数据 (dataset_borkenhagen/)

| 文件 | 行数 | 列数 | 说明 |
|------|:---:|:---:|------|
| `borkenhagen_clean.csv` | 402 | 5 | 清洗后 binding 数据 |
| `borkenhagen_split.csv` | 402 | 4 | 训练/测试划分 |
| `borkenhagen.faa` | 402 seqs | — | HA 序列 FASTA |
| `borkenhagen_aligned.faa` | 403 seqs | — | 含 H3 参考的对齐 |
| `combined.faa` | 403 seqs | — | 合并文件 |
| `h3_ref.faa` | 1 seq | — | H3 参考序列 A/Aichi/2/1968 |

**生成脚本**: `datascripts/borkenhagen_data/`

---

## 数据流全景

```
NCBI Protein (18,680)
  ↓ download_flu.py + supplement_h5.py
raw/{H1,H3,H5,H7}_raw.csv
  ↓ clean_data.py
processed/{H1,H3,H5,H7}_clean.csv (11,714)
  ↓ cluster_strains.py (CD-HIT 99%)
processed/{H1,H3,H5,H7}_clusters.tsv + _consensus.fasta (2,362 clusters)
  ↓ build_jump_labels.py
processed/all_subtypes_simplified.csv (2,362 clusters, 20 列)
  ↓ build_isolate_dataset.py
processed_isolate/all_isolates.csv (11,714 全量，含 unknown + 2026)
  ↓ rebuild_isolate_tables.py（过滤 host=unknown + 分离 2026 + 行序对齐 split）
processed_isolate/all_isolates_clean.csv (11,060, 训练用)
processed_isolate/2026_isolates_clean.csv (201, 前瞻验证)
  ↓ rebuild_isolate_tables.py / build_aligned_dataset.py（查 all_aligned_merged.faa，不重跑 MAFFT）
processed_isolate_MAFFT/all_isolates_aligned.csv (11,060 × 29 列)
processed_isolate_MAFFT/2026_isolates_aligned.csv (201 × 29 列)
```
