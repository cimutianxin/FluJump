# datascripts — FluJump 数据处理脚本

运行顺序: `run_pipeline.py` 一键执行 Phase 1-3

## 主流程

| 脚本 | Phase | 输入 | 输出 | 说明 |
|------|:---:|------|------|------|
| `download_flu.py` | 0 | NCBI Protein | `data/raw/{H1,H3,H5,H7}_raw.csv` | 从 NCBI 下载 HA 序列及元数据 |
| `clean_data.py` | 1 | `data/raw/*.csv` | `data/processed/{subtype}_clean.csv` | 过滤 PDB/非 HA/合成, 改进 host 推断（09-19：`\b` 词边界 + 垃圾 species 内嵌株名提取 + improve-only 覆盖）, 无日期→year=-1 |
| `cluster_strains.py` | 2 | `data/processed/*_clean.csv` | `data/processed/{subtype}_clusters.tsv` + `_consensus.fasta` | CD-HIT 99% identity 聚类 |
| `build_jump_labels.py` | 3 | `*_clean.csv` + `*_clusters.tsv` | `data/processed/all_subtypes_simplified.csv` | 宿主时间线 → 20 列简化标签数据集 |
| `run_pipeline.py` | 1-3 | — | `all_subtypes_simplified.csv` | ★ 一键执行 Phase 1-3 |

## 辅助脚本

| 脚本 | 说明 |
|------|------|
| `supplement_h5.py` | 补充下载 H5 牛源(43) + 人源(634)、H7 人源(255) |
| `compare_thresholds.py` | CD-HIT 参数 sweep (85%-99.9%) |
| `build_isolate_dataset.py` | 从 cluster 级生成 isolate 级全量数据集 `all_isolates.csv`（cluster 映射 key 为 (subtype, accession)，08-02 修复） |
| `rebuild_isolate_tables.py` | 从 all_isolates.csv 重建 clean/2026/aligned 表：过滤 unknown、排除 2026、行序对齐 isolate_split.csv（08-02 新增） |
| `build_aligned_dataset.py` | 从 clean + `all_aligned_merged.faa` 生成 aligned 表（08-02 修复为裸 accession 查对齐序列） |
| `build_h5h7_valtest_splits.py` | H5/H7 内部 5:5 val/test 切分：cluster 臂（主）+ isolate_random 臂（泄露对比），各 5 seeds → `data/splits/h5h7_valtest/` |
| `build_interval_h1357_cluster_split.py` | interval 3 类任务专用：H1357 全体 jump=1 且 interval 有效（1,191 条/44 clusters），cluster 级 70/15/15 随机搜索均衡划分 → `data/splits/interval_h1357_cluster_split.csv`（08-10 新增） |
| `audit_host_label_impact.py` | host 审计标签影响量化：还原 `host_audit_fix_diff.csv` 49 行旧值重算簇标签，对比现行表（09-19 新增） |

## 一键运行

```bash
cd /root/autodl-tmp/FluJump
python datascripts/run_pipeline.py                           # cluster 级
python datascripts/build_isolate_dataset.py                  # isolate 级全量
python datascripts/rebuild_isolate_tables.py                 # clean/2026/aligned（行序对齐 split）
```
