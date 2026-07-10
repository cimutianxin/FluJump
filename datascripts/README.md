# datascripts — FluJump 数据处理脚本

运行顺序: `run_pipeline.py` 一键执行 Phase 1-3

## 主流程

| 脚本 | Phase | 输入 | 输出 | 说明 |
|------|:---:|------|------|------|
| `download_flu.py` | 0 | NCBI Protein | `data/raw/{H1,H3,H5,H7}_raw.csv` | 从 NCBI 下载 HA 序列及元数据 |
| `clean_data.py` | 1 | `data/raw/*.csv` | `data/processed/{subtype}_clean.csv` | 过滤 PDB/非 HA/合成, 改进 host 推断, 无日期→year=-1 |
| `cluster_strains.py` | 2 | `data/processed/*_clean.csv` | `data/processed/{subtype}_clusters.tsv` + `_consensus.fasta` | CD-HIT 99% identity 聚类 |
| `build_jump_labels.py` | 3 | `*_clean.csv` + `*_clusters.tsv` | `data/processed/all_subtypes_simplified.csv` | 宿主时间线 → 20 列简化标签数据集 |
| `run_pipeline.py` | 1-3 | — | `all_subtypes_simplified.csv` | ★ 一键执行 Phase 1-3 |

## 辅助脚本

| 脚本 | 说明 |
|------|------|
| `supplement_h5.py` | 补充下载 H5 牛源(43) + 人源(634)、H7 人源(255) |
| `compare_thresholds.py` | CD-HIT 参数 sweep (85%-99.9%) |
| `build_isolate_dataset.py` | 从 cluster 级生成 isolate 级数据集 (`data/processed_isolate/`) |
| `build_aligned_dataset.py` | MAFFT 对齐到 H3 参考序列 (`data/processed_isolate_MAFFT/`) |

## 一键运行

```bash
cd /root/autodl-tmp/FluJump
python datascripts/run_pipeline.py                           # cluster 级
python datascripts/build_isolate_dataset.py                  # isolate 级
python datascripts/build_aligned_dataset.py                  # MAFFT 对齐
```
