# Isolate 级数据完备性分析

> 2026-07-09 | `data/processed_isolate/all_isolates.csv`

## 1. 数据来源与处理流程

```
NCBI Protein (18,680 条)
  │ download_flu.py + supplement_h5.py
  ▼
raw/ (H1:7170, H3:6673, H5:3337, H7:1500)
  │ clean_data.py: 过滤 PDB(3384) + 长度不符(3580) + 合成(2)
  │ 无日期 586 条保留为 year=-1
  ▼
clean/ (11,714 条)
  │ cluster_strains.py: CD-HIT 99% → 2,362 毒株
  ▼
clusters/ (accession → cluster_id)
  │ build_jump_labels.py: 每 cluster 计算跨物种标签
  ▼
simplified.csv (2,362 毒株, 20 列)
  │ build_isolate_dataset.py: 标签从 cluster 分配给每个 isolate
  ▼
processed_isolate/all_isolates.csv ★ (11,714 isolates, 25 列)
```

## 2. 字段完备性

| 字段 | 缺失 | 缺失率 | 说明 |
|------|:---:|:---:|------|
| ha_sequence | 0 | 0% | 全部有序列 |
| strain_name | 0 | 0% | 全部有 |
| host_category | 453 | 3.9% | Swiss-Prot 条目标记为 unknown |
| serotype | 143 | 1.2% | 部分 GenBank 未标注 |
| collection_date | 584 | 5.0% | year=-1，UniProt/Swiss-Prot |
| geo_location | 432 | 3.7% | |
| country | 432 | 3.7% | |

## 3. 宿主分布

| 宿主 | isolates | jump=1 | jump_h=1 |
|------|:---:|:---:|:---:|
| human | 5,554 (47.4%) | 740 | 740 |
| avian | 2,886 (24.6%) | 391 | 259 |
| swine | 2,638 (22.5%) | 172 | 172 |
| unknown | 453 (3.9%) | 105 | 98 |
| environmental | 88 (0.8%) | 44 | 0 |
| bovine | 43 (0.4%) | 43 | 43 |
| equine | 29 (0.2%) | 0 | 0 |
| canine | 4 | 0 | 0 |
| feline | 3 | 3 | 2 |
| marine_mammal | 3 | 0 | 0 |
| other_mammal | 13 | 0 | 0 |

## 4. 时间分布

| 指标 | 值 |
|------|-----|
| 范围 | 1925 - 2026 |
| 中位数 | 2020 |
| 2000 年后 | 11,026 (99.1%) |
| year=-1 | 584 (5.0%) |

年代分布: 1920s(1) → ... → 2000s(1,297) → 2010s(4,011) → **2020s(5,718)**

## 5. 地理偏差

| 国家 | isolates | 占比 |
|------|:---:|------|
| USA | 7,615 | **65.0%** |
| China | 991 | 8.5% |
| Japan | 300 | 2.6% |
| Taiwan | 293 | 2.5% |
| South Korea | 292 | 2.5% |

## 6. 亚型×宿主交叉

| 亚型 | 主要宿主 |
|------|---------|
| H1 | swine(1930) + human(1904) + avian(229) |
| H3 | human(3129) + swine(705) + avian(701) |
| H5 | avian(1324) + human(360) + unknown(263) |
| H7 | avian(632) + human(161) + unknown(61) |

## 7. 标签分布

| 标签 | isolates | 占比 |
|------|:---:|------|
| label_is_jump=1 | 1,498 | 12.8% |
| label_is_jump_human=1 | 1,314 | 11.2% |

标签来源于 cluster 级别: 每个 isolate 继承其 CD-HIT 99% cluster 的标签。

## 8. 优点

- **样本量大**: 11,714 isolates vs 2,362 clusters（5x）
- **序列完整**: 100% 有 ha_sequence，可直接用于蛋白质语言模型
- **标签丰富**: 双标签 (jump + jump_human) + 时间间隔分类 + 来源宿主
- **时间覆盖广**: 1925-2026，99% 在 2000 年后
- **多宿主**: 11 个宿主类别，覆盖主要人畜共患传播链

## 9. 缺点与风险

| 问题 | 严重度 | 说明 |
|------|:---:|------|
| **标签数据泄漏** | 🔴 高 | isolate 的标签来自其 cluster 的未来行为。建模时必须时间切分 train/test |
| **地理偏差** | 🔴 高 | USA 占 65%，主要来自美国猪流感监测 |
| **class imbalance** | 🟡 中 | jump=1 仅 12.8%，需处理不平衡 |
| **cluster 内冗余** | 🟡 中 | 同一 cluster 的 isolates 序列高度相似（99%），CV 时必须 group-level split |
| **host_category 少量缺失** | 🟢 低 | 3.9% unknown，主要是 Swiss-Prot 条目 |
| **时间零值** | 🟢 低 | 5.0% year=-1，不参与时间计算 |

## 10. 使用建议

```python
# Group-level train/test split (避免数据泄漏)
from sklearn.model_selection import GroupKFold
groups = df['cluster_id']  # 确保同一 cluster 不在 train/test 中同时出现

# 时间切分（如适用）
train = df[df['collection_year'] < '2020']
test  = df[df['collection_year'] >= '2020']
```
