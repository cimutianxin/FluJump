# MAFFT 对齐流程

> 2026-07-10 | 亚型内对齐 → profile merge → gap 列裁剪 → 581 等长

## Pipeline

```
all_isolates_clean.csv (11,261 isolates, 原始不等长)
  │ 按亚型拆分
  ▼
H1.faa (4069)  H3.faa (4597)  H5.faa (1757)  H7.faa (838)
  │ mafft --auto --thread 3
  ▼
H1_aligned (795)  H3_aligned (597)  H5_aligned (606)  H7_aligned (744)
  │ cat → mafft --auto (profile merge)
  ▼
all_aligned_merged.faa (1039 列)
  │ 裁剪 >90% gap 列 → 581 列
  ▼
all_isolates_aligned.csv ★ (11,261 × 581 等长, 27 列)
```

## 长度变化

| 阶段 | 长度 | gap 中位数 | 说明 |
|------|:---:|:---:|------|
| 一次 MAFFT (旧) | 1070 | 504 | --retree, gap 过多 |
| 亚型内 H1 | 795 | — | |
| 亚型内 H3 | 597 | — | |
| 亚型内 H5 | 606 | — | |
| 亚型内 H7 | 744 | — | |
| Profile merge | 1039 | — | 跨亚型 gap 累积 |
| **裁剪 >90% gap** | **581** | **15** | **★ 最终** |

## 最终文件

```
data/processed_isolate_MAFFT/
├── H1_aligned.faa              # H1 内部对齐 (可复现)
├── H3_aligned.faa              # H3 内部对齐
├── H5_aligned.faa              # H5 内部对齐
├── H7_aligned.faa              # H7 内部对齐
├── all_aligned_merged.faa      # 合并后对齐 (1039 列)
└── all_isolates_aligned.csv    ★ 最终数据集 (581 列等长)
```

## 执行命令

```bash
# 导出亚型 FASTA
python -c "..." → H*.faa

# 亚型内 MAFFT
for st in H1 H3 H5 H7; do
    mafft --auto --thread 3 ${st}.faa > ${st}_aligned.faa &
done; wait

# 合并 profile
cat H*_aligned.faa > all_profiles.faa
mafft --auto --thread 12 all_profiles.faa > all_aligned_merged.faa

# 裁剪 >90% gap + 生成 CSV
python build_aligned_dataset.py
```

## 经验

- 11K 序列一次性 MAFFT 不可行 (L-INS-i 内存溢出)
- 亚型内先对齐再 profile merge 是最优方案
- 裁剪 >90% gap 列可将 1039 → 581, 去除跨亚型无意义 gap
- 最终 gap 中位数仅 15, 对齐质量高
