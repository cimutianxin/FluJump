#!/usr/bin/env python3
"""
解析 Borkenhagen 2024 补充材料 Table S1，提取毒株名，按 test/train 分类。
然后用 NCBI EUtils 查询 HA 序列。

用法:
  python datascripts/borkenhagen_data/parse_strains.py
"""

import pandas as pd
import re
from pathlib import Path

MD_PATH = Path("data/dataset_borkenhagen/S1_S2.md")
OUT_DIR = Path("data/dataset_borkenhagen")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 解析 Table S1 ──
with open(MD_PATH) as f:
    text = f.read()

# 提取表格行 (markdown table)
rows = re.findall(r'^\| (.+?) \| (.+?) \| (.+?) \|$', text, re.MULTILINE)
# 跳过表头
data_rows = []
for r in rows:
    strain = r[0].strip()
    source = r[1].strip()
    use = r[2].strip()
    if 'Strain Name' in strain or '---' in strain:
        continue
    # 清理 strain name 中的多空格
    strain = re.sub(r'\s+', ' ', strain)
    data_rows.append({"strain_name": strain, "source": source, "use": use})

df = pd.DataFrame(data_rows)
df["is_test"] = df["use"].str.contains("Test", case=False)
df["is_train"] = df["use"].str.contains("Train", case=False)

# 提取 subtype
df["subtype"] = df["strain_name"].str.extract(r'\(([^)]+)\)')  # (H1N1), (H3N2), etc.
# 尝试从名称中提取
df["ha_subtype"] = df["subtype"].str.extract(r'(H\d+)')

print(f"解析到 {len(df)} 个毒株")
print(f"  Test:  {df['is_test'].sum()}")
print(f"  Train: {df['is_train'].sum()}")
print(f"\nTest 亚型分布:")
print(df[df["is_test"]]["ha_subtype"].value_counts().sort_index())

# 保存
df.to_csv(OUT_DIR / "strains_parsed.csv", index=False)
print(f"\n✓ 已保存 {OUT_DIR / 'strains_parsed.csv'}")

# ── 生成 NCBI 查询列表 ──
# 48 test + 48 train = 96 strains
# 每个 strain 需要查 HA 序列
# 策略: 用 strain name 的部分关键词搜索 NCBI Protein

print(f"\n=== 准备 NCBI 查询 ===")
test_strains = df[df["is_test"]]["strain_name"].tolist()
train_strains = df[df["is_train"]]["strain_name"].tolist()

print(f"Test strains ({len(test_strains)}):")
for s in test_strains[:5]:
    print(f"  {s}")
print(f"  ... (共 {len(test_strains)})")

print(f"\nTrain strains ({len(train_strains)}):")
for s in train_strains[:5]:
    print(f"  {s}")
print(f"  ... (共 {len(train_strains)})")

# 保存查询列表供后续 NCBI 下载
with open(OUT_DIR / "ncbi_query_strains.txt", "w") as f:
    for s in test_strains + train_strains:
        # 简化为 NCBI 搜索词
        query = s.split("(")[0].strip().replace("/", " ")
        f.write(f"{s}\t{query}\n")
print(f"✓ {OUT_DIR / 'ncbi_query_strains.txt'} (供 NCBI 搜索)")
