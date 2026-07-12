#!/usr/bin/env python3
"""
统一 isolate 级别 70/15/15 划分。
只对 H1+H3 做分层划分，H5/H7 全量 holdout。

分层: stratify by (subtype, label_is_jump_human)
  第一层: train (70%) vs temp (30%)
  第二层: val (15%) vs test (15%) — stratify by label only

输出: data/splits/isolate_split.csv
"""

import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_CSV = Path("data/processed_isolate_MAFFT/all_isolates_aligned.csv")
OUT_DIR = Path("data/splits")
SEED = 42
TRAIN = 0.70
VAL = 0.15           # TEST = 1 - 0.70 - 0.15 = 0.15

df = pd.read_csv(DATA_CSV)
print(f"总数据: {len(df)} isolates")

# ── H5 / H7 标记 ──
h5_mask = df["subtype"] == "H5"
h7_mask = df["subtype"] == "H7"

# ── H1 + H3: isolate 级分层 ──
h13 = df[df["subtype"].isin(["H1", "H3"])].copy()
# stratify key: "H1_jh0", "H1_jh1", "H3_jh0", "H3_jh1"
h13["sk"] = h13["subtype"] + "_jh" + h13["label_is_jump_human"].astype(str)

print(f"\nH1+H3: {len(h13)} isolates")
for k, v in h13["sk"].value_counts().sort_index().items():
    print(f"  {k}: {v}")

# 第一层: 70% train, 30% (val+test)
train_idx, temp_idx = train_test_split(
    h13.index,
    test_size=VAL + (1 - TRAIN - VAL),  # 0.30
    random_state=SEED,
    stratify=h13["sk"],
)

# 第二层: temp 中对半分 → val (15%), test (15%)
temp = h13.loc[temp_idx]
val_idx, test_idx = train_test_split(
    temp.index,
    test_size=0.5,                       # 一半给 test
    random_state=SEED,
    stratify=temp["label_is_jump_human"],
)

# ── 写入 split 列 ──
df["split"] = ""
df.loc[h5_mask, "split"] = "h5_holdout"
df.loc[h7_mask, "split"] = "h7_holdout"
df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "val"
df.loc[test_idx, "split"] = "test"

assert df["split"].eq("").sum() == 0, f"有 {df['split'].eq('').sum()} 条未分配!"

# ── 审查 ──
print(f"\n{'='*70}")
print(f"审查: 完整分布")
print(f"{'='*70}")
for sp in ["train", "val", "test", "h5_holdout", "h7_holdout"]:
    sub = df[df["split"] == sp]
    print(f"\n--- {sp}: {len(sub)} isolates ---")
    for st in ["H1", "H3", "H5", "H7"]:
        s = sub[sub["subtype"] == st]
        if len(s) == 0:
            continue
        jh = s["label_is_jump_human"].sum()
        j = s["label_is_jump"].sum()
        ratio = jh / len(s) * 100
        print(f"  {st}: {len(s):>5} 条  "
              f"jh=1:{jh:>4} ({ratio:>4.1f}%)  "
              f"jump=1:{j:>4}")

# 验证比例
h13_total = len(h13)
print(f"\n{'='*70}")
print(f"比例验证 (H1+H3 = {h13_total})")
print(f"{'='*70}")
for sp in ["train", "val", "test"]:
    n = len(df[(df["split"] == sp) & df["subtype"].isin(["H1", "H3"])])
    print(f"  {sp}: {n} ({n/h13_total*100:.1f}%)")

# ── 输出 ──
OUT_DIR.mkdir(parents=True, exist_ok=True)
cols = ["accession", "subtype", "cluster_id", "split"]
df[cols].to_csv(OUT_DIR / "isolate_split.csv", index=False)
print(f"\n✓ data/splits/isolate_split.csv ({len(df)} rows)")

# cluster_split.json (反推)
cluster_split = {
    f"{sp}_clusters": sorted(int(c) for c in
        df[df["split"] == sp]["cluster_id"].unique())
    for sp in ["train", "val", "test", "h5_holdout", "h7_holdout"]
}
cluster_split["meta"] = {
    "level": "isolate",
    "ratios": {"train": TRAIN, "val": VAL, "test": round(1 - TRAIN - VAL, 2)},
    "seed": SEED,
    "stratify": "layer1: subtype+jh, layer2: jh only",
    "h13_only": True,
}
with open(OUT_DIR / "cluster_split.json", "w") as f:
    json.dump(cluster_split, f, indent=2)
print("✓ data/splits/cluster_split.json")
