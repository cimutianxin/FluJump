"""准备 interval 预测所需的数据。

- 加载 all_isolates_aligned.csv + isolate_split.csv
- 筛选 H1+H3 且 label_is_jump=1 且 iso_interval_days 非空
- 生成：
  - labels_interval_days.npy（连续回归目标）
  - labels_interval_cat.npy（分类目标，4 类）
  - masks 和元数据保存为 interval_data_info.json
- 直接复用已有的 ESM-2 embedding（ESM_clf/jump_exp/output/esm_emb_150M_*.npy）
"""

import numpy as np
import pandas as pd
import json
import sys
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    split_df = pd.read_csv(SPLIT_CSV)
    df["split"] = split_df["split"]

    # ── 筛选：H1+H3，jump=1，有 iso_interval_days ──
    mask_subtype = df["subtype"].isin(SUBTYPES)
    mask_jump = df["label_is_jump"] == 1
    # ── 筛选：H1+H3，jump=1，jump_interval_cat 非空 ──
    mask_subtype = df["subtype"].isin(SUBTYPES)
    mask_jump = df["label_is_jump"] == 1
    mask_interval = (df["jump_interval_cat"].notna()) & (df["jump_interval_cat"] != "")

    mask = mask_subtype & mask_jump & mask_interval
    sub = df[mask].copy()

    print(f"筛选结果:")
    print(f"  H1+H3 total:          {mask_subtype.sum():>5}")
    print(f"  + jump=1:             {mask_jump.sum():>5}")
    print(f"  + has jump_interval:  {mask.sum():>5}")

    # ── Split 分布 ──
    print(f"\nSplit 分布:")
    for sp in ["train", "val", "test"]:
        n = (sub["split"] == sp).sum()
        print(f"  {sp}: {n}")

    # ── 分类目标：jump_interval_cat（cluster 级，已修复）──
    cat_target = sub["jump_interval_cat"].values
    cat_map = {l: i for i, l in enumerate(INTERVAL_LABELS)}
    cat_int = np.array([cat_map[c] for c in cat_target], dtype=np.int32)
    np.save(OUT_DIR / "labels_interval_cat.npy", cat_int)
    print(f"\n分类目标 (jump_interval_cat, {INTERVAL_LABELS}):")
    for i, l in enumerate(INTERVAL_LABELS):
        n = (cat_int == i).sum()
        print(f"  class {i} ({l}): {n:>4} ({n/len(cat_int)*100:.1f}%)")

    # ── 回归目标：iso_interval_days（isolate 级，连续值）──
    sub["iso_interval_days"] = pd.to_numeric(sub["iso_interval_days"], errors="coerce")
    reg_target = sub["iso_interval_days"].values.astype(np.float32)
    np.save(OUT_DIR / "labels_interval_days.npy", reg_target)
    print(f"\n回归目标 (iso_interval_days):")
    print(f"  count: {len(reg_target)}")
    print(f"  mean:  {reg_target.mean():.1f}")
    print(f"  std:   {reg_target.std():.1f}")
    print(f"  range: [{reg_target.min():.0f}, {reg_target.max():.0f}]")

    # ── 保存 masks 和元数据 ──
    info = {
        "n_total": len(sub),
        "n_train": int((sub["split"] == "train").sum()),
        "n_val": int((sub["split"] == "val").sum()),
        "n_test": int((sub["split"] == "test").sum()),
        "subtype_dist": sub["subtype"].value_counts().to_dict(),
        "cluster_count": int(sub["cluster_id"].nunique()),
        "reg_mean": float(reg_target.mean()),
        "reg_std": float(reg_target.std()),
        "class_dist": {INTERVAL_LABELS[i]: int((cat_int == i).sum()) for i in range(len(INTERVAL_LABELS))},
        "subtypes": SUBTYPES,
        "emb_types": EMB_TYPES,
        "interval_bins": INTERVAL_BINS,
        "interval_labels": INTERVAL_LABELS,
    }
    with open(OUT_DIR / "interval_data_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    # ── 保存筛选后 accession 列表（用于从全量 embedding 中切片）──
    np.save(OUT_DIR / "interval_accessions.npy", sub["accession"].values)
    np.save(OUT_DIR / "interval_split.npy", sub["split"].values)

    print(f"\nDone — 数据准备完成")


if __name__ == "__main__":
    main()
