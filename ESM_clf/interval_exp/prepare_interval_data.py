"""准备 interval 预测所需的数据（薄封装）。

调用共享 loader（interval_data.load_interval_data，accession 对齐版），
保存标签 / split / accession 快照与数据摘要：
  - labels_interval_cat.npy      4 类分类标签（human_first/<1yr/1-3yr/3yr+）
  - labels_interval_days.npy     clamped raw_days（负值→0，回归目标）
  - labels_interval_days_raw.npy 有符号 raw_days
  - interval_accessions.npy / interval_split.npy
  - interval_data_info.json      数据摘要
"""

import numpy as np
import json
import sys
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from ESM_clf.interval_exp.interval_data import load_interval_data


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("加载数据（accession 对齐版）...")
    data = load_interval_data(verbose=True)

    y_cat = data["y_cat"]
    y_days = data["y_days"]
    y_days_raw = data["y_days_raw"]
    masks = data["masks"]

    np.save(OUT_DIR / "labels_interval_cat.npy", y_cat)
    np.save(OUT_DIR / "labels_interval_days.npy", y_days.astype(np.float32))
    np.save(OUT_DIR / "labels_interval_days_raw.npy", y_days_raw.astype(np.float32))
    np.save(OUT_DIR / "interval_accessions.npy", data["accessions"])
    np.save(OUT_DIR / "interval_split.npy", data["df"]["split"].values)

    print(f"\n分类目标 ({INTERVAL_LABELS}):")
    for i, l in enumerate(INTERVAL_LABELS):
        n = (y_cat == i).sum()
        print(f"  class {i} ({l}): {n:>4} ({n/len(y_cat)*100:.1f}%)")

    print(f"\nSplit × 类别分布:")
    for sp in ["train", "val", "test"]:
        m = masks[sp]
        dist = {INTERVAL_LABELS[i]: int((y_cat[m] == i).sum()) for i in range(len(INTERVAL_LABELS))}
        print(f"  {sp}: n={m.sum():>4}  {dist}")

    print(f"\n回归目标 (clamped days):")
    print(f"  count: {len(y_days)}  mean: {y_days.mean():.1f}  std: {y_days.std():.1f}")
    print(f"  =0: {(y_days == 0).sum()}  >0: {(y_days > 0).sum()}  max: {y_days.max():.0f}")

    info = {
        "n_total": int(len(y_cat)),
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "subtype_dist": {k: int(v) for k, v in
                         pd.Series(data["subtypes"]).value_counts().items()},
        "cluster_count": int(len(np.unique(data["cluster_ids"]))),
        "reg_mean": float(y_days.mean()),
        "reg_std": float(y_days.std()),
        "class_dist": {INTERVAL_LABELS[i]: int((y_cat == i).sum())
                       for i in range(len(INTERVAL_LABELS))},
        "split_class_dist": {
            sp: {INTERVAL_LABELS[i]: int((y_cat[masks[sp]] == i).sum())
                 for i in range(len(INTERVAL_LABELS))}
            for sp in ["train", "val", "test"]
        },
        "subtypes": SUBTYPES,
        "emb_types": EMB_TYPES,
        "interval_bins": INTERVAL_BINS,
        "interval_labels": INTERVAL_LABELS,
        "label_source": "raw iso_interval 重算（未 clamp），human_first 独立成类",
        "alignment": "accession-based merge + accession-based embedding slicing",
    }
    with open(OUT_DIR / "interval_data_info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nDone — 数据准备完成")


if __name__ == "__main__":
    import pandas as pd
    main()
