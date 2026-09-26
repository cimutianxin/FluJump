"""重新生成 labels_*.npy（不重跑 GPU embedding 提取）。

2026-08-02：build_isolate_dataset.py 的 acc2cid key bug 修复后，
aligned CSV 的 label_is_jump / label_is_jump_human 有少量行变化，
需同步更新 labels npy。embedding 本身不变（序列未变）。

安全性检查：aligned CSV 行序必须逐行等于 isolate_split.csv（== embedding 行序）。
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *


def main():
    df = pd.read_csv(DATA_CSV, dtype=str).fillna("")
    sp = pd.read_csv(SPLIT_CSV, dtype=str)

    # 行序硬性校验：aligned == split == embedding
    assert len(df) == len(sp), f"行数不一致: {len(df)} vs {len(sp)}"
    assert (df["accession"].values == sp["accession"].values).all(), "accession 行序不一致"
    assert (df["subtype"].values == sp["subtype"].values).all(), "subtype 行序不一致"

    acc_old = np.load(OUT_DIR / "accessions.npy", allow_pickle=True)
    assert (df["accession"].values == acc_old).all(), "与 embedding accessions.npy 行序不一致"

    for lc in LABEL_COLS:
        vals = df[lc].astype(int).values
        old_path = OUT_DIR / f"labels_{lc}.npy"
        if old_path.exists():
            old = np.load(old_path)
            n_diff = int((old != vals).sum())
            print(f"  labels_{lc}: {vals.shape}, pos={vals.sum()} "
                  f"({vals.sum()/len(vals)*100:.1f}%), 与旧版差异 {n_diff} 行")
        np.save(old_path, vals)

    print("Done — labels 已更新")


if __name__ == "__main__":
    main()
