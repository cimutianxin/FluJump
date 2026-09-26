#!/usr/bin/env python3
"""Interval 3 类任务专用：H1+H3+H5+H7 全体、cluster 级 70/15/15 划分。

与 isolate_split.csv 的区别：
- 覆盖 H1357 四种亚型（不再只用 H1+H3）
- 按 CD-HIT cluster 划分（同一 cluster 不跨 split，消除 cluster 泄漏）
- 仅包含 jump=1 且 interval 有效（raw days 可算）的 isolate —— 即 interval
  任务的真实样本总体（1191 条 / 44 clusters）

划分：cluster 级随机搜索均衡划分 —— 目标样本占比 70/15/15，
  在满足 val/test 三类俱全、各亚型均有代表的候选中，选样本占比与
  类别占比偏差最小的一个（搜索种子固定，结果可复现）。
输出（新文件，不影响 isolate_split.csv）：
  data/splits/interval_h1357_cluster_split.csv   accession,subtype,cluster_id,split
  data/splits/interval_h1357_cluster_split.json  摘要
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ESM_clf.interval_exp.interval_data import _raw_interval_days

DATA_CSV = Path("data/processed_isolate/all_isolates_clean.csv")
OUT_DIR = Path("data/splits")
SEED = 42
SUBTYPES = ["H1", "H3", "H5", "H7"]
Y3_LABELS = ["<1yr", "1-3yr", "3yr+"]
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
N_TRIALS = 5000          # 随机搜索的候选划分个数
# 可行性硬约束：val/test 稀有类最少样本数
MIN_C1, MIN_C2 = 5, 3


def main():
    df = pd.read_csv(DATA_CSV, dtype=str).fillna("")
    sub = df[df["subtype"].isin(SUBTYPES) & (df["label_is_jump"] == "1")].copy()
    sub["_raw_days"] = sub.apply(_raw_interval_days, axis=1)
    sub = sub[sub["_raw_days"].notna()].copy()
    sub["_y3"] = np.digitize(sub["_raw_days"].values.astype(float), [365, 1095])
    print(f"interval 总体（H1357, jump=1, 有效）: {len(sub)} 条, "
          f"{sub['cluster_id'].nunique()} clusters")

    # ── cluster 级表 ──
    rows = []
    for cid, g in sub.groupby("cluster_id"):
        rows.append({
            "cluster_id": cid, "n": len(g),
            "modal_st": g["subtype"].mode()[0],
            "c0": int((g["_y3"] == 0).sum()),
            "c1": int((g["_y3"] == 1).sum()),
            "c2": int((g["_y3"] == 2).sum()),
            "sts": set(g["subtype"]),
        })
    cl = pd.DataFrame(rows)
    N = len(sub)

    def assign(order):
        """按给定顺序把 cluster 贪心装入当前占比最欠的 split。"""
        counts = {sp: 0 for sp in TARGET}
        split_of = {}
        for i in order:
            deficit = {sp: TARGET[sp] - counts[sp] / N for sp in TARGET}
            sp = max(deficit, key=deficit.get)
            split_of[i] = sp
            counts[sp] += cl.loc[i, "n"]
        return split_of

    def score(split_of):
        """可行性硬约束 + 偏差打分（越小越好）。"""
        cl["_sp"] = pd.Series(split_of)
        dev = 0.0
        for sp in ["val", "test"]:
            g = cl[cl["_sp"] == sp]
            n = g["n"].sum()
            # 稀有类与亚型覆盖硬约束
            if g["c1"].sum() < MIN_C1 or g["c2"].sum() < MIN_C2:
                return None
            sts = set().union(*g["sts"]) if len(g) else set()
            if set(SUBTYPES) - sts:
                return None
            # 类别占比偏差（相对全体）
            for c in ["c0", "c1", "c2"]:
                dev += abs(g[c].sum() / n - cl[c].sum() / N)
        # 样本占比偏差
        for sp in TARGET:
            dev += abs(cl.loc[cl["_sp"] == sp, "n"].sum() / N - TARGET[sp])
        return dev

    best, best_seed = None, None
    for t in range(N_TRIALS):
        rng = np.random.default_rng(SEED * 10000 + t)
        order = rng.permutation(len(cl))
        sp_of = assign(order)
        s = score(sp_of)
        if s is not None and (best is None or s < best[0]):
            best, best_seed = (s, sp_of), t
    assert best is not None, "未找到满足约束的划分"
    cl["split"] = pd.Series(best[1])
    print(f"随机搜索: {N_TRIALS} 候选中选中 #{best_seed} (偏差 {best[0]:.4f})")

    sub = sub.merge(cl[["cluster_id", "split"]], on="cluster_id", how="left")
    assert sub["split"].ne("").all()

    # ── 审查 ──
    print(f"\n{'='*70}\n划分结果审查\n{'='*70}")
    for sp in ["train", "val", "test"]:
        g = sub[sub["split"] == sp]
        dist = np.bincount(g["_y3"], minlength=3)
        print(f"\n--- {sp}: {len(g)} 条 ({len(g)/len(sub)*100:.1f}%), "
              f"{g['cluster_id'].nunique()} clusters ---")
        print(f"  3 类: {dict(zip(Y3_LABELS, dist.tolist()))}")
        print(f"  亚型: {g['subtype'].value_counts().to_dict()}")

    # cluster 泄漏自检
    nsp = sub.groupby("cluster_id")["split"].nunique()
    assert (nsp == 1).all(), "存在跨 split 的 cluster!"
    print("\n✓ 无 cluster 跨 split")

    # ── 输出 ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "interval_h1357_cluster_split.csv"
    sub[["accession", "subtype", "cluster_id", "split"]].to_csv(
        out_csv, index=False)

    info = {
        "level": "cluster",
        "subtypes": SUBTYPES,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "seed": SEED,
        "stratify": "随机搜索均衡划分（目标样本占比 70/15/15，约束 val/test 三类+四亚型俱全）",
        "population": "H1357 & label_is_jump=1 & 有效 interval（raw days 可算）",
        "n_total": int(len(sub)),
        "n_clusters": int(sub["cluster_id"].nunique()),
        "split_dist": {
            sp: {
                "n": int((sub["split"] == sp).sum()),
                "n_clusters": int(sub.loc[sub["split"] == sp, "cluster_id"].nunique()),
                "class_dist": dict(zip(Y3_LABELS, np.bincount(
                    sub.loc[sub["split"] == sp, "_y3"], minlength=3).tolist())),
                "subtype_dist": sub.loc[sub["split"] == sp, "subtype"]
                    .value_counts().to_dict(),
            } for sp in ["train", "val", "test"]
        },
    }
    out_json = OUT_DIR / "interval_h1357_cluster_split.json"
    with open(out_json, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\n✓ {out_csv} ({len(sub)} rows)")
    print(f"✓ {out_json}")


if __name__ == "__main__":
    main()
