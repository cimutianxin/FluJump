#!/usr/bin/env python3
"""host 审计（09-19）标签影响量化

今晨的 host_category 修复（host_audit_fix_diff.csv，49 行）是否翻转了 cluster 标签？
方法：把 49 行按 raw_host_category（修复前旧值）还原，按 build_jump_labels 口径
重算簇标签，与现行 all_subtypes_simplified.csv（修复后）对比。
断言关键簇不受影响：H7N9 {109,149,150,151,154,155,198}、(H5,70)、(H1,15)。

输出：data/processed_isolate/host_audit_label_impact.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

PROC = Path("data/processed")
SUBTYPES = ["H1", "H3", "H5", "H7"]
DIFF_CSV = Path("data/processed_isolate/host_audit_fix_diff.csv")
SIMP_CSV = PROC / "all_subtypes_simplified.csv"
OUT_JSON = Path("data/processed_isolate/host_audit_label_impact.json")

CRITICAL = {("H7", str(c)) for c in [109, 149, 150, 151, 154, 155, 198]}
CRITICAL |= {("H5", "70"), ("H1", "15")}


def cluster_labels(df):
    """build_jump_labels 口径：dated 成员的非 unknown host 集合 → 两标签"""
    hosts = defaultdict(set)
    yr = pd.to_numeric(df["collection_year"], errors="coerce")
    m = yr.notna() & (yr > 0) & (df["host_category"] != "unknown")
    for cid, hc in zip(df.loc[m, "cluster_id"], df.loc[m, "host_category"]):
        hosts[cid].add(hc)
    out = {}
    for cid, hs in hosts.items():
        out[cid] = {"label_is_jump": int(len(hs) >= 2),
                    "label_is_jump_human": int("human" in hs and len(hs) >= 2)}
    return out


def main():
    diff = pd.read_csv(DIFF_CSV)
    old_hc = dict(zip(diff["accession"], diff["raw_host_category"]))
    print(f"审计 diff {len(diff)} 行，涉及簇：")

    simp = pd.read_csv(SIMP_CSV)
    simp["cluster_id"] = simp["cluster_id"].astype(str)
    simp_idx = simp.set_index(["subtype", "cluster_id"])

    report = {"n_audit_rows": int(len(diff)), "flips": [], "per_subtype": {}}
    touched_clusters = set()

    for st in SUBTYPES:
        clean = pd.read_csv(PROC / f"{st}_clean.csv", low_memory=False)
        # accession → cluster_id（簇键 = (subtype, cluster_id)，cluster_id 亚型内编号）
        cmap = pd.read_csv(PROC / f"{st}_clusters.tsv", sep="\t",
                           dtype={"cluster_id": str})
        acc2cid = dict(zip(cmap["accession"], cmap["cluster_id"]))
        clean["cluster_id"] = clean["accession"].map(acc2cid)
        clean = clean[clean["cluster_id"].notna()].copy()

        # 还原 49 行到修复前旧值
        reverted = clean.copy()
        m = reverted["accession"].isin(old_hc)
        reverted.loc[m, "host_category"] = reverted.loc[m, "accession"].map(old_hc)
        touched_clusters |= {(st, c) for c in
                             reverted.loc[m, "cluster_id"].unique()}
        print(f"  {st}: 还原 {int(m.sum())} 行，涉及 {reverted.loc[m, 'cluster_id'].nunique()} 簇")

        lab_new = cluster_labels(clean)
        lab_old = cluster_labels(reverted)

        flips = []
        for cid in set(lab_new) | set(lab_old):
            for label in ["label_is_jump", "label_is_jump_human"]:
                v_new = lab_new.get(cid, {}).get(label, 0)
                v_old = lab_old.get(cid, {}).get(label, 0)
                # 与现行 simplified 表交叉验证（新标签应一致）
                key = (st, str(cid))
                if key in simp_idx.index:
                    v_simp = int(simp_idx.loc[key, label])
                    assert v_simp == v_new, f"{key} {label}: 重算 {v_new} != 现行表 {v_simp}"
                if v_new != v_old:
                    flips.append({"cluster_id": str(cid), "label": label,
                                  "old": v_old, "new": v_new})
        report["per_subtype"][st] = {"n_flips": len(flips)}
        report["flips"].extend([{"subtype": st, **f} for f in flips])
        print(f"  {st}: 标签翻转 {len(flips)} 处", flips if flips else "")

    # 关键簇断言：标签不得翻转（成员被触及可接受，如 (H5,70) 内含审计行但标签稳）
    hit = touched_clusters & CRITICAL
    flipped_critical = [f for f in report["flips"]
                        if (f["subtype"], f["cluster_id"]) in CRITICAL]
    print(f"\n关键簇成员被触及: {sorted(hit) if hit else '无'}；"
          f"关键簇标签翻转: {flipped_critical if flipped_critical else '无'}")
    assert not flipped_critical, f"关键簇标签翻转: {flipped_critical}"
    report["critical_clusters_touched"] = sorted(map(list, hit))
    report["critical_clusters_flipped"] = flipped_critical

    # isolate 级影响（翻转簇的成员数）
    if report["flips"]:
        alliso = pd.read_csv("data/processed_isolate/all_isolates.csv",
                             usecols=["subtype", "cluster_id"],
                             dtype={"cluster_id": str})
        n_iso = 0
        for f in report["flips"]:
            n_iso += int(((alliso["subtype"] == f["subtype"])
                          & (alliso["cluster_id"] == f["cluster_id"])).sum())
        report["n_isolate_rows_in_flipped_clusters"] = n_iso
    else:
        report["n_isolate_rows_in_flipped_clusters"] = 0

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n→ {OUT_JSON}")


if __name__ == "__main__":
    main()
