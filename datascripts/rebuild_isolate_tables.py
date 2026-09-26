#!/usr/bin/env python3
"""重建 isolate 级三表（clean / 2026 / aligned），恢复 07-10 原始设计。

背景（2026-08-02 审阅发现）：
- 07-12 重建数据时把 201 条 2026 年 isolate 并入了 clean/aligned（原始设计是
  2026 单独存放，不进 train/val/test），导致 clean/aligned（11261 行）与
  isolate_split.csv / embedding（11060 行）行数与行序都不一致。
- build_isolate_dataset.py 的 acc2cid key bug 已修复（(subtype, accession)），
  all_isolates.csv 已重跑。

本脚本：
1. all_isolates.csv → all_isolates_clean.csv
   （过滤 host_category=unknown + 排除 2026 年，行序重排为 isolate_split.csv 行序，
   使 clean / split / embedding 三者逐行一致）
2. all_isolates.csv → 2026_isolates_clean.csv（2026 年子集，host!=unknown）
3. 用已有 all_aligned_merged.faa（裸 accession 作 key）查对齐序列（不重跑 MAFFT），
   输出 all_isolates_aligned.csv 与 2026_isolates_aligned.csv

运行: /root/miniconda3/envs/env1/bin/python datascripts/rebuild_isolate_tables.py
"""

import pandas as pd
from pathlib import Path

ISOLATE_DIR = Path("data/processed_isolate")
MAFFT_DIR = Path("data/processed_isolate_MAFFT")
SPLIT_CSV = Path("data/splits/isolate_split.csv")

KEY = ["accession", "subtype"]


def main():
    df = pd.read_csv(ISOLATE_DIR / "all_isolates.csv", dtype=str).fillna("")
    sp = pd.read_csv(SPLIT_CSV, dtype=str)

    known_host = df["host_category"] != "unknown"
    is_2026 = df["collection_year"] == "2026"

    # ── 1. clean（排除 unknown host + 2026 年）──
    clean = df[known_host & ~is_2026].copy()
    print(f"clean 候选: {len(clean)} 行")

    key_clean = set(map(tuple, clean[KEY].values))
    key_split = set(map(tuple, sp[KEY].values))
    assert key_clean == key_split, (
        f"key 集合不一致: clean-only={len(key_clean - key_split)}, "
        f"split-only={len(key_split - key_clean)}")
    assert not clean.duplicated(KEY).any(), "clean 存在重复 (accession, subtype)"

    # 重排为 split 行序（== embedding 行序），恢复逐行一致
    order = pd.DataFrame({KEY[0]: sp[KEY[0]], KEY[1]: sp[KEY[1]], "_ord": range(len(sp))})
    clean = clean.merge(order, on=KEY, how="left").sort_values("_ord").drop(columns="_ord")
    assert (clean["accession"].values == sp["accession"].values).all()
    assert (clean["subtype"].values == sp["subtype"].values).all()
    clean.to_csv(ISOLATE_DIR / "all_isolates_clean.csv", index=False)
    print(f"  → all_isolates_clean.csv: {len(clean)} 行（行序 == isolate_split.csv）")

    # ── 2. 2026 子集 ──
    d2026 = df[known_host & is_2026].copy()
    d2026.to_csv(ISOLATE_DIR / "2026_isolates_clean.csv", index=False)
    print(f"  → 2026_isolates_clean.csv: {len(d2026)} 行")

    # ── 3. aligned（用已有 all_aligned_merged.faa 查对齐序列，不重跑 MAFFT）──
    # merged faa 的 header 是裸 accession；跨亚型重复 accession 的序列已验证一致，
    # 因此 accession → 对齐序列映射无歧义
    aligned = {}
    seq_id, chars = None, []
    with open(MAFFT_DIR / "all_aligned_merged.faa") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    aligned[seq_id] = "".join(chars)
                seq_id, chars = line[1:], []
            else:
                chars.append(line)
    if seq_id:
        aligned[seq_id] = "".join(chars)
    print(f"对齐序列数: {len(aligned)}")

    OUT_COLS = [
        "accession", "subtype", "cluster_id", "cluster_size",
        "strain_name", "ha_sequence", "aligned_ha_seq",
        "aligned_length", "gap_count", "gap_ratio",
        "host_category", "serotype",
        "collection_year", "collection_month", "collection_date",
        "geo_location", "country",
        "cluster_hosts",
        "label_is_jump", "jump_interval_cat",
        "label_is_jump_human", "jump_interval_human_cat",
        "label_jump_source",
        "first_human_ym", "first_avian_ym", "first_swine_ym", "first_bovine_ym",
        "iso_interval_days", "iso_interval_cat",
    ]

    def build_aligned_rows(src):
        rows, n_missed = [], 0
        for r in src.to_dict("records"):
            ali = aligned.get(r["accession"], "")
            if not ali:
                n_missed += 1
                continue
            gaps = ali.count("-")
            rows.append({
                "accession": r["accession"], "subtype": r["subtype"],
                "cluster_id": r["cluster_id"], "cluster_size": r["cluster_size"],
                "strain_name": r["strain_name"], "ha_sequence": r["ha_sequence"],
                "aligned_ha_seq": ali, "aligned_length": len(ali),
                "gap_count": gaps, "gap_ratio": round(gaps / max(len(ali), 1), 4),
                "host_category": r["host_category"], "serotype": r["serotype"],
                "collection_year": r["collection_year"],
                "collection_month": r["collection_month"],
                "collection_date": r["collection_date"],
                "geo_location": r["geo_location"], "country": r["country"],
                "cluster_hosts": r["cluster_hosts"],
                "label_is_jump": r["label_is_jump"],
                "jump_interval_cat": r["jump_interval_cat"],
                "label_is_jump_human": r["label_is_jump_human"],
                "jump_interval_human_cat": r["jump_interval_human_cat"],
                "label_jump_source": r["label_jump_source"],
                "first_human_ym": r["first_human_ym"],
                "first_avian_ym": r["first_avian_ym"],
                "first_swine_ym": r["first_swine_ym"],
                "first_bovine_ym": r["first_bovine_ym"],
                "iso_interval_days": r["iso_interval_days"],
                "iso_interval_cat": r["iso_interval_cat"],
            })
        return rows, n_missed

    rows, n_missed = build_aligned_rows(clean)
    out = pd.DataFrame(rows, columns=OUT_COLS)
    out.to_csv(MAFFT_DIR / "all_isolates_aligned.csv", index=False)
    print(f"  → all_isolates_aligned.csv: {len(out)} 行（未匹配 {n_missed}）")

    rows26, n_missed26 = build_aligned_rows(d2026)
    out26 = pd.DataFrame(rows26, columns=OUT_COLS)
    out26.to_csv(MAFFT_DIR / "2026_isolates_aligned.csv", index=False)
    print(f"  → 2026_isolates_aligned.csv: {len(out26)} 行（未匹配 {n_missed26}）")

    print("\nDone — 三表重建完成，行序与 isolate_split.csv / embedding 一致")


if __name__ == "__main__":
    main()
