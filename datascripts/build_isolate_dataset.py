#!/usr/bin/env python3
"""
从 clean 数据生成 isolate 级别数据集，每个 isolate 继承其 CD-HIT cluster 的跨物种标签。
输出: data/processed_isolate/all_isolates.csv
"""

import csv
from collections import defaultdict
from pathlib import Path

DATA = Path("data/processed")
OUT_DIR = Path("data/processed_isolate")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUBTYPES = ["H1", "H3", "H5", "H7"]

# 1. 加载 cluster 标签
print("加载 cluster 标签...")
cid_label = {}
with open(DATA / "all_subtypes_simplified.csv") as f:
    for r in csv.DictReader(f):
        cid_label[(r["subtype"], r["cluster_id"])] = {
            "cluster_size": r["cluster_size"],
            "host_categories": r["host_categories"],
            "label_is_jump": r["label_is_jump"],
            "jump_interval_cat": r["jump_interval_cat"],
            "label_is_jump_human": r["label_is_jump_human"],
            "jump_interval_human_cat": r["jump_interval_human_cat"],
            "label_jump_source": r["label_jump_source"],
            "consensus_ha_seq": r["consensus_ha_seq"],
            "serotype": r["serotype"],
            # 各宿主最早时间
            "first_human_ym": r["first_human_ym"],
            "first_avian_ym": r["first_avian_ym"],
            "first_swine_ym": r["first_swine_ym"],
            "first_bovine_ym": r["first_bovine_ym"],
        }

# 2. 加载 accession → cluster 映射
print("加载 cluster 映射...")
acc2cid = {}
for st in SUBTYPES:
    with open(DATA / f"{st}_clusters.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            acc2cid[r["accession"]] = (st, r["cluster_id"])

# 3. 生成 isolate 级别数据
print("生成 isolate 数据集...")
OUTPUT_COLS = [
    "accession", "subtype", "cluster_id", "cluster_size",
    "strain_name", "ha_sequence", "serotype",
    "host_category", "host_species",
    "collection_year", "collection_month", "collection_date",
    "geo_location", "country",
    "isolation_source",
    # cluster 级标签
    "cluster_hosts",
    "label_is_jump", "jump_interval_cat",
    "label_is_jump_human", "jump_interval_human_cat",
    "label_jump_source",
    # cluster 级时间
    "first_human_ym", "first_avian_ym", "first_swine_ym", "first_bovine_ym",
]

total, nj, nh = 0, 0, 0
with open(OUT_DIR / "all_isolates.csv", "w", encoding="utf-8", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=OUTPUT_COLS)
    writer.writeheader()

    for st in SUBTYPES:
        with open(DATA / f"{st}_clean.csv") as fin:
            for r in csv.DictReader(fin):
                total += 1
                key = acc2cid.get(r["accession"])
                if not key:
                    continue
                lbl = cid_label.get(key, {})

                year = r.get("collection_year", "")
                month = r.get("collection_month", "")
                if year == "-1":
                    date_str = "-1"
                elif month and month != "-1":
                    date_str = f"{year}-{month.zfill(2)}"
                else:
                    date_str = f"{year}-06" if year else ""

                row = {
                    "accession": r["accession"],
                    "subtype": st,
                    "cluster_id": key[1],
                    "cluster_size": lbl.get("cluster_size", ""),
                    "strain_name": r.get("strain_name", ""),
                    "ha_sequence": r.get("ha_sequence", ""),
                    "serotype": lbl.get("serotype", r.get("serotype", "")),
                    "host_category": r.get("host_category", ""),
                    "host_species": r.get("host_species", ""),
                    "collection_year": year,
                    "collection_month": month,
                    "collection_date": date_str,
                    "geo_location": r.get("geo_location", ""),
                    "country": r.get("country", ""),
                    "isolation_source": r.get("isolation_source", ""),
                    "cluster_hosts": lbl.get("host_categories", ""),
                    "label_is_jump": lbl.get("label_is_jump", "0"),
                    "jump_interval_cat": lbl.get("jump_interval_cat", ""),
                    "label_is_jump_human": lbl.get("label_is_jump_human", "0"),
                    "jump_interval_human_cat": lbl.get("jump_interval_human_cat", ""),
                    "label_jump_source": lbl.get("label_jump_source", ""),
                    "first_human_ym": lbl.get("first_human_ym", ""),
                    "first_avian_ym": lbl.get("first_avian_ym", ""),
                    "first_swine_ym": lbl.get("first_swine_ym", ""),
                    "first_bovine_ym": lbl.get("first_bovine_ym", ""),
                }
                writer.writerow(row)

                if lbl.get("label_is_jump") == "1":
                    nj += 1
                if lbl.get("label_is_jump_human") == "1":
                    nh += 1

print(f"\n{'='*60}")
print(f"  输出: {OUT_DIR / 'all_isolates.csv'}")
print(f"  总 isolate: {total}")
print(f"  label_is_jump=1:       {nj} ({100*nj/max(total,1):.1f}%)")
print(f"  label_is_jump_human=1: {nh} ({100*nh/max(total,1):.1f}%)")
print(f"  字段数: {len(OUTPUT_COLS)}")
print(f"{'='*60}")
