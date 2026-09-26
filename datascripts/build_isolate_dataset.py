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

ANIMAL_HOST_FIELDS = ["first_avian_ym", "first_swine_ym", "first_bovine_ym"]


def days_between(ym1, ym2):
    """两个 YYYY-MM 之间的近似天数差 (ym2 - ym1)"""
    if not ym1 or not ym2:
        return None
    try:
        y1, m1 = int(ym1[:4]), int(ym1[5:7])
        y2, m2 = int(ym2[:4]), int(ym2[5:7])
        return (y2 - y1) * 365 + (m2 - m1) * 30
    except (ValueError, IndexError):
        return None


def cat_interval_year(days):
    """天数 → 年分类: negative / <1yr / 1-3yr / 3-5yr / 5yr+"""
    if days is None:
        return ""
    if days < 0:
        return "negative"
    years = days / 365.25
    if years < 1:
        return "<1yr"
    elif years < 3:
        return "1-3yr"
    elif years < 5:
        return "3-5yr"
    return "5yr+"


def compute_isolate_interval(collection_date, host_category, first_human_ym, first_avian_ym, first_swine_ym, first_bovine_ym):
    """
    计算 isolate 级别的跨物种间隔。
    - 动物 isolate: collection_date → first_human_ym（距人类首检还有多久）
    - 人 isolate:   first_animal_ym → collection_date（动物首检后多久感染人）
    返回 (interval_days: int|None, interval_cat: str)
    """
    if not collection_date or collection_date == "-1":
        return None, ""
    if not first_human_ym:
        return None, ""

    # 最早动物宿主时间
    animal_yms = [ym for ym in [first_avian_ym, first_swine_ym, first_bovine_ym] if ym]
    if not animal_yms:
        return None, ""
    first_animal_ym = min(animal_yms)

    if host_category == "human":
        days = days_between(first_animal_ym, collection_date)
    else:
        days = days_between(collection_date, first_human_ym)

    # 人先于动物：跨物种能力已激活，interval=0（最高危）
    if days is not None and days < 0:
        days = 0

    cat = cat_interval_year(days) if days is not None else ""
    return days, cat

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
# 注意：同一 accession 可出现在多个亚型的 tsv 中（260+ 个跨亚型重复），
# 必须用 (subtype, accession) 做 key，否则后处理的亚型会覆盖先处理的，
# 导致 isolate 继承错误亚型的 cluster 标签（2026-08-02 修复）
print("加载 cluster 映射...")
acc2cid = {}
for st in SUBTYPES:
    with open(DATA / f"{st}_clusters.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            acc2cid[(st, r["accession"])] = (st, r["cluster_id"])

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
    # isolate 级 interval（每个 isolate 独立计算）
    "iso_interval_days", "iso_interval_cat",
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
                key = acc2cid.get((st, r["accession"]))
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

                # 计算 isolate 级 interval
                iso_days, iso_cat = compute_isolate_interval(
                    date_str,
                    r.get("host_category", ""),
                    lbl.get("first_human_ym", ""),
                    lbl.get("first_avian_ym", ""),
                    lbl.get("first_swine_ym", ""),
                    lbl.get("first_bovine_ym", ""),
                )
                row["iso_interval_days"] = iso_days if iso_days is not None else ""
                row["iso_interval_cat"] = iso_cat

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
