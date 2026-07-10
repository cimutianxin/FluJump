#!/usr/bin/env python3
"""
从 MAFFT 对齐结果生成 aligned isolate 数据集
对齐基准: H3 reference (A/Aichi/2/1968), 所有序列统一长度
"""

import csv
from pathlib import Path

DATA = Path("data/processed_isolate_MAFFT")
ISOLATE_DIR = Path("data/processed_isolate")

# ── 解析对齐 ──
print("解析 MAFFT 对齐...")
aligned = {}
ref_aligned = ""
with open(DATA / "all_aligned.fasta") as f:
    seq_id = None; seq_chars = []
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if seq_id:
                s = "".join(seq_chars)
                if seq_id.startswith("H3_ref"):
                    ref_aligned = s
                else:
                    aligned[seq_id] = s
            seq_id = line[1:]
            seq_chars = []
        else:
            seq_chars.append(line)
    if seq_id and not seq_id.startswith("H3_ref"):
        aligned[seq_id] = "".join(seq_chars)

aligned_len = len(ref_aligned)
print(f"  对齐后长度: {aligned_len}")
print(f"  对齐序列数: {len(aligned)}")
print(f"  H3 参考 gap 数: {ref_aligned.count('-')}")

# H3 参考中非 gap 位置 → 原始 H3 位置 (1-based)
h3_numbering = {}
h3_pos = 0
for i, aa in enumerate(ref_aligned):
    if aa != "-":
        h3_pos += 1
        h3_numbering[i] = (h3_pos, aa)
print(f"  H3 参考非 gap 位置: {h3_pos}")

# ── 生成数据集 ──
print("\n生成 aligned 数据集...")
with open(ISOLATE_DIR / "all_isolates_clean.csv") as fin:
    isolates = list(csv.DictReader(fin))

OUTPUT = [
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
]

matched = 0
with open(DATA / "all_isolates_aligned.csv", "w", newline="") as fout:
    w = csv.DictWriter(fout, fieldnames=OUTPUT)
    w.writeheader()
    for r in isolates:
        key = f'{r["accession"]}|{r["subtype"]}|{r["host_category"]}'
        ali = aligned.get(key, "")
        if ali:
            matched += 1
            ali_len = len(ali)
            gaps = ali.count("-")
            w.writerow({
                "accession": r["accession"], "subtype": r["subtype"],
                "cluster_id": r["cluster_id"], "cluster_size": r["cluster_size"],
                "strain_name": r["strain_name"],
                "ha_sequence": r["ha_sequence"],
                "aligned_ha_seq": ali,
                "aligned_length": ali_len,
                "gap_count": gaps,
                "gap_ratio": round(gaps / max(ali_len, 1), 4),
                "host_category": r["host_category"],
                "serotype": r["serotype"],
                "collection_year": r["collection_year"],
                "collection_month": r["collection_month"],
                "collection_date": r["collection_date"],
                "geo_location": r["geo_location"],
                "country": r["country"],
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
            })

print(f"  匹配: {matched}/{len(isolates)}")
    print(f"  → {DATA / 'all_isolates_aligned.csv'}")
    print("Done.")
