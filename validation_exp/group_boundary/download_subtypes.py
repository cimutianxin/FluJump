#!/usr/bin/env python3
"""从 NCBI Protein 下载 H10/H4 HA 蛋白序列及元数据

逻辑复用 datascripts/download_flu.py（查询、解析、字段完全一致），
仅改输出目录到本实验自包含的 data/raw/，不动主数据目录。
"""

import csv
import re
import time
import calendar
import sys
from io import StringIO

from Bio import Entrez, SeqIO

sys.path.insert(0, ".")
from validation_exp.group_boundary.config import QUERIES, RAW_DIR

Entrez.email = "research@example.com"
BATCH_SIZE = 500
SLEEP_INTERVAL = 0.4

FIELD_NAMES = [
    "accession", "strain_name", "subtype", "serotype",
    "host_species", "host_category", "is_human", "organism",
    "collection_date_raw", "collection_year", "collection_month", "collection_day",
    "geo_location", "country",
    "isolation_source", "lab_host", "passage_history", "note",
    "ha_sequence", "seq_length", "segment", "definition",
]

MONTH_MAP = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if i > 0}

# 与 download_flu.py 一致的 host_category 推断（顺序重要）
HOST_RULES = [
    (["homo sapiens", "human", "homo", "patient"], "human"),
    (["chicken", "duck", "goose", "quail", "turkey", "mallard",
      "pheasant", "pigeon", "poultry", "wild bird", "teal",
      "gull", "shorebird", "swan", "fowl", "avian"], "avian"),
    (["swine", "pig", "porcine", "hog"], "swine"),
    (["equine", "horse"], "equine"),
    (["canine", "dog"], "canine"),
    (["bovine", "cattle", "cow", "dairy"], "bovine"),
    (["feline", "cat"], "feline"),
    (["ferret", "mink"], "mustelid"),
    (["seal", "whale", "marine"], "marine_mammal"),
    (["mouse", "bat", "rodent"], "other_mammal"),
    (["environment", "water", "air", "sewage"], "environmental"),
]


def parse_collection_date(raw):
    if not raw:
        return None, None, None
    raw = raw.strip()
    if re.match(r"^\d{4}/\d{4}$", raw):
        return None, None, None
    patterns = [
        (r"^(\d{1,2})-([A-Za-z]{3})-(\d{4})$",
         lambda g: (int(g[2]), MONTH_MAP.get(g[1].lower()), int(g[0]))),
        (r"^([A-Za-z]{3})-(\d{4})$",
         lambda g: (int(g[1]), MONTH_MAP.get(g[0].lower()), None)),
        (r"^(\d{4})-(\d{1,2})-(\d{1,2})$",
         lambda g: (int(g[0]), int(g[1]), int(g[2]))),
        (r"^(\d{4})/(\d{1,2})/(\d{1,2})$",
         lambda g: (int(g[0]), int(g[1]), int(g[2]))),
        (r"^(\d{4})-(\d{1,2})$",
         lambda g: (int(g[0]), int(g[1]), None)),
        (r"^(\d{4})$",
         lambda g: (int(g[0]), None, None)),
    ]
    for pat, extractor in patterns:
        m = re.match(pat, raw)
        if m:
            return extractor(m.groups())
    return None, None, None


def infer_host_category(host_species):
    if not host_species:
        return "unknown"
    hs = host_species.lower()
    for keywords, category in HOST_RULES:
        for kw in keywords:
            if kw in hs:
                return category
    return "unknown"


def _get(qualifiers, key):
    vals = qualifiers.get(key, [])
    return vals[0] if vals else ""


def extract_country(geo_location):
    if not geo_location:
        return ""
    return geo_location.split(":")[0].strip()


def extract_strain_name(qualifiers, definition):
    strain = _get(qualifiers, "strain")
    if strain:
        return strain
    isolate = _get(qualifiers, "isolate")
    if isolate:
        return isolate
    if definition:
        m = re.match(r"^([^,;]+)", definition)
        if m:
            return m.group(1).strip()
    return ""


def parse_record(record, subtype):
    feats = record.features
    quals = feats[0].qualifiers if feats else {}
    ha_seq = str(record.seq).upper()
    host_species = _get(quals, "host") or _get(quals, "organism")
    collection_date_raw = _get(quals, "collection_date")
    geo_location = _get(quals, "country") or _get(quals, "geo_loc_name")
    note = _get(quals, "note")
    year, month, day = parse_collection_date(collection_date_raw)
    host_category = infer_host_category(host_species)

    return {
        "accession": record.annotations.get("accessions", [""])[0]
        if record.annotations.get("accessions") else record.id,
        "strain_name": extract_strain_name(quals, record.description),
        "subtype": subtype,
        "serotype": _get(quals, "serotype"),
        "host_species": host_species,
        "host_category": host_category,
        "is_human": 1 if host_category == "human" else 0,
        "organism": _get(quals, "organism"),
        "collection_date_raw": collection_date_raw,
        "collection_year": year if year else "",
        "collection_month": month if month else "",
        "collection_day": day if day else "",
        "geo_location": geo_location,
        "country": extract_country(geo_location),
        "isolation_source": _get(quals, "isolation_source"),
        "lab_host": _get(quals, "lab_host"),
        "passage_history": note or _get(quals, "passage_history"),
        "note": note,
        "ha_sequence": ha_seq,
        "seq_length": len(ha_seq),
        "segment": _get(quals, "segment"),
        "definition": record.description,
    }


def download_subtype(subtype, query):
    output_path = RAW_DIR / f"{subtype}_raw.csv"
    print(f"\n{'='*60}\n  下载 {subtype}: {query}\n{'='*60}")

    handle = Entrez.esearch(db="protein", term=query, retmax=0, usehistory="y")
    result = Entrez.read(handle)
    handle.close()
    count = int(result["Count"])
    webenv, query_key = result["WebEnv"], result["QueryKey"]
    print(f"  → 共 {count} 条记录")
    if count == 0:
        return

    all_rows = []
    for start in range(0, count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, count)
        data = None
        for attempt in range(3):
            try:
                handle = Entrez.efetch(
                    db="protein", rettype="gb", retmode="text",
                    retstart=start, retmax=BATCH_SIZE,
                    webenv=webenv, query_key=query_key)
                data = handle.read()
                handle.close()
                break
            except Exception as e:
                print(f"    ⚠ 第 {attempt+1} 次尝试失败: {e}")
                time.sleep(2)
        if data is None:
            print(f"    ✗ 批次 {start}-{end} 下载失败，跳过")
            continue
        try:
            records = list(SeqIO.parse(StringIO(data), "genbank"))
        except Exception as e:
            print(f"    ✗ 解析失败: {e}")
            continue
        for rec in records:
            all_rows.append(parse_record(rec, subtype))
        print(f"    ✓ {start+1}-{end}/{count}  ({len(all_rows)} 已解析)")
        if end < count:
            time.sleep(SLEEP_INTERVAL)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  ✓ {output_path}: {len(all_rows)} 条")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for subtype, query in QUERIES.items():
        download_subtype(subtype, query)
    print("\n全部完成")


if __name__ == "__main__":
    main()
