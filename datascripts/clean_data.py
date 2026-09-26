#!/usr/bin/env python3
"""
Phase 1: 数据清洗
从 raw CSV 过滤掉 PDB 条目、非 HA 蛋白、无日期记录、合成构建体，
并改进 host_category 推断后输出 clean CSV。
"""

import csv
import re
import sys
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBTYPES = ["H1", "H3", "H5", "H7"]

# ── PDB accession 模式 ────────────────────────────────────
PDB_PATTERN = re.compile(r"^[0-9A-Z]{4}_[A-Z]$")

# ── 改进的 host_category 推断 ─────────────────────────────

# 英文关键词（匹配用 \b 词边界，防止 cat/pig/cow/air/water 等裸子串误伤
# furcata / flycatcher / pigeon / Moscow / Cairina / shearwater 等——09-19 审计）
ENGLISH_RULES = [
    (["homo sapiens", "human", "homo", "patient"], "human"),
    (["chicken", "duck", "goose", "quail", "turkey", "mallard",
      "pheasant", "pigeon", "poultry", "wild bird", "teal",
      "gull", "shorebird", "swan", "fowl",
      "waterbird", "waterfowl", "bluebird", "flycatcher", "guineafowl"], "avian"),
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

# 科学属名
GENUS_RULES = [
    (["anas", "aix", "aythya", "bucephala", "mares", "netta", "dendrocygna",
      "tadorna", "spatula", "mareca", "anser", "branta", "cygnus", "gallus",
      "meleagris", "coturnix", "phasianus", "pavo", "numida", "columba",
      "larus", "sterna", "charadrius", "calidris", "tringa", "arenaria",
      "phalacrocorax", "ardea", "egretta", "pelecanus", "phoenicopterus",
      "accipiter", "buteo", "falco", "aquila", "struthio", "passer",
      "taeniopygia", "corvus", "pica", "sturnus", "turdus", "tyto",
      "cairina"], "avian"),
    (["sus", "porcus"], "swine"),
    (["homo"], "human"),
    (["canis", "lupus"], "canine"),
    (["equus", "caballus"], "equine"),
    (["bos", "taurus", "indicus"], "bovine"),
    (["felis", "catus"], "feline"),
    (["mustela", "neovison", "neogale"], "mustelid"),
    (["phoca", "halichoerus", "mirounga", "leptonychotes", "tursiops",
      "delphinus", "balaenoptera", "megaptera", "orca"], "marine_mammal"),
    (["mus", "rattus", "myotis", "eptesicus", "pipistrellus", "pteropus",
      "rousettus", "rhinolophus"], "other_mammal"),
]

# strain_name 模式（如 A/swine/..., A/duck/...）
STRAIN_HOST_PATTERNS = [
    (r"/swine/|/pig/|/porcine/", "swine"),
    (r"/duck/|/chicken/|/goose/|/quail/|/turkey/|/mallard/|/pheasant/"
     r"|/pigeon/|/poultry/|/teal/|/gull/|/shorebird/|/swan/|/fowl/"
     r"|/anas /|/calidris /|/avian/|/bluebird/", "avian"),
    (r"/human/|/homo /|/patient/", "human"),
    (r"/canine/|/dog/", "canine"),
    (r"/equine/|/horse/", "equine"),
    (r"/bovine/|/cattle/|/cow/", "bovine"),
    (r"/feline/|/cats?/", "feline"),
    (r"/mink/|/ferret/", "mustelid"),
    (r"/seal/|/whale/|/marine/", "marine_mammal"),
    (r"/bat/|/mouse/|/rodent/", "other_mammal"),
    (r"/environment/|/water/|/sewage/|/air/", "environmental"),
]

# host_species 垃圾值：元数据串位（"Influenza A virus (A/...)"）或字面占位符，
# 不可直接用于推断，且旧类别视为不可信（09-19 审计）；
# 串位值内嵌真实株名 "(A/mallard duck/...)"，提取后再走正常推断
GARBAGE_SPECIES_PREFIX = "influenza a virus"
GARBAGE_SPECIES_EXACT = {"environment", "environmental"}


def _extract_embedded_strain(hs_lower: str) -> str:
    """从串位 host_species 提取内嵌株名 "(A/mallard duck/PA/...)" → 'a/mallard duck/...'"""
    m = re.search(r"\((a/[^)]+)\)", hs_lower)
    return m.group(1) if m else ""


def _compile_rules(rules):
    """关键词表 → \b 词边界正则（允许复数 s），防止裸子串误伤
    （cat→furcata/flycatcher、pig→pigeon、cow→Moscow、air→Cairo/Cairina、
    water→shearwater，09-19 审计）"""
    return [([re.compile(r'\b' + re.escape(kw) + r's?\b') for kw in kws], cat)
            for kws, cat in rules]


ENGLISH_RULES_C = _compile_rules(ENGLISH_RULES)


def is_garbage_species(host_species: str) -> bool:
    hs = (host_species or "").lower().strip()
    return hs.startswith(GARBAGE_SPECIES_PREFIX) or hs in GARBAGE_SPECIES_EXACT


def infer_host_category(host_species: str, strain_name: str) -> str:
    """改进的 host_category 推断，结合 host_species 和 strain_name。"""
    if not host_species:
        host_species = ""
    hs_lower = host_species.lower()

    # 快速路径：已知非流感物种
    if "synthetic" in hs_lower:
        return "unknown"
    # 垃圾 species：提取内嵌株名再推断（提取不到则仅靠 strain_name）
    if is_garbage_species(host_species):
        hs_lower = _extract_embedded_strain(hs_lower)

    # 1. 英文关键词匹配（\b 词边界）
    for patterns, category in ENGLISH_RULES_C:
        for pat in patterns:
            if pat.search(hs_lower):
                return category

    # 2. 科学属名匹配（检查完整词边界）
    for genera, category in GENUS_RULES:
        for genus in genera:
            # 匹配独立的属名（作为完整单词或生物学名字开头）
            if re.search(r'\b' + re.escape(genus) + r'\b', hs_lower):
                return category

    # 3. 从 strain_name 推断
    if strain_name:
        sn_lower = strain_name.lower()
        for pattern, category in STRAIN_HOST_PATTERNS:
            if re.search(pattern, sn_lower):
                return category

    return "unknown"


def clean_subtype(subtype: str):
    """清洗一个亚型的数据。"""
    in_path = RAW_DIR / f"{subtype}_raw.csv"
    out_path = OUT_DIR / f"{subtype}_clean.csv"

    if not in_path.exists():
        print(f"  ⚠ {in_path} 不存在，跳过")
        return

    stats = {"total": 0, "kept": 0, "pdb": 0, "bad_len": 0, "no_date": 0,
             "synthetic": 0, "host_improved": 0}

    rows_kept = []
    with open(in_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            stats["total"] += 1

            # 1. 排除 PDB
            if PDB_PATTERN.match(row["accession"]):
                stats["pdb"] += 1
                continue

            # 2. 序列长度过滤
            try:
                slen = int(row["seq_length"])
            except (ValueError, TypeError):
                stats["bad_len"] += 1
                continue
            if slen < 400 or slen > 600:
                stats["bad_len"] += 1
                continue

            # 3. 无日期 → 保留，标记 year=-1
            if not row["collection_year"] or not row["collection_year"].strip():
                row["collection_year"] = "-1"
                row["collection_month"] = "-1"
                stats["no_date"] += 1

            # 4. 排除合成构建体
            hs = (row.get("host_species") or "").lower()
            if "synthetic" in hs:
                stats["synthetic"] += 1
                continue

            # 5. 改进 host_category 推断（09-19 审计后规则）：
            #    improve-only——新推断非 unknown 且与旧值不同才覆盖；
            #    唯一例外是垃圾 species 行允许降级为 unknown
            #   （raw 旧值由下载期裸子串 bug 产生，不可信）
            old_hc = row.get("host_category", "")
            host_species = row.get("host_species") or ""
            strain_name = row.get("strain_name") or ""

            new_hc = infer_host_category(host_species, strain_name)
            if new_hc != old_hc and (new_hc != "unknown"
                                     or is_garbage_species(host_species)):
                stats["host_improved"] += 1
                row["host_category"] = new_hc
                row["is_human"] = "1" if new_hc == "human" else "0"

            rows_kept.append(row)
            stats["kept"] += 1

    # 写输出
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_kept)

    # 统计输出
    print(f"\n  {subtype}: {stats['total']} → {stats['kept']} "
          f"({100*stats['kept']/max(stats['total'],1):.1f}%)")
    print(f"  排除: PDB={stats['pdb']}, 长度={stats['bad_len']}, "
          f"合成={stats['synthetic']}  (无日期=-1: {stats['no_date']})")
    print(f"    host 改进: {stats['host_improved']}")
    return stats


def main():
    print("=" * 60)
    print("  Phase 1: 数据清洗")
    print("=" * 60)

    all_stats = {}
    for subtype in SUBTYPES:
        all_stats[subtype] = clean_subtype(subtype)

    # 总计
    total_in = sum(s["total"] for s in all_stats.values() if s)
    total_out = sum(s["kept"] for s in all_stats.values() if s)
    total_pdb = sum(s["pdb"] for s in all_stats.values() if s)
    total_badlen = sum(s["bad_len"] for s in all_stats.values() if s)
    total_nodate = sum(s["no_date"] for s in all_stats.values() if s)
    total_synth = sum(s["synthetic"] for s in all_stats.values() if s)
    total_improved = sum(s["host_improved"] for s in all_stats.values() if s)

    print(f"\n{'='*60}")
    print(f"  总计: {total_in} → {total_out} ({100*total_out/max(total_in,1):.1f}%)")
    print(f"  排除: PDB={total_pdb}, 长度={total_badlen}, "
          f"无日期={total_nodate}, 合成={total_synth}")
    print(f"  host 改进: {total_improved}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
