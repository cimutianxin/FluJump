#!/usr/bin/env python3
"""H10/H4 单亚型版 Phase 1–3：clean → CD-HIT 聚类 → cluster 级 jump 标签

规则与主 pipeline 完全一致：
  - clean：datascripts/clean_data.py（PDB 排除、长度 400–600、year=-1 标记、
    合成构建体排除、host_category 三段推断）
  - 聚类：datascripts/cluster_strains.py（CD-HIT -c 0.99 -s 0.9 -g 1 -n 5）
  - 标签：datascripts/build_jump_labels.py（cluster 内 ≥2 个非 unknown 宿主
    类别 → is_jump；human+animal → jump_human）

输出（data/processed/ 下）：
  {ST}_clean.csv          清洗后 isolate 表
  {ST}_clusters.tsv       accession → cluster_id
  {ST}_cluster_labels.csv cluster 级标签表
  {ST}_isolates.csv       isolate 级评估表（clean + cluster_id + 簇标签，
                          行序即 embedding 行序）
并打印阳性簇构成明细（宿主类别/年份/serotype），供已知事件核对。
"""

import csv
import re
import subprocess
import sys
from collections import defaultdict, Counter

sys.path.insert(0, ".")
from validation_exp.group_boundary.config import (
    SUBTYPES, RAW_DIR, PROC_DIR, CD_HIT, CDHIT_IDENTITY, CDHIT_COVERAGE,
    SUBTYPE_PATTERN,
)

# ═══════════ clean 规则（复刻 datascripts/clean_data.py）═══════════

PDB_PATTERN = re.compile(r"^[0-9A-Z]{4}_[A-Z]$")

ENGLISH_RULES = [
    (["homo sapiens", "human", "homo", "patient"], "human"),
    (["chicken", "duck", "goose", "quail", "turkey", "mallard",
      "pheasant", "pigeon", "poultry", "wild bird", "teal",
      "gull", "shorebird", "swan", "fowl"], "avian"),
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

GENUS_RULES = [
    (["anas", "aix", "aythya", "bucephala", "mares", "netta", "dendrocygna",
      "tadorna", "spatula", "mareca", "anser", "branta", "cygnus", "gallus",
      "meleagris", "coturnix", "phasianus", "pavo", "numida", "columba",
      "larus", "sterna", "charadrius", "calidris", "tringa", "arenaria",
      "phalacrocorax", "ardea", "egretta", "pelecanus", "phoenicopterus",
      "accipiter", "buteo", "falco", "aquila", "struthio", "passer",
      "taeniopygia", "corvus", "pica", "sturnus", "turdus"], "avian"),
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

STRAIN_HOST_PATTERNS = [
    (r"/swine/|/pig/|/porcine/", "swine"),
    (r"/duck/|/chicken/|/goose/|/quail/|/turkey/|/mallard/|/pheasant/"
     r"|/pigeon/|/poultry/|/teal/|/gull/|/shorebird/|/swan/|/fowl/"
     r"|/anas /|/calidris /|/avian/", "avian"),
    (r"/human/|/homo /|/patient/", "human"),
    (r"/canine/|/dog/", "canine"),
    (r"/equine/|/horse/", "equine"),
    (r"/bovine/|/cattle/|/cow/", "bovine"),
    (r"/feline/|/cat/", "feline"),
    (r"/mink/|/ferret/", "mustelid"),
    (r"/seal/|/whale/|/marine/", "marine_mammal"),
    (r"/bat/|/mouse/|/rodent/", "other_mammal"),
    (r"/environment/|/water/|/sewage/|/air/", "environmental"),
]


def infer_host_category(host_species, strain_name):
    if not host_species:
        host_species = ""
    hs_lower = host_species.lower()
    if "synthetic" in hs_lower:
        return "unknown"
    for keywords, category in ENGLISH_RULES:
        for kw in keywords:
            if kw in hs_lower:
                return category
    for genera, category in GENUS_RULES:
        for genus in genera:
            if re.search(r'\b' + re.escape(genus) + r'\b', hs_lower):
                return category
    if strain_name:
        sn_lower = strain_name.lower()
        for pattern, category in STRAIN_HOST_PATTERNS:
            if re.search(pattern, sn_lower):
                return category
    return "unknown"


def clean_subtype(subtype):
    in_path = RAW_DIR / f"{subtype}_raw.csv"
    out_path = PROC_DIR / f"{subtype}_clean.csv"
    stats = {"total": 0, "kept": 0, "pdb": 0, "bad_len": 0, "no_date": 0,
             "synthetic": 0, "host_improved": 0, "wrong_subtype": 0}
    rows_kept = []
    with open(in_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            stats["total"] += 1
            if PDB_PATTERN.match(row["accession"]):
                stats["pdb"] += 1
                continue
            # 亚型一致性：serotype/strain_name/definition 均不含本亚型标注则排除
            pat = SUBTYPE_PATTERN[subtype]
            if not (pat.search(row.get("serotype") or "")
                    or pat.search(row.get("strain_name") or "")
                    or pat.search(row.get("definition") or "")):
                stats["wrong_subtype"] += 1
                continue
            try:
                slen = int(row["seq_length"])
            except (ValueError, TypeError):
                stats["bad_len"] += 1
                continue
            if slen < 400 or slen > 600:
                stats["bad_len"] += 1
                continue
            if not row["collection_year"] or not row["collection_year"].strip():
                row["collection_year"] = "-1"
                row["collection_month"] = "-1"
                stats["no_date"] += 1
            hs = (row.get("host_species") or "").lower()
            if "synthetic" in hs:
                stats["synthetic"] += 1
                continue
            old_hc = row.get("host_category", "")
            new_hc = infer_host_category(row.get("host_species") or "",
                                         row.get("strain_name") or "")
            if old_hc == "unknown" and new_hc != "unknown":
                stats["host_improved"] += 1
                row["host_category"] = new_hc
                row["is_human"] = "1" if new_hc == "human" else "0"
            rows_kept.append(row)
            stats["kept"] += 1

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_kept)
    print(f"  {subtype}: {stats['total']} → {stats['kept']} clean "
          f"(PDB={stats['pdb']}, 长度={stats['bad_len']}, 合成={stats['synthetic']}, "
          f"杂亚型={stats['wrong_subtype']}, 无日期={stats['no_date']}, "
          f"host改进={stats['host_improved']})")
    return rows_kept


# ═══════════ CD-HIT 聚类（参数同 datascripts/cluster_strains.py）═══════════

def run_cdhit(subtype):
    clean_path = PROC_DIR / f"{subtype}_clean.csv"
    fasta_path = PROC_DIR / f"{subtype}_sequences.fasta"
    with open(clean_path, encoding="utf-8") as fin, \
         open(fasta_path, "w", encoding="utf-8") as fout:
        for row in csv.DictReader(fin):
            if row["ha_sequence"]:
                fout.write(f">{row['accession']}\n{row['ha_sequence']}\n")

    out_base = PROC_DIR / f"{subtype}_cdhit_out"
    cmd = [CD_HIT, "-i", str(fasta_path), "-o", str(out_base),
           "-c", str(CDHIT_IDENTITY), "-s", str(CDHIT_COVERAGE),
           "-g", "1", "-n", "5", "-d", "0"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-500:])
        sys.exit(1)

    # 解析 .clstr → accession → cluster_id（亚型前缀防跨亚型撞号）
    clstr_path = str(out_base) + ".clstr"
    mapping = {}
    cid = None
    with open(clstr_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                cid = f"{subtype}_{line.split()[1]}"
            elif cid and ">" in line:
                acc = line.split(">")[1].split("...")[0].strip()
                mapping[acc] = cid
    fasta_path.unlink(missing_ok=True)
    n_clusters = len(set(mapping.values()))
    print(f"  {subtype}: {len(mapping)} 序列 → {n_clusters} cluster")
    return mapping


# ═══════════ cluster 标签（复刻 datascripts/build_jump_labels.py）═══════════

HOSTS = ["human", "avian", "swine", "canine", "equine", "bovine",
         "feline", "mustelid", "marine_mammal", "other_mammal", "environmental"]


def build_labels(subtype, clean_rows, acc2cid):
    cd = defaultdict(lambda: {"hosts": defaultdict(list), "seros": [], "recs": []})
    for r in clean_rows:
        cid = acc2cid.get(r["accession"])
        if not cid:
            continue
        hc = r.get("host_category", "unknown")
        y = (r.get("collection_year") or "").strip()
        m = (r.get("collection_month") or "").strip()
        if y and y != "-1":
            try:
                yi, mi = int(y), (int(m) if m and m != "-1" else 6)
                cd[cid]["hosts"][hc].append((yi, mi))
            except ValueError:
                pass
        cd[cid]["seros"].append(r.get("serotype", ""))
        cd[cid]["recs"].append(r)

    label_rows = []
    for cid in sorted(cd.keys()):
        c = cd[cid]
        host_ym = {}
        for hc in HOSTS:
            entries = c["hosts"].get(hc, [])
            if entries:
                entries.sort()
                host_ym[hc] = f"{entries[0][0]}-{str(entries[0][1]).zfill(2)}"
            else:
                host_ym[hc] = ""
        present = [h for h in HOSTS if host_ym[h]]
        non_unk = [h for h in present if h != "unknown"]
        label_is_jump = 1 if len(non_unk) >= 2 else 0
        has_human = "human" in present
        has_animal = any(h != "human" for h in non_unk)
        label_is_jump_human = 1 if (has_human and has_animal) else 0
        sc = Counter(s for s in c["seros"] if s)
        years = sorted({y for es in c["hosts"].values() for y, _ in es})
        label_rows.append({
            "cluster_id": cid, "subtype": subtype, "cluster_size": len(c["recs"]),
            "serotype": sc.most_common(1)[0][0] if sc else "",
            "host_categories": "|".join(present) if present else "unknown",
            "year_min": years[0] if years else "",
            "year_max": years[-1] if years else "",
            "first_human_ym": host_ym.get("human", ""),
            "label_is_jump": label_is_jump,
            "label_is_jump_human": label_is_jump_human,
        })

    out_path = PROC_DIR / f"{subtype}_cluster_labels.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(label_rows)

    n = len(label_rows)
    nj = sum(r["label_is_jump"] for r in label_rows)
    nh = sum(r["label_is_jump_human"] for r in label_rows)
    print(f"  {subtype}: {n} cluster, jump={nj}, jump_human={nh} → {out_path}")

    # 阳性簇构成明细（sanity check：已知事件核对）
    for r in label_rows:
        if r["label_is_jump"]:
            print(f"    [jump+] {r['cluster_id']}: size={r['cluster_size']} "
                  f"{r['host_categories']} {r['year_min']}-{r['year_max']} "
                  f"sero={r['serotype']}")
    return {r["cluster_id"]: r for r in label_rows}


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    for subtype in SUBTYPES:
        print(f"\n{'='*60}\n  {subtype}\n{'='*60}")
        clean_rows = clean_subtype(subtype)
        acc2cid = run_cdhit(subtype)
        cl2label = build_labels(subtype, clean_rows, acc2cid)

        # isolate 级评估表（行序 = embedding 行序）
        out_path = PROC_DIR / f"{subtype}_isolates.csv"
        cols = ["accession", "subtype", "strain_name", "host_category",
                "collection_year", "country", "ha_sequence", "cluster_id",
                "label_is_jump", "label_is_jump_human"]
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in clean_rows:
                cid = acc2cid.get(r["accession"], "")
                lab = cl2label.get(cid, {})
                writer.writerow({
                    "accession": r["accession"], "subtype": subtype,
                    "strain_name": r.get("strain_name", ""),
                    "host_category": r.get("host_category", ""),
                    "collection_year": r.get("collection_year", ""),
                    "country": r.get("country", ""),
                    "ha_sequence": r["ha_sequence"],
                    "cluster_id": cid,
                    "label_is_jump": lab.get("label_is_jump", ""),
                    "label_is_jump_human": lab.get("label_is_jump_human", ""),
                })
        print(f"  → {out_path}: {len(clean_rows)} 行")


if __name__ == "__main__":
    main()
