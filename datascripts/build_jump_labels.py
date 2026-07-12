#!/usr/bin/env python3
"""
Phase 3: 宿主时间线 + 简化标签
直接输出 data/processed/all_subtypes_simplified.csv (20 列)
"""

import csv
from collections import defaultdict, Counter
from pathlib import Path

DATA = Path("data/processed")
SUBTYPES = ["H1", "H3", "H5", "H7"]

HOSTS = ["human", "avian", "swine", "canine", "equine", "bovine",
         "feline", "mustelid", "marine_mammal", "other_mammal", "environmental"]

OUTPUT_COLS = [
    "cluster_id", "subtype", "cluster_size", "consensus_ha_seq", "serotype",
    "first_human_ym", "first_avian_ym", "first_swine_ym",
    "first_canine_ym", "first_equine_ym", "first_bovine_ym",
    "first_feline_ym", "first_other_mammal_ym", "first_environmental_ym",
    "host_categories",
    "label_is_jump", "jump_interval_cat",
    "label_is_jump_human", "jump_interval_human_cat",
    "label_jump_source",
]


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


def cat_interval(days, allow_negative=False):
    """天数 → 年分类: negative / <1yr / 1-3yr / 3-5yr / 5yr+"""
    if days is None:
        return ""
    if allow_negative and days < 0:
        return "negative"
    years = abs(days) / 365.25
    if years < 1:
        return "<1yr"
    elif years < 3:
        return "1-3yr"
    elif years < 5:
        return "3-5yr"
    return "5yr+"


def process_subtype(subtype):
    print(f"\n  {subtype}:")
    clean_path = DATA / f"{subtype}_clean.csv"
    cluster_path = DATA / f"{subtype}_clusters.tsv"

    if not clean_path.exists():
        return []

    clean = list(csv.DictReader(open(clean_path)))
    print(f"    {len(clean)} clean records")

    # cluster 映射
    acc2cid = {}
    with open(cluster_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            acc2cid[r["accession"]] = r["cluster_id"]

    # 代表序列 (CD-HIT .clstr 中 * 标记)
    reps = {}
    clstr_file = DATA / f"{subtype}_cdhit_clusters"
    if clstr_file.exists():
        cid = None
        for line in open(clstr_file):
            line = line.strip()
            if line.startswith(">Cluster"):
                cid = line.split()[1]
            elif cid and "*" in line and ">" in line:
                reps[cid] = line.split(">")[1].split("...")[0].strip()

    # 按 cluster 分组
    cd = defaultdict(lambda: {"hosts": defaultdict(list), "seros": [], "recs": []})
    for r in clean:
        cid = acc2cid.get(r["accession"])
        if not cid:
            continue
        hc = r.get("host_category", "unknown")
        y = (r.get("collection_year") or "").strip()
        m = (r.get("collection_month") or "").strip()
        # 跳过无日期的记录（year=-1），不计入宿主时间，但仍算入 cluster 总数
        if y and y != "-1":
            try:
                yi = int(y)
                mi = int(m) if m and m != "-1" else 6
                ym_str = f"{yi}-{str(mi).zfill(2)}"
                cd[cid]["hosts"][hc].append((yi, mi, ym_str))
            except ValueError:
                pass
        cd[cid]["seros"].append(r.get("serotype", ""))
        cd[cid]["recs"].append(r)

    rows = []
    for cid in sorted(cd.keys()):
        c = cd[cid]
        csize = len(c["recs"])

        # serotype
        sc = Counter(s for s in c["seros"] if s)
        sero = sc.most_common(1)[0][0] if sc else ""

        # 各宿主最早时间 YYYY-MM
        host_ym = {}
        for hc in HOSTS:
            entries = c["hosts"].get(hc, [])
            if entries:
                entries.sort(key=lambda x: (x[0], x[1]))
                host_ym[hc] = f"{entries[0][0]}-{str(entries[0][1]).zfill(2)}"
            else:
                host_ym[hc] = ""

        present = [h for h in HOSTS if host_ym[h]]
        non_unk = [h for h in present if h != "unknown"]

        # label_is_jump: 任意 2 宿主
        label_is_jump = 1 if len(non_unk) >= 2 else 0

        # jump_interval_cat: 跨物种最早→最晚宿主的时间跨度（绝对值，非方向性）
        # 修复: 原逻辑仅取 "animal→human 正值"，导致 H1+H3 的 human|swine 全为空
        # 新逻辑: 对所有 ≥2 宿主的 jump cluster 计算时间跨度
        jump_interval_cat = ""
        if label_is_jump:
            all_yms = [host_ym[h] for h in non_unk if host_ym[h]]
            if len(all_yms) >= 2:
                all_yms.sort()
                d = days_between(all_yms[0], all_yms[-1])
                if d is not None and d >= 0:
                    jump_interval_cat = cat_interval(d)

        # label_is_jump_human: 含 human
        has_human = "human" in present
        has_animal = any(h != "human" for h in non_unk)
        label_is_jump_human = 1 if (has_human and has_animal) else 0

        # jump_interval_human_cat: 可负
        jump_interval_human_cat = ""
        if label_is_jump_human:
            animals = [(h, host_ym[h]) for h in non_unk if h != "human"]
            if animals:
                animals.sort(key=lambda x: x[1])
                d = days_between(animals[0][1], host_ym["human"])
                jump_interval_human_cat = cat_interval(d, allow_negative=True)

        # label_jump_source
        label_jump_source = ""
        if label_is_jump_human:
            animals = [(h, host_ym[h]) for h in non_unk if h != "human"]
            if animals:
                animals.sort(key=lambda x: x[1])
                label_jump_source = animals[0][0]

        # 代表序列
        rep_acc = reps.get(cid)
        seq = ""
        for r in c["recs"]:
            if rep_acc and r["accession"] == rep_acc:
                seq = r.get("ha_sequence", "")
                break
        if not seq and c["recs"]:
            seq = c["recs"][0].get("ha_sequence", "")

        rows.append({
            "cluster_id": cid, "subtype": subtype, "cluster_size": csize,
            "consensus_ha_seq": seq, "serotype": sero,
            "first_human_ym": host_ym.get("human", ""),
            "first_avian_ym": host_ym.get("avian", ""),
            "first_swine_ym": host_ym.get("swine", ""),
            "first_canine_ym": host_ym.get("canine", ""),
            "first_equine_ym": host_ym.get("equine", ""),
            "first_bovine_ym": host_ym.get("bovine", ""),
            "first_feline_ym": host_ym.get("feline", ""),
            "first_other_mammal_ym": host_ym.get("other_mammal", ""),
            "first_environmental_ym": host_ym.get("environmental", ""),
            "host_categories": "|".join(present) if present else "unknown",
            "label_is_jump": label_is_jump,
            "jump_interval_cat": jump_interval_cat,
            "label_is_jump_human": label_is_jump_human,
            "jump_interval_human_cat": jump_interval_human_cat,
            "label_jump_source": label_jump_source,
        })

    n = len(rows)
    nj = sum(1 for r in rows if r["label_is_jump"])
    nh = sum(1 for r in rows if r["label_is_jump_human"])
    print(f"    {n} clusters, jump={nj}, jump_human={nh}")
    return rows


def main():
    print("=" * 60)
    print("  Phase 3: Jump 标签 (简化格式)")
    print("=" * 60)

    all_rows = []
    for st in SUBTYPES:
        all_rows.extend(process_subtype(st))

    out_path = DATA / "all_subtypes_simplified.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        writer.writerows(all_rows)

    n = len(all_rows)
    nj = sum(1 for r in all_rows if r["label_is_jump"])
    nh = sum(1 for r in all_rows if r["label_is_jump_human"])
    print(f"\n{'='*60}")
    print(f"  总计: {n} 毒株")
    print(f"  label_is_jump:        {nj} ({100*nj/max(n,1):.1f}%)")
    print(f"  label_is_jump_human:  {nh} ({100*nh/max(n,1):.1f}%)")
    print(f"  → {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
