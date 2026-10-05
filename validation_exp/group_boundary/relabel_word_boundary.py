#!/usr/bin/env python3
"""H10/H4 host 标签按 09-19 词边界口径重算（S5，Major 7 彻底版）

两阶段，与主数据 09-19 修复等价：
  A. 下载端修复：raw csv 的 host_category/is_human 用修复后
     download_subtypes.infer_host_category（\b 词边界）重算——只动 bug 误伤行
     （如 "oyster catcher" 裸子串 cat→feline；主数据 raw 当时同样就地修复）
  B. clean 重跑：过滤规则不变 + improve-only 覆盖（垃圾 species 允许降级），
     序列与簇划分不变（不重跑 CD-HIT），只重算簇级 jump/jump_human 标签。
     {ST}_isolates.csv 行序保持 = embedding 行序（逐行断言 accession 不变）。

旧文件备份为 *.pre_wboundary 后缀；diff 报告写 output/relabel_report.json。
"""

import csv
import json
import shutil
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, ".")
from validation_exp.group_boundary.config import (
    SUBTYPES, RAW_DIR, PROC_DIR, OUT_DIR, SUBTYPE_PATTERN,
)
from validation_exp.group_boundary.prep_clusters import (
    PDB_PATTERN, infer_host_category, is_garbage_species, HOSTS,
)
from validation_exp.group_boundary.download_subtypes import (
    infer_host_category as infer_download,
)


def clean_rows_fixed(subtype):
    """复跑 clean 过滤 + 词边界 host 推断（与修复后 prep_clusters.clean_subtype 等价）"""
    rows = []
    with open(RAW_DIR / f"{subtype}_raw.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if PDB_PATTERN.match(row["accession"]):
                continue
            pat = SUBTYPE_PATTERN[subtype]
            if not (pat.search(row.get("serotype") or "")
                    or pat.search(row.get("strain_name") or "")
                    or pat.search(row.get("definition") or "")):
                continue
            try:
                slen = int(row["seq_length"])
            except (ValueError, TypeError):
                continue
            if slen < 400 or slen > 600:
                continue
            if not row["collection_year"] or not row["collection_year"].strip():
                row["collection_year"] = "-1"
                row["collection_month"] = "-1"
            hs = (row.get("host_species") or "").lower()
            if "synthetic" in hs:
                continue
            old_hc = row.get("host_category", "")
            host_species = row.get("host_species") or ""
            new_hc = infer_host_category(host_species, row.get("strain_name") or "")
            if new_hc != old_hc and (new_hc != "unknown"
                                     or is_garbage_species(host_species)):
                row["host_category"] = new_hc
                row["is_human"] = "1" if new_hc == "human" else "0"
            rows.append(row)
    return rows


def build_cluster_labels(subtype, rows, acc2cid):
    """同 prep_clusters.build_labels 的簇级标签逻辑"""
    cd = defaultdict(lambda: {"hosts": defaultdict(list), "seros": [], "recs": []})
    for r in rows:
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
        sc = Counter(s for s in c["seros"] if s)
        years = sorted({y for es in c["hosts"].values() for y, _ in es})
        label_rows.append({
            "cluster_id": cid, "subtype": subtype, "cluster_size": len(c["recs"]),
            "serotype": sc.most_common(1)[0][0] if sc else "",
            "host_categories": "|".join(present) if present else "unknown",
            "year_min": years[0] if years else "",
            "year_max": years[-1] if years else "",
            "first_human_ym": host_ym.get("human", ""),
            "label_is_jump": 1 if len(non_unk) >= 2 else 0,
            "label_is_jump_human": 1 if ("human" in present
                                         and any(h != "human" for h in non_unk)) else 0,
        })
    return {r["cluster_id"]: r for r in label_rows}


def repair_raw(subtype):
    """阶段A：raw csv 的 host_category/is_human 按词边界下载口径就地修复"""
    path = RAW_DIR / f"{subtype}_raw.csv"
    bak = Path(str(path) + ".pre_wboundary")
    if not bak.exists():
        shutil.copy(path, bak)
    # 修复口径 = 修复后下载推断与备份（下载当时裸子串值）的差异，保证重跑仍可报告
    old_rows = {r["accession"]: r for r in csv.DictReader(open(bak, encoding="utf-8"))}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    fieldnames = list(rows[0].keys())
    diff = {}
    for r in rows:
        old = old_rows.get(r["accession"], {}).get("host_category", r["host_category"])
        new = infer_download(r.get("host_species") or "")
        if new != old:
            diff[r["accession"]] = (old, new)
        if new != r["host_category"]:
            r["host_category"] = new
            r["is_human"] = "1" if new == "human" else "0"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  阶段A raw 修复 {len(diff)} 行:")
    for acc, (o, n) in sorted(diff.items()):
        hs = next(r["host_species"] for r in rows if r["accession"] == acc)
        print(f"    {acc} ({hs}): {o} → {n}")
    return {a: list(v) for a, v in diff.items()}


def _read_old(subtype, suffix):
    """旧状态优先取 .pre_wboundary 备份（首轮运行时创建），保证重跑报告仍对照原始口径"""
    bak = PROC_DIR / f"{subtype}_{suffix}.csv.pre_wboundary"
    cur = PROC_DIR / f"{subtype}_{suffix}.csv"
    src = bak if bak.exists() else cur
    return list(csv.DictReader(open(src, encoding="utf-8")))


def main():
    report = {}
    for st in SUBTYPES:
        print(f"\n{'='*60}\n  {st}\n{'='*60}")
        raw_diff = repair_raw(st)
        old_clean = _read_old(st, "clean")
        old_iso = _read_old(st, "isolates")
        old_lab = {r["cluster_id"]: r for r in _read_old(st, "cluster_labels")}

        new_rows = clean_rows_fixed(st)
        # 保留集必须与旧 clean 完全一致（序列/簇划分不变的前提）
        old_acc = [r["accession"] for r in old_clean]
        new_acc = [r["accession"] for r in new_rows]
        assert old_acc == new_acc, f"{st} clean 保留集变化！"
        acc2cid = {r["accession"]: r["cluster_id"] for r in old_iso}

        # host_category diff
        old_hc = {r["accession"]: r["host_category"] for r in old_clean}
        hc_diff = {r["accession"]: (old_hc[r["accession"]], r["host_category"])
                   for r in new_rows if r["host_category"] != old_hc[r["accession"]]}
        print(f"  host_category 变化 {len(hc_diff)} 行:")
        for acc, (o, n) in sorted(hc_diff.items()):
            print(f"    {acc}: {o} → {n}")

        new_lab = build_cluster_labels(st, new_rows, acc2cid)
        assert set(new_lab) == set(old_lab), f"{st} 簇集合变化！"

        # 簇标签 diff
        flipped, counts = [], {}
        for cid in old_lab:
            for col in ["label_is_jump", "label_is_jump_human"]:
                o, n = int(old_lab[cid][col]), int(new_lab[cid][col])
                if o != n:
                    flipped.append({"cluster_id": cid, "col": col, "old": o, "new": n,
                                    "hosts_old": old_lab[cid]["host_categories"],
                                    "hosts_new": new_lab[cid]["host_categories"]})
        for col in ["label_is_jump", "label_is_jump_human"]:
            counts[col] = {"old": sum(int(r[col]) for r in old_lab.values()),
                           "new": sum(int(r[col]) for r in new_lab.values())}
        print(f"  阳性簇计数: {counts}")
        print(f"  翻转簇: {json.dumps(flipped, ensure_ascii=False)}")

        # 备份（仅首次创建，保证重跑不覆盖原始口径）+ 重写（isolates 行序逐行断言）
        for suffix in ["clean", "cluster_labels", "isolates"]:
            bak = Path(str(PROC_DIR / f"{st}_{suffix}.csv") + ".pre_wboundary")
            if not bak.exists():
                shutil.copy(PROC_DIR / f"{st}_{suffix}.csv", bak)

        with open(PROC_DIR / f"{st}_clean.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
            w.writeheader()
            w.writerows(new_rows)

        lab_cols = list(next(iter(new_lab.values())).keys())
        with open(PROC_DIR / f"{st}_cluster_labels.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=lab_cols)
            w.writeheader()
            w.writerows([new_lab[cid] for cid in sorted(new_lab)])

        hc_new = {r["accession"]: r["host_category"] for r in new_rows}
        iso_cols = list(old_iso[0].keys())
        with open(PROC_DIR / f"{st}_isolates.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=iso_cols)
            w.writeheader()
            for r in old_iso:
                lab = new_lab[r["cluster_id"]]
                r["host_category"] = hc_new[r["accession"]]
                r["label_is_jump"] = lab["label_is_jump"]
                r["label_is_jump_human"] = lab["label_is_jump_human"]
                w.writerow(r)
        new_iso = list(csv.DictReader(open(PROC_DIR / f"{st}_isolates.csv", encoding="utf-8")))
        assert [r["accession"] for r in new_iso] == [r["accession"] for r in old_iso]
        print(f"  → {st}_clean/{st}_cluster_labels/{st}_isolates 已重写（行序不变）")

        report[st] = {"raw_repair": raw_diff,
                      "n_host_category_changed": len(hc_diff),
                      "host_category_changes": {a: list(v) for a, v in hc_diff.items()},
                      "pos_cluster_counts": counts, "flipped_clusters": flipped}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "relabel_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✓ {OUT_DIR}/relabel_report.json")


if __name__ == "__main__":
    main()
