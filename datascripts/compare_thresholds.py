#!/usr/bin/env python3
"""
对比不同 CD-HIT 阈值的 Jump 数量，找到最佳聚类参数。
"""

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("data/processed")
SUBTYPES = ["H1", "H3", "H5", "H7"]

THRESHOLDS = [0.99, 0.97, 0.95, 0.90]  # 要测试的 identity 阈值

HOST_CATEGORIES = [
    "human", "avian", "swine", "canine", "equine",
    "bovine", "feline", "mustelid", "marine_mammal",
    "other_mammal", "environmental", "unknown",
]


def run_cdhit(subtype: str, identity: float) -> dict[str, str]:
    """运行 CD-HIT 并返回 {accession: cluster_id}。"""
    fasta = DATA_DIR / f"{subtype}_sequences.fasta"
    if not fasta.exists():
        # 从 clean CSV 生成 FASTA
        csv_path = DATA_DIR / f"{subtype}_clean.csv"
        with open(csv_path, encoding="utf-8") as fin, \
             open(fasta, "w", encoding="utf-8") as fout:
            reader = csv.DictReader(fin)
            for row in reader:
                seq = row.get("ha_sequence", "")
                if seq:
                    fout.write(f">{row['accession']}\n{seq}\n")

    # Word size: 0.95+ → 5, 0.90-0.95 → 4, 0.85-0.90 → 4
    n_word = 5 if identity >= 0.95 else 4

    out_prefix = DATA_DIR / f"{subtype}_cdhit_{int(identity*100)}"
    cmd = [
        "cd-hit", "-i", str(fasta),
        "-o", str(out_prefix),
        "-c", str(identity),
        "-s", "0.8",
        "-g", "1",
        "-n", str(n_word),
        "-d", "0",
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    clstr_file = Path(str(out_prefix) + ".clstr")
    mapping = {}
    current = None
    with open(clstr_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current = line.split()[1]
            elif current and ">" in line:
                acc = line.split(">")[1].split("...")[0].strip()
                mapping[acc] = current

    # 清理
    for f in [clstr_file, out_prefix]:
        f.unlink(missing_ok=True)
    fasta.unlink(missing_ok=True)

    return mapping


def count_jumps(subtype: str, mapping: dict[str, str]) -> dict:
    """统计该聚类下的 jump 数量。"""
    clean_path = DATA_DIR / f"{subtype}_clean.csv"
    records = []
    with open(clean_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(row)

    cluster_hosts = defaultdict(set)
    for rec in records:
        acc = rec["accession"]
        cid = mapping.get(acc)
        if cid is None:
            continue
        hc = rec.get("host_category", "unknown")
        cluster_hosts[cid].add(hc)

    n_clusters = len(cluster_hosts)
    n_jump = 0
    n_animal_only = 0
    n_human_only = 0
    jump_sources = defaultdict(int)

    for cid, hosts in cluster_hosts.items():
        has_human = "human" in hosts
        animal_hosts = hosts - {"human", "unknown"}
        has_animal = bool(animal_hosts)

        if has_human and has_animal:
            n_jump += 1
            for ah in animal_hosts:
                jump_sources[ah] += 1
        elif has_human:
            n_human_only += 1
        elif has_animal:
            n_animal_only += 1

    return {
        "n_clusters": n_clusters,
        "n_jump": n_jump,
        "n_human_only": n_human_only,
        "n_animal_only": n_animal_only,
        "jump_sources": dict(jump_sources),
    }


def main():
    print("=" * 70)
    print("  CD-HIT 阈值对比：不同 identity 下的 Jump 数量")
    print("=" * 70)

    results = {}

    for ident in THRESHOLDS:
        print(f"\n{'─'*60}")
        print(f"  阈值: {ident*100:.0f}% identity")
        print(f"{'─'*60}")

        total_clusters = 0
        total_jumps = 0
        subtype_results = {}

        for st in SUBTYPES:
            clean_path = DATA_DIR / f"{st}_clean.csv"
            if not clean_path.exists():
                continue
            mapping = run_cdhit(st, ident)
            stats = count_jumps(st, mapping)
            subtype_results[st] = stats
            total_clusters += stats["n_clusters"]
            total_jumps += stats["n_jump"]
            sources = ", ".join(f"{k}:{v}" for k, v in stats["jump_sources"].items())
            print(f"  {st}: {stats['n_clusters']:>5} clusters, "
                  f"{stats['n_jump']:>4} jumps, "
                  f"{stats['n_human_only']:>4} human-only, "
                  f"{stats['n_animal_only']:>5} animal-only"
                  f"  [{sources}]")

        results[ident] = {
            "total_clusters": total_clusters,
            "total_jumps": total_jumps,
            "subtypes": subtype_results,
        }
        print(f"  {'─'*50}")
        print(f"  合计: {total_clusters} clusters, {total_jumps} jumps")

    # 汇总对比
    print(f"\n{'='*70}")
    print(f"  【汇总对比】")
    print(f"  {'Threshold':>10} {'Clusters':>9} {'Jumps':>6} {'Jump%':>7}")
    print(f"  {'─'*10} {'─'*9} {'─'*6} {'─'*7}")
    for ident in THRESHOLDS:
        r = results[ident]
        print(f"  {ident*100:>8.0f}%  {r['total_clusters']:>9} "
              f"{r['total_jumps']:>6} {100*r['total_jumps']/max(r['total_clusters'],1):>6.1f}%")

    # 推荐
    print(f"\n{'='*70}")
    print(f"  【建议】选择 jump 数量足够多但聚类不过于粗糙的阈值。")
    print(f"  97% 或 95% 通常是流感 HA 分析的良好平衡点。")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
