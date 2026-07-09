#!/usr/bin/env python3
"""
Phase 2: 序列聚类
对清洗后的 HA 序列用 CD-HIT（99% identity）聚类，每个 cluster = 一个毒株。
输出: {subtype}_clusters.tsv (accession → cluster_id)
      {subtype}_consensus.fasta (每个 cluster 的代表序列)
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

DATA_DIR = Path("data/processed")
SUBTYPES = ["H1", "H3", "H5", "H7"]

CD_HIT = "cd-hit"
CDHIT_IDENTITY = 0.99
CDHIT_COVERAGE = 0.9  # 短序列需覆盖长序列 90%


def write_fasta(subtype: str) -> Path:
    """从 clean CSV 写出 FASTA 文件用于聚类。"""
    csv_path = DATA_DIR / f"{subtype}_clean.csv"
    fasta_path = DATA_DIR / f"{subtype}_sequences.fasta"

    with open(csv_path, encoding="utf-8") as fin, \
         open(fasta_path, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            acc = row["accession"]
            seq = row["ha_sequence"]
            if seq:
                fout.write(f">{acc}\n{seq}\n")

    return fasta_path


def run_cdhit(fasta_path: Path, subtype: str) -> tuple[Path, Path]:
    """运行 CD-HIT，返回 (cluster_file, rep_fasta)。"""
    cluster_file = DATA_DIR / f"{subtype}_cdhit_clusters"
    rep_fasta = DATA_DIR / f"{subtype}_consensus.fasta"

    cmd = [
        CD_HIT, "-i", str(fasta_path),
        "-o", str(cluster_file),
        "-c", str(CDHIT_IDENTITY),
        "-s", str(CDHIT_COVERAGE),
        "-g", "1",          # 更精确的贪婪聚类
        "-n", "5",          # 99% 用 word size 5
        "-d", "0",          # 不截断描述
    ]
    print(f"  运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ✗ CD-HIT 失败:")
        print(result.stderr[-500:])
        sys.exit(1)

    # 打印统计信息
    for line in result.stdout.split("\n"):
        if "total seqs" in line or "total clusters" in line or "longest" in line:
            print(f"  {line.strip()}")

    # CD-HIT 的输出文件
    actual_cluster = Path(str(cluster_file) + ".clstr")
    actual_rep = Path(str(cluster_file))

    # 重命名为友好名称
    if actual_cluster.exists():
        actual_cluster.rename(cluster_file)
    if actual_rep.exists():
        # 这是代表序列，重命名
        pass  # 保留原样，cluster_file 是代表序列 FASTA

    return cluster_file, rep_fasta


def parse_clusters(cluster_path: Path) -> dict[str, str]:
    """解析 CD-HIT .clstr 文件，返回 {accession: cluster_id}。"""
    mapping = {}
    current_cluster = None

    with open(cluster_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = line.split()[1]  # e.g., "0"
            elif current_cluster is not None and ">" in line:
                # 格式: "0\t567aa, >accession... *" 或 "0\t567aa, >accession... at +/..."
                # 提取 accession
                m = line.split(">")[1].split("...")[0].strip()
                mapping[m] = current_cluster

    return mapping


def write_cluster_tsv(subtype: str, mapping: dict[str, str]):
    """写入 accession → cluster_id 映射。"""
    out_path = DATA_DIR / f"{subtype}_clusters.tsv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("accession\tcluster_id\n")
        for acc in sorted(mapping.keys()):
            f.write(f"{acc}\t{mapping[acc]}\n")
    return out_path


def process_subtype(subtype: str):
    """处理一个亚型的聚类。"""
    print(f"\n  {'='*50}")
    print(f"  {subtype}: 序列聚类 (CD-HIT, -c {CDHIT_IDENTITY})")
    print(f"  {'='*50}")

    csv_path = DATA_DIR / f"{subtype}_clean.csv"
    if not csv_path.exists():
        print(f"  ⚠ {csv_path} 不存在，跳过")
        return

    # 1. 写 FASTA
    fasta_path = write_fasta(subtype)
    seq_count = sum(1 for _ in open(fasta_path)) // 2
    print(f"  序列数: {seq_count}")

    # 2. 运行 CD-HIT
    cluster_file, rep_fasta = run_cdhit(fasta_path, subtype)

    # 3. 解析聚类结果
    mapping = parse_clusters(cluster_file)
    n_clusters = len(set(mapping.values()))
    print(f"  聚类数: {n_clusters}")
    print(f"  平均每簇: {len(mapping)/max(n_clusters,1):.1f} 条")

    # 4. 写 TSV
    tsv_path = write_cluster_tsv(subtype, mapping)
    print(f"  → {tsv_path}")

    # 5. 重命名代表序列文件
    rep_src = Path(str(cluster_file))
    if rep_src.exists():
        rep_src.rename(rep_fasta)
        print(f"  → {rep_fasta}")

    # 清理临时文件
    fasta_path.unlink(missing_ok=True)

    return n_clusters, seq_count


def main():
    print("=" * 60)
    print("  Phase 2: CD-HIT 序列聚类 (99% identity)")
    print("=" * 60)

    totals = {}
    for subtype in SUBTYPES:
        result = process_subtype(subtype)
        if result:
            totals[subtype] = result

    print(f"\n{'='*60}")
    print(f"  聚类汇总:")
    for st, (nc, ns) in totals.items():
        print(f"    {st}: {ns} 序列 → {nc} 聚类 "
              f"({100*nc/max(ns,1):.1f}% 压缩率)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
