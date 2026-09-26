#!/usr/bin/env python3
"""把 H10/H4 raw 序列以 mafft --add --keeplength 并入既有 MAFFT 参考对齐

供朴素基线（max identity to H1+H3 train 人源参考）使用：参考对齐
all_aligned_merged.faa 的列框架保持不变，新序列插入 gap 落入同一坐标系。

输出：output/{ST}_aligned.csv（accession, aligned_ha_seq），长度 = 参考宽度。
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO

sys.path.insert(0, ".")
from validation_exp.group_boundary.config import (
    SUBTYPES, PROC_DIR, OUT_DIR, MERGED_FAA,
)

MAFFT = "/usr/bin/mafft"
THREADS = 16


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 写新序列 fasta
    new_fasta = OUT_DIR / "h10_h4_new.faa"
    n_new = 0
    with open(new_fasta, "w") as f:
        for st in SUBTYPES:
            df = pd.read_csv(PROC_DIR / f"{st}_isolates.csv", dtype=str)
            for _, r in df.iterrows():
                f.write(f">{r['accession']}\n{r['ha_sequence']}\n")
                n_new += 1
    print(f"新序列: {n_new} 条 → {new_fasta}")

    # mafft --add --keeplength：保持参考对齐宽度
    out_fasta = OUT_DIR / "h10_h4_added.faa"
    cmd = [MAFFT, "--add", str(new_fasta), "--keeplength",
           "--thread", str(THREADS), str(MERGED_FAA)]
    print(f"运行: {' '.join(cmd)}")
    with open(out_fasta, "w") as fout:
        result = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stderr[-1000:])
        sys.exit(1)

    # 提取新序列的对齐结果（跳过参考序列）
    ref_ids = {r.id for r in SeqIO.parse(str(MERGED_FAA), "fasta")}
    n_written = {st: 0 for st in SUBTYPES}
    outs = {st: open(OUT_DIR / f"{st}_aligned.csv", "w") for st in SUBTYPES}
    for st in SUBTYPES:
        outs[st].write("accession,aligned_ha_seq\n")
    st_of = {}
    for st in SUBTYPES:
        df = pd.read_csv(PROC_DIR / f"{st}_isolates.csv", dtype=str)
        st_of.update({a: st for a in df["accession"]})
    for rec in SeqIO.parse(str(out_fasta), "fasta"):
        if rec.id in ref_ids:
            continue
        st = st_of.get(rec.id)
        if st:
            outs[st].write(f"{rec.id},{rec.seq}\n")
            n_written[st] += 1
    for st in SUBTYPES:
        outs[st].close()
        print(f"  {st}: {n_written[st]} 条对齐序列 → output/{st}_aligned.csv")


if __name__ == "__main__":
    main()
