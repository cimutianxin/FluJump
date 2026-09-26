#!/usr/bin/env python3
"""Step 0：建立 对齐列 ↔ H3 canonical 编号 映射并验证

做法：把 A/Aichi/2/1968（AAA43178，566 aa）用 mafft --add --keeplength 加入
all_aligned_merged.faa（保持原 1039 列不变），Aichi 行第 n 个非 gap 字符所在列 = 原始位置 n。
编号换算（经 PDB 1HGG/X-31 权威验证）：
  - HA1 canonical = 原始位置 − 16（信号肽 16 aa 不计入；canonical 1 = 原始 17 的 Q）
  - HA2 canonical = 原始位置 − 345（HA2 从原始 346 的 G 开始，融合肽 GLFGAIAGFI）
  验证锚点：canonical 226 = 原始 242 = L，canonical 228 = 原始 244 = S（X-31 实测）。
验证：H3 序列在 canonical 226 应富集 L/I/V、228 富集 S/A/T、190 富集 D/E；
  全体序列在 HA2-1 列（融合肽起点）应 >90% 为 G。
输出：output/col_to_h3.json（列 → {raw_pos, h1: canonical HA1, h2: canonical HA2}）+ 验证报告。
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from validation_exp.site_attribution.config import (
    ALIGNED_CSV, MERGED_FAA, ISOLATES_CSV, MAFFT, H3_REF_ACCESSION, OUT_DIR,
)

SPOT_CHECK = {190: set("DEN"), 225: set("DGNST"), 226: set("LIVQ"), 228: set("SATG")}


def read_fasta(path):
    seqs, sid, buf = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if sid:
                    seqs[sid] = "".join(buf)
                sid, buf = line[1:], []
            else:
                buf.append(line)
    if sid:
        seqs[sid] = "".join(buf)
    return seqs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 取 Aichi 参考序列 ──
    iso = pd.read_csv(ISOLATES_CSV, usecols=["accession", "subtype", "strain_name", "ha_sequence"])
    ref = iso[iso["accession"] == H3_REF_ACCESSION]
    assert len(ref) > 0, f"参考株 {H3_REF_ACCESSION} 不在 {ISOLATES_CSV}"
    ref_seq = ref.iloc[0]["ha_sequence"]
    print(f"参考株: {ref.iloc[0]['strain_name']} ({ref.iloc[0]['subtype']}), {len(ref_seq)} aa")
    assert len(ref_seq) == 566

    # 确认参考株不在原对齐中
    with open(MERGED_FAA) as f:
        for line in f:
            if line.startswith(">"):
                assert H3_REF_ACCESSION not in line, "参考株已在合并对齐中，需改逻辑"

    # ── mafft --add --keeplength（保持 1039 列）──
    ref_faa = OUT_DIR / "h3_ref.faa"
    ref_faa.write_text(f">{H3_REF_ACCESSION}\n{ref_seq}\n")
    out_faa = OUT_DIR / "all_aligned_with_h3ref.faa"
    if not out_faa.exists():
        print(f"运行 mafft --add --keeplength（输入 {MERGED_FAA}）...")
        with open(out_faa, "w") as fout:
            subprocess.run([MAFFT, "--add", str(ref_faa), "--keeplength",
                            "--thread", "4", str(MERGED_FAA)],
                           stdout=fout, check=True)
    seqs = read_fasta(out_faa)
    ref_aln = seqs[H3_REF_ACCESSION]
    L = len(ref_aln)
    print(f"对齐长度: {L}（期望 1039，--keeplength 保持原列）")
    assert all(len(s) == L for s in seqs.values())

    # ── 列 → 原始位置 → canonical H3 编号（HA1: raw−16；HA2: raw−345）──
    col_to_raw, col_to_h1, col_to_h2 = {}, {}, {}
    n = 0
    for i, c in enumerate(ref_aln):
        if c != "-":
            n += 1                      # 1-based 原始位置
            col_to_raw[i] = n
            if 17 <= n <= 345:
                col_to_h1[i] = n - 16   # canonical HA1 编号
            elif n >= 346:
                col_to_h2[i] = n - 345  # canonical HA2 编号
    assert n == 566, f"Aichi 非 gap 字符数 {n} != 566"
    # 锚点自检（X-31/1HGG 实测）：canonical 226=L, 228=S；融合肽 HA2-1=G
    raw_at = {v: k for k, v in col_to_raw.items()}
    assert ref_aln[raw_at[242]] == "L" and ref_aln[raw_at[244]] == "S", "L226/S228 锚点不符"
    assert ref_aln[raw_at[346]] == "G", "融合肽锚点不符"
    print(f"canonical HA1 覆盖列数: {len(col_to_h1)}, HA2: {len(col_to_h2)}"
          f"（其余列为信号肽区/其他亚型插入）")

    # ── 验证：抽查标记位点列的残基身份（H3 序列）+ 融合肽列（全体）──
    df = pd.read_csv(ALIGNED_CSV, usecols=["accession", "subtype", "aligned_ha_seq"])
    h3 = df[df["subtype"] == "H3"]["aligned_ha_seq"]
    print(f"\n验证（H3 序列 n={len(h3)}）:")
    h1_to_col = {v: k for k, v in col_to_h1.items()}
    report = {}
    all_pass = True
    for site, expected in SPOT_CHECK.items():
        col = h1_to_col.get(site)
        residues = Counter(s[col] for s in h3)
        top = residues.most_common(3)
        hit = sum(v for k, v in residues.items() if k in expected) / len(h3)
        ok = hit > 0.5
        all_pass &= ok
        report[site] = {"col": col, "top": top, "expected_frac": round(hit, 4), "pass": ok}
        print(f"  H3 {site} (列 {col}): top={top} 预期残基占比={hit:.3f} {'PASS' if ok else 'FAIL'}")
    # 融合肽列：全体序列应 >90% G
    fp_col = raw_at[346]
    fp = Counter(s[fp_col] for s in df["aligned_ha_seq"])
    fp_g = fp["G"] / len(df)
    all_pass &= fp_g > 0.9
    report["HA2_1_fusion"] = {"col": fp_col, "top": fp.most_common(3),
                              "G_frac": round(fp_g, 4), "pass": fp_g > 0.9}
    print(f"  融合肽 HA2-1 (列 {fp_col}, 全体 n={len(df)}): top={fp.most_common(3)} "
          f"G 占比={fp_g:.3f} {'PASS' if fp_g > 0.9 else 'FAIL'}")

    payload = {
        "ref": H3_REF_ACCESSION, "aln_len": L,
        "col_to_raw": {str(k): v for k, v in col_to_raw.items()},
        "col_to_h3_ha1": {str(k): v for k, v in col_to_h1.items()},
        "col_to_h3_ha2": {str(k): v for k, v in col_to_h2.items()},
        "numbering_note": "canonical H3 = raw-16 (HA1, raw 17-345) / raw-345 (HA2, raw 346-566)，经 PDB 1HGG 验证",
        "spot_check": report, "all_pass": bool(all_pass),
    }
    with open(OUT_DIR / "col_to_h3.json", "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n{'映射验证全部通过' if all_pass else '⚠ 存在 FAIL，需排查后再继续'} → {OUT_DIR / 'col_to_h3.json'}")


if __name__ == "__main__":
    main()
