#!/usr/bin/env python3
"""
FluJump — 一键 Pipeline: raw → clean → cluster → simplified

用法:
  python datascripts/run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = "/root/miniconda3/envs/env1/bin/python"
SCRIPTS = ROOT / "datascripts"

STEPS = [
    ("Phase 1: 数据清洗",   "clean_data.py"),
    ("Phase 2: CD-HIT 聚类", "cluster_strains.py"),
    ("Phase 3: Jump 标签",  "build_jump_labels.py"),
]

def main():
    print("=" * 60)
    print("  FluJump Pipeline  raw → simplified")
    print("=" * 60)

    for name, script in STEPS:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        result = subprocess.run([PY, str(SCRIPTS / script)], cwd=str(ROOT))
        if result.returncode != 0:
            print(f"\n  FAIL: {name} (exit {result.returncode})")
            sys.exit(result.returncode)

    out = ROOT / "data/processed/all_subtypes_simplified.csv"
    if out.exists():
        import csv
        with open(out) as f:
            rows = list(csv.DictReader(f))
        nj = sum(1 for r in rows if r["label_is_jump"] == "1")
        nh = sum(1 for r in rows if r["label_is_jump_human"] == "1")
        print(f"\n{'='*60}")
        print(f"  Pipeline 完成")
        print(f"  {len(rows)} 毒株, jump={nj}, jump_human={nh}")
        print(f"  → {out}")
        print(f"{'='*60}")
    else:
        print(f"\n  ⚠ 输出文件未生成: {out}")
        sys.exit(1)

if __name__ == "__main__":
    main()
