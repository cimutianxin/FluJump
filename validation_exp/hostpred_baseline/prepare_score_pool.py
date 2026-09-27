"""构建 VirHostPRED 全量打分池（步骤 1 前置）。

打分对象（口径与 notes/09-28-00 日志一致）：
  - H5/H7 holdout：isolate_split.csv 中 split ∈ {h5_holdout, h7_holdout}（2,585）
  - 前向评估池：year==2024（2,677）/ year==2025（1,020）/ 2026 表（201）
  - 训练集：split==train 且 year ≤2023 / ≤2024 / ≤2025（3,426 / 5,262 / 5,933）
按 (accession, subtype) 去重，池成员以布尔列标记。

服务端拒绝非标准残基序列（HTTP 400）：统计各池占比，跳过清单落盘
output/skipped_nonstd.csv，打分池仅含标准 20 字母表序列。

产出：
  output/score_pool.csv      —— 待打分全量（标准残基）
  output/skipped_nonstd.csv  —— 被跳过序列 + 非标准字符 + 池成员标记
"""
import sys

import pandas as pd

sys.path.insert(0, ".")
from validation_exp.temporal_validation.config import (
    ALIGNED_CSV, SPLIT_CSV, CLEAN_2026_CSV,
)

OUT_DIR = "validation_exp/hostpred_baseline/output"
POOL_COLS = ["in_h5_holdout", "in_h7_holdout", "in_2024", "in_2025", "in_2026",
             "in_train2023", "in_train2024", "in_train2025"]
STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def main():
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=[
        "accession", "subtype", "ha_sequence", "collection_year"])
    df = split.merge(ali, on=["accession", "subtype"], validate="one_to_one")
    df["year"] = pd.to_numeric(df["collection_year"], errors="coerce")

    df["in_h5_holdout"] = df["split"] == "h5_holdout"
    df["in_h7_holdout"] = df["split"] == "h7_holdout"
    df["in_2024"] = df["year"] == 2024
    df["in_2025"] = df["year"] == 2025
    df["in_2026"] = False
    for c in (2023, 2024, 2025):
        df[f"in_train{c}"] = (df["split"] == "train") & (df["year"] <= c)

    d26 = pd.read_csv(CLEAN_2026_CSV, usecols=["accession", "subtype", "ha_sequence"])
    d26["year"] = 2026
    for c in POOL_COLS:
        d26[c] = c == "in_2026"

    pool = pd.concat([df[["accession", "subtype", "ha_sequence", "year"] + POOL_COLS],
                      d26[["accession", "subtype", "ha_sequence", "year"] + POOL_COLS]],
                     ignore_index=True)
    n0 = len(pool)
    # (accession, subtype) 去重，池标记取并集
    pool = pool.groupby(["accession", "subtype"], as_index=False).agg(
        ha_sequence=("ha_sequence", "first"), year=("year", "first"),
        **{c: (c, "any") for c in POOL_COLS})
    print(f"合并 {n0} 行 -> 去重 {len(pool)} 条唯一 (accession, subtype)")
    print("各池条数:", {c: int(pool[c].sum()) for c in POOL_COLS})

    # 非标准残基统计 + 过滤
    nonstd = pool["ha_sequence"].apply(lambda s: "".join(sorted(set(s) - STD_AA)))
    pool["nonstd_chars"] = nonstd
    skipped = pool[nonstd != ""]
    for c in POOL_COLS:
        n_in, n_skip = int(pool[c].sum()), int(skipped[c].sum())
        print(f"  {c}: {n_skip}/{n_in} 含非标准残基"
              f"（{n_skip / max(n_in, 1) * 100:.2f}%）")
    skipped.to_csv(f"{OUT_DIR}/skipped_nonstd.csv", index=False)

    ok = pool[nonstd == ""].drop(columns=["nonstd_chars"])
    ok.to_csv(f"{OUT_DIR}/score_pool.csv", index=False)
    print(f"待打分 {len(ok)} 条（跳过 {len(skipped)} 条，"
          f"跳过率 {len(skipped) / len(pool) * 100:.3f}%）")
    print("待打分各池:", {c: int(ok[c].sum()) for c in POOL_COLS})


if __name__ == "__main__":
    main()
