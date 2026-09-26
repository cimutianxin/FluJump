#!/usr/bin/env python3
"""G1：H7 反转驱动位点剖面（主数据内，纯 CPU）

mean-pooled probe 的 logit 精确逐位分解 c_i = scaler(h_i)·w（同 site_attribution），
按 (亚型 × label) 聚合每对齐列的平均贡献：
  d_pos[col] = mean(H7,+) − mean(H1H3,+)   —— 反转驱动位点（最负 = 把 H7 阳性推负的列）
  d_neg[col] = mean(H7,−) − mean(H1H3,−)   —— 亚型身份列（对照）
  d_within[col] = mean(H7,+) − mean(H7,−)  —— H7 内部分层列（对照）
  d_pos_H5[col] = mean(H5,+) − mean(H1H3,+) —— Group 1 对照（预期无系统负偏）
L28 双标签 probe 参数复用 site_attribution 已存 npz，并用 H5 holdout AUC 校验。

输出：output/g1_column_contrib_{label}.csv（1039 行）+ g1_top_drivers_{label}.csv
      + g1_summary.json
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from validation_exp.reversal_sites.config import (
    SPLIT_CSV, ALIGNED_CSV, MEAN_EMB_L3, RESIDUE_EMB_L28, RESIDUE_LENS,
    LABELS, PROBE_NPZ, COL_TO_H3_JSON, LABEL_COLS, ALN_LEN, OUT_DIR,
)

GROUPS = ["H1H3", "H5", "H7"]


def col_positions(aln_seq):
    """aligned 序列 → 每个 raw 残基对应的对齐列（0-based）"""
    return np.fromiter((i for i, c in enumerate(aln_seq) if c != "-"),
                       dtype=np.int32, count=len(aln_seq) - aln_seq.count("-"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=["accession", "subtype", "aligned_ha_seq"])
    df = split.merge(ali, on=["accession", "subtype"], how="left",
                     validate="one_to_one")
    assert df["aligned_ha_seq"].notna().all()
    n = len(df)
    subtype = df["subtype"].to_numpy()
    split_arr = df["split"].to_numpy()
    grp = np.where(np.isin(subtype, ["H1", "H3"]), "H1H3", subtype)  # H5/H7 保持
    lens = np.load(RESIDUE_LENS)
    col_maps = [col_positions(s) for s in df["aligned_ha_seq"]]
    for r in range(n):
        assert len(col_maps[r]) == lens[r]

    # H3 编号映射（HA1/HA2 分置）
    m = json.load(open(COL_TO_H3_JSON))
    c2h1 = {int(k): v for k, v in m["col_to_h3_ha1"].items()}
    c2h2 = {int(k): v for k, v in m["col_to_h3_ha2"].items()}

    def h3_label(col):
        if col in c2h1:
            return f"HA1_{c2h1[col]}"
        if col in c2h2:
            return f"HA2_{c2h2[col]}"
        return ""

    res = np.load(RESIDUE_EMB_L28, mmap_mode="r")
    X = np.load(MEAN_EMB_L3).astype(np.float32)

    summary = {}
    for label in LABEL_COLS:
        pz = np.load(PROBE_NPZ[label])
        w, smean, sscale = pz["w"], pz["scaler_mean"], pz["scaler_scale"]
        y = np.load(LABELS[label])
        assert len(y) == n

        # probe 校验：scaler(mean-pool)·w 应复现 H5 holdout AUC 0.7–0.9
        h5 = split_arr == "h5_holdout"
        z = (X[h5] - smean) / sscale
        auc_h5 = float(roc_auc_score(y[h5], z @ w))
        assert 0.65 < auc_h5 < 0.95, f"{label} probe 校验异常: {auc_h5}"
        print(f"[{label}] L28 probe 校验 H5 AUC={auc_h5:.4f}")

        # ── 逐列贡献累加（按 组×label 值）──
        keys = [(g, v) for g in GROUPS for v in (0, 1)]
        csum = {k: np.zeros(ALN_LEN, dtype=np.float64) for k in keys}
        ccnt = {k: np.zeros(ALN_LEN, dtype=np.int64) for k in keys}
        for i0 in range(0, n, 512):
            h = res[i0:i0 + 512].astype(np.float32)
            c = ((h - smean) / sscale) @ w          # (b,573) 每残基贡献
            for j, r in enumerate(range(i0, min(i0 + 512, n))):
                L = int(lens[r])
                k = (grp[r], int(y[r]))
                np.add.at(csum[k], col_maps[r], c[j, :L])
                np.add.at(ccnt[k], col_maps[r], 1)
            if i0 % (512 * 6) == 0:
                print(f"  [{label}] {i0}/{n}", flush=True)

        means = {k: np.where(ccnt[k] > 0, csum[k] / np.maximum(ccnt[k], 1), np.nan)
                 for k in keys}
        out = pd.DataFrame({"aln_col": np.arange(ALN_LEN)})
        out["h3"] = [h3_label(c) for c in out["aln_col"]]
        for k in keys:
            out[f"mean_{k[0]}_{'pos' if k[1] else 'neg'}"] = means[k]
            out[f"n_{k[0]}_{'pos' if k[1] else 'neg'}"] = ccnt[k]
        out["d_pos_H7"] = out["mean_H7_pos"] - out["mean_H1H3_pos"]
        out["d_neg_H7"] = out["mean_H7_neg"] - out["mean_H1H3_neg"]
        out["d_within_H7"] = out["mean_H7_pos"] - out["mean_H7_neg"]
        out["d_pos_H5"] = out["mean_H5_pos"] - out["mean_H1H3_pos"]
        out.to_csv(OUT_DIR / f"g1_column_contrib_{label}.csv", index=False)

        # top 驱动位点（d_pos_H7 最负）
        valid = out.dropna(subset=["d_pos_H7"])
        top = valid.nsmallest(20, "d_pos_H7")[
            ["aln_col", "h3", "d_pos_H7", "d_neg_H7", "d_within_H7", "d_pos_H5"]]
        top.to_csv(OUT_DIR / f"g1_top_drivers_{label}.csv", index=False)
        summary[label] = {
            "probe_h5_auc": round(auc_h5, 4),
            "n_valid_cols": int(len(valid)),
            "d_pos_H7_min": round(float(valid["d_pos_H7"].min()), 3),
            "d_pos_H7_median": round(float(valid["d_pos_H7"].median()), 3),
            "d_pos_H5_median": round(float(valid["d_pos_H5"].median()), 3),
            "frac_cols_H7neg": round(float((valid["d_pos_H7"] < 0).mean()), 3),
            "frac_cols_H5neg": round(float((valid["d_pos_H5"] < 0).mean()), 3),
            "top5": top.head(5).to_dict("records"),
        }
        print(f"[{label}] 有效列 {len(valid)}，d_pos_H7 中位 "
              f"{summary[label]['d_pos_H7_median']}（负列占比 "
              f"{summary[label]['frac_cols_H7neg']}），d_pos_H5 中位 "
              f"{summary[label]['d_pos_H5_median']}（负列占比 "
              f"{summary[label]['frac_cols_H5neg']}）")

    with open(OUT_DIR / "g1_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"→ {OUT_DIR}/g1_summary.json")


if __name__ == "__main__":
    main()
