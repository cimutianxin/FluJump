#!/usr/bin/env python3
"""Part A：logit 逐位点分解 + 已知标记位点富集检验

原理：linear probe logit = w·scaler(mean_i h_i) + b = mean_i[w·scaler(h_i)] + b，
每位点贡献 c_i = w·scaler(h_i) 是精确分解。在全体 1039 个对齐列上统计
贡献的类间分离度（train 上拟合、H5/H7 上验证），检验已知宿主适应标记位点
（H3 编号，经 Step 0 映射到对齐列）是否富集于 top 5%。

输出：
  output/partA_position_ranking_{label}.csv  — 每列的分离度/AUC/是否标记位点
  output/partA_summary.json                  — probe 复现指标 + Fisher 富集检验 + top 位点
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from validation_exp.site_attribution.config import (
    ALIGNED_CSV, SPLIT_CSV, RESIDUE_EMB, RESIDUE_LENS, MEAN_EMB_L3, LABELS,
    MARKER_SITES, LABEL_COLS, RANDOM_SEED, OUT_DIR,
)

C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
TOP_FRAC = 0.05


def col_positions(aln_seq):
    """aligned 序列 → 每个 raw 残基对应的对齐列（0-based）"""
    return np.fromiter((i for i, c in enumerate(aln_seq) if c != "-"),
                       dtype=np.int32, count=len(aln_seq) - aln_seq.count("-"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 数据与行序对齐（embedding 行序 = isolate_split.csv 行序）──
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=["accession", "subtype", "aligned_ha_seq"])
    df = split.merge(ali, on=["accession", "subtype"], how="left", validate="one_to_one")
    assert df["aligned_ha_seq"].notna().all()
    n = len(df)
    split_arr = df["split"].to_numpy()

    X = np.load(MEAN_EMB_L3).astype(np.float32)
    assert X.shape[0] == n
    lens = np.load(RESIDUE_LENS)
    assert len(lens) == n

    # ── 重训 probe（复现 train_probe.py 流程），恢复 w 与 scaler ──
    train_mask = split_arr == "train"
    probes = {}
    for label in LABEL_COLS:
        y_all = np.load(LABELS[label])
        gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": C_VALUES},
                          cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
                          scoring="roc_auc", n_jobs=-1)
        scaler = StandardScaler().fit(X[train_mask])
        gs.fit(scaler.transform(X[train_mask]), y_all[train_mask])
        clf = gs.best_estimator_
        # 复现检查：H5 holdout AUC 应在 0.7–0.9（参照 0.79）
        h5 = split_arr == "h5_holdout"
        auc_h5 = roc_auc_score(y_all[h5], clf.decision_function(scaler.transform(X[h5])))
        print(f"[{label}] best C={gs.best_params_['C']}  H5 AUC={auc_h5:.4f}")
        assert 0.65 < auc_h5 < 0.95, "probe 复现异常，检查流程"
        probes[label] = {"w": clf.coef_[0], "scaler": scaler,
                         "y": y_all, "auc_h5_repro": float(auc_h5)}

    # ── 逐位点贡献：c_i = scaler(h_i)·w，映射到对齐列 ──
    res = np.load(RESIDUE_EMB, mmap_mode="r")   # (11060, 573, 640) fp16
    col_maps = [col_positions(s) for s in df["aligned_ha_seq"]]
    for r in range(n):
        assert len(col_maps[r]) == lens[r], f"行 {r}: 非gap数与序列长度不一致"

    summary = {"probe_repro_auc_h5": {l: round(p["auc_h5_repro"], 4)
                                      for l, p in probes.items()},
               "labels": {}}
    for label, p in probes.items():
        w, scaler, y = p["w"], p["scaler"], p["y"]
        # 保存 probe 参数供 ISM 复用
        np.savez(OUT_DIR / f"probe_{label}.npz", w=w,
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        # 全体 isolate 的 (列, 贡献, 标签, split) 展开为大数组
        chunks = []
        for i0 in range(0, n, 512):
            h = res[i0:i0 + 512].astype(np.float32)          # (b,573,640)
            c = ((h - scaler.mean_) / scaler.scale_) @ w     # (b,573)
            for j, r in enumerate(range(i0, min(i0 + 512, n))):
                L = int(lens[r])
                chunks.append(pd.DataFrame({
                    "col": col_maps[r], "contrib": c[j, :L],
                    "y": y[r], "split": split_arr[r]}))
        big = pd.concat(chunks, ignore_index=True)
        print(f"[{label}] 贡献表: {len(big)} 残基条目")

        # ── 每列分离度与单点 AUC ──
        rows = []
        for col, g in big.groupby("col"):
            rec = {"aln_col": int(col)}
            for name, mask_fn in [("train", lambda s: s == "train"),
                                  ("h5", lambda s: s == "h5_holdout"),
                                  ("h7", lambda s: s == "h7_holdout")]:
                sub = g[mask_fn(g["split"].to_numpy())]
                if len(sub) < 50 or sub["y"].nunique() < 2:
                    rec[f"delta_{name}"] = np.nan
                    rec[f"auc_{name}"] = np.nan
                    continue
                v, t = sub["contrib"].to_numpy(), sub["y"].to_numpy()
                rec[f"delta_{name}"] = float(v[t == 1].mean() - v[t == 0].mean())
                rk = rankdata(v)
                n1 = int(t.sum())
                rec[f"auc_{name}"] = float((rk[t == 1].sum() - n1 * (n1 + 1) / 2)
                                           / (n1 * (len(t) - n1)))
            rows.append(rec)
        rank_df = pd.DataFrame(rows).sort_values("aln_col").reset_index(drop=True)

        # ── H3 编号与标记位点注释 ──
        with open(OUT_DIR / "col_to_h3.json") as f:
            col_to_h3 = {int(k): v for k, v in json.load(f)["col_to_h3_ha1"].items()}
        marker_cols = set()
        for sites in MARKER_SITES.values():
            marker_cols |= {c for c, hn in col_to_h3.items() if hn in sites}
        rank_df["h3_num"] = rank_df["aln_col"].map(col_to_h3)
        rank_df["is_marker"] = rank_df["aln_col"].isin(marker_cols)

        # ── 富集检验：train delta 绝对值 top 5% ──
        valid = rank_df.dropna(subset=["delta_train"]).copy()
        valid["rank"] = valid["delta_train"].abs().rank(ascending=False, method="max")
        k = max(1, int(len(valid) * TOP_FRAC))
        top = set(valid.nsmallest(k, "rank")["aln_col"])
        m_in = sum(1 for c in marker_cols if c in top)
        m_out = len(marker_cols) - m_in
        odds, pval = fisher_exact([[m_in, m_out], [k - m_in, len(valid) - k - m_out]])
        marker_rank = valid[valid["is_marker"]].copy()
        marker_rank["pct"] = marker_rank["rank"] / len(valid) * 100
        summary["labels"][label] = {
            "n_cols": len(valid), "top_k": k,
            "markers_total": len(marker_cols), "markers_in_top5pct": m_in,
            "fisher_odds": round(float(odds), 3), "fisher_p": float(f"{pval:.4g}"),
            "marker_rank_pct": {int(r["h3_num"]): round(float(r["pct"]), 1)
                                for _, r in marker_rank.iterrows()
                                if pd.notna(r["h3_num"])},
            "top10_cols": [
                {"aln_col": int(r["aln_col"]),
                 "h3_num": int(r["h3_num"]) if pd.notna(r["h3_num"]) else None,
                 "delta_train": round(float(r["delta_train"]), 4),
                 "auc_train": round(float(r["auc_train"]), 4),
                 "auc_h5": round(float(r["auc_h5"]), 4) if pd.notna(r["auc_h5"]) else None,
                 "is_marker": bool(r["is_marker"])}
                for _, r in valid.assign(a=valid["delta_train"].abs())
                .nlargest(10, "a").iterrows()],
        }
        print(f"[{label}] 标记位点 {m_in}/{len(marker_cols)} 入 top5% "
              f"(Fisher p={pval:.4g})")
        rank_df.to_csv(OUT_DIR / f"partA_position_ranking_{label}.csv", index=False)

    with open(OUT_DIR / "partA_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"→ {OUT_DIR / 'partA_summary.json'}")


if __name__ == "__main__":
    main()
