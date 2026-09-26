#!/usr/bin/env python3
"""V2：H7 亚型内 logit ↔ 与禽源（阴性）consensus 的序列距离 —— "末层按谱系典型性倒排"检验

动机（B1 解读）：若末层读出退化为"序列/亚型典型性"，则 H7 亚型内 probe logit
在末层应与"与 H7 阴性（禽源）consensus 的一致度"强相关（越典型越不被排负 /
方向取决于符号），而中层该相关应弱；H5 作对照（不应出现末层飙升）。

设计：
  - consensus = H7 holdout 阴性（label_is_jump=0）aligned 序列的列多数票
    （列覆盖率 ≥0.5 才计入）；identity_i = 匹配列数 / 有效列数。
  - logit 复用 g1_probes.npz 已存 probe（StandardScaler + w + intercept），
    sanity：L17/L28 复算 AUC_H7 与 g1 JSON 一致（容差 0.01）。
  - Spearman rho(logit, identity) 逐层 × 双标签 × {H7 全体, H7 阴性子集}；
    H5 同口径对照。阴性子集无 ± 成分混杂，为干净检验。
  - 附：H7+ vs H7− 的 identity 分布对比（rank-biserial / 中位百分位）。

判定（预注册式）：
  T1 H7 阴性子集 |rho| 在 late(28–30) 显著大于 mid(13–22)；
  T2 H5 阴性子集无此末层飙升（对照）。

运行：env1 python（纯 CPU，秒级）
输出：output/v2_h7_lineage_distance.json、figs/v2_h7_lineage_distance.png、
      figs/v2_scatter_{label}.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, ".")
from validation_exp.depth_reversal.config import (
    SPLIT_CSV, ALIGNED_CSV, ALL_LAYERS_NPY, LABELS, MID_LAYERS, LATE_LAYERS,
    LABEL_COLS, OUT_DIR,
)

CONS_COVERAGE_MIN = 0.5     # consensus 列最低覆盖率
AUC_TOL = 0.01


def consensus_identity(aln_seqs):
    """列多数票 consensus + 每条序列与 consensus 的一致度。"""
    arr = np.array([list(s) for s in aln_seqs])          # (n,1039)
    n, L = arr.shape
    cons = np.full(L, "-", dtype=arr.dtype)
    cov = np.zeros(L)
    for c in range(L):
        col = arr[:, c]
        ng = col[col != "-"]
        cov[c] = len(ng) / n
        if cov[c] >= CONS_COVERAGE_MIN:
            vals, cnts = np.unique(ng, return_counts=True)
            cons[c] = vals[cnts.argmax()]
    valid = cons != "-"
    iden = np.full(n, np.nan)
    for i in range(n):
        m = valid & (arr[i] != "-")
        iden[i] = (arr[i][m] == cons[m]).mean() if m.any() else np.nan
    return cons, iden


def logits_from_probe(emb, probe):
    mean, scale, w, b = probe
    return (emb - mean) / scale @ w + b


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figs").mkdir(exist_ok=True)

    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV,
                      usecols=["accession", "subtype", "aligned_ha_seq"])
    df = split.merge(ali, on=["accession", "subtype"], how="left",
                     validate="one_to_one")
    sp = split["split"].to_numpy()
    emb_all = np.load(ALL_LAYERS_NPY)                     # (31,11060,640)
    probes = np.load(OUT_DIR / "g1_probes.npz")
    g1 = json.load(open(OUT_DIR / "g1_layer_geometry.json"))
    n_layers = emb_all.shape[0]

    # ── sanity：L17/L28 复算 AUC_H7 ──
    from sklearn.metrics import roc_auc_score
    m_h7 = sp == "h7_holdout"
    yj = np.load(LABELS["label_is_jump"])
    for l in [17, 28]:
        pr = (probes[f"mean_L{l}_label_is_jump"], probes[f"scale_L{l}_label_is_jump"],
              probes[f"w_L{l}_label_is_jump"], probes[f"intercept_L{l}_label_is_jump"])
        z = logits_from_probe(emb_all[l][m_h7], pr)
        auc = roc_auc_score(yj[m_h7], z)
        ref = g1["per_layer"]["label_is_jump"][l]["auc_h7"]
        assert abs(auc - ref) < AUC_TOL, f"L{l} AUC 复算 {auc:.4f} ≠ g1 {ref:.4f}"
    print("sanity 通过：L17/L28 AUC_H7 与 G1 一致")

    res = {}
    fig_data = {}
    for st, m_st in [("H7", m_h7), ("H5", sp == "h5_holdout")]:
        rows = np.where(m_st)[0]
        cons, iden = consensus_identity(df.loc[m_st, "aligned_ha_seq"].tolist())
        res[st] = {"n": int(m_st.sum()), "labels": {}}
        fig_data[st] = {"iden": iden}
        for c in LABEL_COLS:
            y = np.load(LABELS[c])[rows]
            neg = y == 0
            # identity：以该亚型阴性重建 consensus（对 H7 即禽源典型性）
            cons_c, iden_c = consensus_identity(
                df.loc[m_st, "aligned_ha_seq"][neg].tolist())
            # 重算全体 identity（相对阴性 consensus）
            arr = np.array([list(s) for s in df.loc[m_st, "aligned_ha_seq"]])
            valid = cons_c != "-"
            iden_all = np.array([
                (a[valid & (a != "-")] == cons_c[valid & (a != "-")]).mean()
                for a in arr])
            per_layer = []
            for l in range(n_layers):
                pr = (probes[f"mean_L{l}_{c}"], probes[f"scale_L{l}_{c}"],
                      probes[f"w_L{l}_{c}"], probes[f"intercept_L{l}_{c}"])
                z = logits_from_probe(emb_all[l][rows], pr)
                rho_all = spearmanr(z, iden_all)
                rho_neg = spearmanr(z[neg], iden_all[neg])
                per_layer.append({
                    "layer": l,
                    "rho_all": float(rho_all.statistic),
                    "p_all": float(rho_all.pvalue),
                    "rho_neg": float(rho_neg.statistic),
                    "p_neg": float(rho_neg.pvalue),
                })
            # H7+ vs H7− identity 分布
            iden_pos, iden_neg = iden_all[~neg], iden_all[neg]
            rb = 2 * (rankdata(iden_all)[~neg].mean() - len(iden_pos) / 2
                      ) / len(iden_all) - 1 if len(iden_pos) else np.nan
            res[st]["labels"][c] = {
                "per_layer": per_layer,
                "n_pos": int((~neg).sum()),
                "identity_pos_median": float(np.median(iden_pos)),
                "identity_neg_median": float(np.median(iden_neg)),
                "rank_biserial_pos_vs_neg": float(rb),
                "mid_neg_mean": float(np.mean([p["rho_neg"] for p in per_layer
                                               if p["layer"] in MID_LAYERS])),
                "late_neg_mean": float(np.mean([p["rho_neg"] for p in per_layer
                                                if p["layer"] in LATE_LAYERS])),
                "late_all_mean": float(np.mean([p["rho_all"] for p in per_layer
                                                if p["layer"] in LATE_LAYERS])),
            }
            if st == "H7":
                fig_data[st][c] = {"per_layer": per_layer, "iden_all": iden_all,
                                   "neg": neg, "y": y}
            print(f"{st} {c}: mid_neg ρ 均值 "
                  f"{res[st]['labels'][c]['mid_neg_mean']:+.3f} → "
                  f"late_neg {res[st]['labels'][c]['late_neg_mean']:+.3f} | "
                  f"identity 中位 pos {np.median(iden_pos):.4f} vs "
                  f"neg {np.median(iden_neg):.4f}")

    with open(OUT_DIR / "v2_h7_lineage_distance.json", "w") as f:
        json.dump(res, f, indent=1)

    # ── 图1：ρ(logit, identity) 逐层曲线（阴性子集），H7 vs H5 对照 ──
    gap_h7 = np.array([p["decomp"]["H7"]["within+"] - p["decomp"]["H7"]["within-"]
                       for p in g1["per_layer"]["label_is_jump"]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, c in zip(axes, LABEL_COLS):
        for st, mk in [("H7", "o-"), ("H5", "s--")]:
            pl = res[st]["labels"][c]["per_layer"]
            ax.plot([p["layer"] for p in pl], [p["rho_neg"] for p in pl], mk,
                    ms=3, label=f"{st} negatives")
        ax.axvspan(21, 26, color="gray", alpha=0.12)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("layer")
        ax.set_title(f"rho(logit, avian-consensus identity), {c}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Spearman rho (negatives only)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figs" / "v2_h7_lineage_distance.png", dpi=200)
    plt.close(fig)

    # ── 图2：H7 阴性 L17 vs L28 散点（logit vs identity）──
    for c in LABEL_COLS:
        dat = fig_data["H7"][c]
        neg = dat["neg"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, l in zip(axes, [17, 28]):
            pr = (probes[f"mean_L{l}_{c}"], probes[f"scale_L{l}_{c}"],
                  probes[f"w_L{l}_{c}"], probes[f"intercept_L{l}_{c}"])
            z = logits_from_probe(emb_all[l][m_h7], pr)
            ax.scatter(dat["iden_all"][neg], z[neg], s=4, alpha=0.4,
                       label="neg")
            ax.scatter(dat["iden_all"][~neg], z[~neg], s=8, alpha=0.8, c="r",
                       label="pos")
            ax.set_xlabel("identity to H7-neg consensus")
            ax.set_ylabel(f"logit L{l}")
            r = spearmanr(z[neg], dat["iden_all"][neg])
            ax.set_title(f"L{l}  rho_neg={r.statistic:+.3f} (p={r.pvalue:.1e})")
            ax.legend(fontsize=8)
        fig.suptitle(f"V2: H7 within-subtype logit vs lineage typicality ({c})")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "figs" / f"v2_scatter_{c}.png", dpi=200)
        plt.close(fig)
    print("完成 →", OUT_DIR / "v2_h7_lineage_distance.json")


if __name__ == "__main__":
    main()
