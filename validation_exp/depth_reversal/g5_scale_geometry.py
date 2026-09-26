#!/usr/bin/env python3
"""G5：规模普适核查——650M/3B 上的层几何同构性（H1/H3 跨规模）

在 scale_replication 已提取的全层 mean-pooled embedding（fp16，行序同
isolate_split）上重训每层 probe（协议同 G1：6 C × 5 fold GridSearchCV），
重复 G1 的核心几何量：cos(w,d_H7/H5)、offset 占比、亚型间/内方差比，
按相对深度（layer / (L_max)）与 150M 的 G1 结果对照。

判定：若三规模均呈现"cos(w,d_H7) 随相对深度变号 + offset 主导性随深度
增长"，则几何机制跨规模同构（支撑 RESULT.md §4"最优深度前移"的机制解读）。

输出：output/g5_scale_geometry.json、figs/g5_scale_geometry.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from validation_exp.depth_reversal.config import (
    SPLIT_CSV, LABELS, SCALE_NPY, RIDGE_C_VALUES, CV_FOLDS, RANDOM_SEED,
    LABEL_COLS, OUT_DIR,
)


def fit_probe(Xtr, ytr):
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    scl = StandardScaler().fit(Xtr)
    gs.fit(scl.transform(Xtr), ytr)
    return scl, gs.best_estimator_


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figs").mkdir(exist_ok=True)

    split = pd.read_csv(SPLIT_CSV)
    subtype = split["subtype"].to_numpy()
    sp = split["split"].to_numpy()
    m_train = (sp == "train")
    m_h5 = (sp == "h5_holdout")
    m_h7 = (sp == "h7_holdout")
    m_h13 = np.isin(subtype, ["H1", "H3"])
    ys = {c: np.load(LABELS[c]) for c in LABEL_COLS}

    results = {}
    for scale, path in SCALE_NPY.items():
        arr = np.load(path, mmap_mode="r")
        n_layers, n_rows, dim = arr.shape
        print(f"[{scale}] shape=({n_layers},{n_rows},{dim})", flush=True)
        assert n_rows == len(split)
        scale_res = {c: [] for c in LABEL_COLS}
        w_store = {c: [] for c in LABEL_COLS}
        for li in range(n_layers):
            X = np.asarray(arr[li], dtype=np.float32)
            for label in LABEL_COLS:
                y = ys[label]
                scl, clf = fit_probe(X[m_train], y[m_train])
                Xs = (X - scl.mean_) / scl.scale_
                w = clf.coef_[0].astype(np.float32)
                z = Xs @ w + float(clf.intercept_[0])
                auc_h5 = float(roc_auc_score(y[m_h5], z[m_h5]))
                auc_h7 = float(roc_auc_score(y[m_h7], z[m_h7]))

                ctr_tr = Xs[m_train].mean(0)
                ctr_h5 = Xs[m_h5].mean(0)
                ctr_h7 = Xs[m_h7].mean(0)
                w_hat = unit(w)
                cos_h5 = float(w_hat @ unit(ctr_h5 - ctr_tr))
                cos_h7 = float(w_hat @ unit(ctr_h7 - ctr_tr))
                # 亚型内 ± gap（AUC 的决定量，平移不变性下 offset 无关）
                gap_h7 = float(w @ (Xs[m_h7 & (y == 1)].mean(0)
                                    - Xs[m_h7 & (y == 0)].mean(0)))
                gap_h5 = float(w @ (Xs[m_h5 & (y == 1)].mean(0)
                                    - Xs[m_h5 & (y == 0)].mean(0)))
                gap_tr = float(w @ (Xs[m_train & (y == 1)].mean(0)
                                    - Xs[m_train & (y == 0)].mean(0)))
                # offset 主导性：|offset| / (|offset| + within 离散)
                off_h7 = float(w @ (ctr_h7 - ctr_tr))
                within_h7 = float(np.std((Xs[m_h7] - ctr_h7) @ w))
                # 亚型间/内方差比
                mu_all = Xs.mean(0)
                between, within = 0.0, 0.0
                for st in ["H1", "H3", "H5", "H7"]:
                    mst = (subtype == st) if st in ("H5", "H7") \
                        else (m_h13 & (subtype == st))
                    Xst = Xs[mst]
                    mu_st = Xst.mean(0)
                    between += float(((mu_st - mu_all) ** 2).sum())
                    within += float(((Xst - mu_st) ** 2).sum(1).mean())
                scale_res[label].append({
                    "layer": li, "rel_depth": li / (n_layers - 1),
                    "auc_h5": auc_h5, "auc_h7": auc_h7,
                    "cos_h5": cos_h5, "cos_h7": cos_h7,
                    "gap_h7": gap_h7, "gap_h5": gap_h5, "gap_train": gap_tr,
                    "offset_h7": off_h7, "within_h7_sd": within_h7,
                    "var_ratio_bw": between / (within + 1e-12),
                })
                w_store[label].append(w_hat)
            print(f"[{scale}][L{li:02d}] done", flush=True)

        # 跨层相关
        for label in LABEL_COLS:
            df = pd.DataFrame(scale_res[label])
            rho_cos, p_cos = spearmanr(df["rel_depth"], df["cos_h7"])
            rho_vr, p_vr = spearmanr(df["rel_depth"], df["var_ratio_bw"])
            rho_auc, p_auc = spearmanr(df["cos_h7"], df["auc_h7"])
            rho_gap, p_gap = spearmanr(df["gap_h7"], df["auc_h7"])
            rho_gap57, p_gap57 = spearmanr(df["gap_h5"], df["gap_h7"])
            sign_cos = np.sign(df["cos_h7"].to_numpy())
            flips = [int(df["layer"][i]) for i in range(1, len(sign_cos))
                     if sign_cos[i] != sign_cos[i - 1]]
            sign_gap = np.sign(df["gap_h7"].to_numpy())
            gap_flips = [int(df["layer"][i]) for i in range(1, len(sign_gap))
                         if sign_gap[i] != sign_gap[i - 1]]
            # 每层 w 与末层 w 的余弦（解随深度的旋转）
            Ws = np.stack(w_store[label])
            cos_last = (Ws @ Ws[-1]).tolist()
            # argmax-H7AUC 层与末层的 w 余弦
            best_li = int(df["auc_h7"].idxmax())
            cos_best_last = float(cos_last[best_li])
            results.setdefault(scale, {})[label] = {
                "per_layer": scale_res[label],
                "rho_reldepth_cosH7": float(rho_cos), "p": float(p_cos),
                "rho_reldepth_varratio": float(rho_vr), "p_vr": float(p_vr),
                "rho_cosH7_aucH7": float(rho_auc), "p_auc": float(p_auc),
                "rho_gapH7_aucH7": float(rho_gap), "p_gap": float(p_gap),
                "rho_gapH5_gapH7": float(rho_gap57), "p_gap57": float(p_gap57),
                "cosH7_sign_flip_layers": flips,
                "gapH7_sign_flip_layers": gap_flips,
                "cos_w_layer_w_last": cos_last,
                "best_h7_layer": best_li,
                "cos_w_bestlayer_w_last": cos_best_last,
            }
        results[scale]["n_layers"] = int(n_layers)

    # 对照 150M（G1 结果）
    g1 = json.load(open(OUT_DIR / "g1_layer_geometry.json"))
    cmp150 = {}
    for label in LABEL_COLS:
        df = pd.DataFrame(g1["per_layer"][label])
        gap7 = df["decomp"].apply(lambda d: d["H7"]["within+"]
                                  - d["H7"]["within-"])
        gap5 = df["decomp"].apply(lambda d: d["H5"]["within+"]
                                  - d["H5"]["within-"])
        cmp150[label] = {
            "rel_depth": (df["layer"] / 30).tolist(),
            "cos_h7": df["cos_h7"].tolist(),
            "auc_h7": df["auc_h7"].tolist(),
            "gap_h7": gap7.tolist(),
            "var_ratio_bw": df["var_ratio_bw"].tolist(),
            "rho_reldepth_cosH7": float(spearmanr(df["layer"] / 30,
                                                  df["cos_h7"])[0]),
            "rho_cosH7_aucH7": float(spearmanr(df["cos_h7"], df["auc_h7"])[0]),
            "rho_gapH7_aucH7": float(spearmanr(gap7, df["auc_h7"])[0]),
            "rho_gapH5_gapH7": float(spearmanr(gap5, gap7)[0]),
        }
    results["150m_ref"] = cmp150

    # 图：三规模 H7 亚型内 gap（对 train gap 归一）vs 相对深度
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    g1full = json.load(open(OUT_DIR / "g1_layer_geometry.json"))
    for ai, label in enumerate(LABEL_COLS):
        ax = axes[ai]
        d150 = pd.DataFrame(g1full["per_layer"][label])
        gap7 = d150["decomp"].apply(lambda d: d["H7"]["within+"]
                                    - d["H7"]["within-"])
        ax.plot(d150["layer"] / 30, gap7, "o-", label="150M", color="tab:blue")
        for scale, color in [("650m", "tab:orange"), ("3b", "tab:green")]:
            df = pd.DataFrame(results[scale][label]["per_layer"])
            ax.plot(df["rel_depth"], df["gap_h7"], "s-", label=scale,
                    color=color, alpha=0.8)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("relative depth")
        ax.set_ylabel("within-subtype gap H7 (logit)")
        ax.set_title(f"H7 within-gap vs depth - {label}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figs/g5_scale_geometry.png", dpi=300)
    plt.close(fig)

    with open(OUT_DIR / "g5_scale_geometry.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"→ {OUT_DIR}/g5_scale_geometry.json + figs/")


if __name__ == "__main__":
    main()
