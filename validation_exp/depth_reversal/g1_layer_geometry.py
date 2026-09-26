#!/usr/bin/env python3
"""G1：层几何分析——w 与亚型偏移方向的夹角沿深度变化（H1/H2/H3）

31 层 × 双标签重训 Ridge probe（P0 回归断言，协议同 reversal_sites/g2），
probe 参数存 output/g1_probes.npz 供 G2/G3 复用。

每层在标准化空间计算：
  - test/H5/H7 raw-logit AUC
  - cos(w, d_ST)，d_ST = μ'_ST − μ'_train（ST ∈ {H5,H7}）；cos(w, d_H1vH3)
  - 亚型间/内方差比（4 亚型全行）
  - logit 分解：logit(x) − w·μ'_train = offset[w·(μ'_ST−μ'_train)] + within[w·(x−μ'_ST)]
  - H2 方向竞争：正交基 {a_H1, a_H3, b}（a_ST=亚型内 ± 质心差，b=μ'_H1−μ'_H3），
    share_b = (ŵ·q3)²；亚型轴单变量 train AUC
跨层检验（BH 校正）：Spearman(cos_H7, AUC_H7)、(share_b, AUC_H7/H5)、(share_b, layer) 等。
shuffled null：每层 20 次打乱标签单 fit（固定 C），给 share_b-ρ 零分布。
关键层（13/17/28/30）5 seed 稳定性。

输出：output/g1_probes.npz、output/g1_layer_geometry.json、figs/g1_geometry_{label}.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from validation_exp.depth_reversal.config import (
    SPLIT_CSV, ALL_LAYERS_NPY, LABELS, RIDGE_C_VALUES, CV_FOLDS, RANDOM_SEED,
    LABEL_COLS, REPRO_REF, REPRO_TOL, N_SHUFFLE_NULL, N_LAYERS, OUT_DIR,
)

SEED_STAB_LAYERS = [13, 17, 28, 30]
SEED_STAB_SEEDS = [42, 123, 456, 7, 2024]


def fit_probe(Xtr, ytr, seed=RANDOM_SEED):
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed),
                      scoring="roc_auc", n_jobs=-1)
    scl = StandardScaler().fit(Xtr)
    gs.fit(scl.transform(Xtr), ytr)
    return scl, gs.best_estimator_, float(gs.best_params_["C"])


def bh_adjust(pvals):
    """Benjamini-Hochberg 校正，返回 q 值数组"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[order[i]] * n / (i + 1))
        q[order[i]] = prev
    return q


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figs").mkdir(exist_ok=True)

    split = pd.read_csv(SPLIT_CSV)
    subtype = split["subtype"].to_numpy()
    sp = split["split"].to_numpy()
    m_train = (sp == "train")
    m_test = (sp == "test")
    m_h5 = (sp == "h5_holdout")
    m_h7 = (sp == "h7_holdout")
    m_h1tr = m_train & (subtype == "H1")
    m_h3tr = m_train & (subtype == "H3")
    m_h13 = np.isin(subtype, ["H1", "H3"])          # 4 亚型方差比用
    arr = np.load(ALL_LAYERS_NPY, mmap_mode="r")
    ys = {c: np.load(LABELS[c]) for c in LABEL_COLS}

    probe_store = {}   # 供 npz 保存
    per_layer = {c: [] for c in LABEL_COLS}

    for li in range(N_LAYERS):
        X = np.asarray(arr[li], dtype=np.float32)
        for label in LABEL_COLS:
            y = ys[label]
            scl, clf, best_c = fit_probe(X[m_train], y[m_train])
            m, s = scl.mean_.astype(np.float32), scl.scale_.astype(np.float32)
            w = clf.coef_[0].astype(np.float32)
            b0 = float(clf.intercept_[0])
            Xs = (X - m) / s
            z = Xs @ w + b0

            auc_test = float(roc_auc_score(y[m_test], z[m_test]))
            auc_h5 = float(roc_auc_score(y[m_h5], z[m_h5]))
            auc_h7 = float(roc_auc_score(y[m_h7], z[m_h7]))

            # P0 回归断言（同 reversal_sites/g2 基准）
            if (li, label) in REPRO_REF:
                h5_ref, h7_ref = REPRO_REF[(li, label)]
                assert abs(auc_h5 - h5_ref) < REPRO_TOL, \
                    f"P0 失败 L{li} {label} H5: {auc_h5:.4f} vs {h5_ref}"
                assert abs(auc_h7 - h7_ref) < REPRO_TOL, \
                    f"P0 失败 L{li} {label} H7: {auc_h7:.4f} vs {h7_ref}"

            # ── 标准化空间质心 ──
            ctr = {
                "train": Xs[m_train].mean(0),
                "H1tr": Xs[m_h1tr].mean(0), "H3tr": Xs[m_h3tr].mean(0),
                "H1tr+": Xs[m_h1tr & (y == 1)].mean(0),
                "H1tr-": Xs[m_h1tr & (y == 0)].mean(0),
                "H3tr+": Xs[m_h3tr & (y == 1)].mean(0),
                "H3tr-": Xs[m_h3tr & (y == 0)].mean(0),
                "H5": Xs[m_h5].mean(0), "H5+": Xs[m_h5 & (y == 1)].mean(0),
                "H5-": Xs[m_h5 & (y == 0)].mean(0),
                "H7": Xs[m_h7].mean(0), "H7+": Xs[m_h7 & (y == 1)].mean(0),
                "H7-": Xs[m_h7 & (y == 0)].mean(0),
            }
            d_h5 = ctr["H5"] - ctr["train"]
            d_h7 = ctr["H7"] - ctr["train"]
            d_h1v3 = ctr["H1tr"] - ctr["H3tr"]
            w_hat = unit(w)
            cos_h5 = float(w_hat @ unit(d_h5))
            cos_h7 = float(w_hat @ unit(d_h7))
            cos_h1v3 = float(w_hat @ unit(d_h1v3))

            # ── 亚型间/内方差比（4 亚型全部行，标准化空间）──
            mu_all = Xs.mean(0)
            between = 0.0
            within = 0.0
            for st in ["H1", "H3", "H5", "H7"]:
                mst = (subtype == st) if st in ("H5", "H7") else (m_h13 & (subtype == st))
                Xst = Xs[mst]
                mu_st = Xst.mean(0)
                between += float(((mu_st - mu_all) ** 2).sum())
                within += float(((Xst - mu_st) ** 2).sum(1).mean())
            var_ratio = between / 4.0 / (within / 4.0 + 1e-12)

            # ── logit 分解（相对 train 质心；offset 对组内所有成员相同）──
            decomp = {}
            for st, mst in [("H5", m_h5), ("H7", m_h7)]:
                offset = float(w @ (ctr[st] - ctr["train"]))
                decomp[st] = {
                    "offset": offset,
                    "within+": float(w @ (ctr[f"{st}+"] - ctr[st])),
                    "within-": float(w @ (ctr[f"{st}-"] - ctr[st])),
                    "total+": float(w @ (ctr[f"{st}+"] - ctr["train"])),
                    "total-": float(w @ (ctr[f"{st}-"] - ctr["train"])),
                }

            # ── H2：方向竞争基 {a_H1, a_H3, b} ──
            a1 = ctr["H1tr+"] - ctr["H1tr-"]
            a2 = ctr["H3tr+"] - ctr["H3tr-"]
            A = np.column_stack([a1, a2, d_h1v3])
            Q, _ = np.linalg.qr(A)
            proj = Q.T @ w_hat
            share_a12 = float(proj[0] ** 2 + proj[1] ** 2)
            share_b = float(proj[2] ** 2)
            share_resid = float(max(0.0, 1.0 - (proj ** 2).sum()))
            # 亚型轴单变量标签信息（train，q3 = 宿主轴正交后的亚型身份方向）
            auc_axis_b = float(roc_auc_score(y[m_train], Xs[m_train] @ Q[:, 2]))
            auc_axis_a1 = float(roc_auc_score(y[m_train], Xs[m_train] @ Q[:, 0]))

            per_layer[label].append({
                "layer": li, "best_C": best_c,
                "auc_test": auc_test, "auc_h5": auc_h5, "auc_h7": auc_h7,
                "cos_h5": cos_h5, "cos_h7": cos_h7, "cos_h1v3": cos_h1v3,
                "var_ratio_bw": var_ratio,
                "decomp": decomp,
                "share_within_host": share_a12, "share_subtype": share_b,
                "share_resid": share_resid,
                "auc_axis_subtype": auc_axis_b, "auc_axis_host_h1": auc_axis_a1,
                "sign_w_hat_q3": float(np.sign(proj[2])),
            })
            probe_store[f"w_L{li}_{label}"] = w
            probe_store[f"mean_L{li}_{label}"] = m
            probe_store[f"scale_L{li}_{label}"] = s
            probe_store[f"intercept_L{li}_{label}"] = np.float32(b0)
        print(f"[L{li:02d}] 双标签 probe + 几何完成", flush=True)

    np.savez_compressed(OUT_DIR / "g1_probes.npz", **probe_store)

    # ══ 跨层检验（BH 校正）══
    tests = []
    for label in LABEL_COLS:
        df = pd.DataFrame(per_layer[label])
        for name, x, yv in [
            ("cosH7_vs_aucH7", df["cos_h7"], df["auc_h7"]),
            ("cosH5_vs_aucH5", df["cos_h5"], df["auc_h5"]),
            ("shareB_vs_aucH7", df["share_subtype"], df["auc_h7"]),
            ("shareB_vs_aucH5", df["share_subtype"], df["auc_h5"]),
            ("shareB_vs_layer", df["share_subtype"], df["layer"]),
            ("varRatio_vs_layer", df["var_ratio_bw"], df["layer"]),
            ("cosH7_vs_layer", df["cos_h7"], df["layer"]),
            ("cosH5_vs_layer", df["cos_h5"], df["layer"]),
        ]:
            rho, p = spearmanr(x, yv)
            tests.append({"label": label, "test": name,
                          "rho": float(rho), "p": float(p)})
        # cos 变号层 vs AUC 穿越 0.5 层
        sign_cos = np.sign(df["cos_h7"].to_numpy())
        flip_layers = [int(df["layer"][i]) for i in range(1, len(sign_cos))
                       if sign_cos[i] != sign_cos[i - 1]]
        cross = df[((df["auc_h7"] - 0.5).abs() < 0.06)]["layer"].tolist()
        tests.append({"label": label, "test": "cosH7_sign_flip_layers",
                      "layers": flip_layers, "aucH7_near_half_layers": cross})
    sig = [t for t in tests if "p" in t]
    qs = bh_adjust([t["p"] for t in sig])
    for t, q in zip(sig, qs):
        t["q_bh"] = float(q)

    # ══ shuffled null（share_b 的 ρ 零分布；固定 C 单 fit）══
    # worker 内自行 mmap 读层矩阵，避免大数组跨进程 pickling；
    # 基轴用真实标签定义（诊断乱拟合的 w 读到什么方向的亚型成分）
    ytr_map = {c: ys[c][m_train] for c in LABEL_COLS}
    mask_map = {c: {"h1p": m_h1tr & (ys[c] == 1), "h1m": m_h1tr & (ys[c] == 0),
                    "h3p": m_h3tr & (ys[c] == 1), "h3m": m_h3tr & (ys[c] == 0)}
                for c in LABEL_COLS}

    def null_fit(li, label, C, rep):
        X = np.asarray(np.load(ALL_LAYERS_NPY, mmap_mode="r")[li],
                       dtype=np.float32)
        ytr = ytr_map[label]
        r = np.random.default_rng(RANDOM_SEED * 1000 + rep)
        y_sh = r.permutation(ytr)
        scl = StandardScaler().fit(X[m_train])
        clf = LogisticRegression(max_iter=2000, C=C).fit(
            scl.transform(X[m_train]), y_sh)
        Xs = (X[m_train] - scl.mean_) / scl.scale_
        w_hat = unit(clf.coef_[0])
        mm = mask_map[label]
        # m_train 布尔索引后行序 = train 子集，需把全局 mask 换成 train 内位置
        g = np.where(m_train)[0]
        pos = {k: np.searchsorted(g, np.where(v)[0]) for k, v in mm.items()}
        a1 = Xs[pos["h1p"]].mean(0) - Xs[pos["h1m"]].mean(0)
        a2 = Xs[pos["h3p"]].mean(0) - Xs[pos["h3m"]].mean(0)
        b = Xs[np.searchsorted(g, np.where(m_h1tr)[0])].mean(0) \
            - Xs[np.searchsorted(g, np.where(m_h3tr)[0])].mean(0)
        Q, _ = np.linalg.qr(np.column_stack([a1, a2, b]))
        return li, label, rep, float((Q.T @ w_hat)[2] ** 2)

    null_res = Parallel(n_jobs=-1, verbose=5)(
        delayed(null_fit)(li, label, per_layer[label][li]["best_C"], rep)
        for li in range(N_LAYERS) for label in LABEL_COLS
        for rep in range(N_SHUFFLE_NULL))

    null_share = {(c, rep): {} for c in LABEL_COLS for rep in range(N_SHUFFLE_NULL)}
    for li, label, rep, sb in null_res:
        null_share[(label, rep)][li] = sb

    null_summary = {}
    for label in LABEL_COLS:
        real_rho = [t["rho"] for t in sig
                    if t["label"] == label and t["test"] == "shareB_vs_aucH7"][0]
        auc_h7 = pd.DataFrame(per_layer[label])["auc_h7"].to_numpy()
        rhos = []
        for rep in range(N_SHUFFLE_NULL):
            sb = np.array([null_share[(label, rep)][li] for li in range(N_LAYERS)])
            rhos.append(float(spearmanr(sb, auc_h7)[0]))
        rhos = np.array(rhos)
        null_summary[label] = {
            "real_rho": real_rho,
            "null_rho_mean": float(rhos.mean()), "null_rho_sd": float(rhos.std()),
            "null_rho_min": float(rhos.min()), "null_rho_max": float(rhos.max()),
            "p_onesided_le": float((rhos <= real_rho).mean()),
        }

    # ══ 关键层 5 seed 稳定性 ══
    seed_stab = {}
    for li in SEED_STAB_LAYERS:
        X = np.asarray(arr[li], dtype=np.float32)
        for label in LABEL_COLS:
            y = ys[label]
            ws, aucs5, aucs7 = [], [], []
            for sd in SEED_STAB_SEEDS:
                scl, clf, _ = fit_probe(X[m_train], y[m_train], seed=sd)
                z = ((X - scl.mean_) / scl.scale_) @ clf.coef_[0] + clf.intercept_[0]
                ws.append(unit(clf.coef_[0]))
                aucs5.append(float(roc_auc_score(y[m_h5], z[m_h5])))
                aucs7.append(float(roc_auc_score(y[m_h7], z[m_h7])))
            ws = np.stack(ws)
            cos_mat = ws @ ws.T
            iu = np.triu_indices(len(ws), 1)
            seed_stab[f"L{li}_{label}"] = {
                "w_cos_min": float(cos_mat[iu].min()),
                "w_cos_mean": float(cos_mat[iu].mean()),
                "auc_h5_mean": float(np.mean(aucs5)), "auc_h5_sd": float(np.std(aucs5)),
                "auc_h7_mean": float(np.mean(aucs7)), "auc_h7_sd": float(np.std(aucs7)),
            }

    # ══ 图 ══
    for label in LABEL_COLS:
        df = pd.DataFrame(per_layer[label])
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        ax1.plot(df["layer"], df["auc_h7"], "o-", color="tab:red", label="H7 AUC")
        ax1.plot(df["layer"], df["auc_h5"], "s-", color="tab:blue", label="H5 AUC")
        ax1.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax1.set_xlabel("layer")
        ax1.set_ylabel("holdout AUC (raw logit)")
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.plot(df["layer"], df["cos_h7"], "^--", color="tab:red", alpha=0.6,
                 label="cos(w,d_H7)")
        ax2.plot(df["layer"], df["cos_h5"], "v--", color="tab:blue", alpha=0.6,
                 label="cos(w,d_H5)")
        ax2.plot(df["layer"], df["share_subtype"], ".-", color="tab:green",
                 alpha=0.7, label="share_subtype")
        ax2.axhline(0, color="black", lw=0.6)
        ax2.set_ylabel("cos / share")
        ax2.legend(loc="upper right")
        plt.title(f"G1 层几何 — {label}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"figs/g1_geometry_{label}.png", dpi=300)
        plt.close(fig)

    out = {
        "per_layer": per_layer, "tests": tests,
        "null_share_subtype": null_summary, "seed_stability": seed_stab,
    }
    with open(OUT_DIR / "g1_layer_geometry.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"→ {OUT_DIR}/g1_layer_geometry.json + g1_probes.npz + figs/")


if __name__ == "__main__":
    main()
