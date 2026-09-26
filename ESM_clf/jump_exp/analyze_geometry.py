"""H7 方向反转的几何分析

诊断问题：H7 反转是 embedding 空间的域偏移伪影，还是谱系构成问题？

分析内容（对每个 label × embedding 组合）：
  1. cos(w, d)：probe 权重方向 w 与各亚型质心偏移方向 d 的余弦。
     若 H7 的 cos 强负 → H7 整体落在训练负半空间，反转是域偏移。
  2. H7 逐 cluster 的 w 投影均值 vs 标签：阳性 cluster 是否系统性位于负侧。
  3. PCA 可视化（按亚型着色、阳性高亮），图存 output/figs/。
  4. H7 阳性 cluster 构成（strain_name / 年份），确认是否单一暴发事件。

注意：subtype/cluster_id 用 isolate_split.csv（与 embedding/labels 行序一致），
不用重建后的 aligned CSV（行数与亚型判定已漂移）。

输出：output/geometry_analysis.json + output/figs/pca_*.png
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

FIG_DIR = OUT_DIR / "figs"
SUBTYPES = ["H1", "H3", "H5", "H7"]
EMB_TYPES = ["L1", "L3", "L1L3"]


def train_w(X, y, train_mask):
    """在 H1+H3 train 上训 probe，返回 (scaler, w, best_C)"""
    scl = StandardScaler()
    Xt = scl.fit_transform(X[train_mask])
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                      scoring="roc_auc")
    gs.fit(Xt, y[train_mask])
    return scl, gs.best_estimator_.coef_[0], gs.best_params_["C"]


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    split_df = pd.read_csv(SPLIT_CSV)
    subtypes = split_df["subtype"].values
    clusters = split_df["cluster_id"].values
    train_mask = (split_df["split"] == "train").values
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # accession → strain_name / 年份（用于 H7 阳性构成；名称字段不受亚型重判影响）
    data_df = pd.read_csv(DATA_CSV)
    info = data_df.set_index("accession")[["strain_name", "collection_year", "serotype"]]
    acc = split_df["accession"].values

    results = {}
    for label_col in LABEL_COLS:
        y = np.load(OUT_DIR / f"labels_{label_col}.npy")
        print(f"\n{'='*72}\n  Label: {label_col}\n{'='*72}")
        label_res = {}

        for emb_name in EMB_TYPES:
            X = np.load(OUT_DIR / f"esm_emb_150M_{emb_name}.npy")
            scl, w, best_c = train_w(X, y, train_mask)
            X_std = scl.transform(X)

            # ── 1. cos(w, d)：各亚型质心偏移 vs probe 方向 ──
            mu_train = X_std[train_mask].mean(axis=0)
            cos_res = {}
            for st in SUBTYPES:
                m = subtypes == st
                d = X_std[m].mean(axis=0) - mu_train
                # 同时给亚型内正/负质心各自相对 w 的投影
                proj_pos = float((X_std[m & (y == 1)].mean(axis=0) - mu_train) @ w / np.linalg.norm(w)) if (m & (y == 1)).any() else None
                proj_neg = float((X_std[m & (y == 0)].mean(axis=0) - mu_train) @ w / np.linalg.norm(w)) if (m & (y == 0)).any() else None
                cos_res[st] = {"cos_w_d": cosine(w, d),
                               "proj_pos": proj_pos, "proj_neg": proj_neg}
            print(f"\n  [{emb_name}] best_C={best_c}")
            for st in SUBTYPES:
                r = cos_res[st]
                pp = f"{r['proj_pos']:+.3f}" if r['proj_pos'] is not None else "N/A"
                pn = f"{r['proj_neg']:+.3f}" if r['proj_neg'] is not None else "N/A"
                print(f"    {st}: cos(w,d)={r['cos_w_d']:+.3f}  proj(+)={pp}  proj(-)={pn}")

            # ── 2. H7 逐 cluster 投影 ──
            h7 = subtypes == "H7"
            proj_all = X_std @ w / np.linalg.norm(w)
            cl_rows = []
            for cid in np.unique(clusters[h7]):
                cm = h7 & (clusters == cid)
                cl_rows.append({"cluster_id": int(cid), "n": int(cm.sum()),
                                "label": int(y[cm].max()),
                                "proj_mean": float(proj_all[cm].mean())})
            cl_df = pd.DataFrame(cl_rows).sort_values("proj_mean")
            n_pos = cl_df[cl_df["label"] == 1]
            n_neg = cl_df[cl_df["label"] == 0]
            print(f"    H7 clusters: {len(cl_df)} (阳性 {len(n_pos)})")
            print(f"    阳性 cluster 投影均值: {n_pos['proj_mean'].mean():+.3f} | "
                  f"阴性: {n_neg['proj_mean'].mean():+.3f}")
            print(f"    阳性 cluster 中投影<0 的比例: {(n_pos['proj_mean'] < 0).mean()*100:.0f}%")

            label_res[emb_name] = {"best_C": best_c, "cos": cos_res,
                                   "h7_cluster_proj": cl_rows}

            # ── 3. PCA 图（只对 L3 出图，主 embedding）──
            if emb_name == "L3":
                pca = PCA(n_components=2, random_state=RANDOM_SEED)
                pca.fit(X_std[train_mask])
                Z = pca.transform(X_std)
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = {"H1": "#4C72B0", "H3": "#55A868",
                          "H5": "#DD8452", "H7": "#C44E52"}
                for st in SUBTYPES:
                    m = subtypes == st
                    ax.scatter(Z[m & (y == 0), 0], Z[m & (y == 0), 1],
                               s=4, alpha=0.25, c=colors[st], label=f"{st} neg")
                    ax.scatter(Z[m & (y == 1), 0], Z[m & (y == 1), 1],
                               s=14, alpha=0.9, c=colors[st], marker="^",
                               edgecolors="k", linewidths=0.3, label=f"{st} pos")
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
                ax.set_title(f"ESM-2 L3 PCA — {label_col}")
                ax.legend(markerscale=2, fontsize=7, ncol=2)
                fig.tight_layout()
                fig_path = FIG_DIR / f"pca_L3_{label_col}.png"
                fig.savefig(fig_path, dpi=300)
                plt.close(fig)
                print(f"    图已保存: {fig_path}")

        # ── 4. H7 阳性 cluster 构成 ──
        h7 = subtypes == "H7"
        pos_clusters = sorted(set(clusters[h7 & (y == 1)]))
        comp = []
        print(f"\n  H7 阳性 cluster 构成 ({label_col}):")
        for cid in pos_clusters:
            cm = h7 & (clusters == cid)
            names = info.loc[acc[cm], "strain_name"].astype(str)
            years = info.loc[acc[cm], "collection_year"].astype(str)
            seros = info.loc[acc[cm], "serotype"].astype(str)
            comp.append({"cluster_id": int(cid), "n": int(cm.sum()),
                         "serotypes": sorted(set(seros))[:3],
                         "year_range": f"{years.min()}–{years.max()}",
                         "example": names.iloc[0]})
            print(f"    cluster {cid}: n={cm.sum()}, {sorted(set(seros))[:3]}, "
                  f"{years.min()}–{years.max()}, 例: {names.iloc[0]}")
        label_res["h7_positive_clusters"] = comp
        results[label_col] = label_res

    out_path = OUT_DIR / "geometry_analysis.json"
    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
