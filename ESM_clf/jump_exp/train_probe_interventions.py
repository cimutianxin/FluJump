"""Phase 1 干预实验：亚型方向擦除 + cluster 加权

基于几何分析的结论（H7 整体落在训练负半空间深处，反转源于
宿主/亚型特征混淆），在冻结 embedding 上测试两类修复：

  1. subtype erasure（INLP 式迭代线性擦除）：
     在训练集（H1+H3）embedding 上迭代训练亚型线性分类器，
     将 embedding 投影到其权重方向的正交补，消除亚型身份信息后重训 probe。
  2. cluster 加权：sample_weight = 1/cluster_size，削弱大谱系主导。

条件组合：baseline / erasure / cluster-weight / erasure+cluster-weight
成功标准：H7 原始方向 AUC > 0.5，且 H5 / test AUC 不明显下降。

注意：subtype/cluster_id 用 isolate_split.csv（与 embedding 行序一致）。

输出：output/interventions.json
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

EMB_TYPES = ["L1", "L3", "L1L3"]
ERASURE_ITERS = [1, 3, 5]  # INLP 迭代次数扫描


def load_data():
    split_df = pd.read_csv(SPLIT_CSV)
    subtypes = split_df["subtype"].values
    clusters = split_df["cluster_id"].values
    splits = split_df["split"].values
    emb = {e: np.load(OUT_DIR / f"esm_emb_150M_{e}.npy") for e in EMB_TYPES}
    labels = {lc: np.load(OUT_DIR / f"labels_{lc}.npy") for lc in LABEL_COLS}
    return subtypes, clusters, splits, emb, labels


def inlp_erase(X_train, X_all, subtype_train, n_iter):
    """INLP：迭代训练亚型分类器并投影掉其方向，返回擦除后的全部 embedding"""
    Ztr, Zall = X_train.copy(), X_all.copy()
    for _ in range(n_iter):
        clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                                 random_state=RANDOM_SEED)
        clf.fit(Ztr, subtype_train)
        w = clf.coef_[0]
        w = w / (np.linalg.norm(w) + 1e-12)
        Ztr = Ztr - np.outer(Ztr @ w, w)
        Zall = Zall - np.outer(Zall @ w, w)
    return Ztr, Zall


def fit_probe(Xt, yt, sample_weight=None):
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(Xt_s, yt, sample_weight=sample_weight)
    return scl, gs


def evaluate(X, y, scl, model, masks):
    """返回各 split 的 AUC（logits 原始方向）"""
    out = {}
    for name, m in masks.items():
        if not m.any() or len(np.unique(y[m])) < 2:
            out[name] = None
            continue
        logits = model.decision_function(scl.transform(X[m]))
        out[name] = float(roc_auc_score(y[m], logits))
    return out


def main():
    subtypes, clusters, splits, emb, labels = load_data()
    train_mask = splits == "train"
    eval_masks = {"test": splits == "test",
                  "h5": splits == "h5_holdout",
                  "h7": splits == "h7_holdout"}

    # cluster 权重：1 / cluster_size（在 train 内）
    cl_size = pd.Series(clusters).map(pd.Series(clusters).value_counts()).values
    w_cluster = (1.0 / cl_size)[train_mask]

    # 训练集亚型标签（erasure 用，仅 H1/H3）
    subtype_train = subtypes[train_mask]

    results = {}
    for label_col in LABEL_COLS:
        y = labels[label_col]
        print(f"\n{'='*72}\n  Label: {label_col}\n{'='*72}")
        label_res = {}
        for emb_name in EMB_TYPES:
            X = emb[emb_name]
            Xt = X[train_mask]
            yt = y[train_mask]
            emb_res = {}

            # ── baseline ──
            scl, gs = fit_probe(Xt, yt)
            emb_res["baseline"] = evaluate(X, y, scl, gs, eval_masks)
            r = emb_res["baseline"]
            print(f"  [{emb_name:>5}] baseline          test={r['test']:.3f} "
                  f"h5={r['h5']:.3f} h7={r['h7']:.3f}")

            # ── cluster 加权 ──
            scl, gs = fit_probe(Xt, yt, sample_weight=w_cluster)
            emb_res["cluster_weight"] = evaluate(X, y, scl, gs, eval_masks)
            r = emb_res["cluster_weight"]
            print(f"  [{emb_name:>5}] cluster-weight   test={r['test']:.3f} "
                  f"h5={r['h5']:.3f} h7={r['h7']:.3f}")

            # ── erasure（迭代次数扫描）──
            for it in ERASURE_ITERS:
                Ztr, Zall = inlp_erase(Xt, X, subtype_train, it)
                scl, gs = fit_probe(Ztr, yt)
                emb_res[f"erasure_x{it}"] = evaluate(Zall, y, scl, gs, eval_masks)
                r = emb_res[f"erasure_x{it}"]
                print(f"  [{emb_name:>5}] erasure x{it}       test={r['test']:.3f} "
                      f"h5={r['h5']:.3f} h7={r['h7']:.3f}")

            # ── erasure + cluster 加权（用中间迭代次数）──
            it = ERASURE_ITERS[1]
            Ztr, Zall = inlp_erase(Xt, X, subtype_train, it)
            scl, gs = fit_probe(Ztr, yt, sample_weight=w_cluster)
            emb_res[f"erasure_x{it}+cw"] = evaluate(Zall, y, scl, gs, eval_masks)
            r = emb_res[f"erasure_x{it}+cw"]
            print(f"  [{emb_name:>5}] erasure+cw x{it}   test={r['test']:.3f} "
                  f"h5={r['h5']:.3f} h7={r['h7']:.3f}")

            label_res[emb_name] = emb_res
        results[label_col] = label_res

    out_path = OUT_DIR / "interventions.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
