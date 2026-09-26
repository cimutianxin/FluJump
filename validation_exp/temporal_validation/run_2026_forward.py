#!/usr/bin/env python3
"""2026 前向外推排序验证

probe（H1+H3 train，止于 2025）给 201 条 2026 新分离株打分，检验天然阳性
（cluster 70 的 9 条 H5N1 暴发簇前向样本 + cluster 15 的 1 条 H1 猪源）是否排前。
同协议对比朴素基线（max identity to H1+H3 train 人源参考，§5 与 §1 的正面对比）。

输出：output/forward_2026.json + output/forward_2026_ranking.csv
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
from validation_exp.temporal_validation.config import (
    CLEAN_2026_CSV, ALIGNED_CSV, SPLIT_CSV, MEAN_EMB_L3, LABEL_COLS,
    RIDGE_C_VALUES, CV_FOLDS, RANDOM_SEED, NAIVE_REF_HOST, OUT_DIR,
)

GAP = ord("-")
ALIGNED_2026_CSV = "data/processed_isolate_MAFFT/2026_isolates_aligned.csv"


def encode(seqs):
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    return arr.reshape(len(seqs), -1)


def max_identity(Q, R, chunk_q=256, chunk_r=2048):
    """query × ref 的最大双非gap位 identity"""
    out = np.empty(len(Q), dtype=np.float32)
    for i0 in range(0, len(Q), chunk_q):
        q = Q[i0:i0 + chunk_q]
        best = np.zeros(len(q), dtype=np.float32)
        for j0 in range(0, len(R), chunk_r):
            r = R[j0:j0 + chunk_r]
            both = (q[:, None, :] != GAP) & (r[None, :, :] != GAP)
            m = ((q[:, None, :] == r[None, :, :]) & both).sum(2) / np.maximum(both.sum(2), 1)
            best = np.maximum(best, m.max(1))
        out[i0:i0 + len(q)] = best
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 训练集（isolate_split 行序 = embedding 行序）──
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=[
        "accession", "subtype", "aligned_ha_seq", "host_category",
        "label_is_jump", "label_is_jump_human"])
    df = split.merge(ali, on=["accession", "subtype"], validate="one_to_one")
    X = np.load(MEAN_EMB_L3).astype(np.float32)
    assert len(X) == len(df)
    train_mask = (df["split"] == "train").to_numpy()

    # ── 2026 数据 ──
    c26 = pd.read_csv(CLEAN_2026_CSV)
    a26 = pd.read_csv(ALIGNED_2026_CSV, usecols=["accession", "subtype", "aligned_ha_seq"])
    c26 = c26.merge(a26, on=["accession", "subtype"], validate="one_to_one")
    E26 = np.load(OUT_DIR / "emb_2026_L3.npy")
    assert len(E26) == len(c26) == 201
    y26 = {l: c26[l].to_numpy() for l in LABEL_COLS}

    # ── 朴素基线打分（参考集：H1+H3 train 人源）──
    ref = df[train_mask & (df["host_category"] == NAIVE_REF_HOST)]
    R = encode(ref["aligned_ha_seq"].tolist())
    Q26 = encode(c26["aligned_ha_seq"].tolist())
    naive26 = max_identity(Q26, R)
    print(f"朴素基线参考集: {len(R)} 条；2026 max identity 中位 {np.median(naive26):.3f}")

    results = {"n_2026": 201, "n_positive_jump_human": int(y26["label_is_jump_human"].sum()),
               "probe": {}, "naive": {}}
    ranking_frames = []

    for label in LABEL_COLS:
        y = df[label].to_numpy()
        scaler = StandardScaler().fit(X[train_mask])
        gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                          cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                             random_state=RANDOM_SEED),
                          scoring="roc_auc", n_jobs=-1)
        gs.fit(scaler.transform(X[train_mask]), y[train_mask])
        clf = gs.best_estimator_
        h5 = (df["split"] == "h5_holdout").to_numpy()
        auc_h5 = roc_auc_score(y[h5], clf.decision_function(scaler.transform(X[h5])))
        assert 0.65 < auc_h5 < 0.95, f"probe 复现异常: H5 AUC={auc_h5}"
        print(f"[{label}] best C={gs.best_params_['C']}, H5 复现 AUC={auc_h5:.4f}")

        logit26 = clf.decision_function(scaler.transform(E26))
        auc = roc_auc_score(y26[label], logit26) if y26[label].sum() > 0 else None
        # cluster 级
        c26["_logit"] = logit26
        cl = c26.groupby("cluster_id").agg(
            score=("_logit", "mean"), label=(label, "max"), n=(label, "size"))
        cl_auc = roc_auc_score(cl["label"], cl["score"]) if cl["label"].sum() > 0 else None
        # 朴素基线 AUC
        naive_auc = roc_auc_score(y26[label], naive26) if y26[label].sum() > 0 else None

        # 阳性排名
        order = np.argsort(-logit26)
        ranks = np.empty(201, dtype=int)
        ranks[order] = np.arange(1, 202)
        pos_ranks = sorted(int(ranks[i]) for i in np.where(y26[label] == 1)[0])
        results["probe"][label] = {
            "h5_repro_auc": round(float(auc_h5), 4),
            "auc_2026": round(float(auc), 4) if auc else None,
            "auc_2026_cluster": round(float(cl_auc), 4) if cl_auc else None,
            "positive_ranks_of_201": pos_ranks,
        }
        results["naive"][label] = {
            "auc_2026": round(float(naive_auc), 4) if naive_auc else None}
        print(f"  2026 AUC: probe={auc:.4f} (cluster级 {cl_auc:.4f}) vs 朴素基线={naive_auc:.4f}"
              if auc else f"  2026 无阳性（{label}）")
        print(f"  阳性排名: {pos_ranks}")

        if label == "label_is_jump_human":
            c26["_naive"] = naive26
            c26["_rank"] = ranks
            ranking_frames = c26[["accession", "subtype", "strain_name", "host_category",
                                  "country", "collection_date", "cluster_id",
                                  "label_is_jump_human", "_logit", "_naive", "_rank"]]

    ranking_frames.sort_values("_rank").to_csv(
        OUT_DIR / "forward_2026_ranking.csv", index=False)
    with open(OUT_DIR / "forward_2026.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n→ {OUT_DIR / 'forward_2026.json'} / forward_2026_ranking.csv")


if __name__ == "__main__":
    main()
