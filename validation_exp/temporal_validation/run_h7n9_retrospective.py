#!/usr/bin/env python3
"""H7N9 回溯排序验证：pre-2013 训练 → 2013+ H7 暴发簇排序

训练：H1+H3 且 0 < collection_year < 2013（剔除 year=-1），label_is_jump
（jump_human 因 H3 pre-2013 仅 3 阳性不可用——日志需注明此口径差异）。
评估：H7 且 collection_year >= 2013（301 行、59 cluster，7 个 H7N9 阳性簇）。
原始与翻转（-logit）方向都报告（H7 Group 2 反转是 §3 已知现象）。
对照：pre-2013 朴素基线（max identity to pre-2013 H1+H3 人源）。

输出：output/retro_h7n9.json + output/retro_h7n9_cluster_ranking.csv
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
    ALIGNED_CSV, SPLIT_CSV, MEAN_EMB_L3, RIDGE_C_VALUES, CV_FOLDS,
    RANDOM_SEED, RETRO_TRAIN_YEAR_MAX, RETRO_EVAL_YEAR_MIN,
    H7N9_POS_CLUSTERS, OUT_DIR,
)
from validation_exp.temporal_validation.run_2026_forward import encode, max_identity


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=[
        "accession", "subtype", "aligned_ha_seq", "host_category",
        "collection_year", "label_is_jump", "label_is_jump_human"])
    df = split.merge(ali, on=["accession", "subtype"], validate="one_to_one")
    df["year"] = pd.to_numeric(df["collection_year"], errors="coerce").fillna(-1)
    X = np.load(MEAN_EMB_L3).astype(np.float32)
    assert len(X) == len(df)

    # ── 训练集：H1+H3，0 < year < 2013 ──
    tr = (df["subtype"].isin(["H1", "H3"])
          & (df["year"] > 0) & (df["year"] < RETRO_TRAIN_YEAR_MAX)).to_numpy()
    y = df["label_is_jump"].to_numpy()
    print(f"训练集: {tr.sum()} 行，阳性 {y[tr].sum()}（{y[tr].mean():.1%}）")

    # ── 评估集：H7，year >= 2013 ──
    ev = (df["subtype"] == "H7") & (df["year"] >= RETRO_EVAL_YEAR_MIN)
    ev_df = df[ev].copy()
    y_ev = ev_df["cluster_id"].isin(H7N9_POS_CLUSTERS).astype(int).to_numpy()
    print(f"评估集: {ev.sum()} 行，{ev_df['cluster_id'].nunique()} cluster，"
          f"阳性 {y_ev.sum()} 行 / {ev_df.loc[y_ev == 1, 'cluster_id'].nunique()} 簇")

    # ── 训练 probe ──
    scaler = StandardScaler().fit(X[tr])
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(scaler.transform(X[tr]), y[tr])
    clf = gs.best_estimator_
    print(f"best C={gs.best_params_['C']}")

    logits = clf.decision_function(scaler.transform(X[ev.to_numpy()]))
    # ── 朴素基线：pre-2013 H1+H3 人源参考 ──
    ref = df[tr & (df["host_category"] == "human")]
    naive = max_identity(encode(ev_df["aligned_ha_seq"].tolist()),
                         encode(ref["aligned_ha_seq"].tolist()))
    print(f"朴素基线参考集: {len(ref)} 条")

    # ── 评估（isolate 级 + cluster 级，双向）──
    ev_df["_logit"] = logits
    ev_df["_naive"] = naive
    ev_df["_pos"] = y_ev
    cl = ev_df.groupby("cluster_id").agg(
        logit=("_logit", "mean"), naive=("_naive", "mean"),
        pos=("_pos", "max"), n=("_pos", "size"),
        year_min=("year", "min")).sort_values("logit", ascending=False)

    def both_dir_auc(yv, sv):
        a = roc_auc_score(yv, sv)
        return round(float(a), 4), round(float(roc_auc_score(yv, -sv)), 4)

    results = {
        "train": {"n": int(tr.sum()), "n_pos": int(y[tr].sum()),
                  "label": "label_is_jump（H3 pre-2013 jump_human 仅 3 阳性，不可用）"},
        "eval": {"n_isolates": int(ev.sum()), "n_clusters": int(cl.shape[0]),
                 "n_pos_clusters": int(cl["pos"].sum())},
        "probe": {
            "isolate_auc": dict(zip(["raw", "flipped"], both_dir_auc(y_ev, logits))),
            "cluster_auc": dict(zip(["raw", "flipped"],
                                    both_dir_auc(cl["pos"].to_numpy(),
                                                 cl["logit"].to_numpy()))),
        },
        "naive": {
            "isolate_auc": dict(zip(["raw", "flipped"], both_dir_auc(y_ev, naive))),
            "cluster_auc": dict(zip(["raw", "flipped"],
                                    both_dir_auc(cl["pos"].to_numpy(),
                                                 cl["naive"].to_numpy()))),
        },
        "pos_cluster_ranks": {},
    }
    # 阳性簇名次（原始方向 & 翻转方向）
    for direction, col in [("raw", "logit"), ("flipped", None)]:
        cc = cl.sort_values("logit", ascending=(direction == "flipped"))
        ranks = {int(cid): int(i + 1) for i, cid in enumerate(cc.index)}
        results["pos_cluster_ranks"][direction] = {
            int(cid): ranks[cid] for cid in sorted(H7N9_POS_CLUSTERS)}

    cl.to_csv(OUT_DIR / "retro_h7n9_cluster_ranking.csv")
    with open(OUT_DIR / "retro_h7n9.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== 结果 ===")
    print(f"probe   isolate AUC raw/翻转: {results['probe']['isolate_auc']}, "
          f"cluster AUC: {results['probe']['cluster_auc']}")
    print(f"朴素基线 isolate AUC raw/翻转: {results['naive']['isolate_auc']}, "
          f"cluster AUC: {results['naive']['cluster_auc']}")
    print(f"7 个 H7N9 阳性簇名次（raw）: {results['pos_cluster_ranks']['raw']}")
    print(f"7 个 H7N9 阳性簇名次（翻转）: {results['pos_cluster_ranks']['flipped']}")
    print(f"→ {OUT_DIR / 'retro_h7n9.json'}")


if __name__ == "__main__":
    main()
