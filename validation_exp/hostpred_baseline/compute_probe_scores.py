"""为 ΔAUC 配对比较重算 probe（ESM-2 150M ridge LR）分数，纯 CPU。

两部份：
1) H5/H7 holdout：target-val 选中层（cluster 臂 5 seeds，见
   ESM_clf/jump_exp/output/target_val_layer_select.json；jump/H5=L28×5、
   jump/H7=L17×5、jump_human/H5={L1,L11,L11,L28,L11}、jump_human/H7=L13×5），
   在 split==train 上按 jump_exp.fit_probe 协议重训，对全量 holdout 打分。
   标签用 ESM_clf/jump_exp/output/labels_*.npy（与 isolate_split 行序一致，
   开头断言与 aligned csv 当前标签一致）。
2) 前向三年：复刻 run_multiyear_forward.py / run_2026_forward.py 协议
   （2024/2025：train ≤cutoff + ≤cutoff 重算标签；2026：全量 train + headline
   标签；均 L3=arr[28]），对 eval 年份行打分；并与 ranking CSV 的既有列
   对账（max|Δlogit| 应 ~1e-5 内）。

产出 output/probe_scores_h5h7.csv（长表：…,label,seed,layer,logit）
     output/probe_scores_forward.csv（accession,subtype,year,label,logit）
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
    ALIGNED_CSV, SPLIT_CSV, CLEAN_2026_CSV, ALL_ISOLATES_CSV,
    LABEL_COLS, RIDGE_C_VALUES, CV_FOLDS, RANDOM_SEED,
)

ARR = "ESM_clf/jump_exp/output/esm_emb_150M_all_layers.npy"   # (31, 11060, 640)
LBL = "ESM_clf/jump_exp/output/labels_{}.npy"
TV_JSON = "ESM_clf/jump_exp/output/target_val_layer_select.json"
OUT = "validation_exp/hostpred_baseline/output"
EMB26 = "validation_exp/temporal_validation/output/emb_2026_L3.npy"
L3 = 28      # all_layers 中层索引（L3=倒数第三层=第 28 层，已对账）


def fit_probe(Xt, yt):
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(Xt_s, yt)
    return scl, gs.best_estimator_


def labels_upto(alliso, cutoff):
    yr = alliso["year"]
    m = yr.notna() & (yr > 0) & (yr <= cutoff) & (alliso["host_category"] != "unknown")
    hosts = alliso[m].groupby(["subtype", "cluster_id"])["host_category"].agg(set)
    return {k: {"label_is_jump": int(len(v) >= 2),
                "label_is_jump_human": int("human" in v and len(v) >= 2)}
            for k, v in hosts.items()}


def main():
    arr = np.load(ARR, mmap_mode="r")
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=[
        "accession", "subtype", "cluster_id", "collection_year",
        "label_is_jump", "label_is_jump_human"])
    df = split.merge(ali.drop(columns=["cluster_id"]),  # split 已带 cluster_id
                     on=["accession", "subtype"], validate="one_to_one")
    df["year"] = pd.to_numeric(df["collection_year"], errors="coerce")

    # 对账 1：labels_*.npy == 当前 aligned csv 标签（09-19 修复后口径）
    y = {}
    for l in LABEL_COLS:
        y[l] = np.load(LBL.format(l))
        assert (y[l] == df[l].to_numpy()).all(), f"{l} 标签 npy 与 csv 不一致"
    # 对账 2：all_layers[28] == MEAN_EMB_L3
    from validation_exp.temporal_validation.config import MEAN_EMB_L3
    assert np.allclose(np.asarray(arr[L3]), np.load(MEAN_EMB_L3), atol=1e-5)
    print("对账通过：标签 npy == csv；all_layers[28] == MEAN_EMB_L3")

    # ── H5/H7：target-val 选中层 × 5 seeds ──
    tv = json.load(open(TV_JSON))["per_file"]
    rows = []
    for label in LABEL_COLS:
        for st, split_name in [("h5", "h5_holdout"), ("h7", "h7_holdout")]:
            layers = [tv[f"cluster_seed{s}.csv"][label][st]["selected_layer"]
                      for s in (42, 43, 44, 45, 46)]
            m_tr = (df["split"] == "train").to_numpy()
            m_ev = (df["split"] == split_name).to_numpy()
            aucs = []
            for seed, layer in zip((42, 43, 44, 45, 46), layers):
                scl, clf = fit_probe(np.asarray(arr[layer])[m_tr], y[label][m_tr])
                lg = clf.decision_function(scl.transform(np.asarray(arr[layer])[m_ev]))
                aucs.append(float(roc_auc_score(y[label][m_ev], lg)))
                sub = df.loc[m_ev, ["accession", "subtype", "cluster_id"]].copy()
                sub["label"] = label
                sub["seed"] = seed
                sub["layer"] = layer
                sub["logit"] = lg
                rows.append(sub)
            print(f"[{label}][{st}] layers={layers} "
                  f"isolate AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}", flush=True)
    pd.concat(rows).to_csv(f"{OUT}/probe_scores_h5h7.csv", index=False)

    # ── 前向三年 ──
    alliso = pd.read_csv(ALL_ISOLATES_CSV, usecols=[
        "subtype", "cluster_id", "host_category", "collection_year"])
    alliso["year"] = pd.to_numeric(alliso["collection_year"], errors="coerce")
    X3 = np.asarray(arr[L3])
    fwd_rows = []

    for Y in (2024, 2025):
        cutoff = Y - 1
        pre = labels_upto(alliso, cutoff)
        keys = list(zip(df["subtype"], df["cluster_id"]))
        tr = ((df["split"] == "train") & (df["year"] <= cutoff)).to_numpy()
        ev = (df["year"] == Y).to_numpy()
        for label in LABEL_COLS:
            y_tr = np.array([pre.get(k, {}).get(label, 0) for k in keys])[tr]
            scl, clf = fit_probe(X3[tr], y_tr)
            lg = clf.decision_function(scl.transform(X3[ev]))
            sub = df.loc[ev, ["accession", "subtype"]].copy()
            sub["year"] = Y
            sub["label"] = label
            sub["logit"] = lg
            fwd_rows.append(sub)
            # 对账 3：jump_human 与 ranking CSV 的 logit_L3 列一致
            if label == "label_is_jump_human":
                rk = pd.read_csv(f"validation_exp/temporal_validation/output/"
                                 f"forward_{Y}_ranking.csv")
                chk = sub.merge(rk, on=["accession", "subtype"],
                                validate="one_to_one")
                dmax = float(np.abs(chk["logit"] - chk["logit_L3"]).max())
                print(f"[{Y}][{label}] 与 ranking CSV max|Δ|={dmax:.2e}", flush=True)
                assert dmax < 1e-3, "与既有 ranking CSV 对账失败"

    c26 = pd.read_csv(CLEAN_2026_CSV)
    E26 = np.load(EMB26)
    m_tr = (df["split"] == "train").to_numpy()
    for label in LABEL_COLS:
        scl, clf = fit_probe(X3[m_tr], y[label][m_tr])
        lg = clf.decision_function(scl.transform(E26))
        sub = c26[["accession", "subtype"]].copy()
        sub["year"] = 2026
        sub["label"] = label
        sub["logit"] = lg
        fwd_rows.append(sub)
        if label == "label_is_jump_human":
            rk = pd.read_csv("validation_exp/temporal_validation/output/"
                             "forward_2026_ranking.csv")
            chk = sub.merge(rk, on=["accession", "subtype"], validate="one_to_one")
            dmax = float(np.abs(chk["logit"] - chk["_logit"]).max())
            print(f"[2026][{label}] 与 ranking CSV max|Δ|={dmax:.2e}", flush=True)
            assert dmax < 1e-3, "与既有 ranking CSV 对账失败"

    pd.concat(fwd_rows).to_csv(f"{OUT}/probe_scores_forward.csv", index=False)
    print("→ probe_scores_h5h7.csv / probe_scores_forward.csv")


if __name__ == "__main__":
    main()
