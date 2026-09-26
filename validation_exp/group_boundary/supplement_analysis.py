#!/usr/bin/env python3
"""补充分析：主评估之后的三个收尾统计

1. 全层符号检验：H10/H4 cluster AUC < 0.5 的层数 → 双侧二项 p 值
2. 哺乳跳跃子集：H10 的 avian|environmental 阳性簇是监测共检出，生物学弱；
   仅保留哺乳相关阳性簇（human/swine 等）在预注册层的 cluster AUC（描述性）
3. 已知人源簇（H10_197=H10N8 2013 / H10_22=H10N3 2021）在 jump 与 jump_human
   预注册层上的名次（raw 与 flipped 方向）

输出：output/direction_test_supplement.json
"""

import json
import sys
from math import comb

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from validation_exp.group_boundary.config import (
    SUBTYPES, PROC_DIR, OUT_DIR, SPLIT_CSV, ALL_LAYERS_NPY, LABELS_DIR,
    RIDGE_C_VALUES, CV_FOLDS, RANDOM_SEED, PREREG_LAYERS,
)

MAMMAL_HOSTS = {"human", "swine", "canine", "equine", "bovine", "feline",
                "mustelid", "marine_mammal", "other_mammal"}
KNOWN_HUMAN_CLUSTERS = {"H10_197": "H10N8 江西 2013", "H10_22": "H10N3 江苏 2021"}


def binom_sign_p(k, n):
    """双侧符号检验 p 值"""
    from math import comb as c
    lo = sum(c(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2 * lo / 2 ** n)


def fit_probe(Xt, yt):
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(Xt_s, yt)
    return scl, gs


def main():
    prof = json.load(open(OUT_DIR / "direction_test.json"))["profile"]
    out = {"sign_test": {}, "mammalian_subset": {}, "human_cluster_ranks": {}}

    # ── 1. 符号检验 ──
    for label in ["label_is_jump"]:
        for st in SUBTYPES:
            aucs = [prof[label][st][str(li)]["cluster_auc"] for li in range(31)]
            k = sum(a < 0.5 for a in aucs)
            out["sign_test"][st] = {"n_layers_below_0.5": k, "n_layers": 31,
                                    "p_two_sided": binom_sign_p(k, 31)}
            print(f"符号检验 {st}: {k}/31 层 <0.5, p={out['sign_test'][st]['p_two_sided']:.4f}")

    # ── 2/3. 预注册层重训打分 ──
    arr = np.load(ALL_LAYERS_NPY)
    split_df = pd.read_csv(SPLIT_CSV)
    m_train = (split_df["split"] == "train").values
    ev_df = pd.concat([pd.read_csv(PROC_DIR / f"{st}_isolates.csv", dtype=str)
                       for st in SUBTYPES], ignore_index=True)
    for col in ["label_is_jump", "label_is_jump_human"]:
        ev_df[col] = ev_df[col].astype(int)
    ev_emb = np.load(OUT_DIR / "h10_h4_emb_all_layers.npy")

    # 阳性簇宿主构成（判哺乳相关）
    cl_hosts = ev_df.groupby("cluster_id")["host_category"].apply(
        lambda s: set(s))
    mammal_pos_clusters = {c for c, hs in cl_hosts.items() if hs & MAMMAL_HOSTS}
    # 且是 jump 阳性的簇
    cl_jump = ev_df.groupby("cluster_id")["label_is_jump"].max()
    mammal_pos_clusters &= set(cl_jump[cl_jump == 1].index)
    print(f"哺乳相关 jump 阳性簇: {sorted(mammal_pos_clusters)}")

    for label in ["label_is_jump", "label_is_jump_human"]:
        y = np.load(LABELS_DIR / f"labels_{label}.npy")
        yt = y[m_train]
        for li in PREREG_LAYERS[label]:
            scl, gs = fit_probe(arr[li][m_train], yt)
            sc_all = gs.decision_function(scl.transform(ev_emb[li]))
            m10 = (ev_df["subtype"] == "H10").values
            sub = ev_df[m10].copy()
            sub["_sc"] = sc_all[m10]
            cl = sub.groupby("cluster_id").agg(
                sc=("_sc", "mean"), y=(label, "max"))
            auc_all = roc_auc_score(cl["y"], cl["sc"]) if cl["y"].sum() else None

            # 哺乳子集：阳性限哺乳簇，阴性保持全部
            pos_m = cl.index.isin(mammal_pos_clusters) & (cl["y"] == 1)
            sel = pos_m | (cl["y"] == 0)
            auc_mam = roc_auc_score(cl["y"][sel], cl["sc"][sel]) \
                if cl["y"][sel].sum() else None
            out["mammalian_subset"].setdefault(label, {})[li] = {
                "n_pos_clusters_mammal": int(pos_m.sum()),
                "cluster_auc": float(auc_mam) if auc_mam is not None else None,
                "cluster_auc_flipped": float(roc_auc_score(cl["y"][sel], -cl["sc"][sel]))
                if auc_mam is not None else None,
                "ref_all_pos_cluster_auc": float(auc_all) if auc_all is not None else None,
            }
            print(f"  {label} L{li}: 哺乳子集(+{int(pos_m.sum())}簇) "
                  f"AUC={auc_mam:.3f}（全部阳性簇 {auc_all:.3f}）"
                  if auc_mam is not None else f"  {label} L{li}: 哺乳子集无阳性")

            # 已知人源簇名次
            cl_sorted = cl.sort_values("sc", ascending=False)
            ranks = {c: i + 1 for i, c in enumerate(cl_sorted.index)}
            for c, name in KNOWN_HUMAN_CLUSTERS.items():
                if c in ranks:
                    out["human_cluster_ranks"].setdefault(c, {"event": name})[
                        f"{label}_L{li}"] = {
                        "rank_raw": ranks[c],
                        "rank_flipped": len(cl) - ranks[c] + 1,
                        "n_clusters": len(cl),
                    }

    with open(OUT_DIR / "direction_test_supplement.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n人源簇名次: {json.dumps(out['human_cluster_ranks'], ensure_ascii=False, indent=2)}")
    print(f"✓ {OUT_DIR}/direction_test_supplement.json")


if __name__ == "__main__":
    main()
