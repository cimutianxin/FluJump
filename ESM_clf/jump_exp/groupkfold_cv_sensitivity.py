"""GroupKFold 超参 CV 敏感性（S4，Major 12c）

train_probe.py / target_val_layer_select.py 的 ridge 超参选择用 isolate 级
StratifiedKFold。本脚本仅把 GridSearchCV 的 cv 换成 GroupKFold(5)，
groups = (subtype, cluster_id)（cluster_id 为亚型内编号，拼键必须带 subtype），
其余流程（数据、C 网格、scoring、seed、target-val 选层协议）完全不变。

对照基线（已发表口径，直接读既有产物，不重跑）：
  - 逐层 stratified best_C 与全 holdout 原始 AUC：depth_reversal g1_layer_geometry.json
  - target-val headline（cluster 臂）：target_val_layer_select.json
    （0.836 / 0.918 / 0.688 / 0.854）

输出：output/groupkfold_cv_sensitivity.json
"""

import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

SEEDS = [42, 43, 44, 45, 46]
ARMS = ["cluster", "isolate_random"]
SUBTYPES = ["h5", "h7"]
G1_JSON = "validation_exp/depth_reversal/output/g1_layer_geometry.json"
BASE_TV_JSON = OUT_DIR / "target_val_layer_select.json"


def fit_probe_groupcv(Xt, yt, groups):
    """与 target_val_layer_select.fit_probe 相同，仅 cv 换 GroupKFold(5)"""
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=GroupKFold(n_splits=CV_FOLDS),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(Xt_s, yt, groups=groups)
    return scl, gs


def main():
    arr = np.load(OUT_DIR / "esm_emb_150M_all_layers.npy")  # (31, N, 640)
    split_df = pd.read_csv(SPLIT_CSV)
    m_train = (split_df["split"] == "train").values
    m_h57 = split_df["split"].isin(["h5_holdout", "h7_holdout"]).values
    eval_idx = np.where(m_h57)[0]
    pos_of = {r: p for p, r in enumerate(eval_idx)}

    # train 侧分组键：(subtype, cluster_id)
    gkeys = (split_df["subtype"] + "::" + split_df["cluster_id"].astype(str)).values
    _, groups_train = np.unique(gkeys[m_train], return_inverse=True)
    print(f"train 组数: {groups_train.max() + 1} / {m_train.sum()} 行")

    # ── 10 个 target-val 切分文件 ──
    files = [f"{arm}_seed{s}.csv" for arm in ARMS for s in SEEDS]
    file_rows = {}
    for fname in files:
        sp = pd.read_csv(H5H7_SPLIT_DIR / fname)
        key2row = {(a, s): i for i, (a, s) in
                   enumerate(zip(split_df["accession"], split_df["subtype"]))}
        rows = {}
        for mask_name, g in sp.groupby("split"):
            rows[mask_name] = np.array([key2row[(a, s)] for a, s in
                                        zip(g["accession"], g["subtype"])])
        file_rows[fname] = rows

    # ── 逐 (label, layer) GroupKFold 训练 ──
    per_layer = {c: {} for c in LABEL_COLS}   # label -> layer -> {best_C, auc_h5, auc_h7}
    aucs = {c: {} for c in LABEL_COLS}        # label -> layer -> fname -> mask -> auc
    for label_col in LABEL_COLS:
        y = np.load(OUT_DIR / f"labels_{label_col}.npy")
        yt = y[m_train]
        y_ev = y[eval_idx]
        st_ev = split_df["subtype"].values[eval_idx]
        for li in range(arr.shape[0]):
            scl, gs = fit_probe_groupcv(arr[li][m_train], yt, groups_train)
            sc = gs.decision_function(scl.transform(arr[li][eval_idx]))
            per_layer[label_col][li] = {
                "best_C_groupcv": gs.best_params_["C"],
                "auc_h5": float(roc_auc_score(
                    y_ev[st_ev == "H5"], sc[st_ev == "H5"])),
                "auc_h7": float(roc_auc_score(
                    y_ev[st_ev == "H7"], sc[st_ev == "H7"])),
            }
            per_file = {}
            for fname in files:
                per_file[fname] = {mn: float(roc_auc_score(y[rows], sc[[pos_of[r] for r in rows]]))
                                   for mn, rows in file_rows[fname].items()}
            aucs[label_col][li] = per_file
            if li % 5 == 0 or li == arr.shape[0] - 1:
                print(f"  {label_col} L{li} done", flush=True)

    # ── target-val 选层（协议不变：cluster 臂 val argmax → test 评一次） ──
    group_tv = {}
    for arm in ARMS:
        arm_files = [f"{arm}_seed{s}.csv" for s in SEEDS]
        ares = {}
        for label_col in LABEL_COLS:
            lres = {}
            for st in SUBTYPES:
                picks, tests = [], []
                for fname in arm_files:
                    val_aucs = {li: aucs[label_col][li][fname][f"{st}_val"]
                                for li in aucs[label_col]}
                    best = max(val_aucs, key=val_aucs.get)
                    picks.append(best)
                    tests.append(aucs[label_col][best][fname][f"{st}_test"])
                lres[st] = {"selected_layers": picks,
                            "layer_freq": dict(Counter(picks)),
                            "test_auc_mean": float(np.mean(tests)),
                            "test_auc_std": float(np.std(tests))}
            ares[label_col] = lres
        group_tv[arm] = ares

    # ── 与基线对照 ──
    g1 = json.load(open(G1_JSON))
    base_tv = json.load(open(BASE_TV_JSON))
    compare = {}
    for label_col in LABEL_COLS:
        g1_pl = g1["per_layer"][label_col]
        c_changes = {str(li): {"stratified": g1_pl[li]["best_C"],
                               "groupcv": per_layer[label_col][li]["best_C_groupcv"]}
                     for li in range(31)
                     if g1_pl[li]["best_C"] != per_layer[label_col][li]["best_C_groupcv"]}
        key_layers = {}
        for li in [13, 17, 28]:
            key_layers[f"L{li}"] = {
                "auc_h5": {"stratified": g1_pl[li]["auc_h5"],
                           "groupcv": per_layer[label_col][li]["auc_h5"]},
                "auc_h7": {"stratified": g1_pl[li]["auc_h7"],
                           "groupcv": per_layer[label_col][li]["auc_h7"]},
            }
        tv_cells = {}
        for st in SUBTYPES:
            b = base_tv["summary"]["cluster"][label_col][st]
            g = group_tv["cluster"][label_col][st]
            tv_cells[st] = {
                "baseline": {"selected": b["selected_layers"],
                             "test_auc_mean": b["test_auc_mean"],
                             "test_auc_std": b["test_auc_std"]},
                "groupcv": {"selected": g["selected_layers"],
                            "test_auc_mean": g["test_auc_mean"],
                            "test_auc_std": g["test_auc_std"]},
                "delta_mean": g["test_auc_mean"] - b["test_auc_mean"],
            }
        compare[label_col] = {"n_C_changed": len(c_changes), "C_changes": c_changes,
                              "key_layers": key_layers, "target_val_cluster": tv_cells}

    print(f"\n{'='*76}\nGroupKFold vs Stratified 对照\n{'='*76}")
    for label_col in LABEL_COLS:
        c = compare[label_col]
        print(f"\n[{label_col}] C 变化层数: {c['n_C_changed']}/31 {c['C_changes']}")
        for lk, lv in c["key_layers"].items():
            print(f"  {lk}: H5 {lv['auc_h5']['stratified']:.4f}→{lv['auc_h5']['groupcv']:.4f} "
                  f"H7 {lv['auc_h7']['stratified']:.4f}→{lv['auc_h7']['groupcv']:.4f}")
        for st in SUBTYPES:
            t = c["target_val_cluster"][st]
            print(f"  {st}: baseline {t['baseline']['test_auc_mean']:.3f}±{t['baseline']['test_auc_std']:.3f} "
                  f"{t['baseline']['selected']} → groupcv {t['groupcv']['test_auc_mean']:.3f}±"
                  f"{t['groupcv']['test_auc_std']:.3f} {t['groupcv']['selected']} "
                  f"(Δ{t['delta_mean']:+.3f})")

    out = {"protocol": "train_probe/target_val 全流程不变，仅 GridSearchCV cv: "
                       "StratifiedKFold(5,seed42) → GroupKFold(5), groups=(subtype,cluster_id)",
           "baseline_source": {"per_layer_C_auc": G1_JSON, "target_val": str(BASE_TV_JSON)},
           "per_layer_groupcv": per_layer, "target_val_groupcv": group_tv,
           "compare": compare}
    out_path = OUT_DIR / "groupkfold_cv_sensitivity.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
