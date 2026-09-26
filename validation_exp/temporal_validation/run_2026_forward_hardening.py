#!/usr/bin/env python3
"""2026 前向加固：cluster bootstrap CI / permutation p / ≤2025 标签口径 / L1 对照层

对应 RESULT.md TODO-B4 三子项。背景：09-19 曾以 ad-hoc 方式跑出
output/forward_2026_hardening.json（脚本未留存），本脚本将其固化为可复现流程，
随机项（bootstrap / permutation）以 RANDOM_SEED=42 为准，覆写同名 JSON。

子项：
  ① cluster 级统计推断：65 簇（簇内 mean logit、簇标签 max）上 bootstrap
     B=B_BOOT（弃退化重采样）→ 95% percentile CI + P(AUC>0.5)；
     簇标签 permutation B=B_PERM → p = #(AUC_perm >= AUC_obs)/B。
  ② ≤2025 标签口径：cluster 继承标签有循环风险（簇标签可能由 2026 成员自身贡献），
     从 all_isolates.csv 仅用 year<=2025 有日期成员的非 unknown host 重算簇标签，
     报告变化数与新口径 cluster AUC。
  ③ L1（末层）对照：层选择（L3 经 H5 holdout 选出，而 2026 阳性为 H5N1）的
     最小敏感性检查。

输出：output/forward_2026_hardening.json
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
    CLEAN_2026_CSV, ALIGNED_CSV, SPLIT_CSV, ALL_ISOLATES_CSV,
    MEAN_EMB_L3, MEAN_EMB_L1, LABEL_COLS, RIDGE_C_VALUES, CV_FOLDS,
    RANDOM_SEED, B_BOOT, B_PERM, OUT_DIR,
)

LAYERS = {"L3": MEAN_EMB_L3, "L1": MEAN_EMB_L1}   # L1=末层（对照），L3=主线层


def fit_probe(X_tr, y_tr):
    """同 run_2026_forward.py：StandardScaler + Ridge LR GridSearchCV"""
    scaler = StandardScaler().fit(X_tr)
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(scaler.transform(X_tr), y_tr)
    return scaler, gs.best_estimator_


def cluster_boot(sc, y, rng, B=B_BOOT):
    """cluster 级 bootstrap：重采样簇（含重复），弃退化，95% CI + P(AUC>0.5)"""
    n = len(y)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], sc[idx])))
    v = np.array(vals)
    return {"B_valid": len(v),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "p_auc_gt_0.5": float((v > 0.5).mean())}


def cluster_perm_p(sc, y, rng, B=B_PERM):
    """簇标签 permutation：p = #(AUC_perm >= AUC_obs)/B"""
    obs = float(roc_auc_score(y, sc))
    cnt = 0
    for _ in range(B):
        if roc_auc_score(rng.permutation(y), sc) >= obs:
            cnt += 1
    return obs, cnt / B


def recompute_pre2025_labels():
    """仅用 ≤2025 信息重算簇级 jump / jump_human 标签（口径同 build_jump_labels）

    cluster_id 为亚型内编号（all_isolates 中 693/908 个 id 跨亚型复用），
    必须按 (subtype, cluster_id) 聚合。
    """
    alliso = pd.read_csv(ALL_ISOLATES_CSV,
                         usecols=["subtype", "cluster_id", "host_category",
                                  "collection_year"])
    yr = pd.to_numeric(alliso["collection_year"], errors="coerce")
    m = yr.notna() & (yr > 0) & (yr <= 2025) & (alliso["host_category"] != "unknown")
    hosts = alliso[m].groupby(["subtype", "cluster_id"])["host_category"].agg(set)
    labels = {}
    for (st, cid), hs in hosts.items():
        labels[(st, cid)] = {"label_is_jump": int(len(hs) >= 2),
                             "label_is_jump_human": int("human" in hs and len(hs) >= 2)}
    return labels


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── 训练集（isolate_split 行序 = embedding 行序）──
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=[
        "accession", "subtype", "host_category",
        "label_is_jump", "label_is_jump_human"])
    df = split.merge(ali, on=["accession", "subtype"], validate="one_to_one")
    train_mask = (df["split"] == "train").to_numpy()
    h5_mask = (df["split"] == "h5_holdout").to_numpy()

    # ── 2026 数据 ──
    c26 = pd.read_csv(CLEAN_2026_CSV)
    n26 = len(c26)
    assert n26 == 201

    # ── ≤2025 标签口径：重算并统计变化（簇键 = (subtype, cluster_id)）──
    pre_labels = recompute_pre2025_labels()
    cl26 = c26.groupby(["subtype", "cluster_id"])[LABEL_COLS].max()  # 现行簇标签
    n_no_pre = 0                                                     # 无 ≤2025 成员的簇数
    pre_mat = {}
    for label in LABEL_COLS:
        new_y, missing = [], 0
        for st, cid in cl26.index:
            lab = pre_labels.get((st, cid))
            if lab is None:
                missing += 1
                new_y.append(0)
            else:
                new_y.append(lab[label])
        pre_mat[label] = np.array(new_y)
        n_no_pre = max(n_no_pre, missing)
    label_change = {l: int((pre_mat[l] != cl26[l].to_numpy()).sum()) for l in LABEL_COLS}
    print(f"≤2025 标签口径：变化 {label_change}；无 ≤2025 成员的簇 {n_no_pre} 个"
          f"（共 {len(cl26)} 簇）")

    results = {"n_2026": n26, "n_clusters_2026": int(len(cl26)),
               "label_change_pre2025": label_change, "layers": {}}

    # ── 逐层 × 双标签 ──
    for lname, emb_path in LAYERS.items():
        X = np.load(emb_path).astype(np.float32)
        assert len(X) == len(df)
        E26 = np.load(OUT_DIR / f"emb_2026_{lname}.npy").astype(np.float32)
        assert len(E26) == n26
        results["layers"][lname] = {}

        for label in LABEL_COLS:
            y = df[label].to_numpy()
            scaler, clf = fit_probe(X[train_mask], y[train_mask])
            auc_h5 = float(roc_auc_score(
                y[h5_mask], clf.decision_function(scaler.transform(X[h5_mask]))))
            if lname == "L3":
                assert 0.65 < auc_h5 < 0.95, f"probe 复现异常: {label} H5 AUC={auc_h5}"

            logit26 = clf.decision_function(scaler.transform(E26))
            y26 = c26[label].to_numpy()
            auc_iso = float(roc_auc_score(y26, logit26))

            cg = c26.assign(_logit=logit26).groupby(["subtype", "cluster_id"]).agg(
                score=("_logit", "mean"), label=(label, "max"))  # 行序与 pre_mat 一致
            sc_cl, y_cl = cg["score"].to_numpy(), cg["label"].to_numpy()
            boot = cluster_boot(sc_cl, y_cl, rng)
            auc_cl, p_perm = cluster_perm_p(sc_cl, y_cl, rng)
            auc_cl_pre = float(roc_auc_score(pre_mat[label], sc_cl)) \
                if pre_mat[label].sum() > 0 else None

            order = np.argsort(-logit26)
            ranks = np.empty(n26, dtype=int)
            ranks[order] = np.arange(1, n26 + 1)
            pos_ranks = sorted(int(ranks[i]) for i in np.where(y26 == 1)[0])

            results["layers"][lname][label] = {
                "h5_repro_auc": round(auc_h5, 4),
                "auc_2026_isolate": round(auc_iso, 4),
                "auc_2026_cluster": round(auc_cl, 4),
                "cluster_bootstrap": boot,
                "cluster_permutation_p": p_perm,
                "auc_2026_cluster_pre2025labels": round(auc_cl_pre, 4)
                if auc_cl_pre else None,
                "n_positive_clusters": int(y_cl.sum()),
                "positive_ranks": pos_ranks,
            }
            print(f"[{lname}][{label}] H5 复现 {auc_h5:.4f} | "
                  f"2026 isolate {auc_iso:.4f} cluster {auc_cl:.4f} "
                  f"CI{boot['ci95']} P(>0.5)={boot['p_auc_gt_0.5']:.4f} "
                  f"perm_p={p_perm} pre2025={auc_cl_pre} | 阳性名次 {pos_ranks}",
                  flush=True)

    with open(OUT_DIR / "forward_2026_hardening.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n→ {OUT_DIR / 'forward_2026_hardening.json'}")


if __name__ == "__main__":
    main()
