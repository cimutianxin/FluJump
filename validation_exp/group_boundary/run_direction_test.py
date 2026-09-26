#!/usr/bin/env python3
"""核心评估：H10/H4 零样本方向检验

协议（零样本 = 不用 H10/H4 数据做任何选择）：
  1. 重训 31 层 Ridge LR probe（H1+H3 train，同 target_val_layer_select.fit_probe）
  2. 回归校验：同 probe 在 H5/H7 holdout 上复算 AUC，对照历史 headline
  3. H10/H4 打分：每 (label, layer) isolate 级 + cluster 级（簇内 mean logit）raw AUC
  4. 预注册层判定（config.PREREG_LAYERS）：cluster bootstrap B=1000 →
     p05/p95/P(>0.5)；P(raw>0.5) < 0.05 判为方向显著反转（复现 H7 模式）
  5. 全层剖面（描述性）：31 层 raw AUC，与 H5/H7 并列
  6. 几何检验：预注册层上 H10/H4 正/阴性的 logit 投影分布（对照 H7：
     整体落入 w 负半空间且阳性更负）
  7. 朴素基线一致性：max identity to H1+H3 train 人源参考
  8. jump_human 仅描述性报告（H10 阳性 2 簇 / H4 0 簇，功效不足）

输出：output/direction_test.json + {ST}_cluster_ranking.csv
"""

import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from validation_exp.group_boundary.config import (
    SUBTYPES, PROC_DIR, OUT_DIR, SPLIT_CSV, ALIGNED_CSV,
    ALL_LAYERS_NPY, LABELS_DIR, LABEL_COLS, RIDGE_C_VALUES, CV_FOLDS,
    RANDOM_SEED, PREREG_LAYERS, B_BOOT,
)

GAP = ord("-")


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


def encode(seqs):
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    return arr.reshape(len(seqs), -1)


def max_identity(Q, R, chunk_q=256, chunk_r=2048):
    """query × ref 的最大双非gap位 identity（同 temporal_validation）"""
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


def cluster_boot_auc(y, sc, cl, rng, B=B_BOOT):
    """cluster 级 bootstrap：AUC 分布、p05/p95、P(>0.5)"""
    uniq = np.unique(cl)
    vals = []
    for _ in range(B):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(cl == c)[0] for c in samp])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], sc[idx])))
    v = np.array(vals)
    return {"auc_full": float(roc_auc_score(y, sc)),
            "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)),
            "p_above_0.5": float((v > 0.5).mean()),
            "n_boot": len(v)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── 主数据资产：训练集（H1+H3 train）与 H5/H7 holdout ──
    arr = np.load(ALL_LAYERS_NPY)  # (31, 11060, 640)
    split_df = pd.read_csv(SPLIT_CSV)
    m_train = (split_df["split"] == "train").values
    m_h5 = (split_df["split"] == "h5_holdout").values
    m_h7 = (split_df["split"] == "h7_holdout").values
    n_layers = arr.shape[0]

    # ── H10/H4 评估数据 ──
    ev_df = pd.concat([pd.read_csv(PROC_DIR / f"{st}_isolates.csv", dtype=str)
                       for st in SUBTYPES], ignore_index=True)
    ev_emb = np.load(OUT_DIR / "h10_h4_emb_all_layers.npy")  # (31, N, 640)
    assert ev_emb.shape[1] == len(ev_df)
    for col in LABEL_COLS:
        ev_df[col] = ev_df[col].astype(int)
    ev_mask = {st: (ev_df["subtype"] == st).values for st in SUBTYPES}

    results = {"regression_check": {}, "profile": {}, "preregistered": {},
               "geometry": {}, "naive": {}, "eval_size": {}}
    for st in SUBTYPES:
        results["eval_size"][st] = {
            "n_isolates": int(ev_mask[st].sum()),
            "n_clusters": int(ev_df.loc[ev_mask[st], "cluster_id"].nunique()),
            "n_pos_clusters_jump": int(
                ev_df.loc[ev_mask[st]].groupby("cluster_id")
                ["label_is_jump"].max().sum()),
            "n_pos_clusters_jump_human": int(
                ev_df.loc[ev_mask[st]].groupby("cluster_id")
                ["label_is_jump_human"].max().sum()),
        }
    print(f"评估集规模: {results['eval_size']}")

    # ── 逐 label × layer 训练并打分 ──
    for label in LABEL_COLS:
        y = np.load(LABELS_DIR / f"labels_{label}.npy")
        yt = y[m_train]
        print(f"\n[{label}] train 阳性率 {yt.mean():.1%}")
        results["profile"][label] = {}
        results["regression_check"][label] = {}
        results["preregistered"][label] = {}
        results["geometry"][label] = {}

        for li in range(n_layers):
            scl, gs = fit_probe(arr[li][m_train], yt)
            # 回归校验：H5/H7 全量 holdout AUC
            auc_h5 = float(roc_auc_score(y[m_h5], gs.decision_function(scl.transform(arr[li][m_h5]))))
            auc_h7 = float(roc_auc_score(y[m_h7], gs.decision_function(scl.transform(arr[li][m_h7]))))
            results["regression_check"][label][li] = {"h5": auc_h5, "h7": auc_h7}

            # H10/H4 打分
            sc_all = gs.decision_function(scl.transform(ev_emb[li]))
            for st in SUBTYPES:
                m = ev_mask[st]
                sub = ev_df[m]
                yv = sub[label].values
                sc = sc_all[m]
                iso_auc = float(roc_auc_score(yv, sc)) if yv.sum() > 0 else None
                # cluster 级：簇内 mean logit，标签取簇级
                cl_df = pd.DataFrame({"cl": sub["cluster_id"].values,
                                      "y": yv, "sc": sc})
                cl_g = cl_df.groupby("cl").agg(y=("y", "max"), sc=("sc", "mean"))
                cl_auc = float(roc_auc_score(cl_g["y"], cl_g["sc"])) \
                    if cl_g["y"].sum() > 0 else None
                key = f"{st}"
                results["profile"][label].setdefault(key, {})[li] = {
                    "isolate_auc": iso_auc, "cluster_auc": cl_auc}

                # 预注册层：bootstrap + 几何
                if li in PREREG_LAYERS[label]:
                    boot = None
                    if cl_g["y"].sum() > 0:
                        boot = cluster_boot_auc(
                            cl_g["y"].values, cl_g["sc"].values,
                            cl_g.index.values, rng)
                        boot["n_pos_clusters"] = int(cl_g["y"].sum())
                    results["preregistered"][label].setdefault(st, {})[li] = {
                        "isolate_auc": iso_auc, "cluster_auc": cl_auc,
                        "cluster_bootstrap": boot}
                    # 几何：正/阴性 logit 分布（decision_function 即投影+偏置）
                    results["geometry"][label].setdefault(st, {})[li] = {
                        "proj_pos_mean": float(sc[yv == 1].mean()) if yv.sum() else None,
                        "proj_neg_mean": float(sc[yv == 0].mean()) if (yv == 0).sum() else None,
                        "frac_all_negative": float((sc < 0).mean()),
                        "frac_pos_negative": float((sc[yv == 1] < 0).mean()) if yv.sum() else None,
                    }
            if li % 5 == 0 or li == n_layers - 1:
                print(f"  L{li} done (H5={auc_h5:.3f}, H7={auc_h7:.3f})", flush=True)

        # cluster ranking CSV（label_is_jump 主标签，预注册第一层）
        if label == "label_is_jump":
            li0 = PREREG_LAYERS[label][0]
            scl, gs = fit_probe(arr[li0][m_train], yt)
            sc_all = gs.decision_function(scl.transform(ev_emb[li0]))
            for st in SUBTYPES:
                m = ev_mask[st]
                sub = ev_df[m].copy()
                sub["_logit"] = sc_all[m]
                cl = sub.groupby("cluster_id").agg(
                    logit=("_logit", "mean"),
                    label=("label_is_jump", "max"), n=("label_is_jump", "size"),
                    hosts=("host_category", lambda s: "|".join(sorted(set(s)))),
                    year_min=("collection_year", "min"),
                    year_max=("collection_year", "max"))
                cl = cl.sort_values("logit", ascending=False)
                cl["rank_raw"] = range(1, len(cl) + 1)
                cl["rank_flipped"] = cl["logit"].rank(ascending=True).astype(int)
                cl.to_csv(OUT_DIR / f"{st}_cluster_ranking.csv")

    # ── 朴素基线：max identity to H1+H3 train 人源参考 ──
    print("\n朴素基线打分...")
    ali = pd.read_csv(ALIGNED_CSV, usecols=["accession", "subtype",
                                            "aligned_ha_seq", "host_category"])
    ref_df = split_df.merge(ali, on=["accession", "subtype"], validate="one_to_one")
    ref = ref_df[(ref_df["split"] == "train")
                 & (ref_df["host_category"] == "human")]
    R = encode(ref["aligned_ha_seq"].tolist())
    print(f"  参考集: {len(R)} 条 H1+H3 train 人源")
    for st in SUBTYPES:
        q = pd.read_csv(OUT_DIR / f"{st}_aligned.csv", dtype=str)
        sub = ev_df[ev_mask[st]].copy()
        # mafft --add 丢弃了 7/5026 条序列，naive 基线只对齐存在的行
        q = q.set_index("accession")["aligned_ha_seq"]
        keep = sub["accession"].isin(q.index)
        sub = sub[keep]
        naive_sc = max_identity(encode(q.loc[sub["accession"]].tolist()), R)
        if not keep.all():
            print(f"  {st}: {int((~keep).sum())} 条无对齐序列，naive 基线剔除")
        for label in LABEL_COLS:
            yv = sub[label].values
            if yv.sum() == 0:
                continue
            iso = float(roc_auc_score(yv, naive_sc))
            cl_df = pd.DataFrame({"cl": sub["cluster_id"].values,
                                  "y": yv, "sc": naive_sc})
            cl_g = cl_df.groupby("cl").agg(y=("y", "max"), sc=("sc", "mean"))
            cl_auc = float(roc_auc_score(cl_g["y"], cl_g["sc"])) \
                if cl_g["y"].sum() > 0 else None
            results["naive"].setdefault(label, {})[st] = {
                "isolate_auc_raw": iso,
                "isolate_auc_flipped": float(roc_auc_score(yv, -naive_sc)),
                "cluster_auc_raw": cl_auc,
                "cluster_auc_flipped": float(roc_auc_score(cl_g["y"], -cl_g["sc"]))
                if cl_auc is not None else None,
            }
            print(f"  {st} {label}: naive isolate AUC raw={iso:.3f} "
                  f"flipped={results['naive'][label][st]['isolate_auc_flipped']:.3f}")

    # ── 汇总打印：预注册层判定 ──
    print(f"\n{'='*76}\n预注册层判定（cluster bootstrap, B={B_BOOT}）\n{'='*76}")
    for label in LABEL_COLS:
        for st in SUBTYPES:
            for li, r in sorted(results["preregistered"][label].get(st, {}).items()):
                b = r["cluster_bootstrap"]
                if b is None:
                    print(f"  {label} {st} L{li}: 无阳性簇，跳过")
                    continue
                verdict = "反转显著" if b["p_above_0.5"] < 0.05 else \
                          ("方向为正" if b["p_above_0.5"] > 0.95 else "不显著")
                print(f"  {label} {st} L{li}: cluster AUC={b['auc_full']:.3f} "
                      f"[{b['p05']:.3f},{b['p95']:.3f}] P(>0.5)={b['p_above_0.5']:.2f} "
                      f"(+簇 {b['n_pos_clusters']}) → {verdict}")

    with open(OUT_DIR / "direction_test.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ {OUT_DIR}/direction_test.json")


if __name__ == "__main__":
    main()
