#!/usr/bin/env python3
"""多年份前向复现（2024/2025）——RESULT.md TODO-B7（升档关键②）

把 §5 前向排序验证从"2026 单年 2 簇"扩展为三年（2024/2025/2026）共 17 个
阳性簇的可复现规律。协议复刻 run_2026_forward.py + run_2026_forward_hardening.py，
关键差异：**训练标签按 ≤cutoff 成员重算**（现行标签由 ≤2025 全量数据算出，
对 2024/2025-cutoff 构成未来信息泄漏）。

对每个 eval year Y（cutoff = Y−1）：
  1) train 行 = split==train 且 year≤cutoff，标签用 ≤cutoff 簇成员重算
     （口径同 build_jump_labels；簇键 (subtype, cluster_id)——cluster_id 为亚型内编号）
  2) Ridge LR probe（同 2026 超参）在 L3 / L1（末层对照）上各训一次，
     H5 holdout 复现 AUC 作 sanity
  3) 评估集 = year==Y 全部 isolate（全亚型；H7 为边界外亚型，另报剔除敏感性）
  4) 指标：isolate AUC（probe/naive）、cluster AUC + bootstrap CI + permutation p、
     双口径标签（headline=全量 ever-jump / 因果=≤cutoff）+ 首跳簇名次、
     阳性名次、分层命中（阈值仅 ≤cutoff train 定：高=阳性中位，中=阴性 P95）
  5) 朴素基线：与 ≤cutoff train 人源参考的 max 对齐 identity（参考集同 cutoff）

输出：output/forward_multiyear.json + forward_{2024,2025}_ranking.csv
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
    ALIGNED_CSV, SPLIT_CSV, ALL_ISOLATES_CSV, MEAN_EMB_L3, MEAN_EMB_L1,
    LABEL_COLS, RIDGE_C_VALUES, CV_FOLDS, RANDOM_SEED, B_BOOT, B_PERM,
    NAIVE_REF_HOST, OUT_DIR,
)

YEARS = [2024, 2025]
LAYERS = {"L3": MEAN_EMB_L3, "L1": MEAN_EMB_L1}   # L1=末层（对照），L3=主线层
GAP = ord("-")
TIER_ORD = {"低": 0, "中": 1, "高": 2}


# ── 工具函数（与 run_2026_forward*.py 同实现）──

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


def fit_probe(X_tr, y_tr):
    """StandardScaler + Ridge LR GridSearchCV（同 2026 协议）"""
    scaler = StandardScaler().fit(X_tr)
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(scaler.transform(X_tr), y_tr)
    return scaler, gs.best_estimator_, gs.best_params_["C"]


def cluster_boot(sc, y, rng, B=B_BOOT):
    """cluster 级 bootstrap：重采样簇（含重复），弃退化，95% CI + P(AUC>0.5)"""
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
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
    cnt = sum(float(roc_auc_score(rng.permutation(y), sc) >= obs) for _ in range(B))
    return obs, cnt / B


def labels_upto(alliso, cutoff):
    """仅用 ≤cutoff 有年份成员的非 unknown host 重算簇级标签（口径同 build_jump_labels）"""
    yr = alliso["year"]
    m = yr.notna() & (yr > 0) & (yr <= cutoff) & (alliso["host_category"] != "unknown")
    hosts = alliso[m].groupby(["subtype", "cluster_id"])["host_category"].agg(set)
    return {k: {"label_is_jump": int(len(v) >= 2),
                "label_is_jump_human": int("human" in v and len(v) >= 2)}
            for k, v in hosts.items()}


def map_cluster_labels(df, pre_labels):
    """按 (subtype, cluster_id) 把重算标签映射到行（无 ≤cutoff 成员的簇记 0）"""
    keys = list(zip(df["subtype"], df["cluster_id"]))
    return {l: np.array([pre_labels.get(k, {}).get(l, 0) for k in keys])
            for l in LABEL_COLS}


def tier_metrics(tiers, y):
    """tiers: 低/中/高 数组；y: 0/1。返回高/中及以上 两档的命中指标"""
    res = {}
    for name, mask in [("high", np.array([t == "高" for t in tiers])),
                       ("mid_plus", np.array([t != "低" for t in tiers]))]:
        tp = int((mask & (y == 1)).sum())
        res[name] = {"n": int(mask.sum()), "tp": tp,
                     "precision": round(float(tp / mask.sum()), 4)
                     if mask.sum() else None,
                     "recall": round(float(tp / max(y.sum(), 1)), 4),
                     "fpr": round(float((mask & (y == 0)).sum())
                                  / max((y == 0).sum(), 1), 4)}
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── 主表（isolate_split 行序 = embedding 行序）──
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=[
        "accession", "subtype", "aligned_ha_seq", "host_category", "strain_name",
        "country", "collection_date", "collection_year",
        "label_is_jump", "label_is_jump_human"])
    df = split.merge(ali, on=["accession", "subtype"], validate="one_to_one")
    df["year"] = pd.to_numeric(df["collection_year"], errors="coerce")
    E = {k: np.load(p).astype(np.float32) for k, p in LAYERS.items()}
    for X in E.values():
        assert len(X) == len(df)

    # ── 全量表（≤cutoff 标签重算的底表；含 2026，但 cutoff ≤2024 过滤后等价）──
    alliso = pd.read_csv(ALL_ISOLATES_CSV, usecols=[
        "subtype", "cluster_id", "host_category", "collection_year"])
    alliso["year"] = pd.to_numeric(alliso["collection_year"], errors="coerce")

    results = {"years": {}, "protocol": {
        "train": "split==train & year<=cutoff，标签 ≤cutoff 簇成员重算",
        "eval": "year==Y 全亚型 isolate；headline=全量 ever-jump 簇继承标签",
        "probe": "StandardScaler+Ridge LR GridSearchCV（同 2026）",
        "B_boot": B_BOOT, "B_perm": B_PERM, "seed": RANDOM_SEED}}

    for Y in YEARS:
        cutoff = Y - 1
        print(f"\n{'='*70}\n■ 前向年份 {Y}（训练截止 {cutoff}）\n{'='*70}", flush=True)

        pre_labels = labels_upto(alliso, cutoff)
        pre_mat = map_cluster_labels(df, pre_labels)   # 全表行级 ≤cutoff 标签

        tr = ((df["split"] == "train") & (df["year"] <= cutoff)).to_numpy()
        ev = (df["year"] == Y).to_numpy()
        tr_idx, ev_idx = np.where(tr)[0], np.where(ev)[0]
        y_head = {l: df[l].to_numpy() for l in LABEL_COLS}   # headline（全量口径）

        yres = {"cutoff": cutoff, "n_train": int(tr.sum()),
                "n_train_pos_headline_vs_causal": {
                    l: [int(y_head[l][tr].sum()), int(pre_mat[l][tr].sum())]
                    for l in LABEL_COLS},
                "n_eval": int(ev.sum()),
                "eval_subtypes": {str(k): int(v) for k, v in
                                  df.loc[ev, "subtype"].value_counts().items()},
                "layers": {}, "naive": {}, "tiers": {}}

        # ── 朴素基线（≤cutoff train 人源参考；与 eval 年份不相交，无自匹配）──
        ref_mask = tr & (df["host_category"] == NAIVE_REF_HOST).to_numpy()
        R = encode(df.loc[ref_mask, "aligned_ha_seq"].tolist())
        Q = encode(df.loc[ev, "aligned_ha_seq"].tolist())
        print(f"朴素基线: eval {len(Q)} × 人源参考 {len(R)}", flush=True)
        naive_sc = max_identity(Q, R)

        ev_df = df.loc[ev].reset_index(drop=True)
        ev_noH7 = (ev_df["subtype"] != "H7").to_numpy()

        for lname, X in E.items():
            yres["layers"][lname] = {}
            for label in LABEL_COLS:
                y_tr = pre_mat[label][tr]
                assert y_tr.sum() >= CV_FOLDS * 2, f"{Y}/{label} 训练阳性过少"
                scaler, clf, best_c = fit_probe(X[tr_idx], y_tr)

                # H5 holdout 复现 sanity：同口径（≤cutoff 标签）断言；headline 口径仅记录
                h5 = (df["split"] == "h5_holdout").to_numpy()
                lg_h5 = clf.decision_function(scaler.transform(X[h5]))
                auc_h5 = float(roc_auc_score(pre_mat[label][h5], lg_h5))
                auc_h5_head = float(roc_auc_score(y_head[label][h5], lg_h5))
                # 方向性 sanity（仅 L3 主线层；L1 为对照层，失效本身即证据只记录）
                # 非性能门槛：jump_human H5 是已知最弱格，因果训练下打折属预期
                if lname == "L3":
                    assert auc_h5 > 0.5, \
                        f"probe 方向崩塌: {Y}/{lname}/{label} H5(causal)={auc_h5}"

                logits = clf.decision_function(scaler.transform(X[ev_idx]))
                y_ev = y_head[label][ev]
                y_ev_pre = pre_mat[label][ev]

                # isolate 级
                auc_iso = float(roc_auc_score(y_ev, logits)) if y_ev.sum() else None
                auc_iso_noH7 = float(roc_auc_score(
                    y_ev[ev_noH7], logits[ev_noH7])) if y_ev[ev_noH7].sum() else None
                auc_naive = float(roc_auc_score(y_ev, naive_sc)) if y_ev.sum() else None

                # cluster 级（键 (subtype, cluster_id)）
                cg = ev_df.assign(_l=logits, _n=naive_sc).groupby(
                    ["subtype", "cluster_id"]).agg(
                    score=("_l", "mean"), naive=("_n", "mean"),
                    label=(label, "max"), n=(label, "size"))
                cg["label_pre"] = [pre_labels.get(k, {}).get(label, 0)
                                   for k in cg.index]
                sc_cl, y_cl = cg["score"].to_numpy(), cg["label"].to_numpy()
                boot = cluster_boot(sc_cl, y_cl, rng) if y_cl.sum() else None
                auc_cl, p_perm = cluster_perm_p(sc_cl, y_cl, rng) if y_cl.sum() else (None, None)
                y_pre = cg["label_pre"].to_numpy()
                auc_cl_pre = float(roc_auc_score(y_pre, sc_cl)) if y_pre.sum() else None
                auc_cl_naive = float(roc_auc_score(
                    y_cl, cg["naive"].to_numpy())) if y_cl.sum() else None

                # 首跳簇（headline=1 且 ≤cutoff=0）的簇级名次
                first = cg[(cg["label"] == 1) & (cg["label_pre"] == 0)]
                cranks = cg["score"].rank(ascending=False).astype(int)
                first_ranks = {f"{st}/{cid}": int(cranks.loc[(st, cid)])
                               for st, cid in first.index}

                # 阳性 isolate 名次
                order = np.argsort(-logits)
                ranks = np.empty(len(logits), dtype=int)
                ranks[order] = np.arange(1, len(logits) + 1)
                pos_ranks = sorted(int(ranks[i]) for i in np.where(y_ev == 1)[0])

                yres["layers"][lname][label] = {
                    "best_C": best_c,
                    "h5_repro_auc_causal_labels": round(auc_h5, 4),
                    "h5_repro_auc_headline_labels": round(auc_h5_head, 4),
                    "auc_isolate": round(auc_iso, 4) if auc_iso else None,
                    "auc_isolate_noH7": round(auc_iso_noH7, 4) if auc_iso_noH7 else None,
                    "auc_isolate_naive": round(auc_naive, 4) if auc_naive else None,
                    "auc_cluster": round(auc_cl, 4) if auc_cl else None,
                    "auc_cluster_causal_labels": round(auc_cl_pre, 4) if auc_cl_pre else None,
                    "auc_cluster_naive": round(auc_cl_naive, 4) if auc_cl_naive else None,
                    "cluster_bootstrap": boot, "cluster_permutation_p": p_perm,
                    "n_eval_clusters": int(len(cg)),
                    "n_clusters_no_pre_member": int(
                        sum(1 for k in cg.index if k not in pre_labels)),
                    "n_pos_clusters": int(y_cl.sum()),
                    "n_pos_clusters_causal": int(y_pre.sum()),
                    "n_first_jump_clusters": int(len(first)),
                    "first_jump_cluster_ranks": first_ranks,
                    "positive_ranks": pos_ranks,
                }
                print(f"[{lname}][{label}] C={best_c} H5复现={auc_h5:.4f}(因果)/{auc_h5_head:.4f}(headline) | "
                      f"isolate={auc_iso:.4f}(剔H7 {auc_iso_noH7:.4f}) naive={auc_naive:.4f} | "
                      f"cluster={auc_cl:.4f} CI{boot['ci95']} "
                      f"P(>0.5)={boot['p_auc_gt_0.5']:.3f} perm_p={p_perm} | "
                      f"因果口径={auc_cl_pre} 首跳簇={first_ranks}", flush=True)

                # ── 分层命中（仅 L3 主线层；阈值仅 ≤cutoff train 定）──
                if lname == "L3":
                    lg_tr = clf.decision_function(scaler.transform(X[tr_idx]))
                    thr = {"high": float(np.median(lg_tr[y_tr == 1])),
                           "mid": float(np.quantile(lg_tr[y_tr == 0], 0.95))}
                    tiers = ["高" if v >= thr["high"] else
                             ("中" if v >= thr["mid"] else "低") for v in logits]
                    # trainpct 口径同 risk_scorer：train logits ≤ 该 logit 的比例
                    lg_tr_sorted = np.sort(lg_tr)
                    trainpct = np.searchsorted(lg_tr_sorted, logits) / len(lg_tr)
                    tm = tier_metrics(tiers, y_ev)
                    tag = label.replace("label_is_", "")
                    yres["tiers"][tag] = {
                        "thresholds": thr,
                        "threshold_degenerate": bool(thr["high"] < thr["mid"]),
                        **tm,
                        "pos_trainpct_median": round(
                            float(np.median(trainpct[y_ev == 1])), 4),
                        "tier_table": pd.crosstab(
                            pd.Series(tiers, name="tier"),
                            pd.Series(y_ev, name="y")).to_dict()}
                    print(f"  分层: 高={tm['high']} 中+={tm['mid_plus']} "
                          f"阳性train分位中位={yres['tiers'][tag]['pos_trainpct_median']}",
                          flush=True)

                # 保存 L3 jump_human 排名表
                if lname == "L3" and label == "label_is_jump_human":
                    rk = ev_df.assign(
                        logit_L3=logits, naive=naive_sc, rank=ranks,
                        causal_label=y_ev_pre,
                        first_jump_cluster=((y_ev == 1) & (y_ev_pre == 0)).astype(int))
                    rk[["accession", "subtype", "strain_name", "host_category",
                        "country", "collection_date", "cluster_id",
                        "label_is_jump", "label_is_jump_human", "causal_label",
                        "first_jump_cluster", "logit_L3", "naive", "rank"]
                       ].sort_values("rank").to_csv(
                        OUT_DIR / f"forward_{Y}_ranking.csv", index=False)

        results["years"][str(Y)] = yres

    with open(OUT_DIR / "forward_multiyear.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n→ {OUT_DIR / 'forward_multiyear.json'} + forward_{{2024,2025}}_ranking.csv")


if __name__ == "__main__":
    main()
