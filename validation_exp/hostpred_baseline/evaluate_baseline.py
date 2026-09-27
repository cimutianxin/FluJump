"""VirHostPRED 基线正式评估（步骤 2）。

口径固定（预注册，见任务书与 notes/09-28-00）：
  - 主分数 = p_human（原始概率直接当打分，无符号翻转、无变换、无事后择优）；
  - join 键 (accession, subtype)，簇键 (subtype, cluster_id)；
  - cluster bootstrap / permutation 逐行复刻 run_multiyear_forward.py（B=10000, seed 42）；
  - 簇聚合：score=簇内 mean、label=簇内 max（与参考实现一致）；
  - 分层阈值仅用基线自身分数在 train ≤cutoff 池上定（高=阳性中位、中=阴性 P95）。

格子：
  (a) 描述性：p_human vs host_category==human 的 isolate AUC（H5/H7 holdout +
      三个前向池分别算）——记录工具实际测了什么；
  (b) H5/H7 四格（jump/jump_human × H5/H7，全量 holdout 已打分行）：
      isolate AUC + cluster AUC + 95% CI + 置换 p；ΔAUC vs probe（target-val
      选中层 5 seeds；配对 bootstrap 用 seed42 层，与 target_val 脚本 bootstrap
      口径一致）；
  (c) 前向 2024/2025/2026（headline 簇继承标签）：cluster AUC + CI + 置换 p、
      isolate AUC、阳性名次（2026 isolate 级；2024/2025 阳性簇簇级名次）、
      风险分层；另报 ≤cutoff 因果口径 cluster AUC 对齐 forward_multiyear 字段；
  (d) ΔAUC：基线 vs probe L3 logits（probe_scores_forward.csv，已与 ranking
      CSV 对账）配对 cluster bootstrap（同一批重采样簇），报 Δ 95% CI。

输出 output/hostpred_baseline_{h5h7,forward}.json。
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from validation_exp.temporal_validation.config import (
    ALIGNED_CSV, SPLIT_CSV, CLEAN_2026_CSV, ALL_ISOLATES_CSV,
    LABEL_COLS, RANDOM_SEED, B_BOOT, B_PERM,
)

OUT = "validation_exp/hostpred_baseline/output"


# ── 逐行复刻 run_multiyear_forward.py 的统计实现 ──

def cluster_boot(sc, y, rng, B=B_BOOT):
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
    obs = float(roc_auc_score(y, sc))
    cnt = sum(float(roc_auc_score(rng.permutation(y), sc) >= obs) for _ in range(B))
    return obs, cnt / B


def paired_boot_delta(sc_a, sc_b, y, rng, B=B_BOOT):
    """配对 cluster bootstrap：同一批重采样簇上 Δ=AUC_a−AUC_b。"""
    ds = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        ds.append(float(roc_auc_score(y[idx], sc_a[idx])
                        - roc_auc_score(y[idx], sc_b[idx])))
    d = np.array(ds)
    return {"B_valid": len(d), "delta_mean": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_delta_gt_0": float((d > 0).mean()),
            "p_delta_lt_0": float((d < 0).mean())}


def labels_upto(alliso, cutoff):
    yr = alliso["year"]
    m = yr.notna() & (yr > 0) & (yr <= cutoff) & (alliso["host_category"] != "unknown")
    hosts = alliso[m].groupby(["subtype", "cluster_id"])["host_category"].agg(set)
    return {k: {"label_is_jump": int(len(v) >= 2),
                "label_is_jump_human": int("human" in v and len(v) >= 2)}
            for k, v in hosts.items()}


def tier_metrics(tiers, y):
    res = {}
    for name, mask in [("high", tiers == "高"), ("mid_plus", tiers != "低")]:
        mask = np.asarray(mask)
        tp = int((mask & (y == 1)).sum())
        res[name] = {"n": int(mask.sum()), "tp": tp,
                     "precision": round(float(tp / mask.sum()), 4)
                     if mask.sum() else None,
                     "recall": round(float(tp / max(y.sum(), 1)), 4),
                     "fpr": round(float((mask & (y == 0)).sum())
                                  / max((y == 0).sum(), 1), 4)}
    return res


def cluster_agg(df, score_col, label_col):
    cg = df.groupby(["subtype", "cluster_id"]).agg(
        score=(score_col, "mean"), label=(label_col, "max"), n=(label_col, "size"))
    return cg["score"].to_numpy(), cg["label"].to_numpy(), cg


def cell_stats(ev_df, score_col, label, rng):
    """一个格子的全套指标：isolate AUC + cluster AUC/CI/perm + 阳性名次。"""
    y = ev_df[label].to_numpy()
    sc = ev_df[score_col].to_numpy()
    res = {"n_scored": int(len(ev_df)),
           "n_pos": int(y.sum()),
           "auc_isolate": round(float(roc_auc_score(y, sc)), 4) if y.sum() else None}
    sc_cl, y_cl, cg = cluster_agg(ev_df, score_col, label)
    res["n_eval_clusters"] = int(len(cg))
    res["n_pos_clusters"] = int(y_cl.sum())
    if y_cl.sum():
        res["auc_cluster"] = round(float(roc_auc_score(y_cl, sc_cl)), 4)
        res["cluster_bootstrap"] = cluster_boot(sc_cl, y_cl, rng)
        _, res["cluster_permutation_p"] = cluster_perm_p(sc_cl, y_cl, rng)
    else:
        res["auc_cluster"] = None
        res["cluster_bootstrap"] = None
        res["cluster_permutation_p"] = None
    order = np.argsort(-sc)
    ranks = np.empty(len(sc), dtype=int)
    ranks[order] = np.arange(1, len(sc) + 1)
    res["positive_ranks"] = sorted(int(ranks[i]) for i in np.where(y == 1)[0])
    cranks = cg["score"].rank(ascending=False).astype(int)
    res["positive_cluster_ranks"] = {
        f"{st}/{cid}": int(cranks.loc[(st, cid)])
        for st, cid in cg.index[cg["label"] == 1]}
    return res, cg


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    scores = pd.read_csv(f"{OUT}/scores_all.csv")
    assert scores["p_human"].notna().all()
    scores = scores[["accession", "subtype", "p_human"]]

    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=[
        "accession", "subtype", "cluster_id", "host_category", "collection_year",
        "label_is_jump", "label_is_jump_human"])
    df = split.drop(columns=["cluster_id"]).merge(
        ali, on=["accession", "subtype"], validate="one_to_one")
    df["year"] = pd.to_numeric(df["collection_year"], errors="coerce")
    base = df.merge(scores, on=["accession", "subtype"], how="left",
                    validate="one_to_one")

    alliso = pd.read_csv(ALL_ISOLATES_CSV, usecols=[
        "subtype", "cluster_id", "host_category", "collection_year"])
    alliso["year"] = pd.to_numeric(alliso["collection_year"], errors="coerce")

    skipped = pd.read_csv(f"{OUT}/skipped_nonstd.csv")
    proto = {"score": "p_human 原始概率直读（无变换/无翻转）",
             "cluster_agg": "score=mean, label=max, 键 (subtype, cluster_id)",
             "B_boot": B_BOOT, "B_perm": B_PERM, "seed": RANDOM_SEED,
             "n_skipped_nonstd": int(len(skipped)),
             "leakage_note": "VirHostPRED 训练集为 RefSeq 人源病毒蛋白（几乎必然"
                             "含人流感 HA），序列未公开无法 accession 级核查；"
                             "本基线数字为泄漏方向的上界。"}

    # ═══ H5/H7 四格 + 描述性口径 ═══
    probe57 = pd.read_csv(f"{OUT}/probe_scores_h5h7.csv")
    h57 = {"descriptive_host_human_auc": {}, "cells": {}, "delta_vs_probe": {}}
    for st, split_name in [("H5", "h5_holdout"), ("H7", "h7_holdout")]:
        ev = base[base["split"] == split_name]
        n_tot = len(ev)
        evs = ev[ev["p_human"].notna()]
        h57["descriptive_host_human_auc"][st] = {
            "n_total": int(n_tot), "n_scored": int(len(evs)),
            "auc": round(float(roc_auc_score(
                (evs["host_category"] == "human").astype(int),
                evs["p_human"])), 4)}
        for label in LABEL_COLS:
            res, cg = cell_stats(evs, "p_human", label, rng)
            res["n_total_pool"] = int(n_tot)
            h57["cells"].setdefault(label, {})[st] = res
            # probe：5 seeds 选中层；pair 用 seed42
            pr = probe57[(probe57["label"] == label)]
            pr = pr[pr["subtype"] == st]
            aucs_seeds = []
            for seed, g in pr.groupby("seed"):
                # 簇级聚合 probe logit（与基线同簇键、同 groupby 序）
                gm = evs.merge(g[["accession", "subtype", "logit"]],
                               on=["accession", "subtype"], how="inner")
                sc_p, _, _ = cluster_agg(gm, "logit", label)
                y_cl = cg["label"].to_numpy()
                aucs_seeds.append(float(roc_auc_score(y_cl, sc_p)))
            y_cl = cg["label"].to_numpy()
            gm42 = evs.merge(
                pr[pr["seed"] == 42][["accession", "subtype", "logit"]],
                on=["accession", "subtype"], how="inner")
            sc_p42, y_p42, _ = cluster_agg(gm42, "logit", label)
            assert (y_p42 == y_cl).all()
            delta = paired_boot_delta(cg["score"].to_numpy(), sc_p42, y_cl, rng)
            h57["delta_vs_probe"].setdefault(label, {})[st] = {
                "probe_cluster_auc_seed42_layer": round(
                    float(roc_auc_score(y_cl, sc_p42)), 4),
                "probe_cluster_auc_5seeds_mean": round(float(np.mean(aucs_seeds)), 4),
                "probe_cluster_auc_5seeds_std": round(float(np.std(aucs_seeds)), 4),
                "baseline_minus_probe": delta}
            print(f"[h5h7][{label}][{st}] 基线 cluster "
                  f"{h57['cells'][label][st]['auc_cluster']} vs probe "
                  f"{h57['delta_vs_probe'][label][st]['probe_cluster_auc_5seeds_mean']}"
                  f" ΔCI {delta['ci95']}", flush=True)

    with open(f"{OUT}/hostpred_baseline_h5h7.json", "w") as f:
        json.dump({"protocol": proto, **h57}, f, ensure_ascii=False, indent=2)

    # ═══ 前向三年 ═══
    probe_fwd = pd.read_csv(f"{OUT}/probe_scores_forward.csv")
    c26 = pd.read_csv(CLEAN_2026_CSV)
    c26s = c26.merge(scores, on=["accession", "subtype"], how="left",
                     validate="one_to_one")
    fwd = {"descriptive_host_human_auc": {}, "years": {}}
    YEARS = {2024: (base[base["year"] == 2024], 2023),
             2025: (base[base["year"] == 2025], 2024),
             2026: (c26s, 2025)}
    for Y, (ev, cutoff) in YEARS.items():
        n_tot = len(ev)
        evs = ev[ev["p_human"].notna()].copy()
        fwd["descriptive_host_human_auc"][str(Y)] = {
            "n_total": int(n_tot), "n_scored": int(len(evs)),
            "auc": round(float(roc_auc_score(
                (evs["host_category"] == "human").astype(int),
                evs["p_human"])), 4)}
        pre = labels_upto(alliso, cutoff)
        yres = {"cutoff": cutoff, "n_total": int(n_tot),
                "n_scored": int(len(evs)), "labels": {}, "delta_vs_probe": {}}
        # 分层用 train ≤cutoff 池的基线分数 + 因果标签
        tr_pool = base[(base["split"] == "train") & (base["year"] <= cutoff)]
        tr_pool = tr_pool[tr_pool["p_human"].notna()]
        tr_keys = list(zip(tr_pool["subtype"], tr_pool["cluster_id"]))
        for label in LABEL_COLS:
            res, cg = cell_stats(evs, "p_human", label, rng)
            # 因果口径 cluster AUC
            y_pre = np.array([pre.get(k, {}).get(label, 0) for k in cg.index])
            res["auc_cluster_causal_labels"] = round(
                float(roc_auc_score(y_pre, cg["score"].to_numpy())), 4) \
                if y_pre.sum() else None
            res["n_pos_clusters_causal"] = int(y_pre.sum())
            yres["labels"][label] = res

            # 分层（阈值仅 train ≤cutoff 基线分数；标签因果口径）
            y_tr = np.array([pre.get(k, {}).get(label, 0) for k in tr_keys])
            sc_tr = tr_pool["p_human"].to_numpy()
            thr = {"high": float(np.median(sc_tr[y_tr == 1])),
                   "mid": float(np.quantile(sc_tr[y_tr == 0], 0.95))}
            tiers = np.where(evs["p_human"] >= thr["high"], "高",
                             np.where(evs["p_human"] >= thr["mid"], "中", "低"))
            tm = tier_metrics(tiers, evs[label].to_numpy())
            tr_sorted = np.sort(sc_tr)
            trainpct = np.searchsorted(tr_sorted, evs["p_human"]) / len(sc_tr)
            y_ev = evs[label].to_numpy()
            res["tiers"] = {"thresholds": thr,
                            "threshold_degenerate": bool(thr["high"] < thr["mid"]),
                            **tm,
                            "pos_trainpct_median": round(
                                float(np.median(trainpct[y_ev == 1])), 4)
                            if y_ev.sum() else None,
                            "tier_table": pd.crosstab(
                                pd.Series(tiers, name="tier"),
                                pd.Series(y_ev, name="y")).to_dict()}

            # ΔAUC vs probe（配对 cluster bootstrap）
            pr = probe_fwd[(probe_fwd["year"] == Y) & (probe_fwd["label"] == label)]
            gm = evs.merge(pr[["accession", "subtype", "logit"]],
                           on=["accession", "subtype"], how="inner")
            sc_p, y_p, _ = cluster_agg(gm, "logit", label)
            assert (y_p == cg["label"].to_numpy()).all()
            if cg["label"].sum():
                yres["delta_vs_probe"][label] = {
                    "probe_cluster_auc": round(float(roc_auc_score(
                        y_p, sc_p)), 4),
                    "baseline_minus_probe": paired_boot_delta(
                        cg["score"].to_numpy(), sc_p, y_p, rng)}
            print(f"[fwd {Y}][{label}] 基线 cluster {res['auc_cluster']} "
                  f"CI{res['cluster_bootstrap']['ci95'] if res['cluster_bootstrap'] else None}"
                  f" | Δ vs probe "
                  f"{yres['delta_vs_probe'].get(label, {}).get('baseline_minus_probe', {}).get('ci95')}",
                  flush=True)
        fwd["years"][str(Y)] = yres

    with open(f"{OUT}/hostpred_baseline_forward.json", "w") as f:
        json.dump({"protocol": proto, **fwd}, f, ensure_ascii=False, indent=2)
    print("→ hostpred_baseline_{h5h7,forward}.json")


if __name__ == "__main__":
    main()
