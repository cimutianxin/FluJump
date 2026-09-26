#!/usr/bin/env python3
"""2026 前瞻集：workflow 风险分层验证（TODO-8 评估半）

对 201 条 2026 分离株跑 mainpipeline/risk_scorer 的分层（emb 路径，复用
emb_2026_L3.npy），阈值全部来自 ≤2025 train logits（risk_scorer 内预注册）。
对照：朴素基线（与 H1+H3 train 人源参考的 max identity，自匹配剔除）同规则分层。

功效限制：阳性 10 条 = 2 簇（H5 cluster70 ×9 + H1 cluster15 ×1），报分层命中率
与 precision/recall/FPR，附簇级说明，不作分类性能声称。

输出：output/workflow_2026_eval.json + output/workflow_2026_tiers.csv
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from mainpipeline.risk_scorer import RiskScorer
from validation_exp.temporal_validation.config import (
    CLEAN_2026_CSV, ALIGNED_CSV, SPLIT_CSV, LABEL_COLS, NAIVE_REF_HOST,
    RANDOM_SEED, OUT_DIR,
)

ALIGNED_2026_CSV = "data/processed_isolate_MAFFT/2026_isolates_aligned.csv"
RANKING_CSV = OUT_DIR / "forward_2026_ranking.csv"
GAP = ord("-")
TIER_ORD = {"低": 0, "中": 1, "高": 2}


def encode(seqs):
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    return arr.reshape(len(seqs), -1)


def max_identity(Q, R, q_idx=None, r_idx=None, chunk_q=256, chunk_r=2048):
    """query × ref 的最大双非gap位 identity；q_idx/r_idx 相同位置对（自匹配）剔除"""
    out = np.empty(len(Q), dtype=np.float32)
    for i0 in range(0, len(Q), chunk_q):
        q = Q[i0:i0 + chunk_q]
        best = np.full(len(q), -1.0, dtype=np.float32)
        for j0 in range(0, len(R), chunk_r):
            r = R[j0:j0 + chunk_r]
            both = (q[:, None, :] != GAP) & (r[None, :, :] != GAP)
            m = ((q[:, None, :] == r[None, :, :]) & both).sum(2) / np.maximum(
                both.sum(2), 1)
            if q_idx is not None:          # 剔除自匹配
                same = q_idx[i0:i0 + len(q), None] == r_idx[None, j0:j0 + len(r)]
                m = np.where(same, -1.0, m)
            best = np.maximum(best, m.max(1))
        out[i0:i0 + len(q)] = best
    return out


def tier_metrics(tiers, y):
    """tiers: 低/中/高 数组；y: 0/1。返回高/中及以上 两档的命中指标"""
    res = {}
    for name, mask in [("high", np.array([t == "高" for t in tiers])),
                       ("mid_plus", np.array([t != "低" for t in tiers]))]:
        tp = int((mask & (y == 1)).sum())
        res[name] = {"n": int(mask.sum()), "tp": tp,
                     "precision": round(tp / mask.sum(), 4) if mask.sum() else None,
                     "recall": round(tp / max(y.sum(), 1), 4),
                     "fpr": round(float((mask & (y == 0)).sum()
                                        / max((y == 0).sum(), 1)), 4)}
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scorer = RiskScorer(load_model=False)      # probe + 阈值（仅 train 定界）

    # ── 2026 打分（emb 路径）──
    c26 = pd.read_csv(CLEAN_2026_CSV)
    E26 = np.load(OUT_DIR / "emb_2026_L3.npy")
    assert len(E26) == len(c26) == 201
    recs = [scorer.score_embedding(E26[i], r["subtype"])
            for i, r in c26.iterrows()]
    s26 = pd.DataFrame(recs)

    # 与 forward_2026 同协议 logits 交叉核对（jump_human）
    old = pd.read_csv(RANKING_CSV)[["accession", "_logit"]]
    chk = c26[["accession"]].merge(old, on="accession").merge(
        s26[["logit_jump_human"]], left_index=True, right_index=True)
    rho = spearmanr(chk["_logit"], chk["logit_jump_human"]).statistic
    dmax = float(np.abs(chk["_logit"] - chk["logit_jump_human"]).max())
    assert rho > 0.9999 and dmax < 1e-2, f"与 forward_2026 logits 不一致: ρ={rho}, Δmax={dmax}"
    print(f"logits 交叉核对: ρ={rho:.6f} Δmax={dmax:.2e}", flush=True)

    # ── 朴素基线：train 定界（自匹配剔除）+ 2026 打分 ──
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=["accession", "subtype",
                                            "aligned_ha_seq", "host_category"])
    df = split.merge(ali, on=["accession", "subtype"], validate="one_to_one")
    tr = (df["split"] == "train").to_numpy()
    ref_mask = tr & (df["host_category"] == NAIVE_REF_HOST).to_numpy()
    R = encode(df.loc[ref_mask, "aligned_ha_seq"].tolist())
    tr_idx = np.where(tr)[0]
    Qtr = encode(df.loc[tr, "aligned_ha_seq"].tolist())
    print(f"朴素基线: train 查询 {len(Qtr)} × 人源参考 {len(R)}", flush=True)
    naive_tr = max_identity(Qtr, R, q_idx=tr_idx,
                            r_idx=np.where(ref_mask)[0])
    a26 = pd.read_csv(ALIGNED_2026_CSV, usecols=["accession", "subtype",
                                                 "aligned_ha_seq"])
    c26chk = c26.merge(a26, on=["accession", "subtype"], validate="one_to_one")
    naive26 = max_identity(encode(c26chk["aligned_ha_seq"].tolist()), R)

    # naive 阈值（同规则：高=train 阳性中位，中=train 阴性 P95）
    y_tr = {l: np.load(f"ESM_clf/jump_exp/output/labels_{l}.npy")[tr]
            for l in LABEL_COLS}
    naive_thr = {l: {"high": float(np.median(naive_tr[y_tr[l] == 1])),
                     "mid": float(np.quantile(naive_tr[y_tr[l] == 0], 0.95))}
                 for l in LABEL_COLS}
    for l in LABEL_COLS:
        t = naive_thr[l]
        s26[f"naive_tier_{l.replace('label_is_', '')}"] = [
            "高" if v >= t["high"] else ("中" if v >= t["mid"] else "低")
            for v in naive26]
    s26["naive_score"] = naive26

    # ── 指标汇总 ──
    results = {"n_2026": 201,
               "positives": "10 条 = 2 簇（H5 cluster70 ×9 + H1 cluster15 ×1）",
               "probe_thresholds": scorer.thresholds,
               "naive_thresholds": naive_thr,
               "logit_crosscheck": {"rho": round(float(rho), 6),
                                    "max_abs_diff": dmax}}
    for label in LABEL_COLS:
        tag = label.replace("label_is_", "")
        y = c26[label].to_numpy()
        pt = s26[f"tier_{tag}"].to_numpy()
        nt = s26[f"naive_tier_{tag}"].to_numpy()
        # 簇级命中：阳性簇的最佳分层
        tmp = pd.DataFrame({"cluster": c26["cluster_id"], "y": y,
                            "tier": [TIER_ORD[t] for t in pt]})
        cl_best = (tmp[y == 1].groupby("cluster")["tier"].max()
                   .map({v: k for k, v in TIER_ORD.items()}).to_dict())
        results[tag] = {
            "probe": {"tier_table": pd.crosstab(pd.Series(pt, name="tier"),
                                                pd.Series(y, name="y")).to_dict(),
                      **{k: v for k, v in
                         zip(["high", "mid_plus"],
                             [tier_metrics(pt, y)["high"],
                              tier_metrics(pt, y)["mid_plus"]])},
                      "pos_trainpct_median": round(float(np.median(
                          s26.loc[y == 1, f"trainpct_{tag}"])), 4),
                      "pos_cluster_best_tier": {str(k): v for k, v in cl_best.items()}},
            "naive": {"tier_table": pd.crosstab(pd.Series(nt, name="tier"),
                                                pd.Series(y, name="y")).to_dict(),
                      **tier_metrics(nt, y)},
        }
        print(f"[{tag}] probe 高命中={results[tag]['probe']['high']} | "
              f"naive 高命中={results[tag]['naive']['high']}", flush=True)

    out = pd.concat([c26[["accession", "subtype", "strain_name", "host_category",
                          "cluster_id"] + LABEL_COLS].reset_index(drop=True),
                     s26], axis=1)
    out.to_csv(OUT_DIR / "workflow_2026_tiers.csv", index=False)
    with open(OUT_DIR / "workflow_2026_eval.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"→ {OUT_DIR}/workflow_2026_eval.json / workflow_2026_tiers.csv",
          flush=True)


if __name__ == "__main__":
    main()
