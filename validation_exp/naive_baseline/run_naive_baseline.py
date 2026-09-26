#!/usr/bin/env python3
"""朴素生物学基线：与 H1+H3 人源参考株的全局序列相似度排序

打分（对每条 query 序列）：
  - max_id_human  : 与 H1+H3 train 人源参考集的最大 pairwise identity（主基线）
  - topk_id_human : top-k identity 均值（稳健性变体）
  - diff_id       : max_id_human − max identity to 非人源参考集（双向对照变体）

identity 在 MAFFT 对齐序列（H3 参考株统一坐标）上按双非 gap 位 Hamming 计算。
参考集只用 H1+H3 train split，与 probe 训练信息对齐，避免 H5/H7 人源序列泄漏标签。
评估：test（sanity check）/ h5_holdout / h7_holdout × 两个标签，cluster 级 bootstrap CI。
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from validation_exp.naive_baseline.config import (
    ALIGNED_CSV, SPLIT_CSV, LABEL_COLS, TOPK, N_BOOTSTRAP, RANDOM_SEED,
    ESM_REFERENCE, OUT_DIR,
)

GAP = ord("-")


def encode(seqs):
    """对齐序列（等长）→ uint8 矩阵 (n, L)"""
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    return arr.reshape(len(seqs), -1)


def pairwise_identity(Q, R, chunk_q=256, chunk_r=2048):
    """Q (nq,L) × R (nr,L) → identity 矩阵 (nq,nr)；identity=双非gap位匹配比例"""
    nq, nr = len(Q), len(R)
    out = np.empty((nq, nr), dtype=np.float32)
    for i0 in range(0, nq, chunk_q):
        q = Q[i0:i0 + chunk_q]
        q_valid = q != GAP
        for j0 in range(0, nr, chunk_r):
            r = R[j0:j0 + chunk_r]
            r_valid = r != GAP
            both_valid = q_valid[:, None, :] & r_valid[None, :, :]
            matches = (q[:, None, :] == r[None, :, :]) & both_valid
            denom = both_valid.sum(axis=2)
            out[i0:i0 + len(q), j0:j0 + len(r)] = matches.sum(axis=2) / np.maximum(denom, 1)
    return out


def bootstrap_auc(y, score, cluster_ids, rng, n_boot=N_BOOTSTRAP):
    """cluster 级有放回 bootstrap → (mean, lo, hi, P(AUC>0.5))；按原始打分方向"""
    clusters = np.unique(cluster_ids)
    idx_by_cluster = {c: np.where(cluster_ids == c)[0] for c in clusters}
    aucs = []
    for _ in range(n_boot):
        sample = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in sample])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], score[idx]))
    aucs = np.asarray(aucs)
    return float(aucs.mean()), float(np.percentile(aucs, 2.5)), \
        float(np.percentile(aucs, 97.5)), float((aucs > 0.5).mean())


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 数据加载与 join（按 accession+subtype，严禁按行号）──
    print("加载数据...")
    ali = pd.read_csv(ALIGNED_CSV, usecols=[
        "accession", "subtype", "aligned_ha_seq", "host_category",
        "label_is_jump", "label_is_jump_human", "cluster_id"])
    split = pd.read_csv(SPLIT_CSV, usecols=["accession", "subtype", "split"])
    df = ali.merge(split, on=["accession", "subtype"], how="inner", validate="one_to_one")
    print(f"  join 后行数: {len(df)}（aligned {len(ali)} / split {len(split)}）")
    assert df["aligned_ha_seq"].notna().all()
    lens = df["aligned_ha_seq"].str.len()
    assert lens.nunique() == 1, f"对齐序列长度不一致: {lens.unique()}"
    print(f"  对齐长度: {lens.iloc[0]}")

    # ── 参考集：H1+H3 train，分 human / non-human ──
    train_mask = df["split"] == "train"
    ref_human = df[train_mask & (df["host_category"] == "human")]
    ref_nonhuman = df[train_mask & (df["host_category"] != "human")]
    print(f"  参考集: human {len(ref_human)} / non-human {len(ref_nonhuman)}")
    R_h = encode(ref_human["aligned_ha_seq"].tolist())
    R_n = encode(ref_nonhuman["aligned_ha_seq"].tolist())

    # ── 打分 ──
    Q = encode(df["aligned_ha_seq"].tolist())
    print(f"计算 identity（query {len(Q)} × ref_human {len(R_h)}）...")
    id_h = pairwise_identity(Q, R_h)
    print(f"计算 identity（query {len(Q)} × ref_nonhuman {len(R_n)}）...")
    id_n = pairwise_identity(Q, R_n)

    part_h = np.partition(id_h, len(R_h) - TOPK, axis=1)[:, -TOPK:]
    scores = {
        "max_id_human": id_h.max(axis=1),
        "topk_id_human": part_h.mean(axis=1),
        "diff_id": id_h.max(axis=1) - id_n.max(axis=1),
    }

    # ── 评估 ──
    rng = np.random.default_rng(RANDOM_SEED)
    results = {}
    for split_name in ["test", "h5_holdout", "h7_holdout"]:
        m = (df["split"] == split_name).to_numpy()
        n, clusters = int(m.sum()), df.loc[m, "cluster_id"].to_numpy()
        results[split_name] = {"n": n, "labels": {}, "identity_stats": {}}
        print(f"\n=== {split_name}（n={n}, clusters={len(np.unique(clusters))}）===")
        idh = id_h.max(axis=1)[m]
        results[split_name]["identity_stats"]["max_id_human_quantiles"] = {
            q: round(float(np.quantile(idh, q)), 4) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
        for label in LABEL_COLS:
            y = df.loc[m, label].to_numpy()
            results[split_name]["labels"][label] = {
                "positive_rate": round(float(y.mean()), 4), "scores": {}}
            for sname, s in scores.items():
                sv = s[m]
                auc = roc_auc_score(y, sv)
                auc_flip = roc_auc_score(y, -sv)
                mean, lo, hi, p = bootstrap_auc(y, sv, clusters, rng)
                rec = {"auc": round(float(auc), 4), "auc_flipped": round(float(auc_flip), 4),
                       "bootstrap_mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
                       "p_auc_gt_0.5": round(p, 4)}
                ref = ESM_REFERENCE.get((split_name, label))
                if ref is not None:
                    rec["esm_reference"] = ref
                results[split_name]["labels"][label]["scores"][sname] = rec
                print(f"  {label} / {sname}: AUC={auc:.4f} (翻转 {auc_flip:.4f}) "
                      f"boot={mean:.4f} [{lo:.4f},{hi:.4f}] P(>0.5)={p:.3f}"
                      + (f" | ESM参照 {ref}" if ref else ""))

    payload = {
        "config": {"topk": TOPK, "n_bootstrap": N_BOOTSTRAP, "seed": RANDOM_SEED,
                   "ref": "H1+H3 train split, human vs non-human",
                   "identity": "MAFFT(H3 ref) 双非gap位匹配比例"},
        "results": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = OUT_DIR / "naive_baseline_results.json"
    with open(out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {out}（耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
