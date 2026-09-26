#!/usr/bin/env python3
"""Part B：定点饱和突变（in silico mutagenesis）→ probe logit 变化

设计：
  - 背景序列：H1+H3 train 100 条（jump_human+/− 各 50，cluster 去重分层抽样）
    + H5 holdout 10 条（5+/5−，检验位点效应的跨亚型一致性）
  - 位点面板：~18 个已知标记位点（H3 canonical 编号，MARKER_SITES 并集）
    + 20 个对照位点（按列熵匹配、排除功能位与高缺失列）
  - 每位点 19 种氨基酸饱和突变 → 每条突变序列过 ESM-2（L28 mean-pooled）
    → Δlogit = logit(mut) − logit(wt)，用 Part A 保存的 probe（w + scaler）

分析：
  1) 已知人适应替换（KNOWN_HUMAN_SUBS）Δlogit 方向性（Wilcoxon vs 0）
  2) 标记位点组 vs 对照位点组的 |Δlogit| 比较（Mann-Whitney）
  3) H5 背景上的位点效应与 H1+H3 背景的相关性

输出：output/ism_deltalogit.csv（逐突变）、output/ism_summary.json（汇总检验）
"""

import json
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, wilcoxon
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from validation_exp.site_attribution.config import (
    ALIGNED_CSV, SPLIT_CSV, MARKER_SITES, KNOWN_HUMAN_SUBS, ESM_MODEL,
    LABEL_COLS, ISM_N_BACKGROUND, ISM_N_CONTROL, ISM_BATCH, RANDOM_SEED, OUT_DIR,
)

AAS = "ACDEFGHIKLMNPQRSTVWY"
# 功能位排除区（canonical HA1 区间），对照位点不得落入
EXCLUDE_RANGES = [(130, 140), (185, 200), (218, 232), (156, 160), (320, 345)]


def col_positions(aln_seq):
    return np.fromiter((i for i, c in enumerate(aln_seq) if c != "-"),
                       dtype=np.int32, count=len(aln_seq) - aln_seq.count("-"))


def col_entropy(strings, col):
    c = Counter(s[col] for s in strings if s[col] != "-")
    n = sum(c.values())
    return -sum(v / n * np.log2(v / n) for v in c.values()) if n else 0.0


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── 映射与数据 ──
    with open(OUT_DIR / "col_to_h3.json") as f:
        mp = json.load(f)
    h1_to_col = {v: int(k) for k, v in mp["col_to_h3_ha1"].items()}

    df = pd.read_csv(ALIGNED_CSV, usecols=[
        "accession", "subtype", "ha_sequence", "aligned_ha_seq", "cluster_id",
        "label_is_jump_human"])
    split = pd.read_csv(SPLIT_CSV, usecols=["accession", "subtype", "split"])
    df = df.merge(split, on=["accession", "subtype"], validate="one_to_one")

    # ── 位点面板 ──
    marker_sites = sorted({s for sites in MARKER_SITES.values() for s in sites})
    marker_cols = {s: h1_to_col[s] for s in marker_sites if s in h1_to_col}
    print(f"标记位点 {len(marker_cols)} 个: {sorted(marker_cols)}")

    train_h13 = df[(df["split"] == "train")]
    train_strs = train_h13["aligned_ha_seq"].tolist()
    marker_set = set(marker_cols.values())
    cand_cols = []
    for col, h1 in ((int(k), v) for k, v in mp["col_to_h3_ha1"].items()):
        if col in marker_set:
            continue
        if any(a <= h1 <= b for a, b in EXCLUDE_RANGES):
            continue
        gap_frac = sum(1 for s in train_strs if s[col] == "-") / len(train_strs)
        if gap_frac > 0.05:
            continue
        cand_cols.append((col, col_entropy(train_strs, col)))
    # 对照位点：为每个标记位点找列熵最接近的候选，去重取前 ISM_N_CONTROL
    ctrl_cols, used = [], set()
    for s, mcol in sorted(marker_cols.items()):
        ent = col_entropy(train_strs, mcol)
        best = min(((c, abs(e - ent)) for c, e in cand_cols if c not in used),
                   key=lambda x: x[1], default=None)
        if best:
            ctrl_cols.append(best[0])
            used.add(best[0])
    ctrl_cols = ctrl_cols[:ISM_N_CONTROL]
    print(f"对照位点 {len(ctrl_cols)} 个（列熵匹配）")
    panel = ([(s, c, "marker") for s, c in sorted(marker_cols.items())]
             + [(None, c, "control") for c in ctrl_cols])

    # ── 背景序列（cluster 去重 + 分层）──
    def pick(mask_df, n_pos, n_neg):
        dedup = mask_df.drop_duplicates("cluster_id")
        pos = dedup[dedup["label_is_jump_human"] == 1]
        neg = dedup[dedup["label_is_jump_human"] == 0]
        return pd.concat([
            pos.sample(min(n_pos, len(pos)), random_state=RANDOM_SEED),
            neg.sample(min(n_neg, len(neg)), random_state=RANDOM_SEED)])

    bg_h13 = pick(train_h13, ISM_N_BACKGROUND // 2, ISM_N_BACKGROUND // 2)
    bg_h5 = pick(df[df["split"] == "h5_holdout"], 5, 5)
    backgrounds = pd.concat([bg_h13, bg_h5]).reset_index(drop=True)
    print(f"背景序列: H1+H3 {len(bg_h13)} + H5 {len(bg_h5)}")

    # ── 生成突变面板 ──
    records = []   # (bg_idx, site, col, group, wt, mut, mut_seq)
    for bi, row in backgrounds.iterrows():
        raw = row["ha_sequence"]
        cols = col_positions(row["aligned_ha_seq"])   # raw 索引 → 对齐列
        col2raw = {c: i for i, c in enumerate(cols)}
        for site, col, group in panel:
            ri = col2raw.get(col)
            if ri is None:
                continue                      # 该序列在此列为 gap
            wt = raw[ri]
            for aa in AAS:
                if aa != wt:
                    records.append((bi, site, col, group, wt, aa,
                                    raw[:ri] + aa + raw[ri + 1:]))
    mut_df = pd.DataFrame(records, columns=[
        "bg_idx", "site", "aln_col", "group", "wt", "mut", "mut_seq"])
    print(f"突变序列总数: {len(mut_df)}")

    # ── probe 参数（Part A 保存）──
    probes = {}
    for label in LABEL_COLS:
        z = np.load(OUT_DIR / f"probe_{label}.npz")
        probes[label] = (z["w"], z["scaler_mean"], z["scaler_scale"])

    # ── ESM 提取（L28 = 倒数第三层 mean-pooled）──
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True).to(device).eval()

    def embed(seqs):
        embs = []
        for i in tqdm(range(0, len(seqs), ISM_BATCH), desc="ESM 提取"):
            batch = seqs[i:i + ISM_BATCH]
            inp = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**inp)
            h = out.hidden_states[-3]           # L28
            for j in range(len(batch)):
                m = inp["attention_mask"][j, 1:-1].bool()
                embs.append(h[j, 1:-1][m].mean(dim=0).cpu().numpy())
        return np.stack(embs).astype(np.float32)

    emb_wt = embed(backgrounds["ha_sequence"].tolist())
    emb_mut = embed(mut_df["mut_seq"].tolist())

    # ── Δlogit ──
    for label in LABEL_COLS:
        w, mu, sd = probes[label]
        z_wt = (emb_wt - mu) / sd @ w
        z_mut = (emb_mut - mu) / sd @ w
        mut_df[f"dlogit_{label}"] = z_mut - z_wt[mut_df["bg_idx"].to_numpy()]
    mut_df.to_csv(OUT_DIR / "ism_deltalogit.csv", index=False)

    # ── 分析 ──
    summary = {"n_mutants": len(mut_df), "n_backgrounds": len(backgrounds),
               "n_marker_sites": len(marker_cols), "n_control_sites": len(ctrl_cols),
               "labels": {}}
    for label in LABEL_COLS:
        d = f"dlogit_{label}"
        h13 = mut_df[mut_df["bg_idx"] < len(bg_h13)]      # H1+H3 背景
        # 1) 已知人适应替换方向性（限 wt 匹配的背景）
        ks = []
        for site, wt, mut in KNOWN_HUMAN_SUBS:
            sub = h13[(h13["site"] == site) & (h13["wt"] == wt) & (h13["mut"] == mut)]
            if len(sub) >= 3:
                med = float(sub[d].median())
                p = float(wilcoxon(sub[d]).pvalue) if sub[d].std() > 0 else 1.0
                ks.append({"sub": f"{wt}{site}{mut}", "n": len(sub),
                           "median_dlogit": round(med, 4),
                           "frac_positive": round(float((sub[d] > 0).mean()), 3),
                           "wilcoxon_p": round(p, 5)})
        # 2) 标记组 vs 对照组 |Δlogit|
        g_m = np.abs(h13[h13["group"] == "marker"][d])
        g_c = np.abs(h13[h13["group"] == "control"][d])
        u_stat, u_p = mannwhitneyu(g_m, g_c, alternative="greater")
        # 3) H5 背景：每位点中位 |Δlogit| 与 H1+H3 的相关性
        h5 = mut_df[mut_df["bg_idx"] >= len(bg_h13)]
        site_eff_h13 = h13.groupby("site")[d].apply(lambda x: x.abs().median())
        site_eff_h5 = h5.dropna(subset=["site"]).groupby("site")[d].apply(
            lambda x: x.abs().median())
        common = site_eff_h13.index.intersection(site_eff_h5.index)
        from scipy.stats import spearmanr
        rho, rho_p = spearmanr(site_eff_h13[common], site_eff_h5[common])
        summary["labels"][label] = {
            "known_human_subs": ks,
            "marker_vs_control_absmed": [round(float(g_m.median()), 4),
                                         round(float(g_c.median()), 4)],
            "mannwhitney_p_marker_greater": float(f"{u_p:.4g}"),
            "h5_site_effect_spearman": [round(float(rho), 3), float(f"{rho_p:.4g}")],
        }
        print(f"\n[{label}] 已知替换:")
        for k in ks:
            print(f"  {k['sub']}: n={k['n']} median={k['median_dlogit']} "
                  f"frac+={k['frac_positive']} p={k['wilcoxon_p']}")
        print(f"  标记组|Δ|中位={g_m.median():.4f} vs 对照组={g_c.median():.4f} "
              f"(Mann-Whitney p={u_p:.4g})")
        print(f"  H5 位点效应 Spearman ρ={rho:.3f} (p={rho_p:.4g})")

    with open(OUT_DIR / "ism_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n→ {OUT_DIR / 'ism_summary.json'}（耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
