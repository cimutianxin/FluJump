#!/usr/bin/env python3
"""G4：注意力归因（H6）——逐层 attention mass 的功能/诊断位点分布

子集 ~2700 行分层抽样（train H1/H3 ±、H5 ±、H7 ±，按 label_is_jump 分组，
每组上限 400，seed 42）。eager attention 前向（sdpa 不返回 attentions），
逐层（1..30）计算每个残基位置的 received-attention mass（head/query 平均，
CLS/EOS 排除后残基段归一化为分布 p），按 col_positions 聚合到列坐标系。

判定 H6：
  - func_frac_attn(l) = 列聚合 p 在功能位点集上的质量份额（位点集同 G3）；
  - mid(13–22) vs late(28–30) 差，10k 列标签置换双侧 p；
  - 与 G3 的 func_frac_sep(l) 跨层 Spearman（层 1..30），ρ ≥ 0.6 为两模态互证；
  - 逐层 attention 熵（组均值）mid vs late 对比（描述性）。

smoke test：首个 batch 断言 len(attentions)==30 且形状 (B,heads,L,L)。

运行：HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env1 python（GPU）
输出：output/g4_attention.json、figs/g4_attention_{label}.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from validation_exp.depth_reversal.config import (
    SPLIT_CSV, ALIGNED_CSV, LABELS, COL_TO_H3_JSON, G4_COLUMN_SCORES,
    FUNCTIONAL_SITES, DIAG_JS_QUANTILE, MID_LAYERS, LATE_LAYERS, ESM_MODEL,
    G4_BATCH, G4_SUBSET_PER_GROUP, ALN_LEN, RANDOM_SEED, LABEL_COLS, OUT_DIR,
)

N_PERM = 10000
N_ATTN_LAYERS = 30        # attentions 元组长度（层 1..30）


def col_positions(aln_seq):
    return np.fromiter((i for i, c in enumerate(aln_seq) if c != "-"),
                       dtype=np.int32, count=len(aln_seq) - aln_seq.count("-"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figs").mkdir(exist_ok=True)

    # ── 子集抽样（按 label_is_jump 分组；两标签共用同一行子集）──
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV,
                      usecols=["accession", "subtype", "ha_sequence", "aligned_ha_seq"])
    df = split.merge(ali, on=["accession", "subtype"], how="left",
                     validate="one_to_one")
    yj = np.load(LABELS["label_is_jump"])
    sp = split["split"].to_numpy()
    subtype = split["subtype"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    sel_groups = {
        "H1tr+": (sp == "train") & (subtype == "H1") & (yj == 1),
        "H1tr-": (sp == "train") & (subtype == "H1") & (yj == 0),
        "H3tr+": (sp == "train") & (subtype == "H3") & (yj == 1),
        "H3tr-": (sp == "train") & (subtype == "H3") & (yj == 0),
        "H5+": (sp == "h5_holdout") & (yj == 1),
        "H5-": (sp == "h5_holdout") & (yj == 0),
        "H7+": (sp == "h7_holdout") & (yj == 1),
        "H7-": (sp == "h7_holdout") & (yj == 0),
    }
    sel_idx = []
    for g, msk in sel_groups.items():
        idx = np.where(msk)[0]
        if len(idx) > G4_SUBSET_PER_GROUP:
            idx = rng.choice(idx, G4_SUBSET_PER_GROUP, replace=False)
        sel_idx.append(idx)
        print(f"  子集 {g}: {len(idx)}")
    sel_idx = np.sort(np.concatenate(sel_idx))
    n = len(sel_idx)
    print(f"总子集: {n}")
    seqs = df["ha_sequence"].tolist()
    col_maps = [col_positions(s) for s in df["aligned_ha_seq"]]
    ys = {c: np.load(LABELS[c]) for c in LABEL_COLS}

    # ── 聚合数组：(label, layer1..30, col) sum/cnt；组 = label 依赖 ± ──
    grp_names = ["H1H3tr+", "H1H3tr-", "H5+", "H5-", "H7+", "H7-"]
    m_h13tr = (sp == "train")
    m_h5 = sp == "h5_holdout"
    m_h7 = sp == "h7_holdout"
    grp_of = {}
    for c in LABEL_COLS:
        y = ys[c]
        grp_of[c] = {
            "H1H3tr+": m_h13tr & (y == 1), "H1H3tr-": m_h13tr & (y == 0),
            "H5+": m_h5 & (y == 1), "H5-": m_h5 & (y == 0),
            "H7+": m_h7 & (y == 1), "H7-": m_h7 & (y == 0),
        }
    psum = {c: np.zeros((N_ATTN_LAYERS, len(grp_names), ALN_LEN))
            for c in LABEL_COLS}
    pcnt = {c: np.zeros((len(grp_names), ALN_LEN)) for c in LABEL_COLS}
    ent = {c: {g: [] for g in grp_names} for c in LABEL_COLS}

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, attn_implementation="eager",
                                     output_attentions=True)
    model = model.to(device).eval()

    smoked = False
    for i0 in tqdm(range(0, n, G4_BATCH), desc="注意力提取"):
        rows = sel_idx[i0:i0 + G4_BATCH]
        batch = [seqs[r] for r in rows]
        inp = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inp)
        if not smoked:
            assert hasattr(out, "attentions") and out.attentions is not None
            assert len(out.attentions) == N_ATTN_LAYERS, \
                f"attentions 层数异常: {len(out.attentions)}"
            B, H_, Lq, Lk = out.attentions[0].shape
            print(f"smoke 通过: 30 层 attentions, 单层形状 {(B, H_, Lq, Lk)}")
            smoked = True
        # 逐层 received mass：(B, Lk)
        mass = torch.stack([a.mean(dim=1).sum(dim=1) for a in out.attentions])
        mass = mass.cpu().numpy()                       # (30, B, Lk)
        for j, r in enumerate(rows):
            L = len(seqs[r])
            pos = col_maps[r]
            assert len(pos) == L
            seg = mass[:, j, 1:1 + L]                   # (30, L)
            p = seg / seg.sum(axis=1, keepdims=True)    # 逐层归一化
            for c in LABEL_COLS:
                for gi, g in enumerate(grp_names):
                    if not grp_of[c][g][r]:
                        continue
                    for li in range(N_ATTN_LAYERS):
                        np.add.at(psum[c][li, gi], pos, p[li])
                    ent[c][g].append(
                        (-(p * np.log(p + 1e-12)).sum(axis=1)).tolist())
            for c in LABEL_COLS:
                for gi, g in enumerate(grp_names):
                    if grp_of[c][g][r]:
                        np.add.at(pcnt[c][gi], pos, 1)

    # ── 位点集（同 G3 构建方式）──
    mp = json.load(open(COL_TO_H3_JSON))
    inv_h3 = {}
    for col, num in mp["col_to_h3_ha1"].items():
        inv_h3.setdefault(int(num), []).append(int(col))
    func_cols = set()
    for nums in FUNCTIONAL_SITES.values():
        for num in nums:
            func_cols.update(inv_h3.get(num, []))
    g4cs = pd.read_csv(G4_COLUMN_SCORES)
    valid_mask = g4cs["valid"].to_numpy(dtype=bool)
    js = g4cs["js_div"].to_numpy(dtype=float)
    thr = np.quantile(js[valid_mask & ~np.isnan(js)], DIAG_JS_QUANTILE)
    diag_cols = set(g4cs.index[(valid_mask & (js >= thr))].tolist())
    func_cols &= set(np.where(valid_mask)[0])
    func_arr = np.array(sorted(func_cols))
    diag_arr = np.array(sorted(diag_cols))
    vcols = np.where(valid_mask)[0]

    # G3 读出指数（跨层相关用）
    g3 = json.load(open(OUT_DIR / "g3_site_readout.json"))

    results = {}
    for c in LABEL_COLS:
        mean_p = np.zeros((N_ATTN_LAYERS, len(grp_names), ALN_LEN))
        for gi in range(len(grp_names)):
            mean_p[:, gi] = psum[c][:, gi] / np.maximum(pcnt[c][gi], 1)
        # 以 H7+ 组为主（机制焦点），同时存全组
        gi_h7 = grp_names.index("H7+")
        gi_h5 = grp_names.index("H5+")
        gi_tr = grp_names.index("H1H3tr+")
        fr = {}
        for gname, gi in [("H7+", gi_h7), ("H5+", gi_h5), ("H1H3tr+", gi_tr)]:
            ff = np.array([mean_p[li, gi, func_arr].sum()
                           / mean_p[li, gi].sum()
                           for li in range(N_ATTN_LAYERS)])
            dd = np.array([mean_p[li, gi, diag_arr].sum()
                           / mean_p[li, gi].sum()
                           for li in range(N_ATTN_LAYERS)])
            fr[gname] = {"func_frac": ff.tolist(), "diag_frac": dd.tolist()}

        # 置换（H7+ 组 func_frac mid−late）
        ff_h7 = np.array(fr["H7+"]["func_frac"])
        mid_idx = [l - 1 for l in MID_LAYERS]       # attn 层号 1..30 → 索引 0..29
        late_idx = [l - 1 for l in LATE_LAYERS]
        delta_real = float(ff_h7[mid_idx].mean() - ff_h7[late_idx].mean())
        null_d = []
        mean_p_h7 = mean_p[:, gi_h7]
        tot = mean_p_h7.sum(axis=1)
        for _ in range(N_PERM):
            perm = rng.permutation(vcols)
            fset = perm[:len(func_arr)]
            ff = mean_p_h7[:, fset].sum(axis=1) / tot
            null_d.append(ff[mid_idx].mean() - ff[late_idx].mean())
        p_perm = float((np.abs(null_d) >= abs(delta_real)).mean())

        # 与 G3 sep 指数的跨层相关（G3 层 0..30，取 1..30 对齐）
        g3_ff = np.array(g3[c]["func_frac_per_layer"])[1:N_ATTN_LAYERS + 1]
        rho_g3, p_g3 = spearmanr(ff_h7, g3_ff)

        ent_h7 = np.array(ent[c]["H7+"])            # (n_rows, 30)
        results[c] = {
            "frac_by_group": fr,
            "perm_test_H7+_func_frac_mid_minus_late": {
                "delta": delta_real, "p": p_perm},
            "corr_with_g3_sep_func_frac": {"rho": float(rho_g3),
                                           "p": float(p_g3)},
            "entropy_H7+_mid_mean": float(ent_h7[:, mid_idx].mean()),
            "entropy_H7+_late_mean": float(ent_h7[:, late_idx].mean()),
            "entropy_H7+_per_layer": ent_h7.mean(axis=0).tolist(),
        }

        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        layers = list(range(1, N_ATTN_LAYERS + 1))
        ax1.plot(layers, fr["H7+"]["func_frac"], "o-", color="tab:red",
                 label="attn func_frac (H7+)")
        ax1.plot(layers, fr["H7+"]["diag_frac"], "s-", color="tab:purple",
                 label="attn diag_frac (H7+)")
        ax1.set_xlabel("layer")
        ax1.set_ylabel("attention mass fraction")
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.plot(layers, g3_ff, ".--", color="tab:green", alpha=0.7,
                 label="G3 sep func_frac")
        ax2.set_ylabel("G3 readout index")
        ax2.legend(loc="upper right")
        plt.title(f"G4 注意力位点分布 — {c}（ρ(attn,G3)={rho_g3:.2f}）")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"figs/g4_attention_{c}.png", dpi=300)
        plt.close(fig)

    with open(OUT_DIR / "g4_attention.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"→ {OUT_DIR}/g4_attention.json + figs/")


if __name__ == "__main__":
    main()
