#!/usr/bin/env python3
"""V1 跨规模扩展（S2，Major 11）：650M/3B 的 PR 与 AA identity 解码曲线

设计完全同 v1_aa_identity_decode.py（150M），仅换模型：
  - 650M = facebook/esm2_t33_650M_UR50D（34 hidden states, dim 1280）
  - 3B   = facebook/esm2_t36_3B_UR50D（37 hidden states, dim 2560）
  - 同一批抽样序列/位置（seed 42，每亚型 150 条、每条 ≤120 残基），同一闭式 ridge 解码、
    各向异性、participation ratio 计算。
  - gap_H7 剖面取 g5_scale_geometry.json 对应规模的 per_layer.gap_h7（150M 时取 g1 decomp）。

报告：PR/margin 最小点层号（全局与 L≥2 两口径），对照该规模 target-val 最优层
（650M L5/L3、3B L2/(L1/L2)——RESULT.md §4）。

用法：python v1_aa_identity_decode_scale.py [650m|3b]
输出：output/v1_scale_{650m,3b}.json
"""

import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from validation_exp.depth_reversal.config import (
    SPLIT_CSV, ALIGNED_CSV, RANDOM_SEED, LABEL_COLS, OUT_DIR,
)
from validation_exp.depth_reversal.v1_aa_identity_decode import (
    N_PER_SUBTYPE, N_TRAIN_PER_SUBTYPE, N_POS_PER_SEQ, RIDGE_LAMBDA,
    N_ANISO_PAIRS, N_PR_SUB, AA_ALPHABET, AA_TO_IDX, BATCH,
)

MODELS = {
    "650m": "facebook/esm2_t33_650M_UR50D",
    "3b": "facebook/esm2_t36_3B_UR50D",
}
# 各规模 target-val 最优层（RESULT.md §4，供对照记录）
TARGET_VAL_BEST = {
    "650m": {"label_is_jump": 5, "label_is_jump_human": 3},
    "3b": {"label_is_jump": 2, "label_is_jump_human": [1, 2]},
}


def main(scale):
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=["accession", "subtype", "ha_sequence"])
    df = split.merge(ali, on=["accession", "subtype"], how="left",
                     validate="one_to_one")
    seqs = df["ha_sequence"].tolist()

    # ── 与 V1 完全相同的抽样（同 seed、同顺序）──
    rng = np.random.default_rng(RANDOM_SEED)
    sel_rows, sel_istrain = [], []
    for st in ["H1", "H3", "H5", "H7"]:
        idx = np.where(split["subtype"].to_numpy() == st)[0]
        idx = rng.choice(idx, min(N_PER_SUBTYPE, len(idx)), replace=False)
        sel_rows.append(idx)
        sel_istrain.append(np.arange(len(idx)) < N_TRAIN_PER_SUBTYPE)
    sel_rows = np.concatenate(sel_rows)
    sel_istrain = np.concatenate(sel_istrain)
    n = len(sel_rows)

    pos_of = []
    for r in sel_rows:
        L = len(seqs[r])
        k = min(N_POS_PER_SEQ, L)
        pos_of.append(np.sort(rng.choice(np.arange(1, L + 1), k, replace=False)))

    # ── GPU 前向 ──
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODELS[scale])
    model = EsmModel.from_pretrained(MODELS[scale], output_hidden_states=True
                                     ).to(device).eval()
    n_layers = model.config.num_hidden_layers + 1
    print(f"{scale}: {MODELS[scale]}, {n_layers} hidden states, "
          f"dim {model.config.hidden_size}")

    emb_store = [[] for _ in range(n_layers)]
    aa_labels, aa_istrain = [], []
    n_skip_nonstd = 0
    for i0 in tqdm(range(0, n, BATCH), desc=f"前向提取({scale})"):
        rows = sel_rows[i0:i0 + BATCH]
        batch = [seqs[r] for r in rows]
        inp = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            hs = model(**inp).hidden_states
        for j, r in enumerate(rows):
            pos = pos_of[i0 + j]
            aa = np.fromiter((AA_TO_IDX.get(seqs[r][p - 1], -1) for p in pos),
                             dtype=np.int8, count=len(pos))
            keep = aa >= 0
            n_skip_nonstd += int((~keep).sum())
            pos = pos[keep]
            aa = aa[keep]
            for l in range(n_layers):
                emb_store[l].append(
                    hs[l][j, torch.from_numpy(pos).to(device)]
                    .to(torch.float16).cpu().numpy())
            aa_labels.append(aa)
            aa_istrain.append(np.full(len(aa), sel_istrain[i0 + j], dtype=bool))
        del hs
    aa_labels = np.concatenate(aa_labels).astype(np.int64)
    aa_istrain = np.concatenate(aa_istrain)

    # ── gap_H7 剖面：g5 对应规模 ──
    g5 = json.load(open(OUT_DIR / "g5_scale_geometry.json"))
    gap_h7 = {c: np.array([p["gap_h7"] for p in g5[scale][c]["per_layer"]])
              for c in LABEL_COLS}

    # ── 逐层解码 + 各向异性 + PR（同 V1）──
    per_layer = []
    for l in range(n_layers):
        X = np.concatenate(emb_store[l]).astype(np.float32)
        Xtr, Xte = X[aa_istrain], X[~aa_istrain]
        ytr, yte = aa_labels[aa_istrain], aa_labels[~aa_istrain]
        Ytr = np.eye(len(AA_ALPHABET), dtype=np.float64)[ytr]
        Xd = Xtr.astype(np.float64)
        A = Xd.T @ Xd + RIDGE_LAMBDA * len(Xd) * np.eye(Xd.shape[1])
        W = np.linalg.solve(A, Xd.T @ Ytr)
        Z = Xte.astype(np.float64) @ W
        acc = float((Z.argmax(1) == yte).mean())
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        margin = float(P[np.arange(len(yte)), yte].mean())
        sub = Xtr[rng.choice(len(Xtr), min(N_PR_SUB, len(Xtr)), replace=False)]
        Xn = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
        i1 = rng.integers(0, len(Xn), N_ANISO_PAIRS)
        i2 = rng.integers(0, len(Xn), N_ANISO_PAIRS)
        aniso = float((Xn[i1] * Xn[i2]).sum(1).mean())
        cov = np.cov(sub.T)
        ev = np.linalg.eigvalsh(cov)
        pr = float(ev.sum() ** 2 / (ev ** 2).sum())
        per_layer.append({"layer": l, "aa_acc": acc, "aa_margin": margin,
                          "anisotropy": aniso, "participation_ratio": pr})
        print(f"L{l:02d}: acc={acc:.4f} margin={margin:.4f} "
              f"aniso={aniso:.4f} PR={pr:.1f}", flush=True)
        emb_store[l] = None

    # ── 最小点（全局 vs L≥2）与 gap_H7 相关 ──
    layers = np.array([p["layer"] for p in per_layer])
    res = {"scale": scale, "model": MODELS[scale], "per_layer": per_layer,
           "n_residues": int(len(aa_labels)),
           "n_skip_nonstd": int(n_skip_nonstd),
           "target_val_best_layer": TARGET_VAL_BEST[scale],
           "minima": {}, "tests": {}}
    for metric in ["aa_acc", "aa_margin", "anisotropy", "participation_ratio"]:
        v = np.array([p[metric] for p in per_layer])
        res["minima"][metric] = {
            "global_argmin_layer": int(layers[v.argmin()]),
            "global_min": float(v.min()),
            "ge2_argmin_layer": int(layers[layers >= 2][v[layers >= 2].argmin()]),
            "ge2_min": float(v[layers >= 2].min()),
        }
    for c in LABEL_COLS:
        g = gap_h7[c]
        tests = {}
        for metric in ["aa_acc", "aa_margin", "anisotropy", "participation_ratio"]:
            v = np.array([p[metric] for p in per_layer])
            rho, p = spearmanr(v, g)
            tests[f"rho_{metric}_gapH7"] = {"rho": float(rho), "p": float(p)}
        res["tests"][c] = tests

    out_path = OUT_DIR / f"v1_scale_{scale}.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n最小点: {json.dumps(res['minima'], indent=1)}")
    print("完成 →", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "650m")
