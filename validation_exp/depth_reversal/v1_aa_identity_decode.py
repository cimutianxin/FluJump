#!/usr/bin/env python3
"""V1：逐层氨基酸身份线性解码 + 各向异性 —— "末层被 token 身份特化"检验

动机（B1 解读闭环）：深度翻转若源于"末层被 MLM 预训练目标回收去重建序列身份"，
则随深度应出现 token 身份可解码性 / 各向异性的回升，且回升起点应与 within-gap
变号边界（150M 21–26）对应。

设计：
  - 每亚型（H1/H3/H5/H7）抽 150 条（seed 42），120 条 decode-train / 30 条 decode-test；
    每条序列均匀抽 ≤120 个残基位置（token 坐标 1..L，排除 BOS/EOS）。
  - 每层残基 embedding → 20 类 AA 闭式 ridge 解码（λ=1e-3·n），测 top-1 acc 与
    正确类 softmax 概率（margin）。
  - 各向异性：每层 10k 随机残基对 mean pairwise cos；表示有效维度 participation ratio。
  - 与 G1 的 gap_H7 剖面（decomp.within+ − within−）做跨层 Spearman。

判定（预注册式）：
  T1 末层(28–30) acc/margin 均值 > 中层(13–22)（U 形回升 = 身份特化）；
  T2 ρ(metric, gap_H7) < 0（身份特化越强，H7 within-gap 越负）；
  T3 acc 最低点 ≤ 变号边界（回升不晚于 gap 翻转）。

sanity：layer 0（token embedding 查找表）acc 应 ≈ 1.0。

运行：HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env1 python（GPU，~几分钟）
输出：output/v1_aa_identity_decode.json、figs/v1_aa_identity_decode_{label}.png
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
    SPLIT_CSV, ALIGNED_CSV, ESM_MODEL, MID_LAYERS, LATE_LAYERS,
    RANDOM_SEED, LABEL_COLS, OUT_DIR,
)

N_PER_SUBTYPE = 150          # 每亚型抽样条数
N_TRAIN_PER_SUBTYPE = 120    # 其中 decode-train 条数
N_POS_PER_SEQ = 120          # 每条序列抽样残基数上限
RIDGE_LAMBDA = 1e-3          # 闭式 ridge 系数（×n）
N_ANISO_PAIRS = 10000        # 各向异性随机对数
N_PR_SUB = 10000             # participation ratio 子样本
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ALPHABET)}
BATCH = 8


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figs").mkdir(exist_ok=True)

    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=["accession", "subtype", "ha_sequence"])
    df = split.merge(ali, on=["accession", "subtype"], how="left",
                     validate="one_to_one")
    seqs = df["ha_sequence"].tolist()

    # ── 分层抽样：每亚型 150 条（120 decode-train / 30 decode-test）──
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
    print(f"抽样 {n} 条（decode-train {sel_istrain.sum()} / test {(~sel_istrain).sum()}）")

    # ── 每条序列的抽样 token 位置（1..L，排除 BOS/EOS）──
    pos_of = []
    for r in sel_rows:
        L = len(seqs[r])
        k = min(N_POS_PER_SEQ, L)
        pos_of.append(np.sort(rng.choice(np.arange(1, L + 1), k, replace=False)))

    # ── GPU 前向：收集 31 层抽样残基 embedding ──
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True
                                     ).to(device).eval()

    n_layers = model.config.num_hidden_layers + 1
    emb_store = [[] for _ in range(n_layers)]   # 每层 list[(npos,640) fp16]
    aa_labels, aa_istrain = [], []
    n_skip_nonstd = 0
    for i0 in tqdm(range(0, n, BATCH), desc="前向提取"):
        rows = sel_rows[i0:i0 + BATCH]
        batch = [seqs[r] for r in rows]
        inp = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            hs = model(**inp).hidden_states      # tuple(n_layers) (B,T,640)
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
    print(f"非标准 AA 位置跳过 {n_skip_nonstd}；总残基样本 {len(aa_labels)}")

    # ── 逐层：闭式 ridge 解码 + 各向异性 + participation ratio ──
    gap_h7 = {}
    g1 = json.load(open(OUT_DIR / "g1_layer_geometry.json"))
    for c in LABEL_COLS:
        gap_h7[c] = np.array([p["decomp"]["H7"]["within+"] - p["decomp"]["H7"]["within-"]
                              for p in g1["per_layer"][c]])

    per_layer = []
    for l in range(n_layers):
        X = np.concatenate(emb_store[l]).astype(np.float32)
        Xtr, Xte = X[aa_istrain], X[~aa_istrain]
        ytr, yte = aa_labels[aa_istrain], aa_labels[~aa_istrain]
        Ytr = np.eye(len(AA_ALPHABET), dtype=np.float64)[ytr]
        Xd = Xtr.astype(np.float64)
        A = Xd.T @ Xd + RIDGE_LAMBDA * len(Xd) * np.eye(Xd.shape[1])
        W = np.linalg.solve(A, Xd.T @ Ytr)
        Z = Xte.astype(np.float64) @ W                  # (n_te,20) 解码打分
        pred = Z.argmax(1)
        acc = float((pred == yte).mean())
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        margin = float(P[np.arange(len(yte)), yte].mean())
        # 各向异性（训练侧抽样）
        sub = Xtr[rng.choice(len(Xtr), min(N_PR_SUB, len(Xtr)), replace=False)]
        Xn = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
        i1 = rng.integers(0, len(Xn), N_ANISO_PAIRS)
        i2 = rng.integers(0, len(Xn), N_ANISO_PAIRS)
        aniso = float((Xn[i1] * Xn[i2]).sum(1).mean())
        # participation ratio
        cov = np.cov(sub.T)
        ev = np.linalg.eigvalsh(cov)
        pr = float(ev.sum() ** 2 / (ev ** 2).sum())
        per_layer.append({"layer": l, "aa_acc": acc, "aa_margin": margin,
                          "anisotropy": aniso, "participation_ratio": pr})
        print(f"L{l:02d}: acc={acc:.4f} margin={margin:.4f} "
              f"aniso={aniso:.4f} PR={pr:.1f}")
        emb_store[l] = None

    # ── 判定 ──
    layers = np.array([p["layer"] for p in per_layer])
    res = {"per_layer": per_layer, "n_residues": int(len(aa_labels)),
           "n_skip_nonstd": int(n_skip_nonstd), "tests": {}}
    for c in LABEL_COLS:
        g = gap_h7[c]
        tests = {}
        for metric in ["aa_acc", "aa_margin", "anisotropy", "participation_ratio"]:
            v = np.array([p[metric] for p in per_layer])
            rho, p = spearmanr(v, g)
            tests[f"rho_{metric}_gapH7"] = {"rho": float(rho), "p": float(p)}
            tests[f"{metric}_mid_mean"] = float(v[np.isin(layers, MID_LAYERS)].mean())
            tests[f"{metric}_late_mean"] = float(v[np.isin(layers, LATE_LAYERS)].mean())
        acc = np.array([p["aa_acc"] for p in per_layer])
        tests["acc_argmin_layer"] = int(layers[acc.argmin()])
        tests["acc_late_minus_mid"] = float(acc[np.isin(layers, LATE_LAYERS)].mean()
                                            - acc[np.isin(layers, MID_LAYERS)].mean())
        res["tests"][c] = tests

    with open(OUT_DIR / "v1_aa_identity_decode.json", "w") as f:
        json.dump(res, f, indent=1)

    # ── 图：margin / participation ratio 与 gap_H7 双轴（acc 全层饱和，不画）──
    for c in LABEL_COLS:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, mlab in [
                (axes[0], "aa_margin", "AA decode margin (P(correct))"),
                (axes[1], "participation_ratio", "participation ratio (eff. dim)")]:
            v = [p[metric] for p in per_layer]
            ax.plot(layers, v, "o-", ms=3, color="tab:blue", label=mlab)
            ax.set_xlabel("layer")
            ax.set_ylabel(mlab, color="tab:blue")
            ax2 = ax.twinx()
            ax2.plot(layers, gap_h7[c], "r-", alpha=0.6, label="within-gap H7")
            ax2.axhline(0, color="r", lw=0.5, alpha=0.4)
            ax2.set_ylabel("within-gap H7", color="r")
            ax.axvspan(21, 26, color="gray", alpha=0.15)
            rho = res["tests"][c][f"rho_{metric}_gapH7"]
            ax.set_title(f"{metric} vs depth  (rho with gap={rho['rho']:+.2f}, "
                         f"p={rho['p']:.1e})", fontsize=9)
        fig.suptitle(f"V1: identity re-anchoring vs depth ({c})", y=1.0)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "figs" / f"v1_aa_identity_decode_{c}.png", dpi=200)
        plt.close(fig)
    print("完成 →", OUT_DIR / "v1_aa_identity_decode.json")


if __name__ == "__main__":
    main()
