#!/usr/bin/env python3
"""G6：交叉读出（cross-readout）——表示旋转 vs 读出旋转的分解

深度方向翻转有两个候选载体：
  (i)  读出方向 w 随深度旋转（w_17 ⊥ w_28，cos≈0）；
  (ii) 表示本身随深度改变 H7/H5 亚型内 ± 差异的方向。
用交叉读出分解两者：probe 层 a 的 (w_a, scaler_a) 应用于数据层 b 的 embedding，
计算亚型内 gap 与 holdout AUC。若 w_a 固定在 b 变化时 gap 变号 → 表示贡献；
若 b 固定 a 变化时 gap 变号 → 读出贡献。

输出：output/g6_cross_readout.json、figs/g6_cross_readout_{label}.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from validation_exp.depth_reversal.config import (
    SPLIT_CSV, ALL_LAYERS_NPY, LABELS, LABEL_COLS, OUT_DIR,
)

LAYERS = [13, 17, 22, 28, 30]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figs").mkdir(exist_ok=True)

    sp = pd.read_csv(SPLIT_CSV)["split"].to_numpy()
    m5 = sp == "h5_holdout"
    m7 = sp == "h7_holdout"
    probes = np.load(OUT_DIR / "g1_probes.npz")
    arr = np.load(ALL_LAYERS_NPY, mmap_mode="r")
    Xcache = {b: np.asarray(arr[b], dtype=np.float32) for b in LAYERS}

    results = {}
    for label in LABEL_COLS:
        y = np.load(LABELS[label])
        gap7 = np.zeros((len(LAYERS), len(LAYERS)))
        gap5 = np.zeros_like(gap7)
        auc7 = np.zeros_like(gap7)
        auc5 = np.zeros_like(gap7)
        for ai, a in enumerate(LAYERS):
            w = probes[f"w_L{a}_{label}"]
            m = probes[f"mean_L{a}_{label}"]
            s = probes[f"scale_L{a}_{label}"]
            for bi, b in enumerate(LAYERS):
                Xs = (Xcache[b] - m) / s
                z = Xs @ w
                gap7[ai, bi] = float(w @ (Xs[m7 & (y == 1)].mean(0)
                                          - Xs[m7 & (y == 0)].mean(0)))
                gap5[ai, bi] = float(w @ (Xs[m5 & (y == 1)].mean(0)
                                          - Xs[m5 & (y == 0)].mean(0)))
                auc7[ai, bi] = roc_auc_score(y[m7], z[m7])
                auc5[ai, bi] = roc_auc_score(y[m5], z[m5])
        results[label] = {
            "probe_layers": LAYERS, "data_layers": LAYERS,
            "gap_h7": gap7.tolist(), "gap_h5": gap5.tolist(),
            "auc_h7": auc7.tolist(), "auc_h5": auc5.tolist(),
            # 分解摘要：固定 w_17 读 L22→L28 的 gap 变化（表示贡献）
            #          固定 L22 数据 w_17→w_28 的 gap 变化（读出贡献）
            "decomp_jump_example": {
                "repr_effect_w17_L22toL28": float(
                    gap7[LAYERS.index(17), LAYERS.index(28)]
                    - gap7[LAYERS.index(17), LAYERS.index(22)]),
                "readout_effect_atL22_w17vsw28": float(
                    gap7[LAYERS.index(28), LAYERS.index(22)]
                    - gap7[LAYERS.index(17), LAYERS.index(22)]),
            },
        }

        # 图：gap7 交叉热图（probe 层 × 数据层）
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, mat, name in [(axes[0], gap7, "within-gap H7"),
                              (axes[1], auc7, "AUC H7")]:
            vmax = np.abs(mat).max() if name.startswith("within") else 1.0
            vmin = -vmax if name.startswith("within") else 0.0
            im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(LAYERS)), [f"L{b}" for b in LAYERS])
            ax.set_yticks(range(len(LAYERS)), [f"L{a}" for a in LAYERS])
            ax.set_xlabel("data layer")
            ax.set_ylabel("probe layer")
            ax.set_title(f"{name} - {label}")
            for ii in range(len(LAYERS)):
                for jj in range(len(LAYERS)):
                    ax.text(jj, ii, f"{mat[ii, jj]:.1f}" if name.startswith("within")
                            else f"{mat[ii, jj]:.2f}",
                            ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"figs/g6_cross_readout_{label}.png", dpi=300)
        plt.close(fig)
        print(f"[{label}] 交叉读出完成", flush=True)

    with open(OUT_DIR / "g6_cross_readout.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"→ {OUT_DIR}/g6_cross_readout.json + figs/")


if __name__ == "__main__":
    main()
