#!/usr/bin/env python3
"""G2：残差流逐 block 贡献分解（H4）

残差流恒等式：h_L = h_0 + Σ_{l=1..L} Δ_l，Δ_l = h_l − h_{l−1}
（mean-pool 与差分可交换，直接在全层 mean-pooled npy 上计算）。
对目标层 L ∈ {17, 28, 30} × 双标签，block l 对 logit 的贡献：
  c_l(x) = (Δ_l(x)/s_L)·w_L        （Σ_l c_l = z − b + const）
分组聚合：train±（H1+H3）、H5±、H7± 的 mean c_l 剖面；
派生：分离贡献 sep_l = mean(train+) − mean(train−)、
      推负贡献 neg_l = mean(H7+) − mean(train+)（H5 同法作对照）。

判定 H4：H7 推负贡献是否集中于 L>20 段（|neg_l| 峰值超全段中位 2 倍且
该段贡献占比 >50%）；与 G1 的 cos(w,d_H7) 变号层位对照。

输出：output/g2_residual_stream.json、figs/g2_stream_{label}_L{L}.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from validation_exp.depth_reversal.config import (
    SPLIT_CSV, ALL_LAYERS_NPY, LABELS, LABEL_COLS, MID_LAYERS, OUT_DIR,
)

TARGET_LAYERS = [17, 28, 30]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figs").mkdir(exist_ok=True)

    split = pd.read_csv(SPLIT_CSV)
    sp = split["split"].to_numpy()
    m_train = (sp == "train")
    m_h5 = (sp == "h5_holdout")
    m_h7 = (sp == "h7_holdout")

    probes = np.load(OUT_DIR / "g1_probes.npz")
    arr = np.load(ALL_LAYERS_NPY, mmap_mode="r")

    groups = {}
    ys = {c: np.load(LABELS[c]) for c in LABEL_COLS}

    results = {}
    for label in LABEL_COLS:
        y = ys[label]
        groups = {
            "train+": m_train & (y == 1), "train-": m_train & (y == 0),
            "H5+": m_h5 & (y == 1), "H5-": m_h5 & (y == 0),
            "H7+": m_h7 & (y == 1), "H7-": m_h7 & (y == 0),
        }
        for L in TARGET_LAYERS:
            w = probes[f"w_L{L}_{label}"].astype(np.float32)
            s = probes[f"scale_L{L}_{label}"].astype(np.float32)
            # 逐 block 贡献均值（l = 0..L）
            prof = {g: np.zeros(L + 1, dtype=np.float64) for g in groups}
            prev = np.asarray(arr[0], dtype=np.float32)
            for l in range(L + 1):
                cur = np.asarray(arr[l], dtype=np.float32)
                delta = cur if l == 0 else cur - prev
                c = (delta / s) @ w            # (11060,)
                for g, msk in groups.items():
                    prof[g][l] = float(c[msk].mean())
                prev = cur

            sep = prof["train+"] - prof["train-"]
            neg_h7 = prof["H7+"] - prof["train+"]
            neg_h5 = prof["H5+"] - prof["train+"]
            gap_h7 = prof["H7+"] - prof["H7-"]     # 亚型内 gap（AUC 的决定量）
            gap_h5 = prof["H5+"] - prof["H5-"]
            gap_tr = sep

            # H4 判定：L>20 段的亚型内 gap 贡献占比（|gap_h7| 集中在哪段）
            late_seg = np.array([l for l in range(L + 1) if l > 20],
                                dtype=int)
            if len(late_seg) > 0:
                h7_late_share = float(np.abs(neg_h7[late_seg]).sum()
                                      / (np.abs(neg_h7).sum() + 1e-12))
                gap_late_share = float(np.abs(gap_h7[late_seg]).sum()
                                       / (np.abs(gap_h7).sum() + 1e-12))
                gap_late_sum = float(gap_h7[late_seg].sum())
                h5_late_share = float(np.abs(neg_h5[late_seg]).sum()
                                      / (np.abs(neg_h5).sum() + 1e-12))
            else:   # L≤20 无 L>20 段，指标不适用
                h7_late_share = gap_late_share = None
                gap_late_sum = h5_late_share = None
            med = float(np.median(np.abs(neg_h7)))
            peak_l = int(np.argmax(np.abs(neg_h7)))
            peak_over_med = float(np.abs(neg_h7[peak_l]) / (med + 1e-12))
            gap_total = float(gap_h7.sum())
            gap_mid_sum = float(gap_h7[np.array([l for l in range(L + 1)
                                                 if l in MID_LAYERS],
                                                dtype=int)].sum())

            results[f"{label}_L{L}"] = {
                "blocks": list(range(L + 1)),
                "profile": {g: prof[g].tolist() for g in groups},
                "sep_train": sep.tolist(),
                "push_neg_h7": neg_h7.tolist(),
                "push_neg_h5": neg_h5.tolist(),
                "gap_h7_within": gap_h7.tolist(),
                "gap_h5_within": gap_h5.tolist(),
                "gap_h7_total": gap_total,
                "gap_h7_mid_sum": gap_mid_sum,
                "gap_h7_late20_sum": gap_late_sum,
                "gap_h7_late20_absshare": gap_late_share,
                "gap_train_total": float(gap_tr.sum()),
                "h7_push_neg_late20_share": h7_late_share,
                "h7_push_neg_peak_block": peak_l,
                "h7_push_neg_peak_over_median": peak_over_med,
                "h5_push_neg_late20_share": h5_late_share,
            }

            # 图：亚型内 gap 与分离贡献逐 block 剖面
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(range(L + 1), np.cumsum(gap_h7), "o-", color="tab:red",
                    label="cumsum gap H7 (+ vs -)")
            ax.plot(range(L + 1), np.cumsum(gap_h5), "s-", color="tab:blue",
                    label="cumsum gap H5 (+ vs -)")
            ax.plot(range(L + 1), np.cumsum(gap_tr), ".-", color="tab:green",
                    alpha=0.7, label="cumsum gap train (+ vs -)")
            ax.axhline(0, color="black", lw=0.6)
            if L > 20:
                ax.axvspan(20.5, L + 0.5, color="gray", alpha=0.12)
            ax.set_xlabel("block l (cumulative up to L)")
            ax.set_ylabel("cumulative within-subtype gap")
            ax.legend()
            plt.title(f"G2 residual stream cumulative gap - {label} @ L{L} probe")
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"figs/g2_stream_{label}_L{L}.png", dpi=300)
            plt.close(fig)
            print(f"[{label}][L{L}] gap_H7 total={gap_total:+.2f} "
                  f"mid={gap_mid_sum:+.2f} late20={gap_late_sum} | "
                  f"H7推负 L>20 占比={h7_late_share}", flush=True)

    with open(OUT_DIR / "g2_residual_stream.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"→ {OUT_DIR}/g2_residual_stream.json + figs/")


if __name__ == "__main__":
    main()
