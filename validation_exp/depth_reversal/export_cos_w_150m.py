"""fig5_depth：150M cos(w_best, w_last) 导出

best 层取 target-val 层选择结果（label_is_jump→17、label_is_jump_human→13），
last 层=30。两套口径：
  1. 主拟合：直接读 output/g1_probes.npz 中 g1_layer_geometry.py 存的逐层 probe
     （seed=RANDOM_SEED 的 GridSearchCV 拟合）。
  2. 多种子：seed{42..46} 按 g1 同一 fit_probe 协议在 best/last 层重训，
     逐 seed 计算 cos(w_best, w_last)。

预期（notes/09-26-16）：主拟合 ≈ -0.005/+0.015，|cos| ≤ 0.02；超出在终端报告。

输出：figdata/fig5_depth/cos_w_best_w_last.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from validation_exp.depth_reversal.config import (
    SPLIT_CSV, ALL_LAYERS_NPY, LABELS, LABEL_COLS,
    RIDGE_C_VALUES, CV_FOLDS, RANDOM_SEED,
)

FIG5_DIR = Path("figdata/fig5_depth")
NPZ = Path("validation_exp/depth_reversal/output/g1_probes.npz")
BEST_LAYER = {"label_is_jump": 17, "label_is_jump_human": 13}   # target-val 选择
LAST_LAYER = 30
SEEDS = [42, 43, 44, 45, 46]
COS_ABS_EXPECT = 0.02        # notes/09-26-16：|cos| ≤ 0.02（超出仅报告，不断言失败）


def fit_w(Xtr, ytr, seed):
    """同 g1_layer_geometry.fit_probe：返回标准化空间 w"""
    scl = StandardScaler().fit(Xtr)
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(scl.transform(Xtr), ytr)
    return gs.best_estimator_.coef_[0]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    split = pd.read_csv(SPLIT_CSV)
    m_train = (split["split"] == "train").to_numpy()
    arr = np.load(ALL_LAYERS_NPY, mmap_mode="r")
    npz = np.load(NPZ)

    results = {}
    for label in LABEL_COLS:
        lb, ll = BEST_LAYER[label], LAST_LAYER
        y = np.load(LABELS[label])

        # ── 主拟合（npz 内 g1 已存 probe）──
        cos_main = cos(npz[f"w_L{lb}_{label}"], npz[f"w_L{ll}_{label}"])

        # ── 多种子重训 ──
        Xb = np.asarray(arr[lb], dtype=np.float32)
        Xl = np.asarray(arr[ll], dtype=np.float32)
        cos_per_seed = {}
        for sd in SEEDS:
            wb = fit_w(Xb[m_train], y[m_train], seed=sd)
            wl = fit_w(Xl[m_train], y[m_train], seed=sd)
            c = cos(wb, wl)
            cos_per_seed[str(sd)] = c
            print(f"[{label}] seed={sd}: cos(w_L{lb}, w_L{ll}) = {c:+.6f}",
                  flush=True)

        vals = np.array(list(cos_per_seed.values()))
        results[label] = {
            "best_layer": lb, "last_layer": ll,
            "cos_main_fit_npz": cos_main,
            "cos_per_seed": cos_per_seed,
            "cos_mean": float(vals.mean()),
            "cos_abs_max": float(np.abs(vals).max()),
        }
        flag = "" if max(abs(cos_main), np.abs(vals).max()) <= COS_ABS_EXPECT \
            else "  ⚠ 超出 notes 预期 |cos|≤0.02"
        print(f"[{label}] 主拟合 cos={cos_main:+.6f} | "
              f"5seed mean={vals.mean():+.6f} abs_max={np.abs(vals).max():.6f}{flag}",
              flush=True)

    FIG5_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG5_DIR / "cos_w_best_w_last.json"
    json.dump({"model": "esm2_t30_150M", "best_layer_source": "target-val 层选择",
               "seeds": SEEDS, **results},
              open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"✓ {out_path}")


if __name__ == "__main__":
    main()
