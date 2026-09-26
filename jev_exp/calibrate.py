"""温度缩放校准 + 校准指标评估（JEV 头，复用 train_jev.py 落盘的集成 logit）

流程：
  1. 读 jev_exp/output/scores/ens_{val,test,eval}_L{li}_{label}.npy（3 种子集成 logit s1−s0）
  2. 在 H1+H3 val 上拟合温度 T>0（最小化 NLL，p = σ(score/T)）
     —— val 同时用于 early stop，存在轻微乐观偏差，结论中声明
  3. 在 H1+H3 test（逐层）与 cluster_seed42 的 h5/h7 test（jev_results.json 选中层）上
     报告校准前后 NLL / Brier / ECE(15 bin) + reliability 分箱数据

H7 警示：H7 存在方向反转（分布漂移），H1+H3 val 上拟合的校准不预期在 H7 成立，
如实报告、不翻转。

输出：jev_exp/output/jev_calibration.json
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, ".")
from jev_exp.config import *

EPS = 1e-7
N_BINS = 15


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece_bins(y, p, n_bins=N_BINS):
    """等宽分箱 ECE + reliability 数据"""
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1
                               else p <= edges[i + 1])
        n = int(m.sum())
        if n == 0:
            bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                         "n": 0, "mean_conf": None, "emp_freq": None})
            continue
        conf, freq = float(p[m].mean()), float(y[m].mean())
        ece += n / len(y) * abs(freq - conf)
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": n,
                     "mean_conf": conf, "emp_freq": freq})
    return float(ece), bins


def fit_temperature(scores, y):
    """在 val 上拟合 T（log 空间优化），返回 T 与 val NLL 前后值"""
    def obj(log_t):
        return nll(y, sigmoid(scores / np.exp(log_t)))
    res = minimize_scalar(obj, bounds=(-4.6, 4.6), method="bounded")
    return float(np.exp(res.x))


def eval_cal(scores, y, T):
    p_raw, p_cal = sigmoid(scores), sigmoid(scores / T)
    ece_raw, bins_raw = ece_bins(y, p_raw)
    ece_cal, bins_cal = ece_bins(y, p_cal)
    return {"n": int(len(y)),
            "nll_before": nll(y, p_raw), "nll_after": nll(y, p_cal),
            "brier_before": brier(y, p_raw), "brier_after": brier(y, p_cal),
            "ece_before": ece_raw, "ece_after": ece_cal,
            "bins_before": bins_raw, "bins_after": bins_cal}


def main():
    split_df = pd.read_csv(SPLIT_CSV)
    va_idx = np.where((split_df["split"] == "val").values)[0]
    test_idx = np.where((split_df["split"] == "test").values)[0]
    eval_idx = np.where(split_df["split"].isin(["h5_holdout", "h7_holdout"]).values)[0]
    pos_of = {r: p for p, r in enumerate(eval_idx)}

    key2row = {(a, s): i for i, (a, s) in
               enumerate(zip(split_df["accession"], split_df["subtype"]))}
    sp0 = pd.read_csv(H5H7_SPLIT_DIR / "cluster_seed42.csv")
    mask_rows = {name: np.array([key2row[(a, s)] for a, s in
                                 zip(g["accession"], g["subtype"])])
                 for name, g in sp0.groupby("split")}

    labels = {c: np.load(LABELS_DIR / f"labels_{c}.npy") for c in LABEL_COLS}
    jev = json.load(open(JEV_OUT_DIR / "jev_results.json"))
    sel = jev["per_file"]["cluster_seed42.csv"]   # [label][subtype]["selected_layer"]

    out = {}
    for label_col in LABEL_COLS:
        y = labels[label_col]
        lres = {"temperature": {}, "h13_test": {}, "cluster_seed42": {}}
        for li in CANDIDATE_LAYERS:
            sc_val = np.load(JEV_OUT_DIR / "scores" / f"ens_val_L{li}_{label_col}.npy")
            T = fit_temperature(sc_val, y[va_idx])
            lres["temperature"][f"L{li}"] = T
            sc_test = np.load(JEV_OUT_DIR / "scores" / f"ens_test_L{li}_{label_col}.npy")
            lres["h13_test"][f"L{li}"] = eval_cal(sc_test, y[test_idx], T)
        for st in ["h5", "h7"]:
            best = sel[label_col][st]["selected_layer"]
            rows = mask_rows[f"{st}_test"]
            p = np.array([pos_of[r] for r in rows])
            sc = np.load(JEV_OUT_DIR / "scores" / f"ens_eval_L{best}_{label_col}.npy")[p]
            T = lres["temperature"][f"L{best}"]
            r = eval_cal(sc, y[rows], T)
            r["selected_layer"] = int(best)
            lres["cluster_seed42"][f"{st}_test"] = r
        out[label_col] = lres

    # ── 打印 ──
    print(f"\n{'='*76}\n温度缩放校准（T 拟合于 H1+H3 val）\n{'='*76}")
    for label_col in LABEL_COLS:
        print(f"\n--- {label_col} ---")
        Ts = out[label_col]["temperature"]
        print("  温度: " + "  ".join(f"{k}={v:.3f}" for k, v in Ts.items()))
        for grp in ["h13_test", "cluster_seed42"]:
            for name, r in out[label_col][grp].items():
                if not isinstance(r, dict) or "nll_before" not in r:
                    continue
                print(f"  {grp:>16} {name:>9} (n={r['n']}): "
                      f"NLL {r['nll_before']:.4f}→{r['nll_after']:.4f}  "
                      f"Brier {r['brier_before']:.4f}→{r['brier_after']:.4f}  "
                      f"ECE {r['ece_before']:.4f}→{r['ece_after']:.4f}")

    out_path = JEV_OUT_DIR / "jev_calibration.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")
    print("\n注：H7 存在方向反转，h7_test 上校准恶化属预期（见 AGENTS.md 踩坑记录）。")


if __name__ == "__main__":
    main()
