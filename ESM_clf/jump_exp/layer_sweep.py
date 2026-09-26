"""层扫描：找 H5+H7 方向一致且 AUC 高的 ESM-2 层

用 extract_all_layers.py 提取的 (31, N, 640) 全层 embedding，
对每层在 H1+H3 train 上训 Ridge LR probe，报 test/H5/H7 AUC（原始方向）。

目标：找 H7 原始方向 AUC > 0.5 且 H5/test 不差的层。
也顺带跑 H1+H3+H5 联合训练 → H7（矩阵显示该组合可矫正 jump 方向）。

输出：output/layer_sweep.json
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *


def fit_auc(Xt, yt, Xe, ye):
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(Xt_s, yt)
    logits = gs.decision_function(scl.transform(Xe))
    return float(roc_auc_score(ye, logits))


def main():
    arr = np.load(OUT_DIR / "esm_emb_150M_all_layers.npy")  # (31, N, 640)
    split_df = pd.read_csv(SPLIT_CSV)
    subtypes = split_df["subtype"].values
    splits = split_df["split"].values

    m_train = splits == "train"
    m_test = splits == "test"
    m_h5 = splits == "h5_holdout"
    m_h7 = splits == "h7_holdout"
    # H1+H3+H5 联合训练（H5 holdout 并入训练）
    m_train135 = m_train | m_h5

    results = {}
    for label_col in LABEL_COLS:
        y = np.load(OUT_DIR / f"labels_{label_col}.npy")
        print(f"\n{'='*76}\n  Label: {label_col}")
        print(f"  {'layer':>6} {'test':>7} {'H5':>7} {'H7':>7} | {'H7(135)':>8}")
        print(f"  {'-'*46}")
        label_res = {}
        for li in range(arr.shape[0]):
            X = arr[li]
            auc_test = fit_auc(X[m_train], y[m_train], X[m_test], y[m_test])
            auc_h5 = fit_auc(X[m_train], y[m_train], X[m_h5], y[m_h5])
            auc_h7 = fit_auc(X[m_train], y[m_train], X[m_h7], y[m_h7])
            auc_h7_135 = fit_auc(X[m_train135], y[m_train135], X[m_h7], y[m_h7])
            label_res[f"layer_{li}"] = {"test": auc_test, "h5": auc_h5,
                                        "h7": auc_h7, "h7_train135": auc_h7_135}
            mark = " ← H7方向正确" if auc_h7 > 0.5 else ""
            print(f"  {li:>6} {auc_test:>7.3f} {auc_h5:>7.3f} {auc_h7:>7.3f} | "
                  f"{auc_h7_135:>8.3f}{mark}")
        results[label_col] = label_res

    out_path = OUT_DIR / "layer_sweep.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
