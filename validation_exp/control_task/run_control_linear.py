"""control task · linear probe：打乱 train 标签后 Ridge LR 还能"学"到多少

设计（Hewitt & Liang 2019 control task 的序列级版本）：
  - 仅置换 H1+H3 train 内部的标签（val/test/H5/H7 保持真实标签）
  - 在打乱标签上训 Ridge LR probe（同 jump_exp 的 GridSearchCV 流程）
  - 报两个 AUC 系列：
      * fit_*：对打乱后标签的 train AUC（对噪声的拟合能力 = probe 容量）
      * real_*：对各处真实标签的 AUC（应 ≈0.5，证明无真实信号泄露）
  - real 行（seed=null）为真实标签训练的同款 probe，作对照

输出：output/control_linear.json
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
from validation_exp.control_task.config import *

SPLIT_NAMES = ["train", "val", "test", "h5_holdout", "h7_holdout"]


def fit_probe(Xt, yt):
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(Xt_s, yt)
    return scl, gs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.load(ALL_LAYERS)                    # (31, N, 640)
    split_df = pd.read_csv(SPLIT_CSV)
    splits = split_df["split"].values
    masks = {n: splits == n for n in SPLIT_NAMES}
    m_train = masks["train"]

    results = {}
    for label_col in LABEL_COLS:
        y = np.load(JUMP_OUT / f"labels_{label_col}.npy")
        ytr_real = y[m_train]
        lres = {}
        for seed in [None] + SHUFFLE_SEEDS:      # None = 真实标签对照行
            tag = "real" if seed is None else f"shuffle_{seed}"
            if seed is None:
                ytr = ytr_real
            else:
                rng = np.random.default_rng(seed)
                ytr = rng.permutation(ytr_real)  # 仅 train 内置换
            sres = {}
            for li in CANDIDATE_LAYERS:
                X = arr[li]
                scl, gs = fit_probe(X[m_train], ytr)
                entry = {
                    # 对（打乱后）训练标签的拟合度 = 容量
                    "fit_train_auc": float(roc_auc_score(ytr, gs.decision_function(
                        scl.transform(X[m_train])))),
                    "best_C": gs.best_params_["C"],
                }
                # 对真实标签的各处 AUC（control 下应 ≈0.5）
                for n in SPLIT_NAMES:
                    entry[f"real_{n}_auc"] = float(roc_auc_score(
                        y[masks[n]], gs.decision_function(scl.transform(X[masks[n]]))))
                sres[f"L{li}"] = entry
                print(f"  {label_col} {tag} L{li}: fit_train={entry['fit_train_auc']:.3f} "
                      f"real_test={entry['real_test_auc']:.3f} "
                      f"real_h7={entry['real_h7_holdout_auc']:.3f}", flush=True)
            lres[tag] = sres
        results[label_col] = lres

    out_path = OUT_DIR / "control_linear.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
