"""诊断 H7 迁移 AUC < 0.5：检查是否预测方向反向"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# ── 加载 ──
split_df = pd.read_csv(SPLIT_CSV)
data_df = pd.read_csv(DATA_CSV)
subtypes = data_df["subtype"].values

train_mask = (split_df["split"] == "train").values
test_mask = (split_df["split"] == "test").values
h5_mask = (split_df["split"] == "h5_holdout").values
h7_mask = (split_df["split"] == "h7_holdout").values

for label_col in LABEL_COLS:
    y = np.load(OUT_DIR / f"labels_{label_col}.npy")
    print(f"\n{'='*70}")
    print(f"  {label_col}")
    print(f"  H7: n={h7_mask.sum()}, pos={y[h7_mask].sum()} ({y[h7_mask].mean()*100:.1f}%)")
    print(f"  H5: n={h5_mask.sum()}, pos={y[h5_mask].sum()} ({y[h5_mask].mean()*100:.1f}%)")

    for emb_name in ["L1", "L3", "L1L3"]:
        X = np.load(OUT_DIR / f"esm_emb_150M_{emb_name}.npy")
        Xt, yt = X[train_mask], y[train_mask]
        X7, y7 = X[h7_mask], y[h7_mask]
        X5, y5 = X[h5_mask], y[h5_mask]

        scl = StandardScaler()
        Xt_s = scl.fit_transform(Xt)

        # 用最优 C（从之前的实验：L1=10, L3=1, L1L3=1）
        best_c = 10.0 if emb_name == "L1" else 1.0
        lr = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                                max_iter=5000, random_state=RANDOM_SEED)
        lr.fit(Xt_s, yt)

        X7_s = scl.transform(X7)
        p7 = lr.predict_proba(X7_s)[:, 1]

        X5_s = scl.transform(X5)
        p5 = lr.predict_proba(X5_s)[:, 1]

        auc7 = roc_auc_score(y7, p7) if len(np.unique(y7)) > 1 else None
        auc5 = roc_auc_score(y5, p5) if len(np.unique(y5)) > 1 else None

        # 检查翻转
        p7_flip = 1 - p7
        auc7_flip = roc_auc_score(y7, p7_flip) if len(np.unique(y7)) > 1 else None

        p5_flip = 1 - p5
        auc5_flip = roc_auc_score(y5, p5_flip) if len(np.unique(y5)) > 1 else None

        print(f"\n  [{emb_name}] C={best_c}")
        print(f"    Train pos_rate:    {yt.mean():.3f}")
        print(f"    H7  pred mean:     {p7.mean():.4f}  (labels mean: {y7.mean():.3f})")
        print(f"    H7  pred corr w/ y:{np.corrcoef(p7, y7)[0,1]:.4f}")
        print(f"    H7  AUC (原始):    {auc7:.4f}")
        print(f"    H7  AUC (翻转):    {auc7_flip:.4f}")
        print(f"    H5  pred mean:     {p5.mean():.4f}  (labels mean: {y5.mean():.3f})")
        print(f"    H5  AUC (原始):    {auc5:.4f}")
        print(f"    H5  AUC (翻转):    {auc5_flip:.4f}")

        # 检查 H7 subtype 细分
        for st in ["H5", "H7"]:
            sm = subtypes[h7_mask] == st
            if sm.sum() > 0:
                p_st = p7[sm]
                y_st = y7[sm]
                auc_st = roc_auc_score(y_st, p_st) if len(np.unique(y_st)) > 1 else None
                print(f"    H7_{st} (n={sm.sum()}): pred_mean={p_st.mean():.4f}, AUC={auc_st:.4f}")

        # 检查 train 的 coef 方向
        print(f"    coef mean:         {lr.coef_[0].mean():.6f}")
        print(f"    coef std:          {lr.coef_[0].std():.6f}")
        print(f"    intercept:         {lr.intercept_[0]:.6f}")

print("\nDone")
