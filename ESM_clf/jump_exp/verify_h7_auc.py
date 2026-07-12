"""H7 AUC 精度验证：用 logits + 高精度计算 AUC 原始/翻转"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

split_df = pd.read_csv(SPLIT_CSV)
data_df = pd.read_csv(DATA_CSV)
subtypes = data_df["subtype"].values
train_mask = (split_df["split"] == "train").values
h5_mask = (split_df["split"] == "h5_holdout").values
h7_mask = (split_df["split"] == "h7_holdout").values

for label_col in LABEL_COLS:
    y = np.load(OUT_DIR / f"labels_{label_col}.npy")
    print(f"\n{'='*70}")
    print(f"  {label_col}")

    for emb_name in ["L1", "L3", "L1L3"]:
        X = np.load(OUT_DIR / f"esm_emb_150M_{emb_name}.npy")
        Xt, yt = X[train_mask], y[train_mask]
        X7, y7 = X[h7_mask], y[h7_mask]
        X5, y5 = X[h5_mask], y[h5_mask]

        scl = StandardScaler(); Xt_s = scl.fit_transform(Xt)
        best_c = 10.0 if emb_name == "L1" else 1.0
        lr = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                                max_iter=5000, random_state=RANDOM_SEED)
        lr.fit(Xt_s, yt)

        # —— logits（不经过 sigmoid），避免极端概率的数值问题 ——
        logits7 = lr.decision_function(scl.transform(X7))
        logits5 = lr.decision_function(scl.transform(X5))

        p7 = lr.predict_proba(scl.transform(X7))[:, 1]
        p5 = lr.predict_proba(scl.transform(X5))[:, 1]

        # AUC 原始
        auc7 = roc_auc_score(y7, p7)
        auc5 = roc_auc_score(y5, p5)

        # AUC 翻转：直接用 -logits，数学上等价于 1-p 但不损失精度
        flip7 = roc_auc_score(y7, -logits7)
        flip5 = roc_auc_score(y5, -logits5)

        # 验证 1 - AUC
        one_minus_auc7 = 1.0 - auc7
        one_minus_auc5 = 1.0 - auc5

        # Spearman rank corr（更稳健）
        from scipy.stats import spearmanr
        sp7, _ = spearmanr(p7, y7)
        sp7f, _ = spearmanr(-logits7, y7)

        print(f"\n  [{emb_name}] {label_col}")
        print(f"    H7  pred mean:     {p7.mean():.6f}  (min={p7.min():.6f}, max={p7.max():.6f})")
        print(f"    H7  AUC(p):        {auc7:.6f}")
        print(f"    H7  AUC(-logit):   {flip7:.6f}")
        print(f"    H7  1-AUC(p):      {one_minus_auc7:.6f}")
        print(f"    H7  AUC(p)+AUC(1-p): {auc7+flip7:.6f}")
        print(f"    H7  Spearman(p):   {sp7:.6f}  →  Spearman(-logit): {sp7f:.6f}")
        print(f"    H5  AUC(p):        {auc5:.6f}")
        print(f"    H5  AUC(-logit):   {flip5:.6f}")
        print(f"    H5  AUC(p)+AUC(1-p): {auc5+flip5:.6f}")

print("\nDone")
