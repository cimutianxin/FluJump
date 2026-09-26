"""关键层多 seed 验证：排除层扫描的多重比较偶然性

层扫描在 31 层 × 2 标签上挑最优，存在 selection bias 风险。
对候选层用 5 个 seed 重训 probe，报 test/H5/H7 AUC 的 mean±std。

候选：
  - jump:       layer 26 (H1+H3 → H7=0.705, H5=0.712 双正向), layer 17 (H7=0.893 但 H5 崩)
  - jump_human: layer 13 (H7=0.863), layer 22 (H7=0.740)
  各自由 H1+H3 训练；另加 H1+H3+H5 → H7 的对应配置。

输出：output/verify_layers.json
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

SEEDS = [42, 0, 1, 2, 3]
CANDIDATES = [
    # (label, layer, 是否并入 H5 训练)
    ("label_is_jump", 26, False),
    ("label_is_jump", 26, True),
    ("label_is_jump", 17, False),
    ("label_is_jump_human", 13, False),
    ("label_is_jump_human", 13, True),
    ("label_is_jump_human", 22, False),
]


def one_seed(Xt, yt, Xe, ye, seed, C=1.0):
    scl = StandardScaler()
    lr = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                            max_iter=5000, random_state=seed)
    lr.fit(scl.fit_transform(Xt), yt)
    if len(np.unique(ye)) < 2:
        return None
    return float(roc_auc_score(ye, lr.decision_function(scl.transform(Xe))))


def main():
    arr = np.load(OUT_DIR / "esm_emb_150M_all_layers.npy")
    split_df = pd.read_csv(SPLIT_CSV)
    splits = split_df["split"].values
    m_train, m_test = splits == "train", splits == "test"
    m_h5, m_h7 = splits == "h5_holdout", splits == "h7_holdout"

    results = {}
    for label_col, layer, use_h5 in CANDIDATES:
        y = np.load(OUT_DIR / f"labels_{label_col}.npy")
        X = arr[layer]
        m_tr = m_train | m_h5 if use_h5 else m_train
        key = f"{label_col}|L{layer}|{'H1+H3+H5' if use_h5 else 'H1+H3'}"
        print(f"\n  {key}")
        res = {}
        for name, m_ev in [("test", m_test), ("h5", m_h5), ("h7", m_h7)]:
            if use_h5 and name == "h5":
                res[name] = None  # H5 已入训练，无 holdout 意义
                continue
            aucs = [a for s in SEEDS
                    if (a := one_seed(X[m_tr], y[m_tr], X[m_ev], y[m_ev], s)) is not None]
            res[name] = {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)),
                         "aucs": aucs}
            print(f"    {name}: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  {[f'{a:.3f}' for a in aucs]}")
        results[key] = res

    out_path = OUT_DIR / "verify_layers.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
