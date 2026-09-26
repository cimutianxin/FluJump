"""avg_mid（中层 10–22 平均）统一方案的 cluster bootstrap 验证

avg_mid 是唯一不用测试标签、H7 方向正确的统一架构（jump 0.718 / jump_human 0.732）。
用与 verify_layers_bootstrap.py 相同的 cluster bootstrap（B=20）验证稳健性，
报 H7/H5 AUC 分布与 P(>0.5)。

输出：output/verify_avg_mid_bootstrap.json
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

B = 20
MID_LAYERS = list(range(10, 23))


def fit_gs(Xt, yt, seed):
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(Xt_s, yt)
    return scl, gs


def auc(model, scl, Xe, ye):
    if len(np.unique(ye)) < 2:
        return None
    return float(roc_auc_score(ye, model.decision_function(scl.transform(Xe))))


def main():
    arr = np.load(OUT_DIR / "esm_emb_150M_all_layers.npy")
    X = arr[MID_LAYERS].mean(axis=0)  # (N, 640)
    split_df = pd.read_csv(SPLIT_CSV)
    splits = split_df["split"].values
    clusters = split_df["cluster_id"].values
    m_train, m_test = splits == "train", splits == "test"
    m_h5, m_h7 = splits == "h5_holdout", splits == "h7_holdout"

    rng = np.random.default_rng(RANDOM_SEED)
    results = {}

    for label_col in LABEL_COLS:
        y = np.load(OUT_DIR / f"labels_{label_col}.npy")
        print(f"\n  {label_col} (avg_mid L10-22)", flush=True)

        Xt, yt = X[m_train], y[m_train]
        cl_train = clusters[m_train]
        uniq_cl = np.unique(cl_train)

        scl, gs = fit_gs(Xt, yt, RANDOM_SEED)
        orig = {n: auc(gs, scl, X[m], y[m])
                for n, m in [("test", m_test), ("h5", m_h5), ("h7", m_h7)]}
        print(f"    原始: test={orig['test']:.3f} h5={orig['h5']:.3f} h7={orig['h7']:.3f}",
              flush=True)

        boot = {"test": [], "h5": [], "h7": []}
        for b in range(B):
            sampled_cl = rng.choice(uniq_cl, size=len(uniq_cl), replace=True)
            idx = np.concatenate([np.where(cl_train == c)[0] for c in sampled_cl])
            if len(np.unique(yt[idx])) < 2:
                continue
            scl_b, gs_b = fit_gs(Xt[idx], yt[idx], seed=b)
            for n, m in [("test", m_test), ("h5", m_h5), ("h7", m_h7)]:
                a = auc(gs_b, scl_b, X[m], y[m])
                if a is not None:
                    boot[n].append(a)
            print(f"    b{b:02d}: h7={boot['h7'][-1]:.3f} h5={boot['h5'][-1]:.3f}",
                  flush=True)

        summary = {"orig": orig}
        for n in boot:
            v = np.array(boot[n])
            summary[n] = {"mean": float(v.mean()), "std": float(v.std()),
                          "p05": float(np.percentile(v, 5)),
                          "p95": float(np.percentile(v, 95)),
                          "frac_above_0.5": float((v > 0.5).mean())}
        results[label_col] = summary
        s = summary["h7"]
        print(f"    H7 bootstrap: {s['mean']:.3f} ± {s['std']:.3f} "
              f"[{s['p05']:.3f}, {s['p95']:.3f}], P(>0.5)={s['frac_above_0.5']:.2f}",
              flush=True)

    out_path = OUT_DIR / "verify_avg_mid_bootstrap.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
