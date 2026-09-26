#!/usr/bin/env python3
"""H7N9 回溯的补充对照：区分"时间外推失效"与"跨亚型（H7/Group 2）失效"

对照 1：pre-2013 H1+H3 probe 的 train 5-fold CV AUC（训练本身是否有效）。
对照 2：同一 probe 在 2013+ H1+H3 上的表现（同亚型跨时间是否仍有效）。
若两者都高而 H7 2013+ 近随机，则回溯失败特异于 H7（Group 2），而非时间外推本身。

输出：output/retro_h7n9_controls.json
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
from validation_exp.temporal_validation.config import (
    ALIGNED_CSV, SPLIT_CSV, MEAN_EMB_L3, RIDGE_C_VALUES, CV_FOLDS,
    RANDOM_SEED, RETRO_TRAIN_YEAR_MAX, RETRO_EVAL_YEAR_MIN, OUT_DIR,
)


def main():
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=[
        "accession", "subtype", "collection_year", "label_is_jump"])
    df = split.merge(ali, on=["accession", "subtype"], validate="one_to_one")
    df["year"] = pd.to_numeric(df["collection_year"], errors="coerce").fillna(-1)
    X = np.load(MEAN_EMB_L3).astype(np.float32)
    y = df["label_is_jump"].to_numpy()

    tr = (df["subtype"].isin(["H1", "H3"])
          & (df["year"] > 0) & (df["year"] < RETRO_TRAIN_YEAR_MAX)).to_numpy()
    # 对照 2 评估集：H1+H3，year >= 2013（同亚型跨时间）
    ev_same = (df["subtype"].isin(["H1", "H3"])
               & (df["year"] >= RETRO_EVAL_YEAR_MIN)).to_numpy()

    scaler = StandardScaler().fit(X[tr])
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(scaler.transform(X[tr]), y[tr])

    logits_same = gs.best_estimator_.decision_function(
        scaler.transform(X[ev_same]))

    results = {
        "train_cv_auc": round(float(gs.best_score_), 4),
        "best_C": gs.best_params_["C"],
        "same_subtype_cross_time": {
            "eval": "H1+H3, collection_year >= 2013",
            "n": int(ev_same.sum()),
            "n_pos": int(y[ev_same].sum()),
            "auc": round(float(roc_auc_score(y[ev_same], logits_same)), 4),
        },
    }
    with open(OUT_DIR / "retro_h7n9_controls.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"train CV AUC: {results['train_cv_auc']}（best C={results['best_C']}）")
    print(f"同亚型跨时间（2013+ H1+H3）: n={results['same_subtype_cross_time']['n']}, "
          f"阳性 {results['same_subtype_cross_time']['n_pos']}, "
          f"AUC {results['same_subtype_cross_time']['auc']}")
    print(f"→ {OUT_DIR / 'retro_h7n9_controls.json'}")


if __name__ == "__main__":
    main()
