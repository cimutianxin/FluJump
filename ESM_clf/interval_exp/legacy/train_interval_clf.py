"""Ridge Logistic Regression — interval 分类（4 类：<1yr / 1-3yr / 3-5yr / 5yr+）

对每个 embedding 类型（L1/L3/L1L3）：
  - StandardScaler + LogisticRegression(multinomial, L2) + GridSearchCV
  - 评估：accuracy, balanced accuracy, macro F1, per-class precision/recall
  - 输出 eval_interval_clf.json
"""

import numpy as np
import pandas as pd
import json
import sys
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             classification_report)


def load_data():
    """加载 embedding、标签和 split"""
    # 筛选 mask — 需要和 prepare_interval_data.py 一致的筛选
    df = pd.read_csv(DATA_CSV)
    split_df = pd.read_csv(SPLIT_CSV)
    df["split"] = split_df["split"]

    mask = (df["subtype"].isin(SUBTYPES) &
            (df["label_is_jump"] == 1) &
            (df["jump_interval_cat"].notna()) &
            (df["jump_interval_cat"] != ""))

    split_arr = df.loc[mask, "split"].values
    subtypes_arr = df.loc[mask, "subtype"].values
    idx = df[mask].index.values  # 在全量数据中的行索引

    y_cat = np.load(OUT_DIR / "labels_interval_cat.npy")
    y_reg = np.load(OUT_DIR / "labels_interval_days.npy")

    masks = {
        s: (split_arr == s)
        for s in ["train", "val", "test"]
    }

    emb = {}
    for etype in EMB_TYPES:
        path = EMB_DIR / f"esm_emb_150M_{etype}.npy"
        if path.exists():
            emb[etype] = np.load(path)[idx]  # 按行索引切片
        else:
            emb[etype] = None

    return masks, subtypes_arr, emb, y_cat, y_reg


def train_and_eval(X, y, train_mask, val_mask, test_mask, subtypes, emb_name):
    """单次 (embedding) 组合的训练/评估"""
    Xt, yt = X[train_mask], y[train_mask]
    Xx, yx = X[test_mask], y[test_mask]
    Xv, yv = X[val_mask], y[val_mask] if val_mask.any() else (None, None)

    # 标准化
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    Xx_s = scl.transform(Xx)
    Xv_s = scl.transform(Xv) if Xv is not None else None

    # Multinomial LR + GridSearchCV
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            multi_class="multinomial", random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                      scoring="f1_macro")
    gs.fit(Xt_s, yt)

    # 预测
    yp = gs.predict(Xx_s)
    result = {
        "best_C": gs.best_params_["C"],
        "test": {
            "acc": accuracy_score(yx, yp),
            "balanced_acc": balanced_accuracy_score(yx, yp),
            "macro_f1": f1_score(yx, yp, average="macro"),
            "n": len(yx),
        }
    }

    # Per-class metrics
    for i, label in enumerate(INTERVAL_LABELS):
        if (yx == i).sum() > 0:
            result["test"][f"{label}_prec"] = precision_score(yx, yp, labels=[i], average="macro", zero_division=0)
            result["test"][f"{label}_rec"] = recall_score(yx, yp, labels=[i], average="macro", zero_division=0)

    # Per-subtype test
    sub_test = subtypes[test_mask]
    for st in SUBTYPES:
        sm = (sub_test == st)
        if sm.sum() >= 2 and len(np.unique(yx[sm])) > 1:
            result[f"test_{st}"] = {
                "acc": accuracy_score(yx[sm], yp[sm]),
                "balanced_acc": balanced_accuracy_score(yx[sm], yp[sm]),
                "macro_f1": f1_score(yx[sm], yp[sm], average="macro"),
                "n": int(sm.sum()),
            }

    return result


def main():
    masks, subtypes_arr, emb_dict, y_cat, y_reg = load_data()

    print(f"{'='*60}")
    print(f"  Interval 分类实验")
    print(f"  Classes: {INTERVAL_LABELS}")
    for s in ["train", "val", "test"]:
        m = masks[s]
        unique, counts = np.unique(y_cat[m], return_counts=True)
        dist = {INTERVAL_LABELS[u]: int(c) for u, c in zip(unique, counts)}
        print(f"  {s}: n={m.sum():>4}  {dist}")
    print(f"{'='*60}")

    all_results = {}

    for emb_name in EMB_TYPES:
        if emb_dict[emb_name] is None:
            continue
        X = emb_dict[emb_name]
        dim = X.shape[1]

        print(f"\n  --- {emb_name} (dim={dim}) ---")
        result = train_and_eval(
            X, y_cat, masks["train"], masks["val"], masks["test"],
            subtypes_arr, emb_name
        )
        all_results[emb_name] = result

        r = result["test"]
        print(f"    C={result['best_C']:.4f}  "
              f"acc={r['acc']:.4f}  bal_acc={r['balanced_acc']:.4f}  "
              f"macro_f1={r['macro_f1']:.4f}")

        # Per-subtype
        for st in SUBTYPES:
            if f"test_{st}" in result:
                rs = result[f"test_{st}"]
                print(f"    {st}: acc={rs['acc']:.4f}  macro_f1={rs['macro_f1']:.4f}  n={rs['n']}")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  分类对比汇总")
    print(f"  {'Embedding':<10} {'Acc':>8} {'Bal Acc':>8} {'Macro F1':>9}")
    print(f"  {'-'*40}")
    from math import log2
    print(f"  {'random':<10} {1/4:>8.4f} {1/4:>8.4f} {1/4:>9.4f}")
    print(f"  {'majority':<10} {0.601:>8.4f} {0.25:>8.4f} {'—':>9}")
    for emb_name in EMB_TYPES:
        if emb_name in all_results:
            r = all_results[emb_name]["test"]
            print(f"  {emb_name:<10} {r['acc']:>8.4f} {r['balanced_acc']:>8.4f} {r['macro_f1']:>9.4f}")

    # 保存
    out_path = OUT_DIR / "eval_interval_clf.json"
    json.dump(all_results, open(out_path, "w"), indent=2, default=str)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
