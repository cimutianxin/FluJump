"""有序回归（Ordinal Regression）— Interval 分类（3-class）

使用 mord.LogisticAT（All-Threshold model）利用类别排序信息。
对每个 embedding 类型（L1/L3/L1L3）：
  - StandardScaler + LogisticAT + GridSearchCV over alpha
  - 评估：accuracy, balanced accuracy, macro F1, per-class metrics
  - 输出 eval_interval_ordinal.json

依赖: pip install mord
"""

import numpy as np
import pandas as pd
import json
import sys
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)

try:
    import mord
    HAS_MORD = True
    from mord import LogisticAT
except ImportError:
    HAS_MORD = False
    print("⚠ mord 未安装，请运行: pip install mord")


def load_data():
    """加载 embedding、标签和 split。"""
    df = pd.read_csv(DATA_CSV)
    split_df = pd.read_csv(SPLIT_CSV)
    df["split"] = split_df["split"]

    mask_subtype = df["subtype"].isin(SUBTYPES)
    mask_jump = df["label_is_jump"] == 1

    if USE_CLAMPED_DAYS:
        df["iso_interval_days"] = pd.to_numeric(df["iso_interval_days"], errors="coerce")
        mask_interval = df["iso_interval_days"].notna()
    else:
        mask_interval = (df["jump_interval_cat"].notna()) & (df["jump_interval_cat"] != "")

    mask = mask_subtype & mask_jump & mask_interval
    split_arr = df.loc[mask, "split"].values
    subtypes_arr = df.loc[mask, "subtype"].values
    idx = df[mask].index.values

    y_cat = np.load(OUT_DIR / "labels_interval_cat.npy")

    emb = {}
    for etype in EMB_TYPES:
        path = EMB_DIR / f"esm_emb_150M_{etype}.npy"
        if path.exists():
            emb[etype] = np.load(path)[idx]
        else:
            emb[etype] = None

    masks = {s: (split_arr == s) for s in ["train", "val", "test"]}
    return masks, subtypes_arr, emb, y_cat


def eval_on_test(yp, yx, subtypes_test, config_name, best_params, best_cv_score):
    """统一评估函数。"""
    result = {
        "config": config_name,
        "best_params": best_params,
        "best_cv_score": float(best_cv_score),
        "test": {
            "acc": accuracy_score(yx, yp),
            "balanced_acc": balanced_accuracy_score(yx, yp),
            "macro_f1": f1_score(yx, yp, average="macro"),
            "n": len(yx),
        }
    }

    for i, label in enumerate(INTERVAL_LABELS):
        if (yx == i).sum() > 0:
            result["test"][f"{label}_prec"] = precision_score(
                yx, yp, labels=[i], average="macro", zero_division=0)
            result["test"][f"{label}_rec"] = recall_score(
                yx, yp, labels=[i], average="macro", zero_division=0)

    for st in SUBTYPES:
        sm = (subtypes_test == st)
        if sm.sum() >= 2 and len(np.unique(yx[sm])) > 1:
            result[f"test_{st}"] = {
                "acc": accuracy_score(yx[sm], yp[sm]),
                "balanced_acc": balanced_accuracy_score(yx[sm], yp[sm]),
                "macro_f1": f1_score(yx[sm], yp[sm], average="macro"),
                "n": int(sm.sum()),
            }

    return result


def train_ordinal(X, y, train_mask, test_mask, subtypes, emb_name):
    """训练 LogisticAT + GridSearchCV。"""
    Xt, yt = X[train_mask], y[train_mask]
    Xx, yx = X[test_mask], y[test_mask]

    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    Xx_s = scl.transform(Xx)

    model = LogisticAT(alpha=1.0, max_iter=5000)
    param_grid = {"alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]}

    gs = GridSearchCV(
        model, param_grid,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring=CV_SCORING,
        n_jobs=-1,
    )
    gs.fit(Xt_s, yt)

    yp = gs.predict(Xx_s)
    result = eval_on_test(yp, yx, subtypes[test_mask],
                          f"LogisticAT", gs.best_params_, gs.best_score_)
    return result


def main():
    if not HAS_MORD:
        print("❌ mord 未安装，无法运行有序回归实验。")
        print("   请运行: pip install mord")
        return

    masks, subtypes_arr, emb_dict, y_cat = load_data()

    n_classes = len(INTERVAL_LABELS)
    print(f"{'='*60}")
    print(f"  有序回归（Ordinal Regression）实验")
    print(f"  模型: mord.LogisticAT（All-Threshold）")
    print(f"  Classes: {INTERVAL_LABELS}  (n={n_classes})")
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

        print(f"\n{'='*40}")
        print(f"  {emb_name} (dim={dim})")
        print(f"{'='*40}")

        result = train_ordinal(X, y_cat, masks["train"], masks["test"],
                               subtypes_arr, emb_name)
        all_results[emb_name] = result

        r = result["test"]
        bp = result["best_params"]
        print(f"    best={bp}  cv={result['best_cv_score']:.4f}")
        print(f"    test:  acc={r['acc']:.4f}  bal_acc={r['balanced_acc']:.4f}  "
              f"macro_f1={r['macro_f1']:.4f}")

        recalls = []
        for i, lbl in enumerate(INTERVAL_LABELS):
            k = f"{lbl}_rec"
            if k in r:
                recalls.append(f"{lbl}={r[k]:.3f}")
        print(f"    recall: {', '.join(recalls)}")

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print(f"  有序回归汇总")
    print(f"  {'Emb':<8} {'Acc':>8} {'Bal Acc':>9} {'Macro F1':>9}  {'Best Alpha':>12}")
    print(f"  {'-'*55}")

    rows = []
    for emb_name in EMB_TYPES:
        if emb_name in all_results:
            r = all_results[emb_name]["test"]
            bp = all_results[emb_name]["best_params"]
            rows.append((r["balanced_acc"], emb_name, r, bp))

    rows.sort(key=lambda x: x[0], reverse=True)
    for bal_acc, emb_name, r, bp in rows:
        alpha = bp.get("alpha", "?")
        print(f"  {emb_name:<8} {r['acc']:>8.4f} {r['balanced_acc']:>9.4f} "
              f"{r['macro_f1']:>9.4f}  {str(alpha):>12}")

    # ── 保存 ──
    out_path = OUT_DIR / "eval_interval_ordinal.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
