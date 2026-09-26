"""v2 — 综合 Linear 分类实验：class_weight + ElasticNet + Baseline

对每个 embedding 类型（L1/L3/L1L3）尝试 3 种配置：
  1. Ridge + balanced class_weight（class_weight='balanced'）
  2. ElasticNet + balanced class_weight（penalty='elasticnet', solver='saga'）
  3. Ridge + no weight（baseline）

全部使用：
  - StandardScaler + GridSearchCV（scoring=balanced_accuracy）
  - 5-fold stratified CV
  - 评估：accuracy, balanced accuracy, macro F1, per-class precision/recall
  - 输出 eval_interval_clf_v2.json
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
                             f1_score, precision_score, recall_score)


def load_data():
    """加载 embedding、标签和 split（与 prepare_interval_data.py 一致）。"""
    df = pd.read_csv(DATA_CSV)
    split_df = pd.read_csv(SPLIT_CSV)
    df["split"] = split_df["split"]

    # 筛选与 prepare_interval_data.py 一致
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


def make_gridsearch(model, param_grid):
    """创建 GridSearchCV，统一配置。"""
    return GridSearchCV(
        model,
        param_grid,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring=CV_SCORING,
        n_jobs=-1,
    )


def eval_on_test(model, scl, Xt, yt, Xx, yx, subtypes_test, config_name):
    """统一评估函数。"""
    yp = model.predict(Xx)
    result = {
        "config": config_name,
        "best_params": model.best_params_,
        "best_cv_score": float(model.best_score_),
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
            result["test"][f"{label}_prec"] = precision_score(
                yx, yp, labels=[i], average="macro", zero_division=0)
            result["test"][f"{label}_rec"] = recall_score(
                yx, yp, labels=[i], average="macro", zero_division=0)

    # Per-subtype
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


def train_one_config(X, y, train_mask, val_mask, test_mask, subtypes, emb_name,
                     penalty, class_weight, solver, l1_ratios=None):
    """训练单个 (penalty, class_weight) 组合。"""
    Xt, yt = X[train_mask], y[train_mask]
    Xx, yx = X[test_mask], y[test_mask]

    # 标准化
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    Xx_s = scl.transform(Xx)

    # 构建模型和参数网格
    if penalty == "elasticnet":
        model = LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=5000,
            multi_class="multinomial", class_weight=class_weight,
            random_state=RANDOM_SEED, l1_ratio=0.5,
        )
        param_grid = {
            "C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            "l1_ratio": l1_ratios or [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    else:
        model = LogisticRegression(
            penalty=penalty, solver=solver, max_iter=5000,
            multi_class="multinomial", class_weight=class_weight,
            random_state=RANDOM_SEED,
        )
        param_grid = {"C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]}

    gs = make_gridsearch(model, param_grid)
    gs.fit(Xt_s, yt)

    cw_str = "balanced" if class_weight == "balanced" else "none"
    config_name = f"{penalty}_cw={cw_str}"

    result = eval_on_test(gs, scl, Xt_s, yt, Xx_s, yx,
                          subtypes[test_mask], config_name)
    return result


def main():
    masks, subtypes_arr, emb_dict, y_cat = load_data()

    n_classes = len(INTERVAL_LABELS)

    print(f"{'='*60}")
    print(f"  Interval 分类实验 v2（3-class, clamped days）")
    print(f"  Classes: {INTERVAL_LABELS}  (n={n_classes})")
    for s in ["train", "val", "test"]:
        m = masks[s]
        unique, counts = np.unique(y_cat[m], return_counts=True)
        dist = {INTERVAL_LABELS[u]: int(c) for u, c in zip(unique, counts)}
        print(f"  {s}: n={m.sum():>4}  {dist}")
    print(f"{'='*60}")

    # ── 配置列表 ──
    configs = [
        # (penalty, class_weight, solver, l1_ratios, label)
        ("l2", None, "lbfgs", None, "Ridge (baseline)"),
        ("l2", "balanced", "lbfgs", None, "Ridge + balanced"),
        ("elasticnet", "balanced", "saga", [0.1, 0.3, 0.5, 0.7, 0.9], "ElasticNet + balanced"),
    ]

    all_results = {}

    for emb_name in EMB_TYPES:
        if emb_dict[emb_name] is None:
            continue
        X = emb_dict[emb_name]
        dim = X.shape[1]

        print(f"\n{'='*40}")
        print(f"  {emb_name} (dim={dim})")
        print(f"{'='*40}")

        emb_results = {}

        for penalty, cw, solver, l1_ratios, label in configs:
            print(f"\n  --- {label} ---")
            result = train_one_config(
                X, y_cat,
                masks["train"], masks["val"], masks["test"],
                subtypes_arr, emb_name,
                penalty=penalty, class_weight=cw, solver=solver,
                l1_ratios=l1_ratios,
            )
            emb_results[label] = result

            r = result["test"]
            bp = result["best_params"]
            print(f"    best={bp}  cv={result['best_cv_score']:.4f}")
            print(f"    test:  acc={r['acc']:.4f}  bal_acc={r['balanced_acc']:.4f}  "
                  f"macro_f1={r['macro_f1']:.4f}")

            # Per-class recall
            recalls = []
            for i, lbl in enumerate(INTERVAL_LABELS):
                k = f"{lbl}_rec"
                if k in r:
                    recalls.append(f"{lbl}={r[k]:.3f}")
            print(f"    recall: {', '.join(recalls)}")

            # Per-subtype
            for st in SUBTYPES:
                if f"test_{st}" in result:
                    rs = result[f"test_{st}"]
                    print(f"    {st}: acc={rs['acc']:.4f}  bal_acc={rs['balanced_acc']:.4f}  "
                          f"n={rs['n']}")

        all_results[emb_name] = emb_results

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print(f"  v2 分类对比汇总（按 Balanced Accuracy 排序）")
    print(f"  {'Config':<30} {'Emb':<8} {'Acc':>8} {'Bal Acc':>9} {'Macro F1':>9}")
    print(f"  {'-'*70}")

    # 收集所有结果行并排序
    rows = []
    for emb_name in EMB_TYPES:
        if emb_name not in all_results:
            continue
        for cfg_label, cfg_result in all_results[emb_name].items():
            r = cfg_result["test"]
            rows.append((r["balanced_acc"], cfg_label, emb_name, r))

    rows.sort(key=lambda x: x[0], reverse=True)

    for bal_acc, cfg_label, emb_name, r in rows:
        print(f"  {cfg_label:<30} {emb_name:<8} {r['acc']:>8.4f} {r['balanced_acc']:>9.4f} "
              f"{r['macro_f1']:>9.4f}")

    # ── 保存 ──
    out_path = OUT_DIR / "eval_interval_clf_v2.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
