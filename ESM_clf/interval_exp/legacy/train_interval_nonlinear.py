"""非线性模型 — Interval 分类（3-class）

尝试 SVM (RBF kernel) 和 XGBoost，对不平衡数据加入类别权重。
对每个 embedding 类型（L1/L3/L1L3）：
  - StandardScaler + SVC(kernel='rbf', class_weight='balanced') + GridSearchCV
  - StandardScaler + XGBClassifier + GridSearchCV
  - 评估：accuracy, balanced accuracy, macro F1, per-class metrics
  - 输出 eval_interval_nonlinear.json

依赖: pip install xgboost（SVC 由 sklearn 自带）
"""

import numpy as np
import pandas as pd
import json
import sys
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ xgboost 未安装，请运行: pip install xgboost")


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


def eval_on_test(model, Xx, yx, subtypes_test, config_name, best_params, best_cv_score):
    """统一评估。"""
    yp = model.predict(Xx)
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


def make_gridsearch(model, param_grid):
    return GridSearchCV(
        model, param_grid,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring=CV_SCORING,
        n_jobs=-1,
    )


def train_svm(Xt_s, yt, Xx_s, yx, subtypes_test):
    """SVM RBF kernel + balanced class weight + GridSearchCV。"""
    model = SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED,
                probability=True, max_iter=10000)
    param_grid = {
        "C": [1e-2, 1e-1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1],
    }
    gs = make_gridsearch(model, param_grid)
    gs.fit(Xt_s, yt)
    return eval_on_test(gs, Xx_s, yx, subtypes_test,
                        "SVM_RBF", gs.best_params_, gs.best_score_)


def train_xgb(Xt, yt, Xx, yx, subtypes_test):
    """XGBoost + GridSearchCV。

    注：XGBoost 不需要 StandardScaler（tree-based），但为一致性仍传入 scaled 数据。
    """
    # 计算类别权重
    unique, counts = np.unique(yt, return_counts=True)
    n_total = len(yt)
    # scale_pos_weight 是二分类参数，多分类用 sample_weight
    # 这里用内置的类别权重估计
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(INTERVAL_LABELS),
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "subsample": [0.8, 1.0],
    }
    gs = make_gridsearch(model, param_grid)
    gs.fit(Xt, yt)
    return eval_on_test(gs, Xx, yx, subtypes_test,
                        "XGBoost", gs.best_params_, gs.best_score_)


def main():
    masks, subtypes_arr, emb_dict, y_cat = load_data()

    n_classes = len(INTERVAL_LABELS)
    print(f"{'='*60}")
    print(f"  非线性模型实验（SVM RBF + XGBoost）")
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

        Xt, yt = X[masks["train"]], y_cat[masks["train"]]
        Xx, yx = X[masks["test"]], y_cat[masks["test"]]

        # 标准化（SVM 需要，XGB 不需要但无害）
        scl = StandardScaler()
        Xt_s = scl.fit_transform(Xt)
        Xx_s = scl.transform(Xx)

        emb_results = {}

        # ── SVM RBF ──
        print("\n  --- SVM RBF ---")
        svm_result = train_svm(Xt_s, yt, Xx_s, yx, subtypes_arr[masks["test"]])
        emb_results["SVM_RBF"] = svm_result
        r = svm_result["test"]
        print(f"    best={svm_result['best_params']}  cv={svm_result['best_cv_score']:.4f}")
        print(f"    test:  acc={r['acc']:.4f}  bal_acc={r['balanced_acc']:.4f}  "
              f"macro_f1={r['macro_f1']:.4f}")

        # ── XGBoost ──
        if HAS_XGB:
            print("\n  --- XGBoost ---")
            xgb_result = train_xgb(Xt_s, yt, Xx_s, yx, subtypes_arr[masks["test"]])
            emb_results["XGBoost"] = xgb_result
            r = xgb_result["test"]
            print(f"    best={xgb_result['best_params']}  cv={xgb_result['best_cv_score']:.4f}")
            print(f"    test:  acc={r['acc']:.4f}  bal_acc={r['balanced_acc']:.4f}  "
                  f"macro_f1={r['macro_f1']:.4f}")
        else:
            print("\n  --- XGBoost: 跳过（未安装）---")

        all_results[emb_name] = emb_results

    # ── 汇总 ──
    print(f"\n{'='*80}")
    print(f"  非线性模型汇总（按 Balanced Accuracy 排序）")
    print(f"  {'Emb':<8} {'Model':<12} {'Acc':>8} {'Bal Acc':>9} {'Macro F1':>9}")
    print(f"  {'-'*50}")

    rows = []
    for emb_name in EMB_TYPES:
        if emb_name not in all_results:
            continue
        for model_name, result in all_results[emb_name].items():
            r = result["test"]
            rows.append((r["balanced_acc"], emb_name, model_name, r))

    rows.sort(key=lambda x: x[0], reverse=True)
    for bal_acc, emb_name, model_name, r in rows:
        print(f"  {emb_name:<8} {model_name:<12} {r['acc']:>8.4f} "
              f"{r['balanced_acc']:>9.4f} {r['macro_f1']:>9.4f}")

    # ── 保存 ──
    out_path = OUT_DIR / "eval_interval_nonlinear.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
