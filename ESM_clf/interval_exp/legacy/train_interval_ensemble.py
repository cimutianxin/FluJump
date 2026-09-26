"""概率集成（Ensemble）— Interval 分类（3-class）

从各 Phase 的结果中选出最佳模型，进行 Soft Voting 和 Stacking 集成。

策略：
  1. Soft Voting: 多个模型的 predict_proba 取平均后 argmax
  2. Stacking: 用 LogisticRegression 在 val set 上训练 meta-learner
  3. 跨 embedding 集成: L1 + L3 各自最佳模型的概率平均

依赖 Phase 2/3/4 的结果文件（eval_interval_clf_v2.json 等）。
输出 eval_interval_ensemble.json
"""

import numpy as np
import pandas as pd
import json
import sys
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import mord
    HAS_MORD = True
except ImportError:
    HAS_MORD = False


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


def eval_ensemble(yp, yx, subtypes_test, config_name):
    """统一评估。"""
    result = {
        "config": config_name,
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


def get_best_models_from_results():
    """从各结果文件中解析出最佳模型配置。返回 [(name, model_constructor, embedding_type), ...]"""
    models = []

    # Phase 2: Linear v2
    clf_v2_path = OUT_DIR / "eval_interval_clf_v2.json"
    if clf_v2_path.exists():
        with open(clf_v2_path) as f:
            clf_v2 = json.load(f)
        best_bal = -1
        best_info = None
        for emb_name, emb_results in clf_v2.items():
            for cfg_label, cfg_result in emb_results.items():
                bal = cfg_result["test"]["balanced_acc"]
                if bal > best_bal:
                    best_bal = bal
                    best_info = (emb_name, cfg_label, cfg_result)
        if best_info:
            emb_name, cfg_label, cfg_result = best_info
            bp = cfg_result["best_params"]
            if "Ridge" in cfg_label:
                if "balanced" in cfg_label:
                    cw = "balanced"
                else:
                    cw = None
                models.append({
                    "name": f"L2_{emb_name}_Ridge_cw={cw}",
                    "type": "ridge",
                    "emb": emb_name,
                    "C": bp["C"],
                    "class_weight": cw,
                    "best_bal": best_bal,
                })

    # Phase 3: Ordinal (需要 mord，这里用 LogisticAT 可能不返回 proba，暂跳过)
    # Phase 4: Nonlinear
    nonlinear_path = OUT_DIR / "eval_interval_nonlinear.json"
    if nonlinear_path.exists():
        with open(nonlinear_path) as f:
            nonlinear = json.load(f)
        best_bal = -1
        best_info = None
        for emb_name, emb_results in nonlinear.items():
            for model_name, result in emb_results.items():
                bal = result["test"]["balanced_acc"]
                if bal > best_bal:
                    best_bal = bal
                    best_info = (emb_name, model_name, result)
        if best_info:
            emb_name, model_name, result = best_info
            models.append({
                "name": f"{model_name}_{emb_name}",
                "type": "svm" if "SVM" in model_name else "xgb",
                "emb": emb_name,
                "best_params": result["best_params"],
                "best_bal": best_bal,
            })

    return models


def fit_individual_models(Xt_s, yt, Xv_s, Xx_s, models_info, n_classes):
    """根据模型信息训练各个模型，返回 [(name, proba_on_test, proba_on_val), ...]。

    使用 val set 上训练的模型来生成概率（用于 Stacking 的 meta-features）。
    """
    fitted = []

    for mi in models_info:
        emb_name = mi["emb"]

        if mi["type"] == "ridge":
            model = LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=5000,
                multi_class="multinomial", class_weight=mi.get("class_weight"),
                C=mi.get("C", 1.0), random_state=RANDOM_SEED,
            )
            model.fit(Xt_s[emb_name], yt)
            proba_test = model.predict_proba(Xx_s[emb_name])
            proba_val = model.predict_proba(Xv_s[emb_name])

        elif mi["type"] == "svm":
            bp = mi.get("best_params", {})
            model = SVC(
                kernel="rbf", class_weight="balanced",
                C=bp.get("C", 1.0), gamma=bp.get("gamma", "scale"),
                probability=True, random_state=RANDOM_SEED, max_iter=10000,
            )
            model.fit(Xt_s[emb_name], yt)
            proba_test = model.predict_proba(Xx_s[emb_name])
            proba_val = model.predict_proba(Xv_s[emb_name])

        elif mi["type"] == "xgb" and HAS_XGB:
            bp = mi.get("best_params", {})
            model = XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                max_depth=bp.get("max_depth", 5),
                learning_rate=bp.get("learning_rate", 0.1),
                n_estimators=bp.get("n_estimators", 200),
                subsample=bp.get("subsample", 1.0),
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            model.fit(Xt_s[emb_name], yt)
            proba_test = model.predict_proba(Xx_s[emb_name])
            proba_val = model.predict_proba(Xv_s[emb_name])
        else:
            continue

        fitted.append({
            "name": mi["name"],
            "proba_test": proba_test,
            "proba_val": proba_val,
        })

    return fitted


def main():
    masks, subtypes_arr, emb_dict, y_cat = load_data()
    n_classes = len(INTERVAL_LABELS)

    # 准备数据
    scl = {emb: StandardScaler() for emb in EMB_TYPES if emb_dict[emb] is not None}
    Xt_s, Xv_s, Xx_s = {}, {}, {}
    for emb_name in scl:
        Xt_s[emb_name] = scl[emb_name].fit_transform(emb_dict[emb_name][masks["train"]])
        Xv_s[emb_name] = scl[emb_name].transform(emb_dict[emb_name][masks["val"]])
        Xx_s[emb_name] = scl[emb_name].transform(emb_dict[emb_name][masks["test"]])

    yt, yv, yx = y_cat[masks["train"]], y_cat[masks["val"]], y_cat[masks["test"]]

    print(f"{'='*60}")
    print(f"  集成实验（Ensemble）")
    print(f"  Classes: {INTERVAL_LABELS}")
    print(f"  train/val/test: {len(yt)}/{len(yv)}/{len(yx)}")
    print(f"{'='*60}")

    # 获取最佳模型信息
    models_info = get_best_models_from_results()
    print(f"\n  候选模型 ({len(models_info)}):")
    for mi in models_info:
        print(f"    {mi['name']:<35} bal_acc={mi['best_bal']:.4f}")

    if len(models_info) < 2:
        print("\n  ⚠ 模型不足（需要 ≥2），跳过集成。请先运行 Phase 2-4。")
        return

    # 训练各模型
    fitted = fit_individual_models(Xt_s, yt, Xv_s, Xx_s, models_info, n_classes)
    print(f"\n  成功训练 {len(fitted)} 个模型")

    all_results = {}

    # ── Soft Voting: 概率平均 ──
    avg_proba_test = np.mean([f["proba_test"] for f in fitted], axis=0)
    yp_soft = avg_proba_test.argmax(axis=1)
    all_results["soft_voting"] = eval_ensemble(
        yp_soft, yx, subtypes_arr,
        f"SoftVoting({','.join([f['name'] for f in fitted])})"
    )
    r = all_results["soft_voting"]["test"]
    print(f"\n  Soft Voting:  acc={r['acc']:.4f}  bal_acc={r['balanced_acc']:.4f}  "
          f"macro_f1={r['macro_f1']:.4f}")

    # ── L1+L3 跨 embedding 集成 ──
    l1_models = [f for f in fitted if "L1" in f["name"] and "L1L3" not in f["name"]]
    l3_models = [f for f in fitted if "L3" in f["name"] and "L1L3" not in f["name"]]
    if l1_models and l3_models:
        cross_proba = np.mean(
            [np.mean([m["proba_test"] for m in l1_models], axis=0),
             np.mean([m["proba_test"] for m in l3_models], axis=0)],
            axis=0
        )
        yp_cross = cross_proba.argmax(axis=1)
        all_results["cross_emb_ensemble"] = eval_ensemble(
            yp_cross, yx, subtypes_arr,
            f"CrossEmb(L1_best+L3_best)"
        )
        r = all_results["cross_emb_ensemble"]["test"]
        print(f"  Cross L1+L3:    acc={r['acc']:.4f}  bal_acc={r['balanced_acc']:.4f}  "
              f"macro_f1={r['macro_f1']:.4f}")

    # ── Stacking: LR meta-learner on val set ──
    meta_X_val = np.column_stack([f["proba_val"] for f in fitted])
    meta_X_test = np.column_stack([f["proba_test"] for f in fitted])

    meta_model = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=5000,
        multi_class="multinomial", class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    meta_model.fit(meta_X_val, yv)
    yp_stack = meta_model.predict(meta_X_test)
    all_results["stacking"] = eval_ensemble(
        yp_stack, yx, subtypes_arr,
        f"Stacking({','.join([f['name'][:15] for f in fitted])})"
    )
    r = all_results["stacking"]["test"]
    print(f"  Stacking:        acc={r['acc']:.4f}  bal_acc={r['balanced_acc']:.4f}  "
          f"macro_f1={r['macro_f1']:.4f}")

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print(f"  集成汇总")
    print(f"  {'Method':<25} {'Acc':>8} {'Bal Acc':>9} {'Macro F1':>9}")
    print(f"  {'-'*55}")
    for method, result in all_results.items():
        r = result["test"]
        print(f"  {method:<25} {r['acc']:>8.4f} {r['balanced_acc']:>9.4f} "
              f"{r['macro_f1']:>9.4f}")

    # ── 保存 ──
    out_path = OUT_DIR / "eval_interval_ensemble.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
