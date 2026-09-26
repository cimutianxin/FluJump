"""统一分类实验（final）— Interval 4 类分类（规范化流程）

流程（P1 规范化，替代旧的"test 上反复试模型"做法）：
  1. train 上 GridSearchCV（5-fold stratified, scoring=balanced_accuracy）
  2. 所有 (embedding × 模型) 组合在 **val** 上比 balanced_acc（并列比 macro_f1）
  3. 唯一 winner 在 train+val 上重训（MLP 例外：train 训练 + val early stop，3 seed 投票）
  4. **test 只评估一次**，指标带 bootstrap 95% CI，附 majority / stratified random 基线

候选模型：Ridge LR / ElasticNet LR / Ordinal LogisticAT（balanced sample_weight）
         / SVM-RBF（balanced）/ XGBoost / MLP [256,128]（3 seeds）

输出: output/eval_interval_clf_final.json
"""

import numpy as np
import json
import sys
import warnings
warnings.filterwarnings("ignore")  # sklearn FutureWarning/ConvergenceWarning 刷屏
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from ESM_clf.interval_exp.interval_data import load_interval_data, balanced_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, recall_score)

import torch
from mord import LogisticAT
from xgboost import XGBClassifier
from ESM_clf.interval_exp.mlp_model import train_mlp, predict_mlp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════
# 评估工具
# ═══════════════════════════════════════════════════════════

def clf_metrics(y_true, y_pred):
    """acc / balanced_acc / macro_f1 / per-class recall。"""
    m = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "n": int(len(y_true)),
    }
    for i, label in enumerate(INTERVAL_LABELS):
        if (y_true == i).sum() > 0:
            m[f"{label}_rec"] = float(recall_score(
                y_true, y_pred, labels=[i], average="macro", zero_division=0))
    return m


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """对 test 指标做 bootstrap 95% CI（重采样 test 索引）。"""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    stats = {"acc": [], "balanced_acc": [], "macro_f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats["acc"].append(accuracy_score(y_true[idx], y_pred[idx]))
        stats["balanced_acc"].append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
        stats["macro_f1"].append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    return {k: {"mean": float(np.mean(v)),
                "ci_lo": float(np.percentile(v, 2.5)),
                "ci_hi": float(np.percentile(v, 97.5))}
            for k, v in stats.items()}


def baseline_metrics(y_train, y_test, n_runs=N_RANDOM_BASELINE, seed=RANDOM_SEED):
    """majority 与 stratified random 基线。"""
    out = {}
    # majority
    maj = np.bincount(y_train).argmax()
    out["majority"] = clf_metrics(y_test, np.full_like(y_test, maj))
    # stratified random：按 train 类别分布随机预测，多次平均
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_train, return_counts=True)
    p = counts / counts.sum()
    runs = {"acc": [], "balanced_acc": [], "macro_f1": []}
    for _ in range(n_runs):
        yp = rng.choice(classes, size=len(y_test), p=p)
        runs["acc"].append(accuracy_score(y_test, yp))
        runs["balanced_acc"].append(balanced_accuracy_score(y_test, yp))
        runs["macro_f1"].append(f1_score(y_test, yp, average="macro"))
    out["stratified_random"] = {k: {"mean": float(np.mean(v)),
                                    "std": float(np.std(v))}
                                for k, v in runs.items()}
    return out


# ═══════════════════════════════════════════════════════════
# 候选模型定义
# ═══════════════════════════════════════════════════════════

def make_candidates():
    """返回 [(name, kind, model, param_grid), ...]；kind ∈ sklearn / ordinal / mlp。"""
    c = []
    c.append(("RidgeLR", "sklearn",
              LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                                 random_state=RANDOM_SEED),
              {"C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
               "class_weight": [None, "balanced"]}))
    c.append(("ElasticNetLR", "sklearn",
              LogisticRegression(penalty="elasticnet", solver="saga", max_iter=1000,
                                 tol=1e-3, random_state=RANDOM_SEED),
              {"C": [1e-2, 1e-1, 1.0, 10.0],
               "l1_ratio": [0.5],
               "class_weight": [None, "balanced"]}))
    c.append(("OrdinalAT", "ordinal",
              LogisticAT(max_iter=5000),
              {"alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]}))
    c.append(("SVM_RBF", "sklearn",
              SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED,
                  max_iter=10000),
              {"C": [1e-1, 1.0, 10.0, 100.0],
               "gamma": ["scale", 1e-3, 1e-2]}))
    c.append(("XGBoost", "xgb",
              XGBClassifier(objective="multi:softmax", num_class=len(INTERVAL_LABELS),
                            eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=2),
              {"max_depth": [3, 5], "learning_rate": [0.05, 0.1],
               "n_estimators": [200, 300]}))
    c.append(("MLP", "mlp", None, None))
    return c


def grid_search(model, param_grid, Xt, yt, sample_weight=None):
    """统一 GridSearchCV。"""
    gs = GridSearchCV(
        model, param_grid,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring=CV_SCORING, n_jobs=-1,
    )
    if sample_weight is not None:
        gs.fit(Xt, yt, sample_weight=sample_weight)
    else:
        gs.fit(Xt, yt)
    return gs


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    print(f"设备: {DEVICE}")
    data = load_interval_data(verbose=True)
    X_dict, y = data["X"], data["y_cat"]
    masks, subtypes = data["masks"], data["subtypes"]
    yt, yv, yx = y[masks["train"]], y[masks["val"]], y[masks["test"]]

    print(f"\n{'='*66}")
    print(f"  统一分类实验（final）— {len(INTERVAL_LABELS)} 类: {INTERVAL_LABELS}")
    for s in ["train", "val", "test"]:
        dist = {INTERVAL_LABELS[i]: int((y[masks[s]] == i).sum())
                for i in range(len(INTERVAL_LABELS))}
        print(f"  {s}: n={masks[s].sum():>4}  {dist}")
    print(f"{'='*66}")

    candidates = make_candidates()
    val_table = {}   # (emb, model) -> {val metrics, best_params, kind}

    # ── Phase 1: train 训练 + val 评估 ──
    for emb_name in EMB_TYPES:
        X = X_dict[emb_name]
        if X is None:
            continue
        scl = StandardScaler().fit(X[masks["train"]])
        Xt, Xv = scl.transform(X[masks["train"]]), scl.transform(X[masks["val"]])

        for name, kind, model, grid in candidates:
            key = f"{name}|{emb_name}"
            if kind == "mlp":
                # 3 seeds，val 指标取均值（val 同时用于 early stopping）
                seed_preds, seed_metrics = [], []
                for seed in MLP_SEEDS:
                    m, _ = train_mlp(
                        Xt, yt, Xv, yv, in_dim=X.shape[1],
                        out_dim=len(INTERVAL_LABELS),
                        hidden_dims=MLP_HIDDEN_CONFIGS[0],
                        dropout=MLP_DROPOUT, lr=MLP_LR,
                        weight_decay=MLP_WEIGHT_DECAY, epochs=MLP_EPOCHS,
                        batch_size=MLP_BATCH_SIZE,
                        patience=MLP_EARLY_STOP_PATIENCE,
                        seed=seed, task="clf", device=DEVICE)
                    pv = predict_mlp(m, Xv, task="clf", device=DEVICE)
                    seed_metrics.append(clf_metrics(yv, pv))
                vm = {k: float(np.mean([s[k] for s in seed_metrics]))
                      for k in ["acc", "balanced_acc", "macro_f1"]}
                val_table[key] = {"kind": kind, "emb": emb_name, "name": name,
                                  "val": vm, "best_params": {"hidden": MLP_HIDDEN_CONFIGS[0]}}
            else:
                sw = balanced_sample_weight(yt) if kind in ("ordinal", "xgb") else None
                gs = grid_search(model, grid, Xt, yt, sample_weight=sw)
                vm = clf_metrics(yv, gs.predict(Xv))
                val_table[key] = {"kind": kind, "emb": emb_name, "name": name,
                                  "val": vm, "best_params": gs.best_params_}
            r = val_table[key]["val"]
            print(f"  {key:<24} val: acc={r['acc']:.4f}  "
                  f"bal_acc={r['balanced_acc']:.4f}  macro_f1={r['macro_f1']:.4f}",
                  flush=True)

    # ── Phase 2: val 选择唯一 winner ──
    def _sort_key(k):
        v = val_table[k]["val"]
        return (v["balanced_acc"], v["macro_f1"])
    winner_key = max(val_table, key=_sort_key)
    winner = val_table[winner_key]
    print(f"\n  ★ Winner（val balanced_acc）: {winner_key}  "
          f"bal_acc={winner['val']['balanced_acc']:.4f}  params={winner['best_params']}")

    # ── Phase 3: winner 重训 + test 单次评估 ──
    emb_name = winner["emb"]
    X = X_dict[emb_name]
    m_tr = masks["train"] | masks["val"]
    ytr = y[m_tr]

    if winner["kind"] == "mlp":
        # MLP 无法无 val 重训：保持 train 训练 + val early stop，3 seed 投票
        scl = StandardScaler().fit(X[masks["train"]])
        Xt = scl.transform(X[masks["train"]])
        Xv = scl.transform(X[masks["val"]])
        Xx = scl.transform(X[masks["test"]])
        votes = []
        for seed in MLP_SEEDS:
            m, _ = train_mlp(
                Xt, yt, Xv, yv, in_dim=X.shape[1],
                out_dim=len(INTERVAL_LABELS), hidden_dims=MLP_HIDDEN_CONFIGS[0],
                dropout=MLP_DROPOUT, lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY,
                epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE,
                patience=MLP_EARLY_STOP_PATIENCE, seed=seed, task="clf", device=DEVICE)
            votes.append(predict_mlp(m, Xx, task="clf", device=DEVICE))
        yp = np.apply_along_axis(lambda a: np.bincount(a, minlength=len(INTERVAL_LABELS)).argmax(),
                                 0, np.array(votes))
        refit_note = "train 训练 + val early stop（3 seed 投票，未并入 val）"
    else:
        scl = StandardScaler().fit(X[m_tr])
        Xtr, Xx = scl.transform(X[m_tr]), scl.transform(X[masks["test"]])
        name = winner["name"]
        model, grid = [(m, g) for n, _, m, g in make_candidates() if n == name][0]
        model.set_params(**winner["best_params"])
        sw = balanced_sample_weight(ytr) if winner["kind"] in ("ordinal", "xgb") else None
        if sw is not None:
            model.fit(Xtr, ytr, sample_weight=sw)
        else:
            model.fit(Xtr, ytr)
        yp = model.predict(Xx)
        refit_note = "train+val 重训（best_params）"

    # ── Phase 4: test 报告（一次）──
    test_metrics = clf_metrics(yx, yp)
    test_ci = bootstrap_ci(yx, yp)
    baselines = baseline_metrics(ytr, yx)

    per_subtype = {}
    for st in SUBTYPES:
        sm = subtypes[masks["test"]] == st
        if sm.sum() >= 2 and len(np.unique(yx[sm])) > 1:
            per_subtype[st] = clf_metrics(yx[sm], yp[sm])

    print(f"\n{'='*66}")
    print(f"  TEST（仅一次评估）— {winner_key}  [{refit_note}]")
    print(f"  acc={test_metrics['acc']:.4f}  bal_acc={test_metrics['balanced_acc']:.4f}  "
          f"macro_f1={test_metrics['macro_f1']:.4f}")
    for k, ci in test_ci.items():
        print(f"    {k:<14} 95% CI [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
    print(f"  基线: majority bal_acc={baselines['majority']['balanced_acc']:.4f}  "
          f"random bal_acc={baselines['stratified_random']['balanced_acc']['mean']:.4f}")
    for st, r in per_subtype.items():
        print(f"  {st}: acc={r['acc']:.4f}  bal_acc={r['balanced_acc']:.4f}  n={r['n']}")
    recalls = [f"{l}={test_metrics.get(f'{l}_rec', 0):.3f}" for l in INTERVAL_LABELS]
    print(f"  per-class recall: {', '.join(recalls)}")

    # ── 保存 ──
    result = {
        "protocol": "train GridSearchCV → val 选择 → train+val 重训 → test 单次评估 + bootstrap CI",
        "winner": {"key": winner_key, "emb": emb_name, "model": winner["name"],
                   "best_params": winner["best_params"], "refit": refit_note},
        "val_table": val_table,
        "test": test_metrics,
        "test_bootstrap_ci": test_ci,
        "test_per_subtype": per_subtype,
        "baselines": baselines,
    }
    out_path = OUT_DIR / "eval_interval_clf_final.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
