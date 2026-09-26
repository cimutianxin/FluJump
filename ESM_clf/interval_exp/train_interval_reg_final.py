"""统一回归实验（final）— Interval 天数回归（规范化流程 + 目标改造）

流程（与 train_interval_clf_final.py 一致）：
  1. train 上 GridSearchCV（Ridge alpha）
  2. 所有 (embedding × 目标形式) 组合在 **val** 上比 MAE（并列比 Spearman）
  3. 唯一 winner 在 train+val 上重训
  4. **test 只评估一次**，指标带 bootstrap 95% CI，附 mean / median 基线

目标形式（clamped days，零膨胀 + 重尾）：
  1. raw:    Ridge 直接拟合原始天数（旧 baseline）
  2. log1p:  Ridge 拟合 log1p(days)，预测经 expm1 还原
  3. hurdle: 两段式 — LR 判 P(days>0) × Ridge 对正值拟合 log1p(days)，
             预测 E[days] = P(>0) · expm1(pred_log)

指标：MAE / RMSE / Spearman / Pearson / R²
输出: output/eval_interval_reg_final.json
"""

import numpy as np
import json
import sys
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from ESM_clf.interval_exp.interval_data import load_interval_data
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr


# ═══════════════════════════════════════════════════════════
# 评估工具
# ═══════════════════════════════════════════════════════════

def reg_metrics(y_true, y_pred):
    """MAE / RMSE / Spearman / Pearson / R²。"""
    sp_r, sp_p = spearmanr(y_true, y_pred)
    pe_r, pe_p = pearsonr(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman_r": float(sp_r), "spearman_p": float(sp_p),
        "pearson_r": float(pe_r), "pearson_p": float(pe_p),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """对 test 回归指标做 bootstrap 95% CI。"""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    keys = ["mae", "rmse", "spearman_r", "pearson_r", "r2"]
    stats = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = reg_metrics(y_true[idx], y_pred[idx])
        for k in keys:
            stats[k].append(m[k])
    return {k: {"mean": float(np.mean(v)),
                "ci_lo": float(np.percentile(v, 2.5)),
                "ci_hi": float(np.percentile(v, 97.5))}
            for k, v in stats.items()}


# ═══════════════════════════════════════════════════════════
# 三种目标形式的训练 / 预测
# ═══════════════════════════════════════════════════════════

def _best_ridge(X, y, alphas=None):
    """Ridge + GridSearchCV（5-fold，neg_mean_absolute_error）。"""
    gs = GridSearchCV(
        Ridge(random_state=RANDOM_SEED),
        {"alpha": alphas or RIDGE_ALPHA_VALUES},
        cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    gs.fit(X, y)
    return gs


def fit_predict(form, Xt, yt, Xv, Xx):
    """按目标形式训练并返回 (val_pred, test_pred, best_alpha 信息)。"""
    if form == "raw":
        gs = _best_ridge(Xt, yt)
        return gs.predict(Xv), gs.predict(Xx), {"alpha": gs.best_params_["alpha"]}

    if form == "log1p":
        gs = _best_ridge(Xt, np.log1p(yt))
        pv = np.expm1(gs.predict(Xv))
        px = np.expm1(gs.predict(Xx))
        return np.maximum(pv, 0), np.maximum(px, 0), {"alpha": gs.best_params_["alpha"]}

    if form == "hurdle":
        pos = yt > 0
        clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                                 random_state=RANDOM_SEED)
        clf.fit(Xt, pos.astype(int))
        gs = _best_ridge(Xt[pos], np.log1p(yt[pos]))
        p_pos_v = clf.predict_proba(Xv)[:, 1]
        p_pos_x = clf.predict_proba(Xx)[:, 1]
        pv = p_pos_v * np.expm1(gs.predict(Xv))
        px = p_pos_x * np.expm1(gs.predict(Xx))
        return np.maximum(pv, 0), np.maximum(px, 0), {"alpha": gs.best_params_["alpha"]}

    raise ValueError(form)


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    data = load_interval_data(verbose=True)
    X_dict, y = data["X"], data["y_days"]
    masks = data["masks"]
    yt, yv, yx = y[masks["train"]], y[masks["val"]], y[masks["test"]]

    print(f"\n{'='*66}")
    print(f"  统一回归实验（final）— clamped iso_interval_days")
    for s, ys in [("train", yt), ("val", yv), ("test", yx)]:
        print(f"  {s}: n={len(ys):>4}  mean={ys.mean():.0f}  "
              f"=0:{(ys == 0).sum()}  >0:{(ys > 0).sum()}")
    print(f"{'='*66}")

    FORMS = ["raw", "log1p", "hurdle"]
    val_table = {}

    # ── Phase 1: train 训练 + val 评估 ──
    for emb_name in EMB_TYPES:
        X = X_dict[emb_name]
        if X is None:
            continue
        scl = StandardScaler().fit(X[masks["train"]])
        Xt, Xv = scl.transform(X[masks["train"]]), scl.transform(X[masks["val"]])
        Xx = scl.transform(X[masks["test"]])

        for form in FORMS:
            key = f"{form}|{emb_name}"
            pv, _, info = fit_predict(form, Xt, yt, Xv, Xx)
            vm = reg_metrics(yv, pv)
            val_table[key] = {"form": form, "emb": emb_name, "val": vm, "info": info}
            print(f"  {key:<18} val: MAE={vm['mae']:.1f}  RMSE={vm['rmse']:.1f}  "
                  f"Spearman={vm['spearman_r']:.3f}  R²={vm['r2']:.3f}")

    # ── Phase 2: val 选择唯一 winner（MAE，并列比 Spearman）──
    winner_key = min(val_table, key=lambda k: (val_table[k]["val"]["mae"],
                                               -val_table[k]["val"]["spearman_r"]))
    winner = val_table[winner_key]
    print(f"\n  ★ Winner（val MAE）: {winner_key}  MAE={winner['val']['mae']:.1f}")

    # ── Phase 3: winner 重训（train+val）+ test 单次评估 ──
    emb_name, form = winner["emb"], winner["form"]
    X = X_dict[emb_name]
    m_tr = masks["train"] | masks["val"]
    ytr = y[m_tr]
    scl = StandardScaler().fit(X[m_tr])
    Xtr, Xx = scl.transform(X[m_tr]), scl.transform(X[masks["test"]])

    # 用 train+val 重训：fit_predict 的 val 槽位传入 test 即可取回 test 预测
    _, yp, info = fit_predict(form, Xtr, ytr, Xx, Xx)

    # ── Phase 4: test 报告（一次）──
    test_metrics = reg_metrics(yx, yp)
    test_ci = bootstrap_ci(yx, yp)
    baselines = {
        "train_mean": reg_metrics(yx, np.full_like(yx, ytr.mean())),
        "train_median": reg_metrics(yx, np.full_like(yx, np.median(ytr))),
    }

    print(f"\n{'='*66}")
    print(f"  TEST（仅一次评估）— {winner_key}  [train+val 重训, {info}]")
    print(f"  MAE={test_metrics['mae']:.1f}  RMSE={test_metrics['rmse']:.1f}  "
          f"Spearman={test_metrics['spearman_r']:.3f}  "
          f"Pearson={test_metrics['pearson_r']:.3f}  R²={test_metrics['r2']:.3f}")
    for k, ci in test_ci.items():
        print(f"    {k:<12} 95% CI [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
    print(f"  基线: mean MAE={baselines['train_mean']['mae']:.1f}  "
          f"median MAE={baselines['train_median']['mae']:.1f}")

    # ── 保存 ──
    result = {
        "protocol": "train GridSearchCV → val MAE 选择 → train+val 重训 → test 单次评估 + bootstrap CI",
        "winner": {"key": winner_key, "emb": emb_name, "form": form, "info": info},
        "val_table": val_table,
        "test": test_metrics,
        "test_bootstrap_ci": test_ci,
        "baselines": baselines,
    }
    out_path = OUT_DIR / "eval_interval_reg_final.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
