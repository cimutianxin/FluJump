"""Ridge Linear Regression — interval 回归（预测 iso_interval_days 连续值）

对每个 embedding 类型（L1/L3/L1L3）：
  - StandardScaler + Ridge / Lasso + GridSearchCV over alpha
  - 评估：MAE, RMSE, R², Pearson r
  - 输出 eval_interval_reg.json
"""

import numpy as np
import pandas as pd
import json
import sys
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


def load_data():
    """加载 embedding、标签和 split（与 prepare_interval_data.py 一致）"""
    df = pd.read_csv(DATA_CSV)
    split_df = pd.read_csv(SPLIT_CSV)
    df["split"] = split_df["split"]

    mask = (df["subtype"].isin(SUBTYPES) &
            (df["label_is_jump"] == 1) &
            (df["jump_interval_cat"].notna()) &
            (df["jump_interval_cat"] != ""))

    split_arr = df.loc[mask, "split"].values
    subtypes_arr = df.loc[mask, "subtype"].values
    idx = df[mask].index.values

    y_reg = np.load(OUT_DIR / "labels_interval_days.npy")

    emb = {}
    for etype in EMB_TYPES:
        path = EMB_DIR / f"esm_emb_150M_{etype}.npy"
        if path.exists():
            emb[etype] = np.load(path)[idx]
        else:
            emb[etype] = None

    # Filter NaN (isolates without iso_interval_days)
    valid_mask = ~np.isnan(y_reg)
    print(f"  remove NaN from regression: {valid_mask.sum()} / {len(y_reg)}")
    y_reg = y_reg[valid_mask]
    split_arr = split_arr[valid_mask]
    subtypes_arr = subtypes_arr[valid_mask]
    for etype in EMB_TYPES:
        if emb[etype] is not None:
            emb[etype] = emb[etype][valid_mask]

    masks = {s: (split_arr == s) for s in ["train", "val", "test"]}

    return masks, subtypes_arr, emb, y_reg


def train_and_eval(X, y, train_mask, val_mask, test_mask, subtypes, emb_name, model_name):
    """单次 (embedding, model) 的训练/评估"""
    Xt, yt = X[train_mask], y[train_mask]
    Xx, yx = X[test_mask], y[test_mask]

    # 标准化
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    Xx_s = scl.transform(Xx)

    # Ridge / Lasso + GridSearchCV
    if model_name == "Ridge":
        model = Ridge(random_state=RANDOM_SEED)
    else:
        model = Lasso(random_state=RANDOM_SEED, max_iter=5000)
    param_name = "alpha"

    gs = GridSearchCV(model, {param_name: RIDGE_ALPHA_VALUES},
                      cv=CV_FOLDS, scoring="neg_mean_squared_error")
    gs.fit(Xt_s, yt)

    # 预测
    yp = gs.predict(Xx_s)

    # 指标
    mae = mean_absolute_error(yx, yp)
    rmse = np.sqrt(mean_squared_error(yx, yp))
    r2 = r2_score(yx, yp)
    pearson_r, pearson_p = pearsonr(yx, yp)

    result = {
        f"best_{param_name}": gs.best_params_[param_name],
        "test": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "n": len(yx),
            "y_mean": float(yx.mean()),
            "y_std": float(yx.std()),
            "pred_mean": float(yp.mean()),
            "pred_std": float(yp.std()),
        }
    }

    # Baseline: predict mean
    yp_mean = np.full_like(yx, yt.mean())
    result["baseline_mean"] = {
        "mae": float(mean_absolute_error(yx, yp_mean)),
        "rmse": float(np.sqrt(mean_squared_error(yx, yp_mean))),
        "r2": float(r2_score(yx, yp_mean)),
    }

    # Per-subtype
    sub_test = subtypes[test_mask]
    for st in SUBTYPES:
        sm = (sub_test == st)
        if sm.sum() >= 3:
            result[f"test_{st}"] = {
                "mae": float(mean_absolute_error(yx[sm], yp[sm])),
                "rmse": float(np.sqrt(mean_squared_error(yx[sm], yp[sm]))),
                "r2": float(r2_score(yx[sm], yp[sm])),
                "pearson_r": float(pearsonr(yx[sm], yp[sm])[0]),
                "n": int(sm.sum()),
            }

    return result


def main():
    masks, subtypes_arr, emb_dict, y_reg = load_data()

    print(f"{'='*60}")
    print(f"  Interval 回归实验")
    print(f"  Target: iso_interval_days (days)")
    for s in ["train", "val", "test"]:
        m = masks[s]
        print(f"  {s}: n={m.sum():>4}  mean={y_reg[m].mean():.0f}  std={y_reg[m].std():.0f}")
    print(f"{'='*60}")

    all_results = {}

    for emb_name in EMB_TYPES:
        if emb_dict[emb_name] is None:
            continue
        X = emb_dict[emb_name]
        dim = X.shape[1]

        for model_name in ["Ridge", "Lasso"]:
            print(f"\n  --- {emb_name} + {model_name} (dim={dim}) ---")
            result = train_and_eval(
                X, y_reg, masks["train"], masks["val"], masks["test"],
                subtypes_arr, emb_name, model_name
            )
            key = f"{emb_name}_{model_name}"
            all_results[key] = result

            r = result["test"]
            b = result["baseline_mean"]
            print(f"    alpha={result[f'best_alpha']:.4f}")
            print(f"    MAE={r['mae']:.0f}  (baseline={b['mae']:.0f})")
            print(f"    RMSE={r['rmse']:.0f}  (baseline={b['rmse']:.0f})")
            print(f"    R²={r['r2']:.4f}")
            print(f"    Pearson r={r['pearson_r']:.4f}  (p={r['pearson_p']:.2e})")

            for st in SUBTYPES:
                if f"test_{st}" in result:
                    rs = result[f"test_{st}"]
                    print(f"    {st}: R²={rs['r2']:.4f}  r={rs['pearson_r']:.4f}  n={rs['n']}")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  回归对比汇总")
    print(f"  {'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Pearson r':>10}")
    print(f"  {'-'*55}")
    b = all_results.get(f"{EMB_TYPES[0]}_Ridge", {}).get("baseline_mean", {})
    if b:
        print(f"  {'baseline (mean)':<20} {b['mae']:>8.0f} {b['rmse']:>8.0f} {b['r2']:>8.4f} {'—':>10}")
    for key, result in all_results.items():
        r = result["test"]
        print(f"  {key:<20} {r['mae']:>8.0f} {r['rmse']:>8.0f} {r['r2']:>8.4f} {r['pearson_r']:>10.4f}")

    # 保存
    out_path = OUT_DIR / "eval_interval_reg.json"
    json.dump(all_results, open(out_path, "w"), indent=2, default=str)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
