"""MLP 回归实验 — interval 回归（预测 iso_interval_days）

对每个 (embedding 类型 × hidden_dims 配置 × seed) 组合：
  - StandardScaler + IntervalMLP + MSELoss
  - 评估：MAE, RMSE, R², Pearson r
  - 输出 eval_interval_mlp_reg.json
"""

import numpy as np
import pandas as pd
import json
import sys
import torch
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from ESM_clf.interval_exp.mlp_model import (train_mlp, predict_mlp,
                                             eval_reg_metrics)
from sklearn.preprocessing import StandardScaler


def load_data():
    """加载 embedding、标签和 split（与 prepare_interval_data.py 一致）。"""
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

    # 过滤 NaN
    valid_mask = ~np.isnan(y_reg)
    print(f"  移除 NaN: {valid_mask.sum()} / {len(y_reg)}")
    y_reg = y_reg[valid_mask]
    split_arr = split_arr[valid_mask]
    subtypes_arr = subtypes_arr[valid_mask]
    for etype in EMB_TYPES:
        if emb[etype] is not None:
            emb[etype] = emb[etype][valid_mask]

    masks = {s: (split_arr == s) for s in ["train", "val", "test"]}
    return masks, subtypes_arr, emb, y_reg


def train_one_config(X, y, masks, subtypes, emb_name, hidden_dims, seed,
                     device="cuda"):
    """单次 (embedding, hidden_dims, seed) 的训练/评估。"""
    train_m, val_m, test_m = masks["train"], masks["val"], masks["test"]

    # 标准化
    scl = StandardScaler()
    Xt = scl.fit_transform(X[train_m])
    Xv = scl.transform(X[val_m])
    Xx = scl.transform(X[test_m])

    in_dim = X.shape[1]
    out_dim = 1  # 回归输出单值

    model, history = train_mlp(
        Xt, y[train_m], Xv, y[val_m],
        in_dim=in_dim, out_dim=out_dim,
        hidden_dims=hidden_dims, dropout=MLP_DROPOUT,
        lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY,
        epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE,
        patience=MLP_EARLY_STOP_PATIENCE, seed=seed,
        task="reg", device=device,
    )

    yp = predict_mlp(model, Xx, task="reg", device=device)
    yt = y[test_m]

    result = {
        "hidden_dims": hidden_dims,
        "seed": seed,
        "epochs_run": len(history["train_loss"]),
        "best_val_loss": float(min(history["val_loss"])),
        "test": eval_reg_metrics(yt, yp),
    }

    # Baseline: predict train mean
    yp_mean = np.full_like(yt, y[train_m].mean())
    result["baseline_mean"] = eval_reg_metrics(yt, yp_mean)

    # Per-subtype
    sub_test = subtypes[test_m]
    for st in SUBTYPES:
        sm = (sub_test == st)
        if sm.sum() >= 3:
            result[f"test_{st}"] = eval_reg_metrics(yt[sm], yp[sm])

    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    masks, subtypes_arr, emb_dict, y_reg = load_data()

    print(f"\n{'='*60}")
    print(f"  MLP Interval 回归实验")
    print(f"  Target: iso_interval_days (days)")
    for s in ["train", "val", "test"]:
        m = masks[s]
        print(f"  {s}: n={m.sum():>4}  mean={y_reg[m].mean():.0f}  "
              f"std={y_reg[m].std():.0f}")
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

        for hd in MLP_HIDDEN_CONFIGS:
            hd_key = "x".join(str(d) for d in hd)
            key = f"{emb_name}_h{hd_key}"
            all_results[key] = {"config": {"hidden_dims": hd, "seeds": []}}
            seed_results = []

            for seed in MLP_SEEDS:
                print(f"    hidden={hd}  seed={seed}  ", end="", flush=True)
                r = train_one_config(X, y_reg, masks, subtypes_arr,
                                     emb_name, hd, seed, device)
                seed_results.append(r)
                tr = r["test"]
                print(f"R²={tr['r2']:.4f}  Pearson_r={tr['pearson_r']:.4f}  "
                      f"MAE={tr['mae']:.0f}  epochs={r['epochs_run']}")

            # 平均
            avg_test = {}
            metric_keys = ["mae", "rmse", "r2", "pearson_r", "n"]
            for mk in metric_keys:
                vals = [s["test"][mk] for s in seed_results]
                avg_test[mk] = float(np.mean(vals))
                avg_test[f"{mk}_std"] = float(np.std(vals))

            all_results[key]["test_avg"] = avg_test
            all_results[key]["test_per_seed"] = [
                {"seed": s["seed"], "test": s["test"]} for s in seed_results
            ]

            print(f"    => avg  R²={avg_test['r2']:.4f}±{avg_test['r2_std']:.4f}  "
                  f"r={avg_test['pearson_r']:.4f}±{avg_test['pearson_r_std']:.4f}  "
                  f"MAE={avg_test['mae']:.0f}")

    # ── 保存 ──
    out_path = OUT_DIR / "eval_interval_mlp_reg.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
