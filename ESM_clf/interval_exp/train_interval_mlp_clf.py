"""MLP 分类实验 — interval 4 分类（<1yr / 1-3yr / 3-5yr / 5yr+）

对每个 (embedding 类型 × hidden_dims 配置 × seed) 组合：
  - StandardScaler + IntervalMLP + CrossEntropyLoss
  - 评估：accuracy, balanced accuracy, macro F1, per-class precision/recall
  - 输出 eval_interval_mlp_clf.json
"""

import numpy as np
import pandas as pd
import json
import sys
import torch
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from ESM_clf.interval_exp.mlp_model import (train_mlp, predict_mlp,
                                             eval_clf_metrics)
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
    out_dim = len(INTERVAL_LABELS)

    model, history = train_mlp(
        Xt, y[train_m], Xv, y[val_m],
        in_dim=in_dim, out_dim=out_dim,
        hidden_dims=hidden_dims, dropout=MLP_DROPOUT,
        lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY,
        epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE,
        patience=MLP_EARLY_STOP_PATIENCE, seed=seed,
        task="clf", device=device,
    )

    yp = predict_mlp(model, Xx, task="clf", device=device)
    yt = y[test_m]

    result = {
        "hidden_dims": hidden_dims,
        "seed": seed,
        "epochs_run": len(history["train_loss"]),
        "best_val_loss": float(min(history["val_loss"])),
        "test": eval_clf_metrics(yt, yp, INTERVAL_LABELS),
    }

    # Per-subtype
    sub_test = subtypes[test_m]
    for st in SUBTYPES:
        sm = (sub_test == st)
        if sm.sum() >= 2 and len(np.unique(yt[sm])) > 1:
            result[f"test_{st}"] = eval_clf_metrics(yt[sm], yp[sm], INTERVAL_LABELS)

    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    masks, subtypes_arr, emb_dict, y_cat = load_data()

    print(f"{'='*60}")
    print(f"  MLP Interval 分类实验")
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
                r = train_one_config(X, y_cat, masks, subtypes_arr,
                                     emb_name, hd, seed, device)
                seed_results.append(r)
                tr = r["test"]
                print(f"acc={tr['acc']:.4f}  bal_acc={tr['balanced_acc']:.4f}  "
                      f"macro_f1={tr['macro_f1']:.4f}  "
                      f"epochs={r['epochs_run']}")

            # 平均
            avg_test = {}
            metric_keys = ["acc", "balanced_acc", "macro_f1", "n"]
            for mk in metric_keys:
                vals = [s["test"][mk] for s in seed_results]
                avg_test[mk] = float(np.mean(vals))
                avg_test[f"{mk}_std"] = float(np.std(vals))

            all_results[key]["test_avg"] = avg_test
            all_results[key]["test_per_seed"] = [
                {"seed": s["seed"], "test": s["test"]} for s in seed_results
            ]

            print(f"    => avg  acc={avg_test['acc']:.4f}±{avg_test['acc_std']:.4f}  "
                  f"bal_acc={avg_test['balanced_acc']:.4f}±{avg_test['balanced_acc_std']:.4f}  "
                  f"macro_f1={avg_test['macro_f1']:.4f}±{avg_test['macro_f1_std']:.4f}")

    # ── 保存 ──
    out_path = OUT_DIR / "eval_interval_mlp_clf.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
