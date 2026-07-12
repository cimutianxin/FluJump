"""Ordinal Regression — interval 分类（用顺序信息）

方法：
- Linear Ordinal: mord.LogisticAT（proportional odds model, L2 正则化）
- MLP Ordinal: OrdinalMLP（单 score + K-1 学习阈值 + NLL loss）

对比：multinomial (Linear/MLP) vs ordinal (Linear/MLP)
输出 eval_interval_ordinal.json
"""

import numpy as np
import pandas as pd
import json
import sys
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from ESM_clf.interval_exp.mlp_model import eval_clf_metrics
from ESM_clf.interval_exp.ordinal_model import (
    train_ordinal_mlp, predict_ordinal_mlp,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)


def load_data():
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
        emb[etype] = np.load(path)[idx] if path.exists() else None

    masks = {s: (split_arr == s) for s in ["train", "val", "test"]}
    return masks, subtypes_arr, emb, y_cat


# ── Linear Ordinal (mord) ──

def run_mord(X, y, masks, subtypes, emb_name):
    """mord.LogisticAT + GridSearchCV。"""
    from mord import LogisticAT

    train_m, val_m, test_m = masks["train"], masks["val"], masks["test"]

    scl = StandardScaler()
    Xt = scl.fit_transform(X[train_m])
    Xv = scl.transform(X[val_m])
    Xx = scl.transform(X[test_m])

    # 合并 train+val 用于 GridSearchCV（mord 不支持 validation set in GridSearchCV，
    # 我们用 train 做 CV，val 不用）
    model = LogisticAT(max_iter=5000)
    gs = GridSearchCV(model, {"alpha": [1e-2, 1e-1, 1.0, 10.0, 100.0]},
                      cv=CV_FOLDS, scoring="f1_macro")
    gs.fit(Xt, y[train_m])

    yp = gs.predict(Xx)
    yt = y[test_m]

    result = {
        "best_alpha": float(gs.best_params_["alpha"]),
        "test": eval_clf_metrics(yt, yp, INTERVAL_LABELS),
    }

    sub_test = subtypes[test_m]
    for st in SUBTYPES:
        sm = (sub_test == st)
        if sm.sum() >= 2 and len(np.unique(yt[sm])) > 1:
            result[f"test_{st}"] = eval_clf_metrics(yt[sm], yp[sm], INTERVAL_LABELS)

    return result


# ── MLP Ordinal ──

def run_ordinal_mlp(X, y, masks, subtypes, emb_name, hidden_dims, seed,
                    device="cuda"):
    train_m, val_m, test_m = masks["train"], masks["val"], masks["test"]

    scl = StandardScaler()
    Xt = scl.fit_transform(X[train_m])
    Xv = scl.transform(X[val_m])
    Xx = scl.transform(X[test_m])

    in_dim = X.shape[1]
    num_classes = len(INTERVAL_LABELS)

    model, history, best_val_loss = train_ordinal_mlp(
        Xt, y[train_m], Xv, y[val_m],
        in_dim=in_dim, num_classes=num_classes,
        hidden_dims=hidden_dims, dropout=MLP_DROPOUT,
        lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY,
        epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE,
        patience=MLP_EARLY_STOP_PATIENCE, seed=seed,
        device=device,
    )

    yp = predict_ordinal_mlp(model, Xx, device=device)
    yt = y[test_m]

    result = {
        "hidden_dims": hidden_dims,
        "seed": seed,
        "epochs_run": len(history["train_loss"]),
        "best_val_loss": float(best_val_loss),
        "test": eval_clf_metrics(yt, yp, INTERVAL_LABELS),
    }

    # Learned thresholds
    model.eval()
    with torch.no_grad():
        result["thresholds"] = model.get_thresholds().cpu().numpy().tolist()

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
    print(f"  Ordinal Regression — Interval 分类")
    print(f"  Classes: {INTERVAL_LABELS} (ordered)")
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

        # ── Linear Ordinal (mord) ──
        print(f"  [mord LogisticAT]")
        mord_res = run_mord(X, y_cat, masks, subtypes_arr, emb_name)
        all_results[f"{emb_name}_mord"] = mord_res
        tr = mord_res["test"]
        print(f"    alpha={mord_res['best_alpha']:.4f}  "
              f"acc={tr['acc']:.4f}  bal_acc={tr['balanced_acc']:.4f}  "
              f"macro_f1={tr['macro_f1']:.4f}")

        # ── MLP Ordinal ──
        for hd in MLP_HIDDEN_CONFIGS:
            hd_key = "x".join(str(d) for d in hd)
            key = f"{emb_name}_ordinal_h{hd_key}"
            seed_results = []

            print(f"  [OrdinalMLP hidden={hd}]")
            for seed in MLP_SEEDS:
                print(f"    seed={seed}  ", end="", flush=True)
                r = run_ordinal_mlp(X, y_cat, masks, subtypes_arr,
                                    emb_name, hd, seed, device)
                seed_results.append(r)
                tr = r["test"]
                print(f"acc={tr['acc']:.4f}  bal_acc={tr['balanced_acc']:.4f}  "
                      f"macro_f1={tr['macro_f1']:.4f}  "
                      f"epochs={r['epochs_run']}  "
                      f"thresh={[f'{t:.2f}' for t in r['thresholds']]}")

            # 平均
            avg_test = {}
            for mk in ["acc", "balanced_acc", "macro_f1", "n"]:
                vals = [s["test"][mk] for s in seed_results]
                avg_test[mk] = float(np.mean(vals))
                avg_test[f"{mk}_std"] = float(np.std(vals))

            all_results[key] = {
                "config": {"hidden_dims": hd, "seeds": MLP_SEEDS},
                "test_avg": avg_test,
                "test_per_seed": [
                    {"seed": s["seed"], "test": s["test"],
                     "thresholds": s.get("thresholds", [])}
                    for s in seed_results
                ],
            }

            print(f"    => avg  acc={avg_test['acc']:.4f}±{avg_test['acc_std']:.4f}  "
                  f"bal_acc={avg_test['balanced_acc']:.4f}±{avg_test['balanced_acc_std']:.4f}  "
                  f"macro_f1={avg_test['macro_f1']:.4f}±{avg_test['macro_f1_std']:.4f}")

    # ── 保存 ──
    out_path = OUT_DIR / "eval_interval_ordinal.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
