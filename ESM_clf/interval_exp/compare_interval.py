"""Regression → Classification 对比：将回归预测分箱与分类模型对比"""

import numpy as np
import pandas as pd
import json
import sys
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


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
    y_reg = np.load(OUT_DIR / "labels_interval_days.npy")

    masks = {s: (split_arr == s) for s in ["train", "val", "test"]}

    emb = {}
    for etype in EMB_TYPES:
        path = EMB_DIR / f"esm_emb_150M_{etype}.npy"
        if path.exists():
            emb[etype] = np.load(path)[idx]
        else:
            emb[etype] = None

    return masks, subtypes_arr, emb, y_cat, y_reg


def reg_to_cat(y_reg_pred):
    """将回归预测的连续天数转为 interval 类别"""
    abs_pred = np.abs(y_reg_pred)
    return pd.cut(abs_pred, bins=INTERVAL_BINS, labels=INTERVAL_LABELS,
                  right=False).astype(str)


def main():
    masks, subtypes_arr, emb_dict, y_cat, y_reg = load_data()

    # ── 加载最佳分类模型参数 ──
    clf_results = json.load(open(OUT_DIR / "eval_interval_clf.json"))

    # ── 加载回归模型参数 ──
    reg_results = json.load(open(OUT_DIR / "eval_interval_reg.json"))

    compare = {}

    print(f"{'='*70}")
    print(f"  回归分箱 vs 分类 对比")
    print(f"  Classes: {INTERVAL_LABELS}")
    print(f"{'='*70}")

    for emb_name in ["L1", "L3", "L1L3"]:
        if emb_dict[emb_name] is None:
            continue
        X = emb_dict[emb_name]
        Xt, yt = X[masks["train"]], y_cat[masks["train"]]
        Xx, yx = X[masks["test"]], y_cat[masks["test"]]
        yr_train = y_reg[masks["train"]]
        yr_test = y_reg[masks["test"]]

        scl = StandardScaler()
        Xt_s = scl.fit_transform(Xt)
        Xx_s = scl.transform(Xx)

        print(f"\n--- {emb_name} ---")

        # === 分类模型（最佳参数） ===
        clf_best_c = clf_results[emb_name]["best_C"]
        clf = LogisticRegression(C=clf_best_c, penalty="l2", solver="lbfgs",
                                 max_iter=5000, multi_class="multinomial",
                                 random_state=RANDOM_SEED)
        clf.fit(Xt_s, yt)
        yp_clf = clf.predict(Xx_s)

        acc_clf = accuracy_score(yx, yp_clf)
        bal_clf = balanced_accuracy_score(yx, yp_clf)
        f1_clf = f1_score(yx, yp_clf, average="macro")

        print(f"  分类 (C={clf_best_c}): acc={acc_clf:.4f}  bal_acc={bal_clf:.4f}  macro_f1={f1_clf:.4f}")

        # === 回归模型 → 分箱 ===
        for model_name in ["Ridge", "Lasso"]:
            key = f"{emb_name}_{model_name}"
            if key not in reg_results:
                continue

            alpha = reg_results[key]["best_alpha"]
            if model_name == "Ridge":
                reg = Ridge(alpha=alpha, random_state=RANDOM_SEED)
            else:
                reg = Lasso(alpha=alpha, random_state=RANDOM_SEED, max_iter=5000)

            reg.fit(Xt_s, yr_train)
            yp_reg = reg.predict(Xx_s)

            yp_reg_cat = reg_to_cat(yp_reg)
            cat_map = {l: i for i, l in enumerate(INTERVAL_LABELS)}
            yp_reg_int = np.array([cat_map[c] for c in yp_reg_cat])

            acc_reg = accuracy_score(yx, yp_reg_int)
            bal_reg = balanced_accuracy_score(yx, yp_reg_int)
            f1_reg = f1_score(yx, yp_reg_int, average="macro")

            delta_acc = acc_reg - acc_clf
            delta_f1 = f1_reg - f1_clf

            print(f"  回归→分箱 ({model_name}, alpha={alpha}): acc={acc_reg:.4f}  bal_acc={bal_reg:.4f}  macro_f1={f1_reg:.4f}")
            print(f"    Δ vs 分类: Δacc={delta_acc:+.4f}  Δf1={delta_f1:+.4f}")

            compare[f"{emb_name}_{model_name}"] = {
                "reg2cat_acc": acc_reg,
                "reg2cat_bal_acc": bal_reg,
                "reg2cat_macro_f1": f1_reg,
                "clf_acc": acc_clf,
                "clf_bal_acc": bal_clf,
                "clf_macro_f1": f1_clf,
                "delta_acc": delta_acc,
                "delta_f1": delta_f1,
            }

    # ── 保存 ──
    out_path = OUT_DIR / "compare_reg_vs_clf.json"
    json.dump(compare, open(out_path, "w"), indent=2, default=str)
    print(f"\n  → {out_path}")
    print("Done")


if __name__ == "__main__":
    main()
