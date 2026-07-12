"""Probe — ESM-2 150M L1/L3/L1L3 + Ridge LR

对 label_is_jump 和 label_is_jump_human 分别：
  - H1+H3 train/val/test (70/15/15)
  - H5/H7 holdout 迁移评估
"""

import numpy as np
import pandas as pd
import json
import sys
sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


# ═══════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════

def metrics(y_true, y_pred_prob):
    """返回 dict of metrics；单类时 AUC=None"""
    b = (y_pred_prob >= 0.5).astype(int)
    m = {
        "acc": accuracy_score(y_true, b),
        "prec": precision_score(y_true, b, zero_division=0),
        "rec": recall_score(y_true, b, zero_division=0),
        "f1": f1_score(y_true, b, zero_division=0),
        "n": len(y_true),
        "pos_rate": float(y_true.mean()),
    }
    if len(np.unique(y_true)) > 1:
        m["auc"] = roc_auc_score(y_true, y_pred_prob)
    else:
        m["auc"] = None
    return m


# ═══════════════════════════════════════════════════════════
# 加载
# ═══════════════════════════════════════════════════════════

def load_data():
    """加载 embedding、标签、split 和原始 CSV（含 subtype）"""
    split_df = pd.read_csv(SPLIT_CSV)
    data_df = pd.read_csv(DATA_CSV)
    # embedding 按 CSV 行顺序存储，直接按行对齐
    emb = {}
    for etype in ["L1", "L3", "L1L3"]:
        path = OUT_DIR / f"esm_emb_150M_{etype}.npy"
        emb[etype] = np.load(path) if path.exists() else None

    labels = {}
    for lc in LABEL_COLS:
        path = OUT_DIR / f"labels_{lc}.npy"
        if path.exists():
            labels[lc] = np.load(path)

    return split_df, data_df, emb, labels


def get_split_masks(split_df):
    """返回各 split 的 bool mask

    - train/val/test: H1+H3 only
    - h5_holdout / h7_holdout: 对应的 holdout
    """
    masks = {}
    for s in ["train", "val", "test"]:
        masks[s] = (split_df["split"] == s).values
    masks["h5_holdout"] = (split_df["split"] == "h5_holdout").values
    masks["h7_holdout"] = (split_df["split"] == "h7_holdout").values
    return masks


# ═══════════════════════════════════════════════════════════
# 训练 + 评估
# ═══════════════════════════════════════════════════════════

def train_and_eval(X, y, train_mask, val_mask, test_mask, holdout_masks,
                   subtypes, label_name, emb_name):
    """单次 (embedding, label) 组合的训练/评估流水线"""

    Xt, yt = X[train_mask], y[train_mask]
    Xv, yv = X[val_mask], y[val_mask] if val_mask.any() else (None, None)
    Xx, yx = X[test_mask], y[test_mask]

    # 标准化
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    Xx_s = scl.transform(Xx)
    Xv_s = scl.transform(Xv) if Xv is not None else None

    # GridSearchCV 选最佳 C
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                      scoring="roc_auc")
    gs.fit(Xt_s, yt)
    best_c = gs.best_params_["C"]

    # ── Test set ──
    test_preds = gs.predict_proba(Xx_s)[:, 1]
    result = {}
    result["test"] = metrics(yx, test_preds)
    result["best_C"] = best_c

    # Per-subtype test
    sub_test = subtypes[test_mask]
    for st in ["H1", "H3"]:
        sm = (sub_test == st)
        if sm.sum() >= 2 and len(np.unique(yx[sm])) > 1:
            result[f"test_{st}"] = metrics(yx[sm], test_preds[sm])

    # ── Holdout (迁移) ──
    for ho_name, ho_mask in holdout_masks.items():
        if not ho_mask.any():
            continue
        Xh, yh = X[ho_mask], y[ho_mask]
        Xh_s = scl.transform(Xh)
        ho_preds = gs.predict_proba(Xh_s)[:, 1]

        if len(np.unique(yh)) > 1:
            result[ho_name] = metrics(yh, ho_preds)

        # Per-subtype holdout
        sub_ho = subtypes[ho_mask]
        for st in ["H5", "H7"]:
            sm = (sub_ho == st)
            if sm.sum() >= 2 and len(np.unique(yh[sm])) > 1:
                result[f"{ho_name}_{st}"] = metrics(yh[sm], ho_preds[sm])

    return result


def multi_seed_eval(X, y, train_mask, test_mask, best_c, n_seeds=5):
    """用 best_c 跑多 seed 取 mean±std AUC"""
    Xt, yt = X[train_mask], y[train_mask]
    Xx, yx = X[test_mask], y[test_mask]
    aucs = []
    for seed in range(n_seeds):
        scl = StandardScaler()
        lr = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                                max_iter=5000, random_state=seed)
        Xt_s = scl.fit_transform(Xt)
        Xx_s = scl.transform(Xx)
        lr.fit(Xt_s, yt)
        preds = lr.predict_proba(Xx_s)[:, 1]
        if len(np.unique(yx)) > 1:
            aucs.append(roc_auc_score(yx, preds))
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    split_df, data_df, emb_dict, label_dict = load_data()
    masks = get_split_masks(split_df)
    subtypes = data_df["subtype"].values

    all_results = {}

    for label_col in LABEL_COLS:
        if label_col not in label_dict:
            print(f"跳过 {label_col}: 标签文件不存在")
            continue
        y = label_dict[label_col]

        print(f"\n{'='*60}")
        print(f"  Label: {label_col}")
        print(f"{'='*60}")
        for s in ["train", "val", "test", "h5_holdout", "h7_holdout"]:
            m = masks[s]
            print(f"  {s:>12s}: n={m.sum():5d}  pos={y[m].sum():5d} ({y[m].mean()*100:.1f}%)")

        label_results = {}

        for emb_name in ["L1", "L3", "L1L3"]:
            if emb_dict[emb_name] is None:
                continue
            X = emb_dict[emb_name]
            dim = X.shape[1]

            print(f"\n  --- {emb_name} (dim={dim}) ---")

            result = train_and_eval(
                X, y, masks["train"], masks["val"], masks["test"],
                {"h5_holdout": masks["h5_holdout"], "h7_holdout": masks["h7_holdout"]},
                subtypes, label_col, emb_name
            )

            # Multi-seed
            seed_res = multi_seed_eval(
                X, y, masks["train"], masks["test"], result["best_C"]
            )
            result["5seed_test_auc"] = seed_res

            best_c = result.pop("best_C")
            result["best_C"] = best_c

            # 打印
            print(f"    Best C: {best_c}")
            print(f"    {'Split':<22} {'AUC':>8} {'Acc':>8} {'F1':>8} {'n':>6}")
            print(f"    {'-'*50}")
            for k, v in result.items():
                if k in ("5seed_test_auc", "best_C"):
                    continue
                auc_s = f"{v['auc']:.4f}" if v.get("auc") is not None else "N/A"
                print(f"    {k:<22} {auc_s:>8} {v['acc']:>8.4f} {v['f1']:>8.4f} {v['n']:>6}")
            print(f"    5-seed test AUC: {seed_res['auc_mean']:.4f} ± {seed_res['auc_std']:.4f}")

            label_results[emb_name] = result

        all_results[label_col] = label_results

    # ── 汇总对比表 ──
    print(f"\n{'='*70}")
    print(f"  汇总: ESM-2 150M + Ridge LR vs. Borkenhagen CNN")
    print(f"{'='*70}")
    print(f"  {'Task':<24} {'Emb':>6} {'Test AUC':>10} {'5-seed':>18} {'H5 AUC':>9} {'H7 AUC':>9}")
    print(f"  {'-'*70}")
    # Borkenhagen baseline
    print(f"  {'Borkenhagen CNN (jump)':<24} {'':>6} {'0.9920':>10} {'':>18} {'0.5410':>9} {'0.6620':>9}")
    print(f"  {'Borkenhagen CNN (jump_h)':<24} {'':>6} {'0.9900':>10} {'':>18} {'0.5500':>9} {'0.7240':>9}")
    print(f"  {'-'*70}")

    for label_col in LABEL_COLS:
        if label_col not in all_results:
            continue
        for emb_name in ["L1", "L3", "L1L3"]:
            if emb_name not in all_results[label_col]:
                continue
            r = all_results[label_col][emb_name]
            if label_col == "label_is_jump":
                task_short = "jump"
            else:
                task_short = "jump_h"

            test_auc = f"{r['test']['auc']:.4f}" if r['test'].get('auc') else "N/A"
            seed_str = f"{r['5seed_test_auc']['auc_mean']:.4f}±{r['5seed_test_auc']['auc_std']:.4f}"
            h5_auc = f"{r['h5_holdout']['auc']:.4f}" if r.get('h5_holdout', {}).get('auc') else "N/A"
            h7_auc = f"{r['h7_holdout']['auc']:.4f}" if r.get('h7_holdout', {}).get('auc') else "N/A"
            print(f"  {f'ESM2-150M {emb_name} ({task_short})':<24} {emb_name:>6} {test_auc:>10} {seed_str:>18} {h5_auc:>9} {h7_auc:>9}")

    # ── 保存 ──
    out_path = OUT_DIR / "eval_results.json"
    json.dump(all_results, open(out_path, "w"), indent=2, default=str)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
