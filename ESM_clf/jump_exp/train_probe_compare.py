"""对比 raw vs aligned ESM-2 embedding 的 Ridge LR 表现"""
import numpy as np, pandas as pd, json, sys
sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def metrics(y_true, y_pred_prob):
    b = (y_pred_prob >= 0.5).astype(int)
    m = {"acc": accuracy_score(y_true, b), "f1": f1_score(y_true, b, zero_division=0),
         "prec": precision_score(y_true, b, zero_division=0),
         "rec": recall_score(y_true, b, zero_division=0),
         "n": len(y_true), "pos_rate": float(y_true.mean())}
    m["auc"] = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else None
    return m

# ── 加载 ──
split_df = pd.read_csv(SPLIT_CSV)
data_df = pd.read_csv(DATA_CSV)
subtypes = data_df["subtype"].values
masks = {s: (split_df["split"] == s).values for s in ["train","val","test","h5_holdout","h7_holdout"]}

for label_col in LABEL_COLS:
    y = np.load(OUT_DIR / f"labels_{label_col}.npy")
    print(f"\n{'='*70}")
    print(f"  {label_col}")
    for s in ["train","val","test","h5_holdout","h7_holdout"]:
        m = masks[s]
        print(f"  {s:>12s}: n={m.sum():5d}  pos={y[m].sum():5d} ({y[m].mean()*100:.1f}%)")

    results = {}

    for variant in ["raw", "aligned"]:
        prefix = "esm_emb_150M_aligned_" if variant == "aligned" else "esm_emb_150M_"
        print(f"\n  {'='*50}")
        print(f"  [{variant.upper()}]")

        for emb_name in ["L1", "L3", "L1L3"]:
            path = OUT_DIR / f"{prefix}{emb_name}.npy"
            if not path.exists():
                print(f"    {emb_name}: 文件不存在，跳过")
                continue
            X = np.load(path)
            Xt, yt = X[masks["train"]], y[masks["train"]]
            Xx, yx = X[masks["test"]], y[masks["test"]]

            scl = StandardScaler()
            Xt_s = scl.fit_transform(Xt)
            lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=RANDOM_SEED)
            gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                              cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                              scoring="roc_auc")
            gs.fit(Xt_s, yt)

            Xx_s = scl.transform(Xx)
            tp = gs.predict_proba(Xx_s)[:, 1]

            # H5
            X5_s = scl.transform(X[masks["h5_holdout"]])
            p5 = gs.predict_proba(X5_s)[:, 1]
            auc5 = roc_auc_score(y[masks["h5_holdout"]], p5)

            # H7 — 原始 + -logit 翻转
            X7_s = scl.transform(X[masks["h7_holdout"]])
            y7 = y[masks["h7_holdout"]]
            logits7 = gs.decision_function(X7_s)
            auc7_raw = roc_auc_score(y7, gs.predict_proba(X7_s)[:, 1])
            auc7_flip = roc_auc_score(y7, -logits7)

            r = {"test_auc": roc_auc_score(yx, tp), "best_C": gs.best_params_["C"],
                 "h5_auc": auc5, "h7_auc_raw": auc7_raw, "h7_auc_flip": auc7_flip}
            results[f"{variant}_{emb_name}"] = r

            print(f"    {emb_name:>5s}  dim={X.shape[1]:>4d}  C={gs.best_params_['C']:.4f}  "
                  f"test_AUC={r['test_auc']:.4f}  H5_AUC={auc5:.4f}  "
                  f"H7_raw={auc7_raw:.4f}  H7_flip={auc7_flip:.4f}")

        # ── 汇总对比 ──
        print(f"\n{'='*70}")
        print(f"  {label_col} — raw vs aligned 对比")
        print(f"  {'Embedding':<20} {'Test AUC':>10} {'H5 AUC':>9} {'H7_raw':>9} {'H7_flip':>9}")
        print(f"  {'-'*60}")
        # Borkenhagen baseline
        print(f"  {'CNN (baseline)':<20} {'0.99':>10} {'0.54-0.55':>9} {'0.66-0.72':>9} {'—':>9}")
        for variant in ["raw", "aligned"]:
            for emb_name in ["L1", "L3", "L1L3"]:
                k = f"{variant}_{emb_name}"
                if k in results:
                    r = results[k]
                    label = f"{variant}-{emb_name}"
                    print(f"  {label:<20} {r['test_auc']:>10.4f} {r['h5_auc']:>9.4f} "
                          f"{r['h7_auc_raw']:>9.4f} {r['h7_auc_flip']:>9.4f}")

        # 保存
        out_path = OUT_DIR / f"compare_raw_vs_aligned_{label_col}.json"
        json.dump(results, open(out_path, "w"), indent=2, default=str)

print("\nDone")
