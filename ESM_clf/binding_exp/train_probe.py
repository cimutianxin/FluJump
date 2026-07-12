"""Probe — EsmModel L1, L3, L1+L3 concat + 多 seed"""
import numpy as np, pandas as pd, json, sys, glob
sys.path.insert(0, ".")
from ESM_clf.binding_exp.config import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

split_df = pd.read_csv(SPLIT_CSV)
y = np.load(LABEL_FILE)
tm = split_df["split"].values == "train"
xm = split_df["split"].values == "test"
yt, yx = y[tm], y[xm]
print(f"train={len(yt)}(pos={yt.sum()}) test={len(yx)}(pos={yx.sum()})")

emb_files = sorted(glob.glob(str(OUT_DIR / "esm_emb_150M_esmmodel_*.npy")))
results = {}

for f in emb_files:
    name = f.split("esmmodel_")[-1].replace(".npy", "")
    X = np.load(f)
    dim = X.shape[1]
    Xt, Xx = X[tm], X[xm]
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt); Xx_s = scl.transform(Xx)

    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                      scoring="roc_auc")
    gs.fit(Xt_s, yt)
    p = gs.predict_proba(Xx_s)[:, 1]
    auc = roc_auc_score(yx, p)
    acc = accuracy_score(yx, p >= 0.5)
    print(f"  {name} (dim={dim}): C={gs.best_params_['C']}, AUC={auc:.4f}, Acc={acc:.4f}")

    aucs = []
    for seed in range(5):
        lr2 = LogisticRegression(C=gs.best_params_["C"], penalty="l2", solver="lbfgs",
                                  max_iter=5000, random_state=seed)
        lr2.fit(scl.fit_transform(Xt), yt)
        aucs.append(roc_auc_score(yx, lr2.predict_proba(scl.transform(Xx))[:, 1]))
    results[name] = {"auc": auc, "acc": acc, "C": gs.best_params_["C"], "dim": dim,
                     "auc_mean": np.mean(aucs), "auc_std": np.std(aucs)}
    print(f"    5-seed: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

print(f"\n{'='*55}")
print(f"{'Embedding':<15} {'dim':>5} {'AUC':>8} {'5-seed':>16} {'Acc':>8}")
print(f"{'-'*50}")
print(f"{'Borkenhagen CNN':<15} {'':>5} {'0.9300':>8} {'':>16} {'0.9400':>8}")
for name, r in results.items():
    seed_str = f"{r['auc_mean']:.4f}+/-{r['auc_std']:.4f}"
    print(f"{name:<15} {r['dim']:>5} {r['auc']:>8.4f} {seed_str:>16} {r['acc']:>8.4f}")

json.dump(results, open(OUT_DIR / "probe_final.json", "w"), indent=2)
