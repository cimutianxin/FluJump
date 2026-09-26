"""机制先验实验：逐层 probing——亚型身份 vs 宿主信号在深度上的分布

目的：为"中层 embedding 才能跨亚型迁移"提供事前假设依据。
若中层亚型身份最弱、宿主信号最强，则 L13 能预测 H7 就从"事后挑层"
变成"被机制预测并验证"。

对每层（31 层全层 embedding）：
  1. 亚型 probe：5-fold CV 4 分类（H1/H3/H5/H7）准确率——越高 = 亚型身份越强
  2. 宿主 probe：5-fold CV 3 分类（avian/human/other）准确率
  3. 叠加层扫描的 jump/jump_human test/H5/H7 AUC（读 layer_sweep.json）

输出：output/layer_probing.json + output/figs/layer_probing.png
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

FIG_DIR = OUT_DIR / "figs"


def cv_acc(X, y, n_jobs=-1):
    """5-fold stratified CV 准确率（StandardScaler + LR）"""
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    preds = np.zeros_like(y)
    for tr, te in skf.split(X, y):
        scl = StandardScaler()
        lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000,
                                random_state=RANDOM_SEED)
        lr.fit(scl.fit_transform(X[tr]), y[tr])
        preds[te] = lr.predict(scl.transform(X[te]))
    return float(accuracy_score(y, preds))


def main():
    arr = np.load(OUT_DIR / "esm_emb_150M_all_layers.npy")  # (31, N, 640)
    split_df = pd.read_csv(SPLIT_CSV)
    subtypes = split_df["subtype"].values

    # host_category 按 accession 从 aligned CSV 取（去重；宿主字段不受亚型重判影响）
    data_df = pd.read_csv(DATA_CSV).set_index("accession")
    data_df = data_df[~data_df.index.duplicated(keep="first")]
    host_raw = data_df.loc[split_df["accession"].values, "host_category"].astype(str).values
    host3 = np.array([h if h in ("avian", "human") else "other" for h in host_raw])
    print("宿主 3 类分布:", pd.Series(host3).value_counts().to_dict())

    sweep = json.load(open(OUT_DIR / "layer_sweep.json"))

    results = {}
    for li in range(arr.shape[0]):
        X = arr[li]
        acc_sub = cv_acc(X, subtypes)
        acc_host = cv_acc(X, host3)
        results[f"layer_{li}"] = {"subtype_cv_acc": acc_sub, "host_cv_acc": acc_host}
        print(f"  layer {li:>2}: subtype={acc_sub:.3f} host={acc_host:.3f}", flush=True)

    out_path = OUT_DIR / "layer_probing.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"✓ {out_path}")

    # ── 图：亚型/宿主 probe + jump/jump_human H5/H7 AUC vs 层 ──
    layers = list(range(arr.shape[0]))
    sub_acc = [results[f"layer_{l}"]["subtype_cv_acc"] for l in layers]
    host_acc = [results[f"layer_{l}"]["host_cv_acc"] for l in layers]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(layers, sub_acc, "o-", label="subtype probe (4-class CV acc)")
    axes[0].plot(layers, host_acc, "s-", label="host probe (3-class CV acc)")
    axes[0].axhline(0.25, ls=":", c="gray"); axes[0].axhline(1/3, ls="--", c="gray")
    axes[0].set_ylabel("CV accuracy"); axes[0].legend(fontsize=8)
    axes[0].set_title("Per-layer probing: subtype identity vs host signal")

    for ax, label_col, name in [(axes[1], "label_is_jump", "jump"),
                                (axes[2], "label_is_jump_human", "jump_human")]:
        h5 = [sweep[label_col][f"layer_{l}"]["h5"] for l in layers]
        h7 = [sweep[label_col][f"layer_{l}"]["h7"] for l in layers]
        ax.plot(layers, h5, "o-", label="H5 AUC")
        ax.plot(layers, h7, "s-", label="H7 AUC")
        ax.axhline(0.5, ls=":", c="gray")
        ax.axvspan(12.5, 22.5, alpha=0.1, color="green", label="mid layers (13-22)")
        ax.set_ylabel(f"AUC ({name})"); ax.legend(fontsize=8)
    axes[2].set_xlabel("ESM-2 layer (0=embedding, 30=last)")
    fig.tight_layout()
    fig_path = FIG_DIR / "layer_probing.png"
    fig.savefig(fig_path, dpi=300)
    print(f"✓ {fig_path}")


if __name__ == "__main__":
    main()
