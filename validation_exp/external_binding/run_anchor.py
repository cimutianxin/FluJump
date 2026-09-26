"""外部锚定主实验：jump 方向 vs 实验 binding 表型

两部分：
  1. binding 全层扫描：每层 Ridge probe（沿用 borkenhagen_split.csv 的
     train/test），看 target-val 选出的层 {13,17,28} 在独立实验表型上的排名
  2. jump probe → binding 迁移：H1+H3 train 上训好的 jump/jump_human probe
     （L13/L17/L28）直接给 402 条序列打分，对实验 binding 标签算 AUC
     —— jump 方向的免标签外部验证（α2,6 人型受体结合偏好）

输出：output/external_anchor_results.json
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from validation_exp.external_binding.config import *


def fit_probe(Xt, yt):
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(Xt_s, yt)
    return scl, gs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    barr = np.load(OUT_DIR / "binding_emb_all_layers.npy")   # (31, 402, 640)
    yb = np.load(OUT_DIR / "binding_labels.npy")
    bsp = pd.read_csv(BORK_SPLIT)["split"].values
    bm_tr, bm_te = bsp == "train", bsp == "test"
    print(f"binding: train={bm_tr.sum()} test={bm_te.sum()} (pos={yb[bm_te].sum()})")

    # ── Part 1: binding 全层扫描 ──
    sweep = {}
    print("\n--- binding 全层扫描 ---")
    for li in range(barr.shape[0]):
        X = barr[li]
        scl, gs = fit_probe(X[bm_tr], yb[bm_tr])
        auc = float(roc_auc_score(yb[bm_te],
                                  gs.decision_function(scl.transform(X[bm_te]))))
        sweep[f"L{li}"] = auc
        mark = " ← anchor层" if li in ANCHOR_LAYERS else ""
        print(f"  L{li:>2}: AUC={auc:.3f}{mark}")
    ranked = sorted(sweep, key=sweep.get, reverse=True)
    anchor_rank = {f"L{li}": ranked.index(f"L{li}") + 1 for li in ANCHOR_LAYERS}
    print(f"  anchor 层排名: {anchor_rank} / {len(ranked)}")

    # ── Part 2: jump probe → binding 迁移 ──
    marr = np.load(JUMP_OUT / "esm_emb_150M_all_layers.npy")  # (31, 11060, 640)
    msp = pd.read_csv(MAIN_SPLIT_CSV)["split"].values
    m_tr = msp == "train"
    transfer = {}
    print("\n--- jump probe → binding 迁移 ---")
    for label_col in LABEL_COLS:
        ym = np.load(JUMP_OUT / f"labels_{label_col}.npy")
        lres = {}
        for li in ANCHOR_LAYERS:
            scl, gs = fit_probe(marr[li][m_tr], ym[m_tr])
            sc = gs.decision_function(scl.transform(barr[li]))
            lres[f"L{li}"] = {
                "auc_all": float(roc_auc_score(yb, sc)),
                "auc_test_only": float(roc_auc_score(yb[bm_te], sc[bm_te])),
            }
            print(f"  {label_col} L{li}: AUC(全部402)={lres[f'L{li}']['auc_all']:.3f} "
                  f"AUC(仅test)={lres[f'L{li}']['auc_test_only']:.3f}")
        transfer[label_col] = lres

    out = {"binding_layer_sweep": sweep, "anchor_layer_rank": anchor_rank,
           "jump_probe_to_binding": transfer,
           "note": "Part1 用 binding 自带 split 训测；Part2 probe 只在 H1+H3 train 上训练，"
                   "402 条序列全部为零样本"}
    out_path = OUT_DIR / "external_anchor_results.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
