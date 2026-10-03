"""fig4_boundary panel (b)：几何散点逐点数据导出

复现 analyze_geometry.py 的几何分析流程（label_is_jump_human × L3）：
  H1+H3 train 上 StandardScaler + Ridge LR GridSearchCV → probe w；
  同一 X_std[train] 上 PCA(n_components=2)；逐点导出 PC1/PC2/proj_w。

校验（全部通过才导出）：
  - PC1 解释方差 ≈ 0.462；best_C 与 json 一致；
  - H7 proj_pos≈-5.099、proj_neg≈-3.281；
  - 各亚型 proj_pos/proj_neg 与 output/geometry_analysis.json 对应字段在
    容差 0.1 内一致。注：geometry_analysis.json 生成于 2026-08-01，早于 09-19
    host_label 词边界修复（labels_*.npy 2026-09-19 重生成），标签位移使 w 方向
    轻微旋转，逐字段完全一致不可能；观测最大偏移 ~0.072（H1 proj_pos），
    与 depth_reversal REPRO_TOL=0.06 的"含 09-19 标签修复位移"口径同源。

注意：proj_w = probe decision_function（scaler 变换后 embedding 与 w 点积加偏置，
raw logit）；geometry_analysis.json 中的 proj_pos/proj_neg 为 w 单位化、相对 train
质心的投影（仅用于校验复核，与 proj_w 口径不同）。

输出：figdata/fig4_boundary/panelb_geometry_points.csv（11,060 行）
      figdata/fig4_boundary/panelb_geometry_meta.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

FIG4_DIR = Path("figdata/fig4_boundary")
LABEL = "label_is_jump_human"
EMB_NAME = "L3"
SUBTYPES = ["H1", "H3", "H5", "H7"]
GEOM_JSON = OUT_DIR / "geometry_analysis.json"

# 校验容差：geometry_analysis.json 早于 09-19 标签修复，proj 字段允许 0.1 内偏移
# （观测最大 0.072）；PC1 与标签无关，应精确吻合；H7 锚点值按 notes 3 位小数口径
TOL_JSON = 0.1
TOL_NOTE = 0.1
TOL_NOTE_NEG = 0.01
PC1_EXPECT = 0.462
TOL_PC1 = 5e-4


def train_w(X, y, train_mask):
    """同 analyze_geometry.py：H1+H3 train 上训 probe，返回 (scaler, clf, best_C)"""
    scl = StandardScaler()
    Xt = scl.fit_transform(X[train_mask])
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc")
    gs.fit(Xt, y[train_mask])
    return scl, gs.best_estimator_, gs.best_params_["C"]


def main():
    split_df = pd.read_csv(SPLIT_CSV)
    subtypes = split_df["subtype"].values
    train_mask = (split_df["split"] == "train").values

    X = np.load(OUT_DIR / f"esm_emb_150M_{EMB_NAME}.npy")
    y = np.load(OUT_DIR / f"labels_{LABEL}.npy")
    assert len(X) == len(y) == len(split_df), "embedding/label/split 行数不一致"

    # ── probe 训练 + 标准化（流程同 analyze_geometry.py）──
    scl, clf, best_c = train_w(X, y, train_mask)
    w = clf.coef_[0]
    X_std = scl.transform(X)
    proj_w = X_std @ w + clf.intercept_[0]          # decision_function（raw logit）

    # ── PCA（拟合集合与参数同 analyze_geometry.py）──
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca.fit(X_std[train_mask])
    Z = pca.transform(X_std)
    evr = pca.explained_variance_ratio_

    # ── 校验：各亚型 proj_pos/proj_neg（w 单位化 + 相对 train 质心口径）──
    mu_train = X_std[train_mask].mean(axis=0)
    proj_check = {}
    for st in SUBTYPES:
        m = subtypes == st
        proj_check[st] = {
            "proj_pos": float((X_std[m & (y == 1)].mean(axis=0) - mu_train)
                              @ w / np.linalg.norm(w)),
            "proj_neg": float((X_std[m & (y == 0)].mean(axis=0) - mu_train)
                              @ w / np.linalg.norm(w)),
        }
    ref = json.load(open(GEOM_JSON))[LABEL][EMB_NAME]
    max_dev = 0.0
    for st in SUBTYPES:
        for k in ["proj_pos", "proj_neg"]:
            dev = abs(proj_check[st][k] - ref["cos"][st][k])
            max_dev = max(max_dev, dev)
            assert dev < TOL_JSON, \
                f"{st} {k} 与 geometry_analysis.json 偏差超容差: " \
                f"{proj_check[st][k]} vs {ref['cos'][st][k]} (dev={dev:.4f})"
    assert abs(proj_check["H7"]["proj_pos"] - (-5.099)) < TOL_NOTE, \
        f"H7 proj_pos≠-5.099: {proj_check['H7']['proj_pos']}"
    assert abs(proj_check["H7"]["proj_neg"] - (-3.281)) < TOL_NOTE_NEG, \
        f"H7 proj_neg≠-3.281: {proj_check['H7']['proj_neg']}"
    assert abs(evr[0] - PC1_EXPECT) < TOL_PC1, f"PC1 解释方差≠0.462: {evr[0]}"
    assert best_c == ref["best_C"], f"best_C 不一致: {best_c} vs {ref['best_C']}"
    print(f"✓ 校验通过: best_C={best_c}, PC1={evr[0]:.4f}, PC2={evr[1]:.4f}, "
          f"H7 proj_pos={proj_check['H7']['proj_pos']:+.4f} "
          f"proj_neg={proj_check['H7']['proj_neg']:+.4f}, "
          f"vs json 最大偏差={max_dev:.4f}（09-19 标签修复位移内）")

    # ── 导出逐点数据 ──
    FIG4_DIR.mkdir(parents=True, exist_ok=True)
    out = split_df[["accession", "subtype", "split", "cluster_id"]].copy()
    out[LABEL] = y
    out["PC1"] = Z[:, 0]
    out["PC2"] = Z[:, 1]
    out["proj_w"] = proj_w
    csv_path = FIG4_DIR / "panelb_geometry_points.csv"
    out.to_csv(csv_path, index=False)
    assert len(out) == 11060, f"行数≠11060: {len(out)}"
    print(f"✓ {csv_path}（{len(out)} 行）")

    meta = {
        "label": LABEL, "embedding": f"esm_emb_150M_{EMB_NAME}",
        "probe": "StandardScaler + Ridge LR GridSearchCV（同 analyze_geometry.py）",
        "pca_fit": "X_std[split==train]，PCA(n_components=2, random_state=42)",
        "best_C": best_c,
        "explained_variance_ratio": [float(v) for v in evr],
        "proj_w_definition": "decision_function = X_std @ w + intercept（raw logit）",
        "proj_check_w_unit_rel_train_centroid": proj_check,
        "proj_ref_geometry_analysis_json": {st: ref["cos"][st] for st in SUBTYPES},
        "max_abs_dev_vs_json": max_dev,
        "dev_note": "geometry_analysis.json(2026-08-01) 早于 09-19 host_label 修复"
                    "（labels 2026-09-19 重生成），proj 字段存在 ≤0.1 标签位移偏差",
        "ref": str(GEOM_JSON),
    }
    meta_path = FIG4_DIR / "panelb_geometry_meta.json"
    json.dump(meta, open(meta_path, "w"), indent=2, ensure_ascii=False)
    print(f"✓ {meta_path}")


if __name__ == "__main__":
    main()
