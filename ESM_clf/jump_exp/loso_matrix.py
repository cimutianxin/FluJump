"""留一/留二亚型迁移矩阵（LOSO matrix）

目的：诊断 H7 方向反转的根因。对每种训练亚型组合训练 Ridge LR probe，
在其余亚型上计算 AUC，检验：
  - 反转是否 H7 特有，还是与 HA 系统发育组（Group1: H1/H5, Group2: H3/H7）相关
  - 反向训练（如 H7→H1/H3）是否可行（Phase 3 反向对照实验的数据来源）

训练组合：
  - 单亚型 4 种：H1 / H3 / H5 / H7
  - 两两组合 6 种
  - 三组合（留一）4 种
评估：在每种非训练亚型的全部 isolate 上算 AUC（原始方向，不翻转）。

输出：ESM_clf/jump_exp/output/loso_matrix.json
"""

import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

SUBTYPES = ["H1", "H3", "H5", "H7"]
EMB_TYPES = ["L1", "L3", "L1L3"]


def load_data():
    """加载 embedding / 标签 / 亚型列（embedding 按 CSV 行顺序对齐）

    注意：subtype 用 isolate_split.csv（11060 行），不用 DATA_CSV——
    当前 aligned CSV 已于 2026-07-12 重建（11261 行，267 条 subtype 重判），
    与 embedding/labels 的行序不再一致。split CSV 与 accessions.npy 逐行对齐。
    """
    split_df = pd.read_csv(SPLIT_CSV)
    subtypes = split_df["subtype"].values
    emb = {e: np.load(OUT_DIR / f"esm_emb_150M_{e}.npy") for e in EMB_TYPES}
    labels = {lc: np.load(OUT_DIR / f"labels_{lc}.npy") for lc in LABEL_COLS}
    return subtypes, emb, labels


def train_probe(X, y, train_mask):
    """StandardScaler + GridSearchCV Ridge LR，返回 (scaler, 模型, best_C)"""
    scl = StandardScaler()
    Xt = scl.fit_transform(X[train_mask])
    yt = y[train_mask]
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(Xt, yt)
    return scl, gs.best_estimator_, gs.best_params_["C"]


def eval_subtypes(X, y, scl, model, subtypes, train_set):
    """在每种非训练亚型上算 AUC（logits 方向）"""
    res = {}
    for st in SUBTYPES:
        if st in train_set:
            continue
        m = subtypes == st
        ys = y[m]
        if len(np.unique(ys)) < 2:
            res[st] = None
            continue
        logits = model.decision_function(scl.transform(X[m]))
        res[st] = float(roc_auc_score(ys, logits))
    return res


def main():
    subtypes, emb, labels = load_data()

    # 训练组合：单亚型 + 两两 + 三组合
    train_sets = [(st,) for st in SUBTYPES]
    train_sets += list(combinations(SUBTYPES, 2))
    train_sets += list(combinations(SUBTYPES, 3))

    all_results = {}
    for label_col in LABEL_COLS:
        y = labels[label_col]
        print(f"\n{'='*72}")
        print(f"  Label: {label_col}")
        print(f"{'='*72}")
        label_res = {}
        for emb_name in EMB_TYPES:
            X = emb[emb_name]
            emb_res = {}
            for ts in train_sets:
                train_mask = np.isin(subtypes, list(ts))
                # 训练集需同时有两类
                if len(np.unique(y[train_mask])) < 2:
                    continue
                scl, model, best_c = train_probe(X, y, train_mask)
                aucs = eval_subtypes(X, y, scl, model, subtypes, set(ts))
                key = "+".join(ts)
                emb_res[key] = {"best_C": best_c, "n_train": int(train_mask.sum()),
                                "pos_train": int(y[train_mask].sum()), "auc": aucs}
                auc_str = "  ".join(
                    f"{st}:{f'{a:.3f}' if a is not None else 'N/A':>6}" for st, a in aucs.items())
                print(f"  [{emb_name:>5}] train={key:<10} → {auc_str}")
            label_res[emb_name] = emb_res
        all_results[label_col] = label_res

    out_path = OUT_DIR / "loso_matrix.json"
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")

    # ── 汇总：L3 的 4x4 单亚型矩阵（论文用主表）──
    print(f"\n{'='*72}")
    print("  单亚型 → 单亚型 AUC 矩阵（L3 embedding，行=训练，列=测试）")
    print(f"{'='*72}")
    for label_col in LABEL_COLS:
        print(f"\n  [{label_col}]")
        header = "  train\\test " + "".join(f"{st:>8}" for st in SUBTYPES)
        print(header)
        for tr in SUBTYPES:
            row = f"  {tr:<10}"
            for te in SUBTYPES:
                if tr == te:
                    row += f"{'—':>8}"
                else:
                    a = all_results[label_col]["L3"][tr]["auc"][te]
                    row += f"{a:>8.3f}" if a is not None else f"{'N/A':>8}"
            print(row)


if __name__ == "__main__":
    main()
