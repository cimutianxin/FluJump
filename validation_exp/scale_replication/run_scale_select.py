"""规模复现主实验：650M/3B 的 target-val 层选择 + 全层方向翻转曲线

复刻 ESM_clf/jump_exp/target_val_layer_select.py 的协议（probe 只依赖
(label, layer) 与 H1+H3 train，训一次在 10 切分文件 × 4 mask 上评估），
同时记录每层在全量 h5/h7 holdout 的 AUC（方向翻转诊断曲线，与 150M 对比）。

输出（每规模）：
  output/scale_{size}_select.json   target-val 协议结果（两臂汇总 + bootstrap）
  output/scale_{size}_sweep.json    全层 holdout AUC 曲线（诊断）
"""

import json
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from validation_exp.scale_replication.config import *


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


def run_size(size):
    arr = np.load(OUT_DIR / f"esm_emb_{size}_all_layers.npy", mmap_mode="r")
    n_layers = arr.shape[0]
    split_df = pd.read_csv(SPLIT_CSV)
    splits = split_df["split"].values
    m_train = splits == "train"
    m_test = splits == "test"
    eval_idx = np.where(splits == "h5_holdout")[0].tolist() + \
        np.where(splits == "h7_holdout")[0].tolist()
    eval_idx = np.array(eval_idx)
    pos_of = {r: p for p, r in enumerate(eval_idx)}

    key2row = {(a, s): i for i, (a, s) in
               enumerate(zip(split_df["accession"], split_df["subtype"]))}
    files = [f"{arm}_seed{s}.csv" for arm in ARMS for s in SEEDS]
    file_rows, file_cl = {}, {}
    for fname in files:
        sp = pd.read_csv(H5H7_SPLIT_DIR / fname)
        rows, cls = {}, {}
        for mask_name, g in sp.groupby("split"):
            rows[mask_name] = np.array([key2row[(a, s)] for a, s in
                                        zip(g["accession"], g["subtype"])])
            cls[mask_name] = g["cluster_id"].values
        file_rows[fname] = rows
        file_cl[fname] = cls

    aucs = {c: {} for c in LABEL_COLS}   # [label][layer][fname|FULL][mask]
    sweep = {c: {} for c in LABEL_COLS}
    ens_eval = {}

    for label_col in LABEL_COLS:
        y = np.load(LABELS_DIR / f"labels_{label_col}.npy")
        yt = y[m_train]
        for li in range(n_layers):
            t0 = time.time()
            X = np.asarray(arr[li], dtype=np.float32)
            scl, gs = fit_probe(X[m_train], yt)
            sc_eval = gs.decision_function(scl.transform(X[eval_idx]))
            ens_eval[(label_col, li)] = sc_eval
            # 诊断曲线：全量 holdout + H1+H3 test（按 split 标记定位，不假设行序）
            eval_splits = splits[eval_idx]
            p_h5 = eval_splits == "h5_holdout"
            p_h7 = eval_splits == "h7_holdout"
            sweep[label_col][f"L{li}"] = {
                "test_h13": float(roc_auc_score(
                    y[m_test], gs.decision_function(scl.transform(X[m_test])))),
                "h5_full": float(roc_auc_score(y[eval_idx][p_h5], sc_eval[p_h5])),
                "h7_full": float(roc_auc_score(y[eval_idx][p_h7], sc_eval[p_h7])),
            }
            per_file = {}
            for fname in files:
                per_mask = {}
                for mask_name, rows in file_rows[fname].items():
                    p = np.array([pos_of[r] for r in rows])
                    per_mask[mask_name] = float(roc_auc_score(y[rows], sc_eval[p]))
                per_file[fname] = per_mask
            aucs[label_col][li] = per_file
            if li % 5 == 0 or li == n_layers - 1:
                s = sweep[label_col][f"L{li}"]
                print(f"  [{size}] {label_col} L{li}: h5={s['h5_full']:.3f} "
                      f"h7={s['h7_full']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ── 分亚型选层：val argmax → test 评一次 ──
    per_file_res = {}
    for fname in files:
        fres = {}
        for label_col in LABEL_COLS:
            lres = {}
            for st in SUBTYPES:
                val_aucs = {li: aucs[label_col][li][fname][f"{st}_val"]
                            for li in range(n_layers)}
                best = max(val_aucs, key=val_aucs.get)
                lres[st] = {"selected_layer": best, "val_auc": val_aucs[best],
                            "test_auc": aucs[label_col][best][fname][f"{st}_test"]}
            fres[label_col] = lres
        per_file_res[fname] = fres

    # ── 汇总 ──
    summary = {}
    for arm in ARMS:
        arm_files = [f"{arm}_seed{s}.csv" for s in SEEDS]
        ares = {}
        for label_col in LABEL_COLS:
            lres = {}
            for st in SUBTYPES:
                picks = [per_file_res[f][label_col][st]["selected_layer"] for f in arm_files]
                tests = [per_file_res[f][label_col][st]["test_auc"] for f in arm_files]
                lres[st] = {"selected_layers": picks, "layer_freq": dict(Counter(picks)),
                            "test_auc_mean": float(np.mean(tests)),
                            "test_auc_std": float(np.std(tests))}
            ares[label_col] = lres
        summary[arm] = ares

    # ── cluster_seed42 test cluster bootstrap ──
    fname0 = "cluster_seed42.csv"
    rng = np.random.default_rng(RANDOM_SEED)
    boot = {}
    for label_col in LABEL_COLS:
        y = np.load(LABELS_DIR / f"labels_{label_col}.npy")
        lres = {}
        for st in SUBTYPES:
            best = per_file_res[fname0][label_col][st]["selected_layer"]
            rows = file_rows[fname0][f"{st}_test"]
            cl = file_cl[fname0][f"{st}_test"]
            p = np.array([pos_of[r] for r in rows])
            sc, yt2 = ens_eval[(label_col, best)][p], y[rows]
            uniq_cl = np.unique(cl)
            vals = []
            for _ in range(B_BOOT):
                samp = rng.choice(uniq_cl, size=len(uniq_cl), replace=True)
                idx_b = np.concatenate([np.where(cl == c)[0] for c in samp])
                if len(np.unique(yt2[idx_b])) < 2:
                    continue
                vals.append(float(roc_auc_score(yt2[idx_b], sc[idx_b])))
            v = np.array(vals)
            lres[st] = {"layer": best, "auc_full": float(roc_auc_score(yt2, sc)),
                        "p05": float(np.percentile(v, 5)),
                        "p95": float(np.percentile(v, 95)),
                        "p_above_0.5": float((v > 0.5).mean())}
        boot[label_col] = lres

    # ── 打印 ──
    print(f"\n{'='*76}\n[{size}] target-val 选层 → test（5 seeds, cluster 臂）\n{'='*76}")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            s = summary["cluster"][label_col][st]
            print(f"  {label_col:>22} {st}: 选层 {s['layer_freq']}  "
                  f"test AUC = {s['test_auc_mean']:.3f} ± {s['test_auc_std']:.3f}")
    print(f"[{size}] bootstrap (cluster_seed42):")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            b = boot[label_col][st]
            print(f"  {label_col:>22} {st} L{b['layer']}: AUC={b['auc_full']:.3f} "
                  f"[{b['p05']:.3f}, {b['p95']:.3f}] P(>0.5)={b['p_above_0.5']:.2f}")

    sel = {"per_file": per_file_res, "summary": summary,
           "bootstrap_cluster_seed42": boot}
    json.dump(sel, open(OUT_DIR / f"scale_{size}_select.json", "w"), indent=2)
    json.dump(sweep, open(OUT_DIR / f"scale_{size}_sweep.json", "w"), indent=2)
    print(f"✓ [{size}] 结果已保存")


def main():
    for size in MODELS:
        print(f"\n{'#'*76}\n# 规模: {size}\n{'#'*76}")
        run_size(size)


if __name__ == "__main__":
    main()
