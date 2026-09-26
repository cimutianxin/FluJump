"""CNN 中间层 probing + target-val 选择 — §1 选择预算对齐实验

给 CNN 与 ESM 完全同等的表示选择预算：
  候选表示（8 个/标签）：conv1–5（block 输出 mean-pool）+ dense_stage1 +
    dense_stage2_<label> + logit_stage2_<label>（现有 baseline 头）
  协议：与 ESM_clf/jump_exp/target_val_layer_select.py 逐条对齐——
    同一 10 个切分文件（cluster/isolate_random × 5 seeds）、
    同一 Ridge LR probe（StandardScaler + GridSearchCV，C 网格/5-fold/seed42）、
    val argmax 选表示 → test 评一次、cluster_seed42 test 侧 cluster bootstrap。

注意：所有 probe 打分用 decision_function（logit 口径）。历史 eval 脚本用
float32 sigmoid 概率算 AUC，logit 幅度 ±100 时饱和退化为并列秩，jump_human
H5 全量 AUC 被低估为 0.549（logit 真值 0.702）——本脚本不受影响。

从项目根目录运行：/root/miniconda3/envs/borkenhagen/bin/python \
    Borkenhagen/cnn_baseline/probe_target_val.py
输出：output/probe_target_val.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "Borkenhagen/cnn_baseline")
from config import *  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
ARMS = ["cluster", "isolate_random"]
SUBTYPES = ["h5", "h7"]
LABEL_COLS = ["label_is_jump", "label_is_jump_human"]
H5H7_SPLIT_DIR = Path("data/splits/h5h7_valtest")
B_BOOT = 1000

RIDGE_C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CV_FOLDS = 5

FEAT_DIR = OUTPUT_DIR / "interm_feats"
ESM_TVS_JSON = Path("ESM_clf/jump_exp/output/target_val_layer_select.json")


def candidates_for(label_col):
    suffix = "jump" if label_col == "label_is_jump" else "jump_human"
    return [f"conv{i}" for i in range(1, 6)] + [
        "dense_stage1", f"dense_stage2_{suffix}", f"logit_stage2_{suffix}"]


def fit_probe(Xt, yt):
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                            random_state=RANDOM_SEED)
    gs = GridSearchCV(lr, {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(Xt_s, yt)
    return scl, gs


def main():
    split_df = pd.read_csv(SPLIT_CSV)
    df = pd.read_csv(INPUT_CSV)
    assert len(df) == len(split_df) and \
        (df["accession"].values == split_df["accession"].values).all(), "行序不一致"
    m_train = (split_df["split"] == "train").values
    m_h57 = split_df["split"].isin(["h5_holdout", "h7_holdout"]).values
    eval_idx = np.where(m_h57)[0]
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

    feats = {}
    for label_col in LABEL_COLS:
        for c in candidates_for(label_col):
            if c not in feats:
                feats[c] = np.load(FEAT_DIR / f"{c}.npy")
                if feats[c].ndim == 1:
                    feats[c] = feats[c].reshape(-1, 1)

    aucs = {c: {} for c in LABEL_COLS}   # aucs[label][cand][fname][mask]
    scores = {}                          # (label, cand) -> eval 分数
    full_holdout = {c: {} for c in LABEL_COLS}
    for label_col in LABEL_COLS:
        y = df[label_col].values.astype(int)
        yt = y[m_train]
        for cand in candidates_for(label_col):
            X = feats[cand]
            scl, gs = fit_probe(X[m_train], yt)
            sc = gs.decision_function(scl.transform(X[eval_idx]))
            scores[(label_col, cand)] = sc
            per_file = {}
            for fname in files:
                per_mask = {}
                for mask_name, rows in file_rows[fname].items():
                    p = np.array([pos_of[r] for r in rows])
                    per_mask[mask_name] = float(roc_auc_score(y[rows], sc[p]))
                per_file[fname] = per_mask
            aucs[label_col][cand] = per_file
            # 描述性参照：全量 holdout（不用于选择）
            full_holdout[label_col][cand] = {
                st: float(roc_auc_score(
                    y[split_df["split"].values == f"{st}_holdout"],
                    sc[split_df["split"].values[m_h57] == f"{st}_holdout"]))
                for st in SUBTYPES}
            print(f"  {label_col} {cand} done", flush=True)

    # ── val argmax 选表示 → test 评一次 ──
    per_file_res = {}
    for fname in files:
        fres = {}
        for label_col in LABEL_COLS:
            lres = {}
            for st in SUBTYPES:
                val_mask, test_mask = f"{st}_val", f"{st}_test"
                val_aucs = {c: aucs[label_col][c][fname][val_mask]
                            for c in candidates_for(label_col)}
                best = max(val_aucs, key=val_aucs.get)
                lres[st] = {"selected": best,
                            "val_auc": val_aucs[best],
                            "test_auc": aucs[label_col][best][fname][test_mask],
                            "val_aucs_all": val_aucs}
            fres[label_col] = lres
        per_file_res[fname] = fres
        print(f"  选择完成: {fname}", flush=True)

    # ── 汇总 ──
    summary = {}
    for arm in ARMS:
        arm_files = [f"{arm}_seed{s}.csv" for s in SEEDS]
        ares = {}
        for label_col in LABEL_COLS:
            lres = {}
            for st in SUBTYPES:
                picks = [per_file_res[f][label_col][st]["selected"]
                         for f in arm_files]
                tests = [per_file_res[f][label_col][st]["test_auc"]
                         for f in arm_files]
                lres[st] = {"selected": picks,
                            "pick_freq": dict(Counter(picks)),
                            "test_auc_mean": float(np.mean(tests)),
                            "test_auc_std": float(np.std(tests))}
            ares[label_col] = lres
        summary[arm] = ares

    # ── cluster_seed42 test 侧 cluster bootstrap ──
    fname0 = "cluster_seed42.csv"
    rng = np.random.default_rng(RANDOM_SEED)
    boot_res = {}
    for label_col in LABEL_COLS:
        y = df[label_col].values.astype(int)
        lres = {}
        for st in SUBTYPES:
            best = per_file_res[fname0][label_col][st]["selected"]
            test_mask = f"{st}_test"
            rows = file_rows[fname0][test_mask]
            cl = file_cl[fname0][test_mask]
            p = np.array([pos_of[r] for r in rows])
            sc = scores[(label_col, best)][p]
            yt = y[rows]
            uniq_cl = np.unique(cl)
            vals = []
            for _ in range(B_BOOT):
                samp = rng.choice(uniq_cl, size=len(uniq_cl), replace=True)
                idx = np.concatenate([np.where(cl == c)[0] for c in samp])
                if len(np.unique(yt[idx])) < 2:
                    continue
                vals.append(float(roc_auc_score(yt[idx], sc[idx])))
            v = np.array(vals)
            lres[st] = {"selected": best,
                        "auc_full": float(roc_auc_score(yt, sc)),
                        "p05": float(np.percentile(v, 5)),
                        "p95": float(np.percentile(v, 95)),
                        "p_above_0.5": float((v > 0.5).mean())}
        boot_res[label_col] = lres

    # ── ESM 对照（同协议 cluster 臂）──
    esm = json.load(open(ESM_TVS_JSON))
    esm_cluster = esm["summary"]["cluster"]

    # ── 打印 ──
    print(f"\n{'='*76}\nCNN probing + target-val 选择（cluster 臂, 5 seeds）\n{'='*76}")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            s = summary["cluster"][label_col][st]
            e = esm_cluster[label_col][st]
            print(f"  {label_col:>22} {st}: CNN 选择后 test AUC = "
                  f"{s['test_auc_mean']:.3f} ± {s['test_auc_std']:.3f} "
                  f"(选中 {s['pick_freq']}) | ESM 同协议 {e['test_auc_mean']:.3f}")
    print(f"\ncluster_seed42 test bootstrap:")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            b = boot_res[label_col][st]
            print(f"  {label_col:>22} {st} [{b['selected']}]: AUC={b['auc_full']:.3f} "
                  f"[{b['p05']:.3f}, {b['p95']:.3f}] P(>0.5)={b['p_above_0.5']:.2f}")
    print(f"\n全量 holdout 描述性参照（未用于选择）:")
    for label_col in LABEL_COLS:
        for cand in candidates_for(label_col):
            fh = full_holdout[label_col][cand]
            print(f"  {label_col:>22} {cand:>22}: h5={fh['h5']:.3f} h7={fh['h7']:.3f}")

    out = {"candidates": {c: candidates_for(c) for c in LABEL_COLS},
           "per_file": per_file_res, "summary": summary,
           "bootstrap_cluster_seed42": boot_res,
           "full_holdout_descriptive": full_holdout,
           "note": "probe 全部 decision_function（logit）打分；选择仅用 val；"
                   "epoch/checkpoint 级选择不在预算内（ESM 侧无此自由度）"}
    out_path = OUTPUT_DIR / "probe_target_val.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
