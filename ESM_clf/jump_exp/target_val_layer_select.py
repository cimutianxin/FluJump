"""target-val 层选择实验：H5/H7 内部 val 选层，test 只评一次

协议：H1+H3 train 训 Ridge LR probe（每层一个），用 data/splits/h5h7_valtest/
下的切分文件在 H5/H7 内部 val 上选层，对应 test 上评一次——
选择过程不接触 test 标签，替代此前"全量 H5/H7 看 AUC 选层"的做法。

两臂对比：
  - cluster 臂（主）：cluster 整体归入 val/test（5 seeds）
  - isolate_random 臂（对比）：按行随机切（5 seeds）
  两臂 test AUC 之差 = 近重复序列横跨 val/test 带来的泄露量。

参照层（历史 headline，本身源于全量 H5/H7 观察，仅作对照）：
  jump: L26 / L17 / L28；jump_human: L13 / L28

不确定性：对 cluster_seed42 选中配置做 test 侧 cluster bootstrap（B=1000）。

输出：output/target_val_layer_select.json
"""

import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

SEEDS = [42, 43, 44, 45, 46]
ARMS = ["cluster", "isolate_random"]
SUBTYPES = ["h5", "h7"]
REF_LAYERS = {"label_is_jump": [26, 17, 28], "label_is_jump_human": [13, 28]}
B_BOOT = 1000


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
    arr = np.load(OUT_DIR / "esm_emb_150M_all_layers.npy")  # (31, N, 640)
    split_df = pd.read_csv(SPLIT_CSV)
    m_train = (split_df["split"] == "train").values
    m_h57 = split_df["split"].isin(["h5_holdout", "h7_holdout"]).values
    eval_idx = np.where(m_h57)[0]
    pos_of = {r: p for p, r in enumerate(eval_idx)}  # 全表行号 → eval 数组下标

    # (accession, subtype) → 全表行号（embedding 行序 == isolate_split 行序）
    key2row = {(a, s): i for i, (a, s) in
               enumerate(zip(split_df["accession"], split_df["subtype"]))}

    # ── 加载 10 个切分文件，构建 4 类 mask 的行号 ──
    files = [f"{arm}_seed{s}.csv" for arm in ARMS for s in SEEDS]
    file_rows = {}   # fname -> {mask_name: 全表行号数组}
    file_cl = {}     # fname -> {mask_name: cluster_id 数组（bootstrap 用）}
    for fname in files:
        sp = pd.read_csv(H5H7_SPLIT_DIR / fname)
        rows, cls = {}, {}
        for mask_name, g in sp.groupby("split"):
            r = np.array([key2row[(a, s)] for a, s in
                          zip(g["accession"], g["subtype"])])
            rows[mask_name] = r
            cls[mask_name] = g["cluster_id"].values
        file_rows[fname] = rows
        file_cl[fname] = cls

    # ── 逐 (label, layer) 训 probe，记录各文件各 mask 的 AUC ──
    # aucs[label][layer][fname][mask] = auc；scores[(label, layer)] = eval 分数数组
    aucs = {c: {} for c in LABEL_COLS}
    scores = {}
    for label_col in LABEL_COLS:
        y = np.load(OUT_DIR / f"labels_{label_col}.npy")
        yt = y[m_train]
        for li in range(arr.shape[0]):
            scl, gs = fit_probe(arr[li][m_train], yt)
            sc = gs.decision_function(scl.transform(arr[li][eval_idx]))
            scores[(label_col, li)] = sc
            per_file = {}
            for fname in files:
                per_mask = {}
                for mask_name, rows in file_rows[fname].items():
                    p = np.array([pos_of[r] for r in rows])
                    per_mask[mask_name] = float(roc_auc_score(y[rows], sc[p]))
                per_file[fname] = per_mask
            aucs[label_col][li] = per_file
            if li % 5 == 0 or li == arr.shape[0] - 1:
                print(f"  {label_col} L{li} done", flush=True)

    # ── 分亚型选层：val argmax → test 评一次 ──
    per_file_res = {}
    for fname in files:
        fres = {}
        for label_col in LABEL_COLS:
            lres = {}
            for st in SUBTYPES:
                val_mask, test_mask = f"{st}_val", f"{st}_test"
                val_aucs = {li: aucs[label_col][li][fname][val_mask]
                            for li in aucs[label_col]}
                best = max(val_aucs, key=val_aucs.get)
                lres[st] = {
                    "selected_layer": best,
                    "val_auc": val_aucs[best],
                    "test_auc": aucs[label_col][best][fname][test_mask],
                    "ref_test_auc": {f"L{rl}": aucs[label_col][rl][fname][test_mask]
                                     for rl in REF_LAYERS[label_col]},
                }
            fres[label_col] = lres
        per_file_res[fname] = fres
        print(f"  选层完成: {fname}", flush=True)

    # ── 汇总：各臂选择频率 + test AUC mean±std + 泄露差值 ──
    summary, leakage = {}, {}
    for arm in ARMS:
        arm_files = [f"{arm}_seed{s}.csv" for s in SEEDS]
        ares = {}
        for label_col in LABEL_COLS:
            lres = {}
            for st in SUBTYPES:
                picks = [per_file_res[f][label_col][st]["selected_layer"]
                         for f in arm_files]
                tests = [per_file_res[f][label_col][st]["test_auc"]
                         for f in arm_files]
                lres[st] = {
                    "selected_layers": picks,
                    "layer_freq": dict(Counter(picks)),
                    "test_auc_mean": float(np.mean(tests)),
                    "test_auc_std": float(np.std(tests)),
                }
            ares[label_col] = lres
        summary[arm] = ares
    for label_col in LABEL_COLS:
        leakage[label_col] = {
            st: summary["isolate_random"][label_col][st]["test_auc_mean"]
                - summary["cluster"][label_col][st]["test_auc_mean"]
            for st in SUBTYPES
        }

    # ── cluster_seed42 选中配置的 test 侧 cluster bootstrap ──
    fname0 = "cluster_seed42.csv"
    rng = np.random.default_rng(RANDOM_SEED)
    boot_res = {}
    for label_col in LABEL_COLS:
        y = np.load(OUT_DIR / f"labels_{label_col}.npy")
        lres = {}
        for st in SUBTYPES:
            best = per_file_res[fname0][label_col][st]["selected_layer"]
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
            lres[st] = {"layer": best,
                        "auc_full": float(roc_auc_score(yt, sc)),
                        "p05": float(np.percentile(v, 5)),
                        "p95": float(np.percentile(v, 95)),
                        "p_above_0.5": float((v > 0.5).mean())}
        boot_res[label_col] = lres

    # ── 打印主表 ──
    print(f"\n{'='*76}")
    print("主结果：val 选层 → test 评一次（5 seeds）")
    print(f"{'='*76}")
    for arm in ARMS:
        print(f"\n--- {arm} 臂 ---")
        for label_col in LABEL_COLS:
            for st in SUBTYPES:
                s = summary[arm][label_col][st]
                print(f"  {label_col:>22} {st}: 选层 {s['layer_freq']}  "
                      f"test AUC = {s['test_auc_mean']:.3f} ± {s['test_auc_std']:.3f}")
    print(f"\n泄露量化（isolate_random − cluster, test AUC 均值差）:")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            print(f"  {label_col:>22} {st}: {leakage[label_col][st]:+.3f}")
    print(f"\ncluster_seed42 test cluster bootstrap (B={B_BOOT}):")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            b = boot_res[label_col][st]
            print(f"  {label_col:>22} {st} L{b['layer']}: AUC={b['auc_full']:.3f} "
                  f"[{b['p05']:.3f}, {b['p95']:.3f}] P(>0.5)={b['p_above_0.5']:.2f}")

    out = {"per_file": per_file_res, "summary": summary,
           "leakage_delta": leakage, "bootstrap_cluster_seed42": boot_res,
           "ref_layers_note": "参照层源于全量 H5/H7 观察（历史 headline），仅作对照"}
    out_path = OUT_DIR / "target_val_layer_select.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
