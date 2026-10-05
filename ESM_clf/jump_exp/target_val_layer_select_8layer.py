"""target-val 层选择：候选层限 8 层的选择预算对照（S1，Major 8）

协议与 target_val_layer_select.py 完全一致（seeds 42–46、cluster/isolate_random 两臂、
raw-logit val argmax、test 评一次），唯一改动：候选层集合从 0–30 全部 31 层限制为 8 层。
两套候选：
  A 均匀覆盖+关键层：{0, 4, 9, 13, 17, 22, 26, 30}（含 L13/L17 与首末层）
  B 不含 L13/L17 稳健性：{0, 4, 8, 12, 16, 20, 24, 30}

输出：output/target_val_layer_select_8layer.json
"""

import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *
from ESM_clf.jump_exp.target_val_layer_select import fit_probe, SEEDS, ARMS, SUBTYPES

CANDIDATE_SETS = {
    "A_uniform_with_L13L17": [0, 4, 9, 13, 17, 22, 26, 30],
    "B_uniform_no_L13L17": [0, 4, 8, 12, 16, 20, 24, 30],
}
BASELINE_JSON = OUT_DIR / "target_val_layer_select.json"


def main():
    arr = np.load(OUT_DIR / "esm_emb_150M_all_layers.npy")  # (31, N, 640)
    split_df = pd.read_csv(SPLIT_CSV)
    m_train = (split_df["split"] == "train").values
    m_h57 = split_df["split"].isin(["h5_holdout", "h7_holdout"]).values
    eval_idx = np.where(m_h57)[0]
    pos_of = {r: p for p, r in enumerate(eval_idx)}

    key2row = {(a, s): i for i, (a, s) in
               enumerate(zip(split_df["accession"], split_df["subtype"]))}
    files = [f"{arm}_seed{s}.csv" for arm in ARMS for s in SEEDS]
    file_rows = {}
    for fname in files:
        sp = pd.read_csv(H5H7_SPLIT_DIR / fname)
        rows = {}
        for mask_name, g in sp.groupby("split"):
            rows[mask_name] = np.array([key2row[(a, s)] for a, s in
                                        zip(g["accession"], g["subtype"])])
        file_rows[fname] = rows

    # 只需拟合两套候选的并集
    cand_union = sorted(set(CANDIDATE_SETS["A_uniform_with_L13L17"])
                        | set(CANDIDATE_SETS["B_uniform_no_L13L17"]))
    print(f"候选并集 {len(cand_union)} 层: {cand_union}")

    aucs = {c: {} for c in LABEL_COLS}
    for label_col in LABEL_COLS:
        y = np.load(OUT_DIR / f"labels_{label_col}.npy")
        yt = y[m_train]
        for li in cand_union:
            scl, gs = fit_probe(arr[li][m_train], yt)
            sc = gs.decision_function(scl.transform(arr[li][eval_idx]))
            aucs[label_col][li] = {
                fname: {mn: float(roc_auc_score(y[rows], sc[[pos_of[r] for r in rows]]))
                        for mn, rows in file_rows[fname].items()}
                for fname in files}
            print(f"  {label_col} L{li} done", flush=True)

    # ── 各候选集 × 各臂选层汇总 ──
    results = {}
    for set_name, cand in CANDIDATE_SETS.items():
        set_res = {}
        for arm in ARMS:
            arm_files = [f"{arm}_seed{s}.csv" for s in SEEDS]
            ares = {}
            for label_col in LABEL_COLS:
                lres = {}
                for st in SUBTYPES:
                    picks, tests = [], []
                    for fname in arm_files:
                        val_aucs = {li: aucs[label_col][li][fname][f"{st}_val"]
                                    for li in cand}
                        best = max(val_aucs, key=val_aucs.get)
                        picks.append(best)
                        tests.append(aucs[label_col][best][fname][f"{st}_test"])
                    lres[st] = {"selected_layers": picks,
                                "layer_freq": dict(Counter(picks)),
                                "test_auc_mean": float(np.mean(tests)),
                                "test_auc_std": float(np.std(tests))}
                ares[label_col] = lres
            set_res[arm] = ares
        results[set_name] = set_res

    # ── 与全 31 层基线对照 ──
    base = json.load(open(BASELINE_JSON))
    print(f"\n{'='*76}\n8 层候选 vs 全 31 层（cluster 臂 test AUC mean±SD）\n{'='*76}")
    compare = {}
    for set_name, cand in CANDIDATE_SETS.items():
        compare[set_name] = {}
        print(f"\n--- {set_name}: {cand} ---")
        for label_col in LABEL_COLS:
            compare[set_name][label_col] = {}
            for st in SUBTYPES:
                b = base["summary"]["cluster"][label_col][st]
                g = results[set_name]["cluster"][label_col][st]
                compare[set_name][label_col][st] = {
                    "baseline_31layer": {"test_auc_mean": b["test_auc_mean"],
                                         "test_auc_std": b["test_auc_std"],
                                         "selected": b["selected_layers"]},
                    "candidate_8layer": {"test_auc_mean": g["test_auc_mean"],
                                         "test_auc_std": g["test_auc_std"],
                                         "selected": g["selected_layers"]},
                    "delta_mean": g["test_auc_mean"] - b["test_auc_mean"],
                }
                print(f"  {label_col:>22} {st}: 31层 {b['test_auc_mean']:.3f}±{b['test_auc_std']:.3f} "
                      f"→ 8层 {g['test_auc_mean']:.3f}±{g['test_auc_std']:.3f} "
                      f"(Δ{g['test_auc_mean']-b['test_auc_mean']:+.3f}) 选层 {g['layer_freq']}")

    out = {"protocol": "同 target_val_layer_select.py，唯一改动候选层限 8 层",
           "candidate_sets": CANDIDATE_SETS,
           "baseline_source": str(BASELINE_JSON),
           "results": results, "compare": compare}
    out_path = OUT_DIR / "target_val_layer_select_8layer.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
