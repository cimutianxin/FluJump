"""CNN baseline 在 h5h7_valtest 新切分下的重评估

不重训：stage2  checkpoint 的训练/早停全部在 H1+H3 train/val 上完成，
从未接触 H5/H7，因此在新协议下依然合规——只需把评估子集从
全量 holdout 换成 h5_test/h7_test（与 ESM probe 的 target-val 实验
ESM_clf/jump_exp/target_val_layer_select.py 完全同一切分，保证可比）。

两臂（各 5 seeds）：
  - cluster 臂（主）：h5_test/h7_test 为 cluster 级切分
  - isolate_random 臂（对比）：按行随机切
另报 cluster_seed42 的 test 侧 cluster bootstrap（B=1000）与全量 holdout AUC（延续旧口径）。

输出：output/eval_h5h7_valtest.json
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from config import *
from data_prep import HADataset, get_df
from model import BorkenhagenCNN

SEEDS = [42, 43, 44, 45, 46]
ARMS = ["cluster", "isolate_random"]
SUBTYPES = ["h5", "h7"]
LABEL_COLS = ["label_is_jump", "label_is_jump_human"]
H5H7_SPLIT_DIR = Path("data/splits/h5h7_valtest")
B_BOOT = 1000


@torch.no_grad()
def predict_scores(model, df, device, bs=256):
    ds = HADataset(df["aligned_ha_seq"].tolist(), df["label"].values)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    model.eval()
    preds = []
    for batch in loader:
        x = batch[0].to(device)
        # 用 raw logits：float32 sigmoid 在 |logit|>88 饱和退化 AUC（见 evaluate.py）
        preds.extend(model(x).squeeze(-1).cpu().numpy())
    return np.array(preds)


def metrics(labels, preds):
    b = (preds >= 0.0).astype(int)  # logit>=0 ⇔ p>=0.5
    return {
        "auc": float(roc_auc_score(labels, preds)) if len(np.unique(labels)) > 1 else None,
        "acc": float(accuracy_score(labels, b)),
        "prec": float(precision_score(labels, b, zero_division=0)),
        "rec": float(recall_score(labels, b, zero_division=0)),
        "f1": float(f1_score(labels, b, zero_division=0)),
    }


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    df = get_df()
    split_df = pd.read_csv(SPLIT_CSV)

    # (accession, subtype) → 行号（aligned CSV 行序 == isolate_split 行序）
    key2row = {(a, s): i for i, (a, s) in
               enumerate(zip(split_df["accession"], split_df["subtype"]))}

    # 10 个切分文件 × 4 mask 的行号与 cluster_id
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

    m_h57 = split_df["split"].isin(["h5_holdout", "h7_holdout"]).values
    holdout_rows = np.where(m_h57)[0]

    per_file, boot_res, holdout_ref = {}, {}, {}
    rng = np.random.default_rng(RANDOM_SEED)

    for label_col in LABEL_COLS:
        ckpt = MODEL_DIR / f"stage2_{label_col}_best.pt"
        print(f"\n{'='*60}\n评估: {label_col} <- {ckpt.name}\n{'='*60}")
        model = BorkenhagenCNN().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        # 全量 H5/H7 预测一次（isolate_split 行序），各 mask 只取子集
        ho_df = df.iloc[holdout_rows].copy()
        ho_df["label"] = ho_df[label_col].astype(int)
        scores_ho = predict_scores(model, ho_df, device)
        pos_of = {r: p for p, r in enumerate(holdout_rows)}
        y_all = ho_df["label"].values

        # 全量 holdout 参照（旧口径）
        holdout_ref[label_col] = {
            st: metrics(y_all[ho_df["subtype"].values == st.upper()],
                        scores_ho[ho_df["subtype"].values == st.upper()])
            for st in SUBTYPES
        }

        # 10 个文件 × 4 mask
        fres = {}
        for fname in files:
            mres = {}
            for mask_name, rows in file_rows[fname].items():
                p = np.array([pos_of[r] for r in rows])
                mres[mask_name] = metrics(y_all[p], scores_ho[p])
            fres[fname] = mres
        per_file[label_col] = fres

        # cluster_seed42 test 侧 cluster bootstrap
        fname0 = "cluster_seed42.csv"
        lres = {}
        for st in SUBTYPES:
            test_mask = f"{st}_test"
            rows = file_rows[fname0][test_mask]
            cl = file_cl[fname0][test_mask]
            p = np.array([pos_of[r] for r in rows])
            sc, yt = scores_ho[p], y_all[p]
            uniq_cl = np.unique(cl)
            vals = []
            for _ in range(B_BOOT):
                samp = rng.choice(uniq_cl, size=len(uniq_cl), replace=True)
                idx = np.concatenate([np.where(cl == c)[0] for c in samp])
                if len(np.unique(yt[idx])) < 2:
                    continue
                vals.append(float(roc_auc_score(yt[idx], sc[idx])))
            v = np.array(vals)
            lres[st] = {"auc_full": float(roc_auc_score(yt, sc)),
                        "p05": float(np.percentile(v, 5)),
                        "p95": float(np.percentile(v, 95)),
                        "p_above_0.5": float((v > 0.5).mean())}
        boot_res[label_col] = lres

    # ── 汇总：各臂 test AUC mean±std + 泄露差值 ──
    summary, leakage = {}, {}
    for arm in ARMS:
        arm_files = [f"{arm}_seed{s}.csv" for s in SEEDS]
        ares = {}
        for label_col in LABEL_COLS:
            lres = {}
            for st in SUBTYPES:
                tests = [per_file[label_col][f][f"{st}_test"]["auc"] for f in arm_files]
                lres[st] = {"test_auc_mean": float(np.mean(tests)),
                            "test_auc_std": float(np.std(tests)),
                            "test_aucs": tests}
            ares[label_col] = lres
        summary[arm] = ares
    for label_col in LABEL_COLS:
        leakage[label_col] = {
            st: summary["isolate_random"][label_col][st]["test_auc_mean"]
                - summary["cluster"][label_col][st]["test_auc_mean"]
            for st in SUBTYPES
        }

    # ── 打印 ──
    print(f"\n{'='*76}\nCNN baseline 在新切分下的 test AUC（5 seeds）\n{'='*76}")
    for arm in ARMS:
        print(f"\n--- {arm} 臂 ---")
        for label_col in LABEL_COLS:
            for st in SUBTYPES:
                s = summary[arm][label_col][st]
                print(f"  {label_col:>22} {st}: test AUC = "
                      f"{s['test_auc_mean']:.3f} ± {s['test_auc_std']:.3f}")
    print(f"\n泄露量化（isolate_random − cluster）:")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            print(f"  {label_col:>22} {st}: {leakage[label_col][st]:+.3f}")
    print(f"\ncluster_seed42 test cluster bootstrap (B={B_BOOT}):")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            b = boot_res[label_col][st]
            print(f"  {label_col:>22} {st}: AUC={b['auc_full']:.3f} "
                  f"[{b['p05']:.3f}, {b['p95']:.3f}] P(>0.5)={b['p_above_0.5']:.2f}")
    print(f"\n全量 holdout 参照（旧口径）:")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            print(f"  {label_col:>22} {st}: AUC={holdout_ref[label_col][st]['auc']:.3f}")

    out = {"per_file": per_file, "summary": summary, "leakage_delta": leakage,
           "bootstrap_cluster_seed42": boot_res, "holdout_full_ref": holdout_ref,
           "note": "stage2 checkpoint 未重训；其选择全部在 H1+H3 val 上，未接触 H5/H7"}
    out_path = OUTPUT_DIR / "eval_h5h7_valtest.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
