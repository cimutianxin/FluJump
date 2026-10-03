"""MLP 头 jump/jump_human 跨亚型迁移实验（fig2 容量消融缺口：linear → MLP → transformer）

协议与 target_val_layer_select.py（linear 头）完全对齐，仅把 probe 头换成 MLP：
  - 每层 640 → 256 → 128 → 1，ReLU，隐层间 dropout 0.1
  - BCEWithLogitsLoss(pos_weight=train neg/pos)，Adam lr=1e-3, weight_decay=1e-4,
    batch 512，最多 200 epoch；isolate_split 的 H1+H3 val split 上 early stopping
    （val raw-logit AUC，patience 15，恢复最优权重）；torch.manual_seed(42)

口径（踩坑记录，强制）：
  - 一切 AUC 用 raw logit（sigmoid 之前），禁止概率、禁止符号翻转
  - H5/H7 切分行号一律经 (accession, subtype) 映射回全表行号，禁止位置对齐

输出：
  - output/mlp_probe_results.json（schema 仿 target_val_layer_select.json：
    per_file / summary / leakage_delta，外加 within_dist_test_auc[label][layer]）
  - figdata/fig2_transfer/mlp_head_summary.csv（cluster 臂 20 行长表）
  - figdata/fig2_transfer/mlp_head_within_dist.csv（62 行）
"""

import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

SEEDS = [42, 43, 44, 45, 46]
ARMS = ["cluster", "isolate_random"]
SUBTYPES = ["h5", "h7"]

# ── MLP 训练超参 ──
HIDDEN = (256, 128)
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
MAX_EPOCHS = 200
PATIENCE = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIG_DIR = Path("figdata/fig2_transfer")


class MLPProbe(nn.Module):
    """640 → 256 → 128 → 1，ReLU，隐层间 dropout"""

    def __init__(self, in_dim=640):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN[0]), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xt, yt, Xv, yv):
    """训练一个 MLP probe，返回 (scaler, model)。early stop 用 val raw-logit AUC。"""
    torch.manual_seed(RANDOM_SEED)
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt).astype(np.float32)
    Xv_s = scl.transform(Xv).astype(np.float32)

    model = MLPProbe(Xt_s.shape[1]).to(DEVICE)
    n_pos, n_neg = yt.sum(), len(yt) - yt.sum()
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    Xt_t = torch.from_numpy(Xt_s).to(DEVICE)
    yt_t = torch.from_numpy(yt.astype(np.float32)).to(DEVICE)
    Xv_t = torch.from_numpy(Xv_s).to(DEVICE)
    n = len(yt)

    best_auc, best_state, wait = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = crit(model(Xt_t[idx]), yt_t[idx])
            loss.backward()
            opt.step()
        # val raw-logit AUC early stopping
        model.eval()
        with torch.no_grad():
            va = roc_auc_score(yv, model(Xv_t).cpu().numpy())
        if va > best_auc:
            best_auc, wait = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return scl, model


def raw_logits(scl, model, X):
    """raw logit 打分（sigmoid 之前）"""
    Xs = torch.from_numpy(scl.transform(X).astype(np.float32)).to(DEVICE)
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xs), 4096):
            outs.append(model(Xs[i:i + 4096]).cpu().numpy())
    return np.concatenate(outs)


def main():
    arr = np.load(OUT_DIR / "esm_emb_150M_all_layers.npy")  # (31, N, 640)
    split_df = pd.read_csv(SPLIT_CSV)
    m_train = (split_df["split"] == "train").values
    m_val = (split_df["split"] == "val").values      # H1+H3 内部 val（early stopping）
    m_test = (split_df["split"] == "test").values    # H1+H3 内部 test（同分布口径）
    m_h57 = split_df["split"].isin(["h5_holdout", "h7_holdout"]).values
    eval_idx = np.where(m_h57)[0]
    pos_of = {r: p for p, r in enumerate(eval_idx)}  # 全表行号 → eval 数组下标

    # (accession, subtype) → 全表行号（embedding 行序 == isolate_split 行序）
    key2row = {(a, s): i for i, (a, s) in
               enumerate(zip(split_df["accession"], split_df["subtype"]))}

    # ── 加载 10 个切分文件，构建各 mask 的全表行号 ──
    files = [f"{arm}_seed{s}.csv" for arm in ARMS for s in SEEDS]
    file_rows = {}
    for fname in files:
        sp = pd.read_csv(H5H7_SPLIT_DIR / fname)
        file_rows[fname] = {
            mask_name: np.array([key2row[(a, s)] for a, s in
                                 zip(g["accession"], g["subtype"])])
            for mask_name, g in sp.groupby("split")
        }

    # ── 逐 (label, layer) 训 MLP，记录 H1+H3 test AUC 与 H5/H7 全量 raw logit ──
    within_dist = {c: {} for c in LABEL_COLS}   # label -> layer -> H1+H3 test AUC
    aucs = {c: {} for c in LABEL_COLS}          # label -> layer -> fname -> mask -> auc
    for label_col in LABEL_COLS:
        y = np.load(OUT_DIR / f"labels_{label_col}.npy")
        for li in range(arr.shape[0]):
            scl, model = train_mlp(arr[li][m_train], y[m_train],
                                   arr[li][m_val], y[m_val])
            within_dist[label_col][li] = float(
                roc_auc_score(y[m_test], raw_logits(scl, model, arr[li][m_test])))
            sc = raw_logits(scl, model, arr[li][eval_idx])  # H5/H7 全量 raw logit
            per_file = {}
            for fname in files:
                per_file[fname] = {
                    mask_name: float(roc_auc_score(
                        y[rows], sc[np.array([pos_of[r] for r in rows])]))
                    for mask_name, rows in file_rows[fname].items()
                }
            aucs[label_col][li] = per_file
            if li % 5 == 0 or li == arr.shape[0] - 1:
                print(f"  {label_col} L{li} done "
                      f"(h13_test_auc={within_dist[label_col][li]:.4f})", flush=True)

    # ── 分亚型选层：val argmax → test 评一次（与 linear 完全同逻辑）──
    per_file_res = {}
    for fname in files:
        fres = {}
        for label_col in LABEL_COLS:
            lres = {}
            for st in SUBTYPES:
                val_aucs = {li: aucs[label_col][li][fname][f"{st}_val"]
                            for li in aucs[label_col]}
                best = max(val_aucs, key=val_aucs.get)
                lres[st] = {
                    "selected_layer": best,
                    "val_auc": val_aucs[best],
                    "test_auc": aucs[label_col][best][fname][f"{st}_test"],
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

    # ── 打印主表 ──
    print(f"\n{'='*76}")
    print("MLP 头主结果：val 选层 → test 评一次（5 seeds）")
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
    print(f"\nwithin_dist（H1+H3 test AUC）范围:")
    for label_col in LABEL_COLS:
        v = list(within_dist[label_col].values())
        print(f"  {label_col}: min={min(v):.4f} max={max(v):.4f}")

    out = {"per_file": per_file_res, "summary": summary,
           "leakage_delta": leakage,
           "within_dist_test_auc": within_dist,
           "protocol_note": "MLP 640-256-128-1, raw-logit AUC, 无符号翻转；"
                            "选层只用 H5/H7 内部 val，test 评一次（同 linear 协议）"}
    out_path = OUT_DIR / "mlp_probe_results.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")

    # ── figdata 长表 ──
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in SEEDS:
        fname = f"cluster_seed{s}.csv"
        for label_col in LABEL_COLS:
            for st in SUBTYPES:
                r = per_file_res[fname][label_col][st]
                rows.append({"label": label_col, "holdout": st, "seed": s,
                             "selected_layer": r["selected_layer"],
                             "val_auc": r["val_auc"], "test_auc": r["test_auc"]})
    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "mlp_head_summary.csv", index=False)
    rows2 = [{"label": c, "layer": li, "h13_test_auc": within_dist[c][li]}
             for c in LABEL_COLS for li in range(arr.shape[0])]
    pd.DataFrame(rows2).to_csv(FIG_DIR / "mlp_head_within_dist.csv", index=False)
    print(f"✓ figdata 已保存: {FIG_DIR}/mlp_head_summary.csv ({len(df)} 行), "
          f"mlp_head_within_dist.csv ({len(rows2)} 行)")


if __name__ == "__main__":
    main()
