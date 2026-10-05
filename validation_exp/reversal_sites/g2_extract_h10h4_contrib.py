#!/usr/bin/env python3
"""G2：H10/H4 逐位贡献提取（GPU）

重训 L28/L17/L13 × 双标签 Ridge probe（H1+H3 train，回归断言 H5/H7 headline），
对 H10/H4 isolate 前向时在线计算逐位贡献标量 c_i = scaler(h_i)·w，
只存贡献矩阵 (N,576) fp16 + 长度，不落盘 per-residue embedding。

行序 = group_boundary 的 concat(H10, H4)（同 emb_row_index.csv）。
输出：output/g2_h10h4_contrib.npz（contrib_L{28,17,13}_{label}、lens、accession）

运行：HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env1 python
"""

import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from validation_exp.reversal_sites.config import (
    SPLIT_CSV, ALL_LAYERS_NPY, LABELS, GB_PROC, GB_SUBTYPES, ESM_MODEL,
    RIDGE_C_VALUES, CV_FOLDS, RANDOM_SEED, LABEL_COLS, LAYER_IDX, OUT_DIR,
)

BATCH_SIZE = 8
MAXLEN = 576          # HA 最长 ~570 + 冗余

# 回归校验基准：group_boundary/direction_test.json regression_check
# （10-06 S5 重跑口径：主数据 09-19 词边界修复后标签；jump_human 格不受修复影响，
# 数值与 09-11 一致）。容差 ±0.06 覆盖重训数值噪声。
REPRO_REF = {   # (layer_idx, label) -> (h5_ref, h7_ref)
    (28, "label_is_jump"): (0.8255, 0.2764),
    (28, "label_is_jump_human"): (0.7936, 0.2059),
    (17, "label_is_jump"): (0.3665, 0.9246),
    (17, "label_is_jump_human"): (0.2646, 0.6277),
    (13, "label_is_jump"): (0.2331, 0.7941),
    (13, "label_is_jump_human"): (0.2523, 0.8632),
}
REPRO_TOL = 0.06


def fit_probe(Xtr, ytr):
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    scl = StandardScaler().fit(Xtr)
    gs.fit(scl.transform(Xtr), ytr)
    return scl, gs.best_estimator_


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 重训 probe（全层 embedding 行序 = isolate_split）──
    arr = np.load(ALL_LAYERS_NPY, mmap_mode="r")   # (31,11060,640)
    split = pd.read_csv(SPLIT_CSV)
    m_train = (split["split"] == "train").values
    m_h5 = (split["split"] == "h5_holdout").values
    m_h7 = (split["split"] == "h7_holdout").values

    probes = {}   # (layer_name, label) -> (w, mean, scale)
    for lname, li in LAYER_IDX.items():
        X = np.asarray(arr[li]).astype(np.float32)
        for label in LABEL_COLS:
            y = np.load(LABELS[label])
            scl, clf = fit_probe(X[m_train], y[m_train])
            auc_h5 = float(roc_auc_score(
                y[m_h5], clf.decision_function(scl.transform(X[m_h5]))))
            auc_h7 = float(roc_auc_score(
                y[m_h7], clf.decision_function(scl.transform(X[m_h7]))))
            print(f"[{lname}][{label}] H5={auc_h5:.4f} H7(raw)={auc_h7:.4f}",
                  flush=True)
            # 回归断言：对照 group_boundary 同层基准（容差含 09-19 标签修复位移）
            h5_ref, h7_ref = REPRO_REF[(li, label)]
            assert abs(auc_h5 - h5_ref) < REPRO_TOL, \
                f"H5 复现异常 {lname} {label}: {auc_h5} vs {h5_ref}"
            assert abs(auc_h7 - h7_ref) < REPRO_TOL, \
                f"H7 复现异常 {lname} {label}: {auc_h7} vs {h7_ref}"
            probes[(lname, label)] = (clf.coef_[0].astype(np.float32),
                                      scl.mean_.astype(np.float32),
                                      scl.scale_.astype(np.float32))

    # ── H10/H4 序列 ──
    df = pd.concat([pd.read_csv(GB_PROC / f"{st}_isolates.csv", dtype=str)
                    for st in GB_SUBTYPES], ignore_index=True)
    seqs = df["ha_sequence"].tolist()
    n = len(seqs)
    print(f"H10+H4 isolates: {n}")

    keys = [(ln, lb) for ln in LAYER_IDX for lb in LABEL_COLS]
    contrib = {k: np.zeros((n, MAXLEN), dtype=np.float16) for k in keys}
    lens = np.zeros(n, dtype=np.int32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True)
    model = model.to(device).eval()

    wv = {k: torch.tensor(p[0], device=device) for k, p in probes.items()}
    mv = {k: torch.tensor(p[1], device=device) for k, p in probes.items()}
    sv = {k: torch.tensor(p[2], device=device) for k, p in probes.items()}

    for i in tqdm(range(0, n, BATCH_SIZE), desc="逐位贡献提取"):
        batch = seqs[i:i + BATCH_SIZE]
        inp = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inp)
        for j in range(len(batch)):
            mask = inp["attention_mask"][j, 1:-1].bool()   # 去 CLS/EOS
            L = int(mask.sum())
            r = i + j
            lens[r] = L
            for lname, li in LAYER_IDX.items():
                h = out.hidden_states[li][j, 1:-1][mask].float()  # (L,640)
                for label in LABEL_COLS:
                    k = (lname, label)
                    c = ((h - mv[k]) / sv[k]) @ wv[k]             # (L,)
                    contrib[k][r, :L] = c.half().cpu().numpy()

    np.savez_compressed(OUT_DIR / "g2_h10h4_contrib.npz",
                        **{f"contrib_{ln}_{lb}": contrib[(ln, lb)]
                           for ln, lb in keys},
                        lens=lens,
                        accession=df["accession"].to_numpy(),
                        subtype=df["subtype"].to_numpy())
    print(f"→ {OUT_DIR}/g2_h10h4_contrib.npz（{len(keys)} 个贡献矩阵, shape "
          f"{next(iter(contrib.values())).shape}）")


if __name__ == "__main__":
    main()
