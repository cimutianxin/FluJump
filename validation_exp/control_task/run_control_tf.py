"""control task · transformer 头：容量测量（固定 epoch，不早停）

对每 (layer, label, shuffle_seed) 训练 ResidueTransformerClassifier
（直接 import ESM_tf_clf.model，不复制代码）：
  - 固定 FIXED_EPOCHS 轮、不早停 → 末态对（打乱后）train 标签的 AUC = 纯容量记忆
  - 同时跟踪逐 epoch 的"best-val 状态"（val 用真实标签）→ 近似正式训练的
    early-stop 协议产物，报其对真实标签的 val/test/H5/H7 AUC（应 ≈0.5）
  - real 行（seed=null）= 真实标签训练的同款协议对照

输出：output/control_tf.json
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from validation_exp.control_task.config import *
from ESM_tf_clf.model import ResidueTransformerClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_NAMES = ["train", "val", "test", "h5_holdout", "h7_holdout"]


def train_control(emb, lens, y, tr_idx, va_idx, ytr_labels):
    """固定 FIXED_EPOCHS 轮训练；返回 (末态模型, best-val-状态模型, 轨迹)"""
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    model = ResidueTransformerClassifier().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    ytr_t = torch.from_numpy(ytr_labels).float().to(DEVICE)
    ltr = torch.from_numpy(lens[tr_idx]).long().to(DEVICE)
    tr_t = torch.from_numpy(tr_idx).to(DEVICE)
    va_t = torch.from_numpy(va_idx).to(DEVICE)
    yva_real = y[va_idx]                       # val 始终用真实标签
    lva = torch.from_numpy(lens[va_idx]).long().to(DEVICE)

    best_va_auc, best_state = 0.0, None
    for epoch in range(1, FIXED_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(len(tr_idx))
        for b in range(0, len(perm), BATCH_SIZE):
            bi = torch.from_numpy(perm[b:b + BATCH_SIZE]).to(DEVICE)
            x = emb[tr_t[bi]]
            l = ltr[bi]
            x = x[:, :int(l.max())].float()
            opt.zero_grad()
            loss = loss_fn(model(x, l), ytr_t[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # 逐 epoch 跟踪 val（真实标签）AUC，保留 best-val 状态
        model.eval()
        with torch.no_grad():
            preds = []
            for b in range(0, len(va_idx), 64):
                bi = torch.arange(b, min(b + 64, len(va_idx)), device=DEVICE)
                x = emb[va_t[bi]]
                l = lva[bi]
                x = x[:, :int(l.max())].float()
                preds.append(model(x, l).cpu().numpy())
            va_auc = roc_auc_score(yva_real, np.concatenate(preds))
        if va_auc > best_va_auc:
            best_va_auc = va_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"    e{epoch}: val_real_auc={va_auc:.3f}", flush=True)

    best_model = ResidueTransformerClassifier().to(DEVICE)
    best_model.load_state_dict(best_state)
    return model, best_model


@torch.no_grad()
def predict(model, emb, lens, rows):
    model.eval()
    preds = []
    rows_t = torch.from_numpy(np.asarray(rows)).to(DEVICE)
    lr_ = torch.from_numpy(lens[rows]).long().to(DEVICE)
    for b in range(0, len(rows), 64):
        bi = torch.arange(b, min(b + 64, len(rows)), device=DEVICE)
        x = emb[rows_t[bi]]
        l = lr_[bi]
        x = x[:, :int(l.max())].float()
        preds.append(model(x, l).cpu().numpy())
    return np.concatenate(preds)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_df = pd.read_csv(SPLIT_CSV)
    splits = split_df["split"].values
    masks = {n: splits == n for n in SPLIT_NAMES}
    idx = {n: np.where(masks[n])[0] for n in SPLIT_NAMES}
    tr_idx, va_idx = idx["train"], idx["val"]
    lens = np.load(TF_OUT / "residue_lens.npy")

    results = {}
    for li in TF_LAYERS:
        emb_np = np.load(TF_OUT / f"residue_emb_L{li}.npy", mmap_mode="r")
        emb = torch.from_numpy(np.asarray(emb_np)).to(DEVICE)
        print(f"\n=== L{li} 已加载 ===", flush=True)
        for label_col in LABEL_COLS:
            y = np.load(JUMP_OUT / f"labels_{label_col}.npy")
            ytr_real = y[tr_idx]
            for seed in [None] + SHUFFLE_SEEDS:
                tag = "real" if seed is None else f"shuffle_{seed}"
                if seed is None:
                    ytr = ytr_real
                else:
                    rng = np.random.default_rng(seed)
                    ytr = rng.permutation(ytr_real)
                t0 = time.time()
                final_m, best_m = train_control(emb, lens, y, tr_idx, va_idx, ytr)

                entry = {}
                # 容量：末态对训练所用标签（打乱/真实）的 train AUC
                sc_tr = predict(final_m, emb, lens, tr_idx)
                entry["fit_train_auc_final"] = float(roc_auc_score(ytr, sc_tr))
                # best-val 状态（近似 early-stop 协议）对真实标签的各处 AUC
                for n in SPLIT_NAMES:
                    sc = predict(best_m, emb, lens, idx[n])
                    entry[f"bestval_real_{n}_auc"] = float(roc_auc_score(y[idx[n]], sc))
                # 末态对真实标签的 train AUC（shuffle 时应 ≈0.5，real 时反映拟合）
                entry["final_real_train_auc"] = float(roc_auc_score(ytr_real, sc_tr))

                results[f"{label_col}|L{li}|{tag}"] = entry
                print(f"  {label_col} L{li} {tag}: fit_train={entry['fit_train_auc_final']:.3f} "
                      f"bestval_real_test={entry['bestval_real_test_auc']:.3f} "
                      f"bestval_real_h7={entry['bestval_real_h7_holdout_auc']:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        del emb
        torch.cuda.empty_cache()

    out_path = OUT_DIR / "control_tf.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
