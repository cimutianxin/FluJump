"""Interval 3 类分类 — 冻结 ESM-2 per-residue embedding + 可训练 Transformer 头

任务（2026-08-10 新定义）：human_first(raw<0) 与 <1yr 合并 → 3 类
  0 <1yr (raw_days < 365) / 1 1-3yr / 2 3yr+

数据集（2026-08-10 v2）：H1+H3+H5+H7 全体（jump=1 且 interval 有效，1191 条
/ 44 clusters），**cluster 级 70/15/15 划分**（无 cluster 跨 split），
见 datascripts/build_interval_h1357_cluster_split.py。

协议（与 ESM_clf/interval_exp/train_interval_clf_final.py 对齐）：
  1. 数据走共享 loader（accession 对齐），emb_rows 切片 residue embedding
  2. 每 (层 L13/17/28 × seed 42/123/456) 训练，CE + balanced class_weight，
     early stop 依据 val balanced_acc；3 seed softmax 概率平均集成
  3. val 上按 balanced_acc（并列比 macro_f1）选唯一 winner 层
  4. test 只评估一次：acc / balanced_acc / macro_f1 / per-class recall
     + bootstrap 95% CI + majority / stratified random 基线 + 分亚型指标

输出: ESM_tf_clf/output/tf_interval3_h1357_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, recall_score)

sys.path.insert(0, ".")
from ESM_tf_clf.config import (CANDIDATE_LAYERS, BATCH_SIZE, LR, WEIGHT_DECAY,
                               EPOCHS, PATIENCE, TRAIN_SEEDS, RANDOM_SEED,
                               OUT_DIR)
from ESM_tf_clf.model import ResidueTransformerClassifier
from ESM_clf.interval_exp.interval_data import load_interval_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INTERVAL3_LABELS = ["<1yr", "1-3yr", "3yr+"]
N_CLS = len(INTERVAL3_LABELS)
N_BOOT = 1000
N_RANDOM_BASELINE = 1000

# ── 数据集：H1357 全体 + cluster 级 70/15/15 划分 ──
INTERVAL_SUBTYPES = ["H1", "H3", "H5", "H7"]
INTERVAL_SPLIT_CSV = Path("data/splits/interval_h1357_cluster_split.csv")
OUT_JSON = "tf_interval3_h1357_results.json"


# ═══════════════════════════════════════════════════════════
# 评估工具
# ═══════════════════════════════════════════════════════════

def clf_metrics(y_true, y_pred):
    """acc / balanced_acc / macro_f1 / per-class recall。"""
    m = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "n": int(len(y_true)),
    }
    for i, label in enumerate(INTERVAL3_LABELS):
        if (y_true == i).sum() > 0:
            m[f"{label}_rec"] = float(recall_score(
                y_true, y_pred, labels=[i], average="macro", zero_division=0))
    return m


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOT, seed=RANDOM_SEED):
    """对 test 指标做 bootstrap 95% CI（重采样 test 索引）。"""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    stats = {"acc": [], "balanced_acc": [], "macro_f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats["acc"].append(accuracy_score(y_true[idx], y_pred[idx]))
        stats["balanced_acc"].append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
        stats["macro_f1"].append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    return {k: {"mean": float(np.mean(v)),
                "ci_lo": float(np.percentile(v, 2.5)),
                "ci_hi": float(np.percentile(v, 97.5))}
            for k, v in stats.items()}


def baseline_metrics(y_train, y_test, n_runs=N_RANDOM_BASELINE, seed=RANDOM_SEED):
    """majority 与 stratified random 基线。"""
    out = {}
    maj = np.bincount(y_train).argmax()
    out["majority"] = clf_metrics(y_test, np.full_like(y_test, maj))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_train, return_counts=True)
    p = counts / counts.sum()
    runs = {"acc": [], "balanced_acc": [], "macro_f1": []}
    for _ in range(n_runs):
        yp = rng.choice(classes, size=len(y_test), p=p)
        runs["acc"].append(accuracy_score(y_test, yp))
        runs["balanced_acc"].append(balanced_accuracy_score(y_test, yp))
        runs["macro_f1"].append(f1_score(y_test, yp, average="macro"))
    out["stratified_random"] = {k: {"mean": float(np.mean(v)),
                                    "std": float(np.std(v))}
                                for k, v in runs.items()}
    return out


# ═══════════════════════════════════════════════════════════
# 训练 / 预测
# ═══════════════════════════════════════════════════════════

def train_one(emb, lens, y, tr_idx, va_idx, cls_w, seed):
    """训练一个 (layer, seed) 的 3 类 transformer 头。

    emb: GPU 上 (N, 573, 640) fp16 张量（仅 interval 样本）；
    early stop 依据 val balanced_acc。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ResidueTransformerClassifier(out_dim=N_CLS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=cls_w)

    ytr = torch.from_numpy(y[tr_idx]).long().to(DEVICE)
    ltr = torch.from_numpy(lens[tr_idx]).long().to(DEVICE)
    tr_idx_t = torch.from_numpy(tr_idx).to(DEVICE)
    va_idx_t = torch.from_numpy(va_idx).to(DEVICE)
    yva = y[va_idx]
    lva = torch.from_numpy(lens[va_idx]).long().to(DEVICE)

    best_ba, best_state, patience = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = np.random.permutation(len(tr_idx))
        for b in range(0, len(perm), BATCH_SIZE):
            bi = torch.from_numpy(perm[b:b + BATCH_SIZE]).to(DEVICE)
            x = emb[tr_idx_t[bi]]                  # (B, 573, 640) fp16
            l = ltr[bi]
            x = x[:, :int(l.max())].float()
            opt.zero_grad()
            loss = loss_fn(model(x, l), ytr[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # val balanced_acc（early stop 依据）
        model.eval()
        with torch.no_grad():
            preds = []
            for b in range(0, len(va_idx), 64):
                bi = torch.arange(b, min(b + 64, len(va_idx)), device=DEVICE)
                x = emb[va_idx_t[bi]]
                l = lva[bi]
                x = x[:, :int(l.max())].float()
                preds.append(model(x, l).argmax(-1).cpu().numpy())
            va_ba = balanced_accuracy_score(yva, np.concatenate(preds))
        if va_ba > best_ba:
            best_ba, patience = va_ba, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_ba, epoch


@torch.no_grad()
def predict_proba(model, emb, lens, rows, bs=64):
    """rows: numpy 行号数组；返回这些行上的 softmax 概率 (len(rows), N_CLS)。"""
    model.eval()
    probs = []
    rows_t = torch.from_numpy(np.asarray(rows)).to(DEVICE)
    lr = torch.from_numpy(lens[rows]).long().to(DEVICE)
    for b in range(0, len(rows), bs):
        bi = torch.arange(b, min(b + bs, len(rows)), device=DEVICE)
        x = emb[rows_t[bi]]
        l = lr[bi]
        x = x[:, :int(l.max())].float()
        probs.append(torch.softmax(model(x, l), dim=-1).cpu().numpy())
    return np.concatenate(probs)


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    print(f"设备: {DEVICE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_interval_data(subtypes=INTERVAL_SUBTYPES,
                              split_csv=INTERVAL_SPLIT_CSV, verbose=True)
    y, masks, emb_rows = data["y_cat3"], data["masks"], data["emb_rows"]
    subtypes = data["subtypes"]
    tr_idx = np.where(masks["train"])[0]
    va_idx = np.where(masks["val"])[0]
    te_idx = np.where(masks["test"])[0]
    yt, yv, yx = y[tr_idx], y[va_idx], y[te_idx]

    lens_all = np.load(OUT_DIR / "residue_lens.npy")
    lens = lens_all[emb_rows]

    # balanced class_weight（由 train 频率倒数）
    counts = np.bincount(yt, minlength=N_CLS).astype(np.float64)
    w = len(yt) / (N_CLS * counts)
    cls_w = torch.from_numpy(w).float().to(DEVICE)

    print(f"\n{'='*66}")
    print(f"  Interval 3 类分类 — Transformer 头  {INTERVAL3_LABELS}")
    for s, ys in [("train", yt), ("val", yv), ("test", yx)]:
        dist = {INTERVAL3_LABELS[i]: int((ys == i).sum()) for i in range(N_CLS)}
        print(f"  {s}: n={len(ys):>4}  {dist}")
    print(f"  class_weight: {np.round(w, 3).tolist()}")
    print(f"{'='*66}")

    val_table, ens_val, ens_test, train_meta = {}, {}, {}, {}

    # ── Phase 1: 每层 3 seed 训练 + val 评估 ──
    for li in CANDIDATE_LAYERS:
        emb_np = np.load(OUT_DIR / f"residue_emb_L{li}.npy", mmap_mode="r")
        emb = torch.from_numpy(np.asarray(emb_np[emb_rows])).to(DEVICE)  # fp16
        print(f"\n=== L{li} 已加载 {tuple(emb.shape)} ===", flush=True)

        pv_seeds, px_seeds = [], []
        for seed in TRAIN_SEEDS:
            t0 = time.time()
            model, best_ba, n_ep = train_one(emb, lens, y, tr_idx, va_idx,
                                             cls_w, seed)
            pv_seeds.append(predict_proba(model, emb, lens, va_idx))
            px_seeds.append(predict_proba(model, emb, lens, te_idx))
            train_meta[f"L{li}|s{seed}"] = {"best_val_bal_acc": float(best_ba),
                                            "epochs": n_ep}
            print(f"  L{li} s{seed}: val_bal_acc={best_ba:.4f} "
                  f"epochs={n_ep} ({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()

        ens_val[li] = np.mean(pv_seeds, axis=0)
        ens_test[li] = np.mean(px_seeds, axis=0)
        vm = clf_metrics(yv, ens_val[li].argmax(-1))
        val_table[li] = vm
        print(f"  L{li} 集成 val: acc={vm['acc']:.4f}  "
              f"bal_acc={vm['balanced_acc']:.4f}  macro_f1={vm['macro_f1']:.4f}",
              flush=True)
        del emb
        torch.cuda.empty_cache()

    # ── Phase 2: val 选择唯一 winner 层 ──
    winner = max(val_table, key=lambda li: (val_table[li]["balanced_acc"],
                                            val_table[li]["macro_f1"]))
    print(f"\n  ★ Winner 层（val balanced_acc）: L{winner}  "
          f"bal_acc={val_table[winner]['balanced_acc']:.4f}  "
          f"macro_f1={val_table[winner]['macro_f1']:.4f}")

    # ── Phase 3: test 单次评估 ──
    yp = ens_test[winner].argmax(-1)
    test_metrics = clf_metrics(yx, yp)
    test_ci = bootstrap_ci(yx, yp)
    baselines = baseline_metrics(yt, yx)

    per_subtype = {}
    for st in INTERVAL_SUBTYPES:
        sm = subtypes[te_idx] == st
        if sm.sum() >= 2 and len(np.unique(yx[sm])) > 1:
            per_subtype[st] = clf_metrics(yx[sm], yp[sm])

    print(f"\n{'='*66}")
    print(f"  TEST（仅一次评估）— Transformer 头 L{winner}（3 seed 概率集成）")
    print(f"  acc={test_metrics['acc']:.4f}  bal_acc={test_metrics['balanced_acc']:.4f}  "
          f"macro_f1={test_metrics['macro_f1']:.4f}")
    for k, ci in test_ci.items():
        print(f"    {k:<14} 95% CI [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
    print(f"  基线: majority bal_acc={baselines['majority']['balanced_acc']:.4f}  "
          f"random bal_acc={baselines['stratified_random']['balanced_acc']['mean']:.4f}")
    recalls = [f"{l}={test_metrics.get(f'{l}_rec', 0):.3f}" for l in INTERVAL3_LABELS]
    print(f"  per-class recall: {', '.join(recalls)}")
    for st, r in per_subtype.items():
        print(f"  {st}: acc={r['acc']:.4f}  bal_acc={r['balanced_acc']:.4f}  n={r['n']}")

    # ── 保存 ──
    result = {
        "protocol": "train（CE+balanced class_weight, early stop on val bal_acc）"
                    " → 3 seed 概率集成 → val 选层 → test 单次评估 + bootstrap CI",
        "dataset": {"subtypes": INTERVAL_SUBTYPES,
                    "split_csv": str(INTERVAL_SPLIT_CSV),
                    "split_level": "cluster (无跨 split 泄漏)"},
        "labels": INTERVAL3_LABELS,
        "class_weight": w.tolist(),
        "winner_layer": int(winner),
        "val_table": {f"L{li}": val_table[li] for li in val_table},
        "test": test_metrics,
        "test_bootstrap_ci": test_ci,
        "test_per_subtype": per_subtype,
        "baselines": baselines,
        "train_meta": train_meta,
        "split_class_dist": {
            s: {INTERVAL3_LABELS[i]: int((y[masks[s]] == i).sum())
                for i in range(N_CLS)}
            for s in ["train", "val", "test"]
        },
    }
    out_path = OUT_DIR / OUT_JSON
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
