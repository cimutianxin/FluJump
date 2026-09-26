"""训练 transformer 头并按 target-val 协议评估（与 linear 实验严格可比）

协议（复刻 ESM_clf/jump_exp/target_val_layer_select.py）：
  - probe 只依赖 (label, layer) 与 H1+H3 train/val，与切分文件无关
  - 每 (label, layer) 训 3 个种子，logit 平均集成
  - early stop 用 H1+H3 val AUC（不碰 H5/H7 任何标签）
  - 在 10 个 h5h7 切分文件 × 4 mask 上评 AUC；每 (文件, label, 亚型)
    在候选层 {13,17,28} 上按 val AUC argmax 选层 → test 评一次
  - cluster_seed42 选中配置做 test 侧 cluster bootstrap（B=1000）
  - 与 linear probe（target_val_layer_select.json）并排对比

输出：ESM_tf_clf/output/tf_probe_results.json
"""

import json
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from ESM_tf_clf.config import *
from ESM_tf_clf.model import ResidueTransformerClassifier

SEEDS = [42, 43, 44, 45, 46]
ARMS = ["cluster", "isolate_random"]
SUBTYPES = ["h5", "h7"]
B_BOOT = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one(emb, lens, y, tr_idx, va_idx, seed):
    """训练一个 (layer, label, seed) 的 transformer 头，返回 best val AUC 与模型

    emb: GPU 上的 (N, 573, 640) fp16 张量；lens/y/tr_idx/va_idx 为 numpy
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ResidueTransformerClassifier().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    ytr = torch.from_numpy(y[tr_idx]).float().to(DEVICE)
    ltr = torch.from_numpy(lens[tr_idx]).long().to(DEVICE)
    tr_idx_t = torch.from_numpy(tr_idx).to(DEVICE)
    va_idx_t = torch.from_numpy(va_idx).to(DEVICE)
    yva = y[va_idx]
    lva = torch.from_numpy(lens[va_idx]).long().to(DEVICE)

    best_auc, best_state, patience = 0.0, None, 0
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

        # val AUC（early stop 依据，H1+H3 val，不碰 H5/H7）
        model.eval()
        with torch.no_grad():
            preds = []
            for b in range(0, len(va_idx), 64):
                bi = torch.arange(b, min(b + 64, len(va_idx)), device=DEVICE)
                x = emb[va_idx_t[bi]]
                l = lva[bi]
                x = x[:, :int(l.max())].float()
                preds.append(model(x, l).cpu().numpy())
            va_auc = roc_auc_score(yva, np.concatenate(preds))
        if va_auc > best_auc:
            best_auc, patience = va_auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_auc, epoch


@torch.no_grad()
def predict(model, emb, lens, rows, bs=64):
    """rows: numpy 行号数组；返回这些行上的 logit"""
    model.eval()
    preds = []
    rows_t = torch.from_numpy(np.asarray(rows)).to(DEVICE)
    lr = torch.from_numpy(lens[rows]).long().to(DEVICE)
    for b in range(0, len(rows), bs):
        bi = torch.arange(b, min(b + bs, len(rows)), device=DEVICE)
        x = emb[rows_t[bi]]
        l = lr[bi]
        x = x[:, :int(l.max())].float()
        preds.append(model(x, l).cpu().numpy())
    return np.concatenate(preds)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_df = pd.read_csv(SPLIT_CSV)
    m_train = (split_df["split"] == "train").values
    m_val = (split_df["split"] == "val").values
    m_test = (split_df["split"] == "test").values
    m_h57 = split_df["split"].isin(["h5_holdout", "h7_holdout"]).values
    tr_idx, va_idx = np.where(m_train)[0], np.where(m_val)[0]
    test_idx, eval_idx = np.where(m_test)[0], np.where(m_h57)[0]
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

    lens = np.load(OUT_DIR / "residue_lens.npy")
    labels = {c: np.load(LABELS_DIR / f"labels_{c}.npy") for c in LABEL_COLS}

    # aucs[label][layer][fname][mask]；h13test[label][layer]；ens logits 供 bootstrap
    aucs = {c: {} for c in LABEL_COLS}
    h13test = {c: {} for c in LABEL_COLS}
    ens_eval = {}   # (label, layer) -> eval_idx 上的集成 logit
    train_meta = {}

    for li in CANDIDATE_LAYERS:
        emb_np = np.load(OUT_DIR / f"residue_emb_L{li}.npy", mmap_mode="r")
        emb = torch.from_numpy(np.asarray(emb_np)).to(DEVICE)   # fp16 整层上 GPU
        print(f"\n=== L{li} 已加载 {tuple(emb.shape)} ===", flush=True)
        for label_col in LABEL_COLS:
            y = labels[label_col]
            logits_seeds, test_seeds = [], []
            for seed in TRAIN_SEEDS:
                t0 = time.time()
                model, best_va, n_ep = train_one(emb, lens, y, tr_idx, va_idx, seed)
                logits_seeds.append(predict(model, emb, lens, eval_idx))
                test_seeds.append(predict(model, emb, lens, test_idx))
                train_meta[f"{label_col}|L{li}|s{seed}"] = {
                    "best_val_auc": float(best_va), "epochs": n_ep}
                print(f"  {label_col} L{li} s{seed}: val_auc={best_va:.4f} "
                      f"epochs={n_ep} ({time.time()-t0:.0f}s)", flush=True)
                del model
                torch.cuda.empty_cache()
            ens_eval[(label_col, li)] = np.mean(logits_seeds, axis=0)
            h13test[label_col][li] = float(roc_auc_score(
                y[test_idx], np.mean(test_seeds, axis=0)))
            per_file = {}
            for fname in files:
                per_mask = {}
                for mask_name, rows in file_rows[fname].items():
                    p = np.array([pos_of[r] for r in rows])
                    per_mask[mask_name] = float(roc_auc_score(
                        y[rows], ens_eval[(label_col, li)][p]))
                per_file[fname] = per_mask
            aucs[label_col][li] = per_file
        del emb
        torch.cuda.empty_cache()

    # ── 分亚型选层：val argmax → test 评一次 ──
    per_file_res = {}
    for fname in files:
        fres = {}
        for label_col in LABEL_COLS:
            lres = {}
            for st in SUBTYPES:
                val_mask, test_mask = f"{st}_val", f"{st}_test"
                val_aucs = {li: aucs[label_col][li][fname][val_mask]
                            for li in CANDIDATE_LAYERS}
                best = max(val_aucs, key=val_aucs.get)
                lres[st] = {"selected_layer": best, "val_auc": val_aucs[best],
                            "test_auc": aucs[label_col][best][fname][test_mask]}
            fres[label_col] = lres
        per_file_res[fname] = fres

    # ── 汇总：两臂 5-seed mean±std + 选层频率 + 泄露差值 ──
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
                lres[st] = {"selected_layers": picks,
                            "layer_freq": dict(Counter(picks)),
                            "test_auc_mean": float(np.mean(tests)),
                            "test_auc_std": float(np.std(tests))}
            ares[label_col] = lres
        summary[arm] = ares
    for label_col in LABEL_COLS:
        leakage[label_col] = {
            st: summary["isolate_random"][label_col][st]["test_auc_mean"]
                - summary["cluster"][label_col][st]["test_auc_mean"]
            for st in SUBTYPES}

    # ── cluster_seed42 test 侧 cluster bootstrap ──
    fname0 = "cluster_seed42.csv"
    rng = np.random.default_rng(RANDOM_SEED)
    boot_res = {}
    for label_col in LABEL_COLS:
        y = labels[label_col]
        lres = {}
        for st in SUBTYPES:
            best = per_file_res[fname0][label_col][st]["selected_layer"]
            test_mask = f"{st}_test"
            rows, cl = file_rows[fname0][test_mask], file_cl[fname0][test_mask]
            p = np.array([pos_of[r] for r in rows])
            sc, yt = ens_eval[(label_col, best)][p], y[rows]
            uniq_cl = np.unique(cl)
            vals = []
            for _ in range(B_BOOT):
                samp = rng.choice(uniq_cl, size=len(uniq_cl), replace=True)
                idx = np.concatenate([np.where(cl == c)[0] for c in samp])
                if len(np.unique(yt[idx])) < 2:
                    continue
                vals.append(float(roc_auc_score(yt[idx], sc[idx])))
            v = np.array(vals)
            lres[st] = {"layer": best, "auc_full": float(roc_auc_score(yt, sc)),
                        "p05": float(np.percentile(v, 5)),
                        "p95": float(np.percentile(v, 95)),
                        "p_above_0.5": float((v > 0.5).mean())}
        boot_res[label_col] = lres

    # ── 打印（含与 linear probe 的并排对比）──
    lin = json.load(open("ESM_clf/jump_exp/output/target_val_layer_select.json"))
    print(f"\n{'='*76}\nTransformer 头：val 选层 → test 评一次（5 seeds）\n{'='*76}")
    for arm in ARMS:
        print(f"\n--- {arm} 臂 ---")
        for label_col in LABEL_COLS:
            for st in SUBTYPES:
                s = summary[arm][label_col][st]
                print(f"  {label_col:>22} {st}: 选层 {s['layer_freq']}  "
                      f"test AUC = {s['test_auc_mean']:.3f} ± {s['test_auc_std']:.3f}")
    print(f"\nH1+H3 test AUC（sanity，逐层）:")
    for label_col in LABEL_COLS:
        print(f"  {label_col}: " + "  ".join(
            f"L{li}={h13test[label_col][li]:.3f}" for li in CANDIDATE_LAYERS))
    print(f"\n泄露量化（isolate_random − cluster）:")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            print(f"  {label_col:>22} {st}: {leakage[label_col][st]:+.3f}")
    print(f"\ncluster_seed42 test cluster bootstrap (B={B_BOOT}):")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            b = boot_res[label_col][st]
            print(f"  {label_col:>22} {st} L{b['layer']}: AUC={b['auc_full']:.3f} "
                  f"[{b['p05']:.3f}, {b['p95']:.3f}] P(>0.5)={b['p_above_0.5']:.2f}")
    print(f"\n与 linear probe 并排（cluster 臂 test AUC mean）：")
    print(f"  {'标签':>22} {'亚型':>4} {'linear':>8} {'tf':>8}")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            lm = lin["summary"]["cluster"][label_col][st]["test_auc_mean"]
            tm = summary["cluster"][label_col][st]["test_auc_mean"]
            print(f"  {label_col:>22} {st:>4} {lm:>8.3f} {tm:>8.3f}")

    out = {"per_file": per_file_res, "summary": summary, "leakage_delta": leakage,
           "bootstrap_cluster_seed42": boot_res, "h13_test_per_layer": h13test,
           "train_meta": train_meta,
           "protocol": "frozen ESM per-residue → TransformerEncoder → attention pool → FFN; "
                       "early stop on H1+H3 val; 层选择在 h5/h7 val; test 只评一次"}
    out_path = OUT_DIR / "tf_probe_results.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
