"""Jump 分类 — ESM per-residue → attention pooling → 单层 linear（自适应池化档）

目的：在 mean-pool + Ridge LR（linear probe）与 transformer 头之间插入最小
容量档 —— pooling 可学习（可学习 query 的 masked softmax 加权求和），
无 transformer 块 / FFN，隔离"自适应池化"本身的贡献。

资源策略：磁盘不足（每层 fp16 8.1GB 无法落盘）→ hidden states 常驻 RAM。
注意容器 cgroup 内存上限仅 ~62GB（/proc/meminfo 是宿主机的 500GB，不可用），
按 cgroup 剩余额度分批（每批 ~5 层），每批重新做一次 ESM 前向提取。
全程只写结果 JSON。

协议（复刻 train_tf_probe.py / target_val_layer_select.py，保证三方可比）：
  - 候选层 = 全部 31 层 hidden states（与 linear probe 层扫描范围一致）
  - 每 (层 × label × seed 42/123/456) 训练，early stop on H1+H3 val AUC
  - 3 seed logit 平均集成；10 个 h5h7 切分文件 × 4 mask 评 AUC
  - 每 (文件, label, 亚型) 按 val AUC argmax 选层 → test 评一次
  - cluster_seed42 选中配置 test 侧 cluster bootstrap（B=1000）
  - 与 linear probe / transformer 头并排对比

输出：ESM_tf_clf/output/adaptive_pool_results.json
"""

import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from ESM_tf_clf.config import (DATA_CSV, SPLIT_CSV, H5H7_SPLIT_DIR,
                               ACCESSIONS_NPY, LABELS_DIR, ESM_MODEL, ESM_DIM,
                               MAX_LEN, LABEL_COLS, BATCH_SIZE, LR,
                               WEIGHT_DECAY, EPOCHS, PATIENCE, TRAIN_SEEDS,
                               RANDOM_SEED, OUT_DIR)
from ESM_tf_clf.model import AttentionPoolClassifier

SEEDS = [42, 43, 44, 45, 46]
ARMS = ["cluster", "isolate_random"]
SUBTYPES = ["h5", "h7"]
B_BOOT = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACT_BATCH = 8
LAYER_GB = 8.1             # 单层 fp16 (11060, 573, 640) 占用
RESERVE_GB = 12.0          # 给模型/解释器/系统预留


def _mem_available_gb():
    """cgroup 感知的可用内存（/proc/meminfo 是宿主机值，容器内不可用）。"""
    host = 0.0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                host = int(line.split()[1]) / 1e6
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            cap = f.read().strip()
        with open("/sys/fs/cgroup/memory.current") as f:
            cur = int(f.read().strip()) / 1e9
        if cap != "max":
            return min(host, float(cap) / 1e9 - cur)
    except FileNotFoundError:
        pass
    return host


# ═══════════════════════════════════════════════════════════
# Stage A: 提取 per-residue hidden states 到 RAM
# ═══════════════════════════════════════════════════════════

def extract_layers(layer_list, seqs, n):
    """一次前向，把 layer_list 中各层 per-residue hidden states 存入 RAM。

    返回 ({li: fp16 (n, MAX_LEN, 640)}, lens int32 (n,))，右侧 zero-pad。
    """
    maps = {li: np.zeros((n, MAX_LEN, ESM_DIM), dtype=np.float16)
            for li in layer_list}
    lens = np.zeros(n, dtype=np.int32)

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True)
    model = model.to(DEVICE).eval()

    for i in tqdm(range(0, n, EXTRACT_BATCH), desc="提取 per-residue (RAM)"):
        batch_seqs = seqs[i:i + EXTRACT_BATCH]
        inp = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        hs = {li: out.hidden_states[li].half().cpu().numpy()
              for li in layer_list}
        for j, seq in enumerate(batch_seqs):
            L = len(seq)
            lens[i + j] = L
            for li in layer_list:
                maps[li][i + j, :L] = hs[li][j, 1:1 + L]     # 去 CLS/EOS
    del model
    torch.cuda.empty_cache()
    return maps, lens


def sanity_check(maps, lens, layer_list):
    """抽查：per-residue mean ≈ all_layers 同层 mean-pooled（余弦 ≈ 1）。"""
    pooled = np.load("ESM_clf/jump_exp/output/esm_emb_150M_all_layers.npy",
                     mmap_mode="r")
    for li in [layer_list[0], layer_list[len(layer_list) // 2], layer_list[-1]]:
        a = maps[li][0, :lens[0]].astype(np.float32).mean(axis=0)
        b = np.asarray(pooled[li, 0])
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(f"  抽查 L{li} 行0: mean-pool 与 all_layers 余弦相似度 = {cos:.4f}")


# ═══════════════════════════════════════════════════════════
# Stage B: 训练（attention pooling + linear）
# ═══════════════════════════════════════════════════════════

def train_one(emb, lens, y, tr_idx, va_idx, seed):
    """训练一个 (layer, label, seed) 的 attention-pool 头，返回 best val AUC 模型。

    emb: GPU 上的 (N, 573, 640) fp16 张量；lens/y/tr_idx/va_idx 为 numpy。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AttentionPoolClassifier().to(DEVICE)
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


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"设备: {DEVICE}")

    # ── split 与 h5h7 切分文件（与 train_tf_probe.py 一致）──
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

    labels = {c: np.load(LABELS_DIR / f"labels_{c}.npy") for c in LABEL_COLS}

    # ── 序列（行序锚定 accessions.npy，raw ha_sequence）──
    acc_order = np.load(ACCESSIONS_NPY, allow_pickle=True)
    data_df = pd.read_csv(DATA_CSV).set_index("accession")
    data_df = data_df[~data_df.index.duplicated(keep="first")]
    seqs = data_df.loc[acc_order, "ha_sequence"].tolist()
    n = len(seqs)
    print(f"总 isolates: {n}")

    # ── 全部 31 层；按 cgroup 可用内存分批（每批重新前向提取）──
    ALL_LAYERS = list(range(31))
    avail = _mem_available_gb()
    per_batch = max(1, int((avail - RESERVE_GB) / LAYER_GB))
    layer_batches = [ALL_LAYERS[i:i + per_batch]
                     for i in range(0, len(ALL_LAYERS), per_batch)]
    print(f"可用内存 {avail:.0f}GB → 每批 {per_batch} 层，"
          f"共 {len(layer_batches)} 批: "
          f"{[f'{b[0]}-{b[-1]}' for b in layer_batches]}")

    # scores[(label, li)] = eval_idx 上的集成 logit；h13test[(label, li)] = test AUC
    scores, h13test, train_meta = {}, {}, {}
    lens = None
    for layer_list in layer_batches:
        t0 = time.time()
        maps, lens = extract_layers(layer_list, seqs, n)
        print(f"提取完成 {layer_list[0]}..{layer_list[-1]} "
              f"({time.time()-t0:.0f}s)", flush=True)
        sanity_check(maps, lens, layer_list)

        for li in layer_list:
            emb = torch.from_numpy(maps[li]).to(DEVICE)   # fp16 整层上 GPU
            for label_col in LABEL_COLS:
                y = labels[label_col]
                logits_seeds, test_seeds = [], []
                for seed in TRAIN_SEEDS:
                    t1 = time.time()
                    model, best_va, n_ep = train_one(emb, lens, y,
                                                     tr_idx, va_idx, seed)
                    logits_seeds.append(predict(model, emb, lens, eval_idx))
                    test_seeds.append(predict(model, emb, lens, test_idx))
                    train_meta[f"{label_col}|L{li}|s{seed}"] = {
                        "best_val_auc": float(best_va), "epochs": n_ep}
                    print(f"  {label_col} L{li} s{seed}: val_auc={best_va:.4f} "
                          f"epochs={n_ep} ({time.time()-t1:.0f}s)", flush=True)
                    del model
                scores[(label_col, li)] = np.mean(logits_seeds, axis=0)
                h13test[(label_col, li)] = float(roc_auc_score(
                    y[test_idx], np.mean(test_seeds, axis=0)))
            del emb
            maps[li] = None
            torch.cuda.empty_cache()
        del maps

    # ── Stage C: 10 切分文件 × 4 mask 的 AUC 表 ──
    aucs = {c: {} for c in LABEL_COLS}
    for label_col in LABEL_COLS:
        y = labels[label_col]
        for li in ALL_LAYERS:
            per_file = {}
            for fname in files:
                per_mask = {}
                for mask_name, rows in file_rows[fname].items():
                    p = np.array([pos_of[r] for r in rows])
                    per_mask[mask_name] = float(roc_auc_score(
                        y[rows], scores[(label_col, li)][p]))
                per_file[fname] = per_mask
            aucs[label_col][li] = per_file

    # ── 分亚型选层：val argmax → test 评一次 ──
    per_file_res = {}
    for fname in files:
        fres = {}
        for label_col in LABEL_COLS:
            lres = {}
            for st in SUBTYPES:
                val_mask, test_mask = f"{st}_val", f"{st}_test"
                val_aucs = {li: aucs[label_col][li][fname][val_mask]
                            for li in ALL_LAYERS}
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
            sc, yt = scores[(label_col, best)][p], y[rows]
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

    # ── 打印 + 与 linear / tf 头并排（cluster 臂 test AUC mean）──
    lin = json.load(open("ESM_clf/jump_exp/output/target_val_layer_select.json"))
    tf = json.load(open(OUT_DIR / "tf_probe_results.json"))
    print(f"\n{'='*76}\nAttention pooling：val 选层 → test 评一次（5 seeds）\n{'='*76}")
    for arm in ARMS:
        print(f"\n--- {arm} 臂 ---")
        for label_col in LABEL_COLS:
            for st in SUBTYPES:
                s = summary[arm][label_col][st]
                print(f"  {label_col:>22} {st}: 选层 {s['layer_freq']}  "
                      f"test AUC = {s['test_auc_mean']:.3f} ± {s['test_auc_std']:.3f}")
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
    print(f"\n三方法并排（cluster 臂 test AUC mean）：")
    print(f"  {'标签':>22} {'亚型':>4} {'linear':>8} {'attnpool':>8} {'tf':>8}")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            lm = lin["summary"]["cluster"][label_col][st]["test_auc_mean"]
            am = summary["cluster"][label_col][st]["test_auc_mean"]
            tm = tf["summary"]["cluster"][label_col][st]["test_auc_mean"]
            print(f"  {label_col:>22} {st:>4} {lm:>8.3f} {am:>8.3f} {tm:>8.3f}")

    out = {"per_file": per_file_res, "summary": summary,
           "leakage_delta": leakage, "bootstrap_cluster_seed42": boot_res,
           "h13_test_per_layer": {f"{c}|L{li}": h13test[(c, li)]
                                  for (c, li) in h13test},
           "train_meta": train_meta,
           "protocol": "frozen ESM per-residue → attention pooling（可学习 query）"
                       " → 单层 linear；early stop on H1+H3 val；层选择在 h5/h7 val；"
                       "test 只评一次；候选层 0..30 全扫；hidden states 常驻 RAM"}
    out_path = OUT_DIR / "adaptive_pool_results.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
