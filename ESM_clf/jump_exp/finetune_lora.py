"""Phase 2：LoRA 微调 ESM-2 150M + 分类头，检验能否矫正 H7 方向

冻结 mean-pool + 线性 probe 的所有低成本修复（erasure/cluster 加权/训练组合）
都无法让 jump_human 在 H7 上方向转正。本实验放开表征层：LoRA 微调 + 分类头，
让模型把"jump 方向"重塑为跨亚型一致。

配置矩阵（4 个 run）：
  - 训练集：H1+H3（现方案）/ H1+H3+H5（矩阵中最有希望的组合）
  - 标签：label_is_jump / label_is_jump_human
评估：val 选最佳 epoch，报 test/H5/H7 原始方向 AUC（logits）。

行序与 embedding 实验一致（split CSV 为准，序列按 accession 从 aligned CSV 取）。

输出：output/finetune_lora.json
"""

import json
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

# ── 超参数 ──
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LR = 1e-4
EPOCHS = 4
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4 个 run：(名称, 训练 split mask 函数, 标签列)
RUNS = [
    ("h13_jump", ["train"], "label_is_jump"),
    ("h13_jumph", ["train"], "label_is_jump_human"),
    ("h135_jump", ["train", "h5_holdout"], "label_is_jump"),
    ("h135_jumph", ["train", "h5_holdout"], "label_is_jump_human"),
]


class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], float(self.labels[i])


class EsmClassifier(nn.Module):
    """ESM-2 + LoRA + mean-pool 线性分类头"""

    def __init__(self):
        super().__init__()
        base = EsmModel.from_pretrained(ESM_MODEL)
        lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                              lora_dropout=LORA_DROPOUT, bias="none",
                              target_modules=["query", "key", "value"])
        self.esm = get_peft_model(base, lora_cfg)
        self.head = nn.Linear(base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.esm(input_ids=input_ids,
                     attention_mask=attention_mask).last_hidden_state
        # mean pooling（去 CLS/EOS）
        mask = attention_mask.clone()
        mask[:, 0] = 0
        mask[:, -1] = 0
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)


@torch.no_grad()
def eval_auc(model, loader):
    model.eval()
    logits_all, y_all = [], []
    for ids, attn, y in loader:
        logits = model(ids.to(DEVICE), attn.to(DEVICE))
        logits_all.append(logits.cpu().numpy())
        y_all.append(y.numpy())
    logits_all = np.concatenate(logits_all)
    y_all = np.concatenate(y_all)
    if len(np.unique(y_all)) < 2:
        return None
    return float(roc_auc_score(y_all, logits_all))


def collate(batch, tokenizer):
    seqs, ys = zip(*batch)
    enc = tokenizer(list(seqs), return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"], torch.tensor(ys)


def make_loader(seqs, labels, tokenizer, shuffle):
    ds = SeqDataset(seqs, labels)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      collate_fn=lambda b: collate(b, tokenizer))


def run_one(name, train_splits, label_col, seqs, labels, splits, tokenizer):
    m_train = np.isin(splits, train_splits)
    m_val = splits == "val"
    eval_sets = {"test": splits == "test",
                 "h5": splits == "h5_holdout",
                 "h7": splits == "h7_holdout"}

    y = labels[label_col]
    train_loader = make_loader(seqs[m_train], y[m_train], tokenizer, True)
    val_loader = make_loader(seqs[m_val], y[m_val], tokenizer, False)
    eval_loaders = {k: make_loader(seqs[m], y[m], tokenizer, False)
                    for k, m in eval_sets.items()}

    model = EsmClassifier().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    pos_weight = torch.tensor([(y[m_train] == 0).sum() / max((y[m_train] == 1).sum(), 1)],
                              device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        t0, tot_loss, nb = time.time(), 0.0, 0
        for ids, attn, yb in train_loader:
            logits = model(ids.to(DEVICE), attn.to(DEVICE))
            loss = loss_fn(logits, yb.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            nb += 1
        val_auc = eval_auc(model, val_loader)
        print(f"    ep{ep+1}: loss={tot_loss/nb:.4f} val_auc={val_auc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val_auc is not None and val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    res = {"best_val_auc": best_val, "pos_weight": float(pos_weight)}
    for k, loader in eval_loaders.items():
        res[k] = eval_auc(model, loader)
    print(f"    [{name}] best_val={best_val:.4f} test={res['test']:.4f} "
          f"h5={res['h5']:.4f} h7={res['h7']:.4f}", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return res


def main():
    split_df = pd.read_csv(SPLIT_CSV)
    splits = split_df["split"].values
    acc = split_df["accession"].values
    data_df = pd.read_csv(DATA_CSV).set_index("accession")
    # aligned CSV 含重复 accession（同序列不同亚型），先去重保证行数对齐
    data_df = data_df[~data_df.index.duplicated(keep="first")]
    seqs = data_df.loc[acc, "ha_sequence"].values
    assert len(seqs) == len(acc), f"行数不匹配: {len(seqs)} vs {len(acc)}"
    labels = {lc: np.load(OUT_DIR / f"labels_{lc}.npy") for lc in LABEL_COLS}
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)

    results = {}
    for name, train_splits, label_col in RUNS:
        print(f"\n{'='*60}\n  Run: {name} (train={'+'.join(train_splits)}, label={label_col})\n{'='*60}",
              flush=True)
        results[name] = run_one(name, train_splits, label_col,
                                seqs, labels, splits, tokenizer)

    out_path = OUT_DIR / "finetune_lora.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
