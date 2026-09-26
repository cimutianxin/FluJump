#!/usr/bin/env python3
"""提取 H10/H4 isolates 的 ESM-2 150M 全 31 层 mean-pooled embedding

逻辑照 ESM_clf/jump_exp/extract_all_layers.py：raw 序列（主方案），
mean pooling 排除 CLS/EOS。行序 = data/processed/{ST}_isolates.csv
（先 H10 后 H4 拼接），输出 (31, N, 640) 与行序索引 CSV。

运行：HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env1 python
"""

import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from validation_exp.group_boundary.config import (
    SUBTYPES, PROC_DIR, OUT_DIR, ESM_MODEL,
)

BATCH_SIZE = 8


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(PROC_DIR / f"{st}_isolates.csv", dtype=str)
                    for st in SUBTYPES], ignore_index=True)
    seqs = df["ha_sequence"].tolist()
    print(f"总 isolates: {len(seqs)}（{'+'.join(SUBTYPES)}）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True).to(device).eval()
    n_layers = model.config.num_hidden_layers + 1  # 31（含 embedding 层 0）
    print(f"层数: {n_layers}")

    all_layers = [[] for _ in range(n_layers)]
    for i in tqdm(range(0, len(seqs), BATCH_SIZE), desc="提取全层 embedding"):
        batch_seqs = seqs[i:i + BATCH_SIZE]
        inp = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inp)
        for j in range(len(batch_seqs)):
            mask = inp["attention_mask"][j, 1:-1].bool()  # 去 CLS/EOS
            for li in range(n_layers):
                h = out.hidden_states[li][j, 1:-1][mask].mean(dim=0)
                all_layers[li].append(h.cpu().numpy())

    arr = np.stack([np.stack(layer) for layer in all_layers])  # (31, N, 640)
    np.save(OUT_DIR / "h10_h4_emb_all_layers.npy", arr)
    df[["accession", "subtype"]].to_csv(OUT_DIR / "emb_row_index.csv", index=False)
    print(f"  → {OUT_DIR}/h10_h4_emb_all_layers.npy: {arr.shape}")
    print(f"  → {OUT_DIR}/emb_row_index.csv: {len(df)} 行")


if __name__ == "__main__":
    main()
