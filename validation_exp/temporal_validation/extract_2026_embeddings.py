#!/usr/bin/env python3
"""提取 2026 年 201 条分离株的 ESM-2 L28(=L3) mean-pooled embedding

行序 = data/processed_isolate/2026_isolates_clean.csv 行序。
输出：output/emb_2026_L3.npy (201, 640) + accessions 校验打印。
运行须离线模式：HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1。
"""

import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from validation_exp.temporal_validation.config import (
    CLEAN_2026_CSV, ESM_MODEL, OUT_DIR,
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_2026_CSV)
    seqs = df["ha_sequence"].tolist()
    print(f"2026 序列: {len(seqs)}，前3个 accession: {df['accession'].head(3).tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True).to(device).eval()

    embs = []
    for i in tqdm(range(0, len(seqs), 8), desc="提取 2026 L3 embedding"):
        batch = seqs[i:i + 8]
        inp = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inp)
        h = out.hidden_states[-3]               # L28 = probe 主线 L3
        for j in range(len(batch)):
            mask = inp["attention_mask"][j, 1:-1].bool()
            embs.append(h[j, 1:-1][mask].mean(dim=0).cpu().numpy())

    arr = np.stack(embs).astype(np.float32)
    np.save(OUT_DIR / "emb_2026_L3.npy", arr)
    print(f"emb_2026_L3.npy: {arr.shape}（行序 == 2026_isolates_clean.csv）")


if __name__ == "__main__":
    main()
