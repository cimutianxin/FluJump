"""402 条 Borkenhagen binding 序列的全 31 层 mean-pooled embedding（150M）

序列处理与 ESM_clf/binding_exp 一致（去 gap 后喂 raw）。
行序 == borkenhagen_clean.csv / borkenhagen_split.csv 行序。

输出：output/binding_emb_all_layers.npy (31, 402, 640) + binding_labels.npy
"""

import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from validation_exp.external_binding.config import *


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BORK_CSV)
    seqs = df["ha_sequence"].str.replace("-", "", regex=False).tolist()
    print(f"binding 序列: {len(seqs)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True).to(device).eval()
    n_layers = model.config.num_hidden_layers + 1
    print(f"层数: {n_layers}")

    all_layers = [[] for _ in range(n_layers)]
    for i in tqdm(range(0, len(seqs), 8), desc="提取 binding 全层 embedding"):
        batch = seqs[i:i + 8]
        inp = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inp)
        for j in range(len(batch)):
            mask = inp["attention_mask"][j, 1:-1].bool()
            for li in range(n_layers):
                h = out.hidden_states[li][j, 1:-1][mask].mean(dim=0)
                all_layers[li].append(h.cpu().numpy())

    arr = np.stack([np.stack(l) for l in all_layers])
    np.save(OUT_DIR / "binding_emb_all_layers.npy", arr)
    np.save(OUT_DIR / "binding_labels.npy", df["binding_label"].values)
    print(f"  binding_emb_all_layers.npy: {arr.shape}")
    print(f"  标签分布: {df['binding_label'].value_counts().to_dict()}")
    print("Done")


if __name__ == "__main__":
    main()
