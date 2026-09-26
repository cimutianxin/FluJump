"""提取 ESM-2 650M / 3B 全层 mean-pooled embedding（11,060 条 raw 序列）

行序与现有 150M embedding 严格一致（accessions.npy 锚定，复刻
ESM_clf/jump_exp/extract_all_layers.py 的行序逻辑）。fp16 存储。

输出（每规模）：
  output/esm_emb_{size}_all_layers.npy  fp16 (n_layers+1, 11060, dim)

下载：huggingface.co 本机不可达，需 HF_ENDPOINT=https://hf-mirror.com
"""

import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from validation_exp.scale_replication.config import *


def extract_one(size):
    model_name = MODELS[size]
    out_path = OUT_DIR / f"esm_emb_{size}_all_layers.npy"
    if out_path.exists():
        print(f"[{size}] 已存在，跳过: {out_path}")
        return

    acc_order = np.load(ACCESSIONS_NPY, allow_pickle=True)
    data_df = pd.read_csv(DATA_CSV).set_index("accession")
    data_df = data_df[~data_df.index.duplicated(keep="first")]
    seqs = data_df.loc[acc_order, "ha_sequence"].tolist()
    n = len(seqs)
    print(f"[{size}] {model_name}, {n} isolates")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
    n_layers = model.config.num_hidden_layers + 1
    dim = model.config.hidden_size
    print(f"[{size}] 层数: {n_layers}, dim: {dim}")

    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                    shape=(n_layers, n, dim))
    bs = EXTRACT_BATCH[size]
    for i in tqdm(range(0, n, bs), desc=f"[{size}] 提取全层 embedding"):
        batch = seqs[i:i + bs]
        inp = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inp)
        for j in range(len(batch)):
            mask = inp["attention_mask"][j, 1:-1].bool()
            for li in range(n_layers):
                h = out.hidden_states[li][j, 1:-1][mask].mean(dim=0)
                arr[li, i + j] = h.cpu().numpy().astype(np.float16)
    arr.flush()
    print(f"[{size}] 已保存: {out_path} {arr.shape}")

    # 抽查：与 fp32 重算首序列倒数第二层的 mean-pool 余弦
    inp = tokenizer([seqs[0]], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    ref = out.hidden_states[-1][0, 1:-1].mean(dim=0).cpu().numpy()
    got = arr[-1, 0].astype(np.float32)
    cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
    print(f"[{size}] 抽查首序列末层余弦 = {cos:.4f}")

    del model
    torch.cuda.empty_cache()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for size in MODELS:
        extract_one(size)
    print("Done")


if __name__ == "__main__":
    main()
