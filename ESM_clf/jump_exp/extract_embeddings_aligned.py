"""ESM-2 150M 对齐序列嵌入 — 使用 MAFFT 对齐序列，排除 gap token

与 extract_embeddings.py 的区别：
1. 使用 aligned_ha_seq（含 '-' gap）而非 ha_sequence
2. mean pooling 时排除 gap token（id=30），避免未充分训练的 gap embedding 引入噪声
3. 输出文件名加 _aligned 后缀
"""

import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

GAP_TOKEN_ID = 30  # ESM-2 中 '-' 的 token id（已验证）

# ── 加载数据 ──
df = pd.read_csv(DATA_CSV)
print(f"总 isolates: {len(df)}")

# 防呆（2026-08-02）：同 extract_embeddings.py，强制校验行序与 split 逐行一致
_sp = pd.read_csv(SPLIT_CSV, dtype=str)
assert len(df) == len(_sp), f"DATA_CSV 行数 {len(df)} != split 行数 {len(_sp)}"
assert (df["accession"].values == _sp["accession"].values).all(), "DATA_CSV 行序与 split 不一致"
assert (df["subtype"].values == _sp["subtype"].values).all(), "DATA_CSV subtype 行序与 split 不一致"

# 使用对齐序列
seqs = df["aligned_ha_seq"].tolist()
print(f"对齐序列长度: {len(seqs[0])}")

# ── 加载 ESM-2 150M ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True).to(device).eval()

# ── 提取 L1/L3/L1L3，排除 gap ──
L1, L3, L13 = [], [], []
BATCH_SIZE = 4  # 对齐序列更长，保持小 batch

for i in tqdm(range(0, len(seqs), BATCH_SIZE), desc="提取 aligned embedding"):
    batch_seqs = seqs[i:i + BATCH_SIZE]
    inp = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inp)

    for j in range(len(batch_seqs)):
        # 排除 CLS/EOS + gap token
        ids_j = inp["input_ids"][j, 1:-1]
        mask_j = inp["attention_mask"][j, 1:-1].bool()
        valid = mask_j & (ids_j != GAP_TOKEN_ID)  # 排除 gap

        if valid.sum() == 0:
            # 理论上不会发生，但防御性编程
            h1 = out.hidden_states[-1][j, 1:-1][mask_j].mean(dim=0)
            h3 = out.hidden_states[-3][j, 1:-1][mask_j].mean(dim=0)
        else:
            h1 = out.hidden_states[-1][j, 1:-1][valid].mean(dim=0)
            h3 = out.hidden_states[-3][j, 1:-1][valid].mean(dim=0)

        L1.append(h1.cpu().numpy())
        L3.append(h3.cpu().numpy())
        L13.append(torch.cat([h1, h3]).cpu().numpy())

# ── 保存 ──
OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, arr in [("L1", L1), ("L3", L3), ("L1L3", L13)]:
    fname = OUT_DIR / f"esm_emb_150M_aligned_{name}.npy"
    np.save(fname, np.stack(arr))
    print(f"  {fname}: {np.stack(arr).shape}")

print("Done — aligned 嵌入提取完成")
