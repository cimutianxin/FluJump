"""ESM-2 150M 嵌入提取 — EsmModel + L1/L3/L1L3 mean pooling

对 all_isolates_aligned.csv 中全部 isolate 提取 embedding。
使用原始 ha_sequence（无 gap），与 ESM-2 的预训练分布一致。
"""

import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

# ── 加载数据 ──
df = pd.read_csv(DATA_CSV)
print(f"总 isolates: {len(df)}")
print(f"  subtype 分布: {df['subtype'].value_counts().to_dict()}")

# 防呆（2026-08-02）：embedding 按 DATA_CSV 行序存储，下游全部按行对齐。
# 若 DATA_CSV 被重建导致行数/行序变化，直接重跑会生成与 split/labels 错位的 npy。
# 此处强制校验行序与 isolate_split.csv 逐行一致。
_sp = pd.read_csv(SPLIT_CSV, dtype=str)
assert len(df) == len(_sp), f"DATA_CSV 行数 {len(df)} != split 行数 {len(_sp)}"
assert (df["accession"].values == _sp["accession"].values).all(), "DATA_CSV 行序与 split 不一致"
assert (df["subtype"].values == _sp["subtype"].values).all(), "DATA_CSV subtype 行序与 split 不一致"

seqs = df["ha_sequence"].tolist()

# ── 加载 ESM-2 150M (EsmModel, 无 LM head) ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True).to(device).eval()

# ── 提取 L1 (last), L3 (3rd-from-last), L1+L3 concat ──
L1, L3, L13 = [], [], []
BATCH_SIZE = 4

for i in tqdm(range(0, len(seqs), BATCH_SIZE), desc="提取 embedding"):
    batch_seqs = seqs[i:i + BATCH_SIZE]
    inp = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inp)

    for j in range(len(batch_seqs)):
        mask = inp["attention_mask"][j, 1:-1].bool()  # 去掉 CLS / EOS
        h1 = out.hidden_states[-1][j, 1:-1][mask].mean(dim=0)   # L1
        h3 = out.hidden_states[-3][j, 1:-1][mask].mean(dim=0)   # L3
        L1.append(h1.cpu().numpy())
        L3.append(h3.cpu().numpy())
        L13.append(torch.cat([h1, h3]).cpu().numpy())

# ── 保存 ──
OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, arr in [("L1", L1), ("L3", L3), ("L1L3", L13)]:
    fname = OUT_DIR / f"esm_emb_150M_{name}.npy"
    np.save(fname, np.stack(arr))
    print(f"  {fname}: {np.stack(arr).shape}")

# ── 保存标签和 accession ──
for lc in LABEL_COLS:
    vals = df[lc].fillna(0).astype(int).values
    np.save(OUT_DIR / f"labels_{lc}.npy", vals)
    print(f"  labels_{lc}: {vals.shape}, pos={vals.sum()} ({vals.sum()/len(vals)*100:.1f}%)")

np.save(OUT_DIR / "accessions.npy", df["accession"].values)
print(f"  accessions: {len(df)}")

print("Done — 嵌入提取完成")
