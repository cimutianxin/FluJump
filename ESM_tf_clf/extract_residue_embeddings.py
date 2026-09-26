"""提取 ESM-2 150M 候选层 {13,17,28} 的 per-residue embedding（冻结，不微调）

为 transformer 头服务：mean-pooled 向量没有序列结构，transformer 需要
(N, L, 640) 的残基级 hidden states。行序与 ESM_clf/jump_exp 现有
embedding/labels 严格一致（accessions.npy 锚定）。

输入：raw ha_sequence（主方案，无 gap）
输出（ESM_tf_clf/output/）：
  residue_emb_L{13,17,28}.npy  fp16 memmap (11060, 573, 640)，右侧 zero-pad
  residue_lens.npy             int32 (11060,)，各序列真实残基数
"""

import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from ESM_tf_clf.config import *


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 行序：与现有 embedding 严格一致（复刻 extract_all_layers.py）──
    acc_order = np.load(ACCESSIONS_NPY, allow_pickle=True)
    data_df = pd.read_csv(DATA_CSV).set_index("accession")
    data_df = data_df[~data_df.index.duplicated(keep="first")]  # 531 行重复 accession
    seqs = data_df.loc[acc_order, "ha_sequence"].tolist()
    assert len(seqs) == len(acc_order), f"行数不匹配: {len(seqs)} vs {len(acc_order)}"
    n = len(seqs)
    print(f"总 isolates: {n}")

    # ── memmap 预分配 ──
    maps = {li: np.lib.format.open_memmap(OUT_DIR / f"residue_emb_L{li}.npy",
                                          mode="w+", dtype=np.float16,
                                          shape=(n, MAX_LEN, ESM_DIM))
            for li in CANDIDATE_LAYERS}
    lens = np.zeros(n, dtype=np.int32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True).to(device).eval()

    BATCH = 8
    for i in tqdm(range(0, n, BATCH), desc="提取 per-residue embedding"):
        batch_seqs = seqs[i:i + BATCH]
        inp = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inp)
        for j, seq in enumerate(batch_seqs):
            L = len(seq)
            lens[i + j] = L
            for li in CANDIDATE_LAYERS:
                h = out.hidden_states[li][j, 1:1 + L]        # 去 CLS/EOS，(L, 640)
                maps[li][i + j, :L] = h.cpu().numpy().astype(np.float16)
                maps[li][i + j, L:] = 0

    for li in CANDIDATE_LAYERS:
        maps[li].flush()
        print(f"  residue_emb_L{li}.npy: {maps[li].shape}")
    np.save(OUT_DIR / "residue_lens.npy", lens)
    print(f"  residue_lens.npy: {lens.shape}, len 范围 [{lens.min()}, {lens.max()}]")

    # ── 抽查：per-residue mean ≈ 全层 mean-pooled（all_layers 同层）──
    pooled = np.load("ESM_clf/jump_exp/output/esm_emb_150M_all_layers.npy", mmap_mode="r")
    for li in CANDIDATE_LAYERS:
        a = maps[li][0, :lens[0]].astype(np.float32).mean(axis=0)
        b = pooled[li, 0]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(f"  抽查 L{li} 行0: mean-pool 与 all_layers 余弦相似度 = {cos:.4f}")
    print("Done")


if __name__ == "__main__":
    main()
