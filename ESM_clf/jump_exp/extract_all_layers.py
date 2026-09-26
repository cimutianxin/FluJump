"""提取 ESM-2 150M 全部 31 层 hidden states 的 mean-pooled embedding

为层扫描（layer sweep）服务：现有 L1(=last)/L3(=3rd-from-last) 不足以
定位"H5+H7 方向一致且 AUC 高"的层。一次前向 dump 全部层。

输入序列：旧版 11060 行顺序（与现有 embedding/labels 对齐）——
通过 accessions.npy + split CSV 确定行序，序列从 aligned CSV 按 accession 取
（序列本身不受亚型重判影响，但为保险起见按 accession 对齐）。

输出：output/esm_emb_150M_all_layers.npy，shape (31, 11060, 640)
"""

import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

OUT_PATH = OUT_DIR / "esm_emb_150M_all_layers.npy"


def main():
    # ── 行序：与现有 embedding 严格一致（accessions.npy）──
    # 注意：07-12 重建的 aligned CSV 含 531 行重复 accession（同序列不同亚型），
    # 需先去重，否则 .loc 会返回 11600 行错位
    acc_order = np.load(OUT_DIR / "accessions.npy", allow_pickle=True)
    data_df = pd.read_csv(DATA_CSV).set_index("accession")
    data_df = data_df[~data_df.index.duplicated(keep="first")]
    seqs = data_df.loc[acc_order, "ha_sequence"].tolist()
    assert len(seqs) == len(acc_order), f"行数不匹配: {len(seqs)} vs {len(acc_order)}"
    print(f"总 isolates: {len(seqs)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True).to(device).eval()
    n_layers = model.config.num_hidden_layers + 1  # 31（含 embedding 层 0）
    print(f"层数: {n_layers}")

    all_layers = [[] for _ in range(n_layers)]
    BATCH_SIZE = 4
    for i in tqdm(range(0, len(seqs), BATCH_SIZE), desc="提取全部层 embedding"):
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
    np.save(OUT_PATH, arr)
    print(f"  {OUT_PATH}: {arr.shape}")
    print("Done — 全层 embedding 提取完成")


if __name__ == "__main__":
    main()
