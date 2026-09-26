"""ResidueTransformerClassifier — per-residue embedding 上的 Transformer 分类头

结构：Linear(640→d) + 可学习位置编码 → NUM_LAYERS × TransformerEncoderLayer
      → attention pooling（可学习 query，masked softmax）→ FFN → logit
      （out_dim 参数支持多分类，默认 1 = 二分类 logit）

AttentionPoolClassifier — 最小容量档：ESM 与 linear 之间仅换 adaptive pooling
      （可学习 query 的 masked softmax 加权求和，无 transformer 块/FFN）
"""

import numpy as np
import torch
import torch.nn as nn

from ESM_tf_clf.config import (D_MODEL, NHEAD, NUM_LAYERS, DIM_FF, DROPOUT,
                               ESM_DIM, MAX_LEN)


class ResidueTransformerClassifier(nn.Module):
    def __init__(self, esm_dim=ESM_DIM, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT,
                 max_len=MAX_LEN, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(esm_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x, lengths):
        """
        x: (B, L, esm_dim) per-residue embedding（pad 位置为 0）
        lengths: (B,) 各序列真实长度
        返回: out_dim=1 时 (B,)；否则 (B, out_dim) logits
        """
        B, L, _ = x.shape
        pad_mask = torch.arange(L, device=x.device)[None, :] >= lengths[:, None]
        h = self.proj(x) + self.pos_emb[:, :L]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, h, h, key_padding_mask=pad_mask)
        out = self.ffn(pooled.squeeze(1))                # (B, out_dim)
        return out.squeeze(-1) if self.out_dim == 1 else out


class AttentionPoolClassifier(nn.Module):
    """per-residue embedding → attention pooling（可学习 query）→ 单层 linear。

    最小容量档：相对 mean-pool + Ridge LR 只多一个可学习 query（640 维），
    无 transformer 块 / FFN，用于隔离"自适应池化"本身的贡献。
    """

    def __init__(self, esm_dim=ESM_DIM, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.query = nn.Parameter(torch.randn(esm_dim) * 0.02)
        self.head = nn.Linear(esm_dim, out_dim)

    def forward(self, x, lengths):
        """
        x: (B, L, esm_dim) per-residue embedding（pad 位置为 0）
        lengths: (B,) 各序列真实长度
        返回: out_dim=1 时 (B,)；否则 (B, out_dim) logits
        """
        B, L, D = x.shape
        pad_mask = torch.arange(L, device=x.device)[None, :] >= lengths[:, None]
        scores = x @ self.query / (D ** 0.5)                 # (B, L)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        w = torch.softmax(scores, dim=-1)                    # (B, L)
        pooled = (w.unsqueeze(-1) * x).sum(dim=1)            # (B, D)
        out = self.head(pooled)                              # (B, out_dim)
        return out.squeeze(-1) if self.out_dim == 1 else out


def pad_collate(xs, lens):
    """变长序列右侧 zero-pad 成 batch 张量"""
    B = len(xs)
    L = max(int(l) for l in lens)
    out = np.zeros((B, L, xs[0].shape[-1]), dtype=np.float32)
    for i, (x, l) in enumerate(zip(xs, lens)):
        out[i, :l] = x[:l]
    return torch.from_numpy(out), torch.tensor([int(l) for l in lens])
