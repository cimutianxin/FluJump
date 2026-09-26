"""ClassQueryPrototypeHead — JEV 启发的类别查询 + 原型打分分类头

结构（对应 jev_exp/算法实现考虑.md §1/§2，冻结 ESM-2 per-residue embedding 之上）：

    x (B, L, 640) → Linear(640→d) 投影
    2 个可学习类别查询 q_c 通过 cross-attention 并行读取残基表示 → z_c (B, 2, d)
    2 个可学习类别原型 e_c，分数 s_c = cos(z_c, e_c) / τ（τ 可学习，CLIP 式 log 参数化）
    logits = (s_0, s_1) → CrossEntropyLoss

AUC/排序一律用 logit 口径 score = s_1 − s_0（不用 softmax 概率，见 AGENTS.md 踩坑记录）。
容量档位与 attention pooling 臂相当：无 transformer encoder、无 FFN。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ESM_tf_clf.config import ESM_DIM
from jev_exp.config import (JEV_D_MODEL, JEV_NHEAD, JEV_DROPOUT, JEV_N_CLASS,
                            JEV_TAU_INIT, JEV_INV_TAU_MAX)


class ClassQueryPrototypeHead(nn.Module):
    def __init__(self, esm_dim=ESM_DIM, d_model=JEV_D_MODEL, nhead=JEV_NHEAD,
                 dropout=JEV_DROPOUT, n_class=JEV_N_CLASS, tau_init=JEV_TAU_INIT):
        super().__init__()
        self.n_class = n_class
        self.proj = nn.Linear(esm_dim, d_model)
        self.class_queries = nn.Parameter(torch.randn(1, n_class, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.prototypes = nn.Parameter(torch.randn(n_class, d_model) * 0.02)
        # CLIP 式 log 参数化：score = cos(·,·) * exp(log_inv_tau)
        self.log_inv_tau = nn.Parameter(
            torch.tensor(float(np.log(1.0 / tau_init))))

    def forward(self, x, lengths):
        """
        x: (B, L, esm_dim) per-residue embedding（pad 位置为 0）
        lengths: (B,) 各序列真实长度
        返回: (B, n_class) 打分 logits（softmax 前的 s_c）
        """
        B, L, _ = x.shape
        pad_mask = torch.arange(L, device=x.device)[None, :] >= lengths[:, None]
        h = self.proj(x)                                      # (B, L, d)
        q = self.class_queries.expand(B, -1, -1)              # (B, C, d)
        z, _ = self.attn(q, h, h, key_padding_mask=pad_mask)  # (B, C, d)
        z_n = F.normalize(z, dim=-1)
        e_n = F.normalize(self.prototypes, dim=-1)
        inv_tau = self.log_inv_tau.exp().clamp(max=JEV_INV_TAU_MAX)
        scores = torch.einsum("bcd,cd->bc", z_n, e_n) * inv_tau
        return scores
