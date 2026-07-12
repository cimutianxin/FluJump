"""Ordinal Regression 模型（学一个连续 score + K-1 个阈值）。

方法：Cumulative Link Model (proportional odds)
  - 模型输出单个 score (batch, 1)
  - 学习 K-1 个单调递增阈值
  - P(y ≤ k) = σ(threshold_k - score)
  - P(y = k) = P(y ≤ k) - P(y ≤ k-1)
  - Loss: NLL

同时提供 Linear Ordinal baseline（基于 mord.LogisticAT）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy


class OrdinalMLP(nn.Module):
    """Ordinal Regression MLP。

    输出单个 score + K-1 个可学习阈值。

    Args:
        in_dim: 输入维度
        hidden_dims: 隐层列表
        num_classes: 类别数 K
        dropout: Dropout 概率
    """

    def __init__(self, in_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        # Feature extractor → single score
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))  # single score
        self.net = nn.Sequential(*layers)

        # K-1 thresholds, stored in increasing order via cumsum(softplus)
        self.num_classes = num_classes
        self.threshold_raw = nn.Parameter(torch.zeros(num_classes - 1))

    def get_thresholds(self):
        """返回单调递增的阈值向量 (K-1,)。"""
        return torch.cumsum(F.softplus(self.threshold_raw), dim=0)

    def forward(self, x):
        """返回 score (batch, 1)。"""
        return self.net(x)

    def predict_proba(self, x):
        """返回类别概率 (batch, K)。"""
        score = self.forward(x)  # (batch, 1)
        thresholds = self.get_thresholds()  # (K-1,)

        # P(y ≤ k) = sigmoid(threshold_k - score), k = 0..K-2
        # shape: (batch, K-1)
        cum_probs = torch.sigmoid(thresholds.unsqueeze(0) - score)

        # P(y=0) = cum_probs[:, 0]
        # P(y=k) = cum_probs[:, k] - cum_probs[:, k-1]  for 0 < k < K-1
        # P(y=K-1) = 1 - cum_probs[:, -1]
        probs = torch.cat([
            cum_probs[:, :1],                          # P(y=0)
            cum_probs[:, 1:] - cum_probs[:, :-1],      # P(y=1..K-2)
            1 - cum_probs[:, -1:],                     # P(y=K-1)
        ], dim=1)
        # Clamp for numerical stability
        return torch.clamp(probs, 1e-10, 1.0)


def ordinal_nll_loss(score, y_true, thresholds, num_classes):
    """计算 ordinal NLL loss。

    Args:
        score: (batch, 1) 模型输出的连续分数
        y_true: (batch,) long tensor 真实类别
        thresholds: (K-1,) 阈值向量
        num_classes: K
    Returns:
        scalar loss
    """
    cum_probs = torch.sigmoid(thresholds.unsqueeze(0) - score)  # (batch, K-1)
    probs = torch.cat([
        cum_probs[:, :1],
        cum_probs[:, 1:] - cum_probs[:, :-1],
        1 - cum_probs[:, -1:],
    ], dim=1)
    probs = torch.clamp(probs, 1e-10, 1.0)
    log_probs = torch.log(probs)
    return F.nll_loss(log_probs, y_true)


def _to_tensor(arr, dtype=None):
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr)
        return t.to(dtype) if dtype is not None else t
    if dtype is not None:
        arr = arr.to(dtype)
    return arr


def _make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(_to_tensor(X, dtype=torch.float32),
                       _to_tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_ordinal_mlp(X_train, y_train, X_val, y_val, in_dim, num_classes,
                       hidden_dims, dropout=0.3, lr=1e-3, weight_decay=1e-4,
                       epochs=200, batch_size=64, patience=20, seed=42,
                       device="cuda"):
    """训练 Ordinal MLP。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = OrdinalMLP(in_dim, hidden_dims, num_classes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    y_train_t = y_train.astype(np.int64)
    y_val_t = y_val.astype(np.int64)

    train_loader = _make_loader(X_train, y_train_t, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val_t, batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            score = model(bx)
            thresholds = model.get_thresholds()
            loss = ordinal_nll_loss(score, by, thresholds, num_classes)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                score = model(bx)
                thresholds = model.get_thresholds()
                loss = ordinal_nll_loss(score, by, thresholds, num_classes)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_state)
    return model, history, best_val_loss


def predict_ordinal_mlp(model, X, device="cuda"):
    """预测类别 (batch,) int64。"""
    model.eval()
    loader = _make_loader(X, np.zeros(len(X), dtype=np.int64),
                          batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            probs = model.predict_proba(bx)
            preds.append(probs.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)
