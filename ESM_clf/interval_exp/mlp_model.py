"""MLP probe 模型 + 训练/评估工具函数。

支持：
- 分类（CrossEntropyLoss）和回归（MSELoss）
- 可配置 hidden dims、dropout、early stopping
- 多 seed 平均评估
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from scipy.stats import pearsonr
import copy


class IntervalMLP(nn.Module):
    """可配置隐层的 MLP probe。

    Args:
        in_dim: 输入维度（embedding 维度）
        hidden_dims: 隐层大小列表，如 [256, 128] 表示两层
        out_dim: 输出维度（分类=类别数，回归=1）
        dropout: Dropout 概率
    """

    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _to_tensor(arr, dtype=None):
    """Convert numpy array to torch tensor. Keeps original dtype if dtype=None."""
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr)
        return t.to(dtype) if dtype is not None else t
    if dtype is not None:
        arr = arr.to(dtype)
    return arr


def _make_loader(X, y, batch_size, shuffle=True, y_dtype=None):
    ds = TensorDataset(_to_tensor(X, dtype=torch.float32),
                       _to_tensor(y, dtype=y_dtype))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_mlp(X_train, y_train, X_val, y_val, in_dim, out_dim,
              hidden_dims, dropout=0.3, lr=1e-3, weight_decay=1e-4,
              epochs=200, batch_size=64, patience=20, seed=42,
              task="clf", device="cuda"):
    """训练 MLP，返回最佳模型和训练历史。

    Args:
        task: "clf"（分类，CrossEntropyLoss）或 "reg"（回归，MSELoss）
    Returns:
        best_model, history_dict
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = IntervalMLP(in_dim, hidden_dims, out_dim, dropout).to(device)

    if task == "clf":
        criterion = nn.CrossEntropyLoss()
        y_train_t = y_train.astype(np.int64)
        y_val_t = y_val.astype(np.int64)
        y_dtype = torch.long
    else:
        criterion = nn.MSELoss()
        y_train_t = y_train.astype(np.float32).reshape(-1, 1)
        y_val_t = y_val.astype(np.float32).reshape(-1, 1)
        y_dtype = torch.float32

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    train_loader = _make_loader(X_train, y_train_t, batch_size, shuffle=True, y_dtype=y_dtype)
    val_loader = _make_loader(X_val, y_val_t, batch_size, shuffle=False, y_dtype=y_dtype)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = criterion(pred, by)
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
    return model, history


def predict_mlp(model, X, task="clf", device="cuda"):
    """用训练好的 MLP 做预测。

    Returns:
        np.ndarray: 分类返回 (N,) int，回归返回 (N,) float
    """
    model.eval()
    loader = _make_loader(X, np.zeros(len(X)), batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            out = model(bx)
            if task == "clf":
                preds.append(out.argmax(dim=1).cpu().numpy())
            else:
                preds.append(out.squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


def eval_clf_metrics(y_true, y_pred, labels):
    """计算分类指标。"""
    result = {
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "n": len(y_true),
    }
    for i, label in enumerate(labels):
        if (y_true == i).sum() > 0:
            result[f"{label}_prec"] = precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
            result[f"{label}_rec"] = recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
    return result


def eval_reg_metrics(y_true, y_pred):
    """计算回归指标。"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    r, p = pearsonr(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "n": len(y_true),
        "y_mean": float(y_true.mean()),
        "y_std": float(y_true.std()),
        "pred_mean": float(y_pred.mean()),
        "pred_std": float(y_pred.std()),
    }
