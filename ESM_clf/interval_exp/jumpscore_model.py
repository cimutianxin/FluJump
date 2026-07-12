"""JumpScorer: ESM embedding → 标量 JumpScore，与 iso_interval_days 做秩相关。

模型架构：
  ESM embedding (640/1280-dim) → MLP → JumpScore (scalar)

训练策略：
  1. Pairwise Ranking Loss (MarginRankingLoss)
     对 batch 内所有 pair (i,j) 其中 interval_i ≠ interval_j，
     要求 score 的排序与 interval 一致
  2. 可选 Combined Loss: ranking + MSE regression

评估指标：
  Spearman ρ (秩相关系数), Kendall τ, Pearson r
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau, pearsonr
import copy


class JumpScorer(nn.Module):
    """输入 ESM embedding，输出标量 JumpScore。

    分数越低 → interval 越小 → 跨物种能力越强（越危）。
    """

    def __init__(self, in_dim, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_all_pairs(scores, intervals):
    """从 batch 内构造所有 pair。

    label = +1 表示 score_i 应 > score_j（即 interval_i > interval_j）
    label = -1 表示 score_i 应 < score_j（即 interval_i < interval_j）
    """
    B = scores.shape[0]
    device = scores.device

    idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=device)
    int_i = intervals[idx_i]
    int_j = intervals[idx_j]
    valid = int_i != int_j
    idx_i, idx_j = idx_i[valid], idx_j[valid]
    int_i, int_j = int_i[valid], int_j[valid]

    if len(idx_i) == 0:
        return None, None, None

    s1 = scores[idx_i].squeeze(-1)
    s2 = scores[idx_j].squeeze(-1)
    targets = torch.where(
        int_i > int_j,
        torch.ones_like(int_i, dtype=torch.float32),
        -torch.ones_like(int_i, dtype=torch.float32),
    )
    return s1, s2, targets


def _to_tensor(arr, dtype=None):
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr)
        return t.to(dtype) if dtype is not None else t
    if dtype is not None:
        arr = arr.to(dtype)
    return arr


def _make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        _to_tensor(X, dtype=torch.float32),
        _to_tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_jumpscore(
    X_train, y_train, X_val, y_val, in_dim,
    hidden_dims=(256, 128), dropout=0.3,
    lr=1e-3, weight_decay=1e-4, margin=1.0,
    epochs=200, batch_size=64, patience=20,
    use_mse_weight=0.0, seed=42, device="cuda",
):
    """训练 JumpScorer。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = JumpScorer(in_dim, hidden_dims, dropout).to(device)
    rank_loss_fn = nn.MarginRankingLoss(margin=margin)
    mse_loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    y_train_t = y_train.astype(np.float32)
    y_val_t = y_val.astype(np.float32)

    train_loader = _make_loader(X_train, y_train_t, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val_t, batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_spearman": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            scores = model(bx)

            s1, s2, targets = make_all_pairs(scores, by)
            if s1 is None:
                continue

            rank_loss = rank_loss_fn(s1, s2, targets)
            loss = rank_loss

            if use_mse_weight > 0:
                mse_loss = mse_loss_fn(scores.squeeze(-1), by)
                loss = rank_loss + use_mse_weight * mse_loss

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_val_scores, all_val_intervals = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                scores = model(bx)

                s1, s2, targets = make_all_pairs(scores, by)
                if s1 is not None:
                    val_losses.append(rank_loss_fn(s1, s2, targets).item())

                all_val_scores.append(scores.squeeze(-1).cpu().numpy())
                all_val_intervals.append(by.cpu().numpy())

        avg_train = np.mean(train_losses) if train_losses else 0.0
        avg_val = np.mean(val_losses) if val_losses else 0.0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        val_sp, _ = spearmanr(
            np.concatenate(all_val_scores), np.concatenate(all_val_intervals)
        )
        history["val_spearman"].append(float(val_sp))

        scheduler.step(avg_val if val_losses else float("inf"))

        if avg_val < best_val_loss and val_losses:
            best_val_loss = avg_val
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if (epoch + 1) % 20 == 0:
            print(
                f"    epoch {epoch+1:3d}  "
                f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  "
                f"val_ρ={val_sp:.4f}"
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, history, best_val_loss


def evaluate_jumpscore(model, X, y, device="cuda"):
    """评估 JumpScorer：Spearman ρ, Kendall τ, Pearson r。"""
    model.eval()
    loader = _make_loader(X, y.astype(np.float32), batch_size=256, shuffle=False)
    all_scores, all_intervals = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            scores = model(bx).squeeze(-1).cpu().numpy()
            all_scores.append(scores)
            all_intervals.append(by.numpy())

    pred = np.concatenate(all_scores)
    true = np.concatenate(all_intervals)

    sp_r, sp_p = spearmanr(pred, true)
    kt_r, kt_p = kendalltau(pred, true)
    pr_r, pr_p = pearsonr(pred, true)

    return {
        "spearman_r": float(sp_r),
        "spearman_p": float(sp_p),
        "kendall_tau": float(kt_r),
        "kendall_p": float(kt_p),
        "pearson_r": float(pr_r),
        "pearson_p": float(pr_p),
        "n": len(true),
        "score_mean": float(pred.mean()),
        "score_std": float(pred.std()),
        "interval_mean": float(true.mean()),
        "interval_std": float(true.std()),
    }
