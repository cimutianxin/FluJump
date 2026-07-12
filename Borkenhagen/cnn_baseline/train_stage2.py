"""Stage 2 Jump 微调 — label_is_jump & label_is_jump_human
Borkenhagen 2024: class + subtype 双重加权, LR=1e-3, grad clip
"""

import torch, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import json, time

from config import *
from data_prep import get_df, prepare_stage2_data, create_dataloaders
from model import BorkenhagenCNN


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for x, y, w in loader:
        x, y, w = x.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y, weight=w, reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
    return avg_loss, auc, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in loader:
        if len(batch) == 3:
            x, y, w = batch; w = w.to(device)
        else:
            x, y = batch; w = None
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(-1)
        loss = (F.binary_cross_entropy_with_logits(logits, y, weight=w, reduction="mean")
                if w is not None else F.binary_cross_entropy_with_logits(logits, y))
        total_loss += loss.item() * x.size(0)
        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    preds_bin = (np.array(all_preds) >= 0.5).astype(int)
    return {
        "loss": avg_loss,
        "auc": roc_auc_score(all_labels, all_preds),
        "accuracy": accuracy_score(all_labels, preds_bin),
        "precision": precision_score(all_labels, preds_bin, zero_division=0),
        "recall": recall_score(all_labels, preds_bin, zero_division=0),
        "f1": f1_score(all_labels, preds_bin, zero_division=0),
    }


def run_stage2(label_col):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Stage 2: {label_col} (LR={S2_LR}, clip={CLIP_GRAD_NORM})")
    print(f"{'='*60}")

    df = get_df()
    train_ds, val_ds, test_ds = prepare_stage2_data(df, label_col)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=S2_BATCH_SIZE
    )

    model = BorkenhagenCNN().to(device)
    ckpt = torch.load(MODEL_DIR / "stage1_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print("已加载 Stage 1 最佳模型")

    model.freeze_conv()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {n_trainable:,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=S2_LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, S2_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_auc, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - t0
        print(f"E{epoch:3d} | tr_loss={train_loss:.4f} tr_auc={train_auc:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
              f"val_f1={val_metrics['f1']:.4f} | {elapsed:.1f}s")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / f"stage2_{label_col}_best.pt")
            print(f"  ★ best val_auc={best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= S2_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(
        MODEL_DIR / f"stage2_{label_col}_best.pt", map_location=device, weights_only=True
    ))
    test_metrics = evaluate(model, test_loader, device)

    print(f"\n=== {label_col} 测试 ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    return {"label": label_col, "best_val_auc": best_val_auc, "best_epoch": best_epoch,
            "test": {k: float(v) for k, v in test_metrics.items()}}


def main():
    all_results = {}
    for label_col in ["label_is_jump", "label_is_jump_human"]:
        all_results[label_col] = run_stage2(label_col)

    with open(OUTPUT_DIR / "stage2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}\n汇总\n{'='*60}")
    for label, r in all_results.items():
        t = r["test"]
        print(f"{label}: AUC={t['auc']:.4f} Acc={t['accuracy']:.4f} "
              f"Prec={t['precision']:.4f} Rec={t['recall']:.4f} F1={t['f1']:.4f}")


if __name__ == "__main__":
    main()
