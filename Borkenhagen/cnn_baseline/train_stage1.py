"""Stage 1 宿主预训练 — avian vs human+swine"""

import torch, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import json, time

from config import *
from data_prep import get_df, prepare_stage1_data, create_dataloaders
from model import build_model


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, preds, labels = 0, [], []
    for x, y, w in loader:
        x, y, w = x.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y, weight=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
    loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, (np.array(preds) >= 0.5).astype(int))
    return loss, auc, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, preds, labels = 0, [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        w = batch[2].to(device) if len(batch) == 3 else None
        logits = model(x).squeeze(-1)
        loss = (F.binary_cross_entropy_with_logits(logits, y, weight=w)
                if w is not None else F.binary_cross_entropy_with_logits(logits, y))
        total_loss += loss.item() * x.size(0)
        preds.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(y.cpu().numpy())
    loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, (np.array(preds) >= 0.5).astype(int))
    return loss, auc, acc


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    df = get_df()
    train_ds, val_ds = prepare_stage1_data(df)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=S1_BATCH_SIZE)

    model = build_model(device)
    opt = optim.Adam(model.parameters(), lr=S1_LR)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best_auc, best_ep, patience = 0, 0, 0
    for epoch in range(1, S1_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_auc, tr_acc = train_epoch(model, train_loader, opt, device)
        val_loss, val_auc, val_acc = evaluate(model, val_loader, device)
        sched.step(val_loss)

        print(f"E{epoch:3d} | tr_loss={tr_loss:.4f} tr_auc={tr_auc:.4f} tr_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f} | "
              f"lr={opt.param_groups[0]['lr']:.6f} | {time.time()-t0:.1f}s")

        if val_auc > best_auc:
            best_auc, best_ep, patience = val_auc, epoch, 0
            torch.save(model.state_dict(), MODEL_DIR / "stage1_best.pt")
            print(f"  ★ best val_auc={best_auc:.4f}")
        else:
            patience += 1
            if patience >= S1_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}"); break

    json.dump({"best_val_auc": best_auc, "best_epoch": best_ep},
              open(OUTPUT_DIR / "stage1_results.json", "w"), indent=2)
    print(f"\nStage 1 完成 | best val AUC={best_auc:.4f} (epoch {best_ep})")


if __name__ == "__main__":
    main()
