"""评估 — test + per-subtype + H5/H7 holdout"""

import torch, numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                              precision_score, recall_score, f1_score)
from torch.utils.data import DataLoader

from config import *
from data_prep import get_df, HADataset
from model import BorkenhagenCNN


@torch.no_grad()
def predict_ds(model, ds, device, bs=256):
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    model.eval()
    preds, labels = [], []
    for batch in loader:
        x = batch[0].to(device)
        logits = model(x).squeeze(-1)
        # AUC/排序必须用 raw logits：float32 sigmoid 在 |logit|>88 饱和，
        # 并列秩会把 AUC 拉向 0.5（jump_human H5 曾被低估 0.549 vs 真值 0.702）
        preds.extend(logits.cpu().numpy())
        labels.extend(batch[1].numpy())
    return np.array(preds), np.array(labels)


def metrics(preds, labels):
    b = (preds >= 0.0).astype(int)  # logit>=0 ⇔ p>=0.5
    return {
        "auc": roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else None,
        "acc": accuracy_score(labels, b),
        "prec": precision_score(labels, b, zero_division=0),
        "rec": recall_score(labels, b, zero_division=0),
        "f1": f1_score(labels, b, zero_division=0),
    }


def evaluate_model(model, df, label_col, device):
    """test (H1+H3) + per-subtype + H5/H7 holdout"""
    test_df = df[df["split"] == "test"].copy()
    test_df["label"] = test_df[label_col].astype(int)
    test_ds = HADataset(test_df["aligned_ha_seq"].tolist(), test_df["label"].values)
    preds, labels = predict_ds(model, test_ds, device)

    results = {"test": metrics(preds, labels)}
    for st in ["H1", "H3"]:
        m = test_df["subtype"] == st
        if m.sum() >= 2 and len(np.unique(labels[m])) > 1:
            results[f"test_{st}"] = metrics(preds[m], labels[m])

    for ho in ["h5_holdout", "h7_holdout"]:
        ho_df = df[df["split"] == ho].copy()
        ho_df["label"] = ho_df[label_col].astype(int)
        hds = HADataset(ho_df["aligned_ha_seq"].tolist(), ho_df["label"].values)
        hp, hl = predict_ds(model, hds, device)
        if len(np.unique(hl)) > 1:
            results[ho] = metrics(hp, hl)

        for st in ["H5", "H7"]:
            m = ho_df["subtype"] == st
            if m.sum() >= 2 and len(np.unique(hl[m])) > 1:
                results[f"{ho}_{st}"] = metrics(hp[m], hl[m])

    return results


def plot_rocs(all_curves, path):
    fig, axes = plt.subplots(1, len(all_curves), figsize=(7*len(all_curves), 6))
    if len(all_curves) == 1:
        axes = [axes]
    for ax, (title, curves) in zip(axes, all_curves.items()):
        for name, (fpr, tpr, auc_val) in curves.items():
            ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.2)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(title); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    df = get_df()
    all_results = {}

    for label_col in ["label_is_jump", "label_is_jump_human"]:
        ckpt = MODEL_DIR / f"stage2_{label_col}_best.pt"
        if not ckpt.exists():
            print(f"跳过 {label_col}: 模型不存在"); continue

        print(f"\n{'='*55}\n评估: {label_col}\n{'='*55}")
        model = BorkenhagenCNN().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        results = evaluate_model(model, df, label_col, device)
        all_results[label_col] = results

        print(f"\n{'Split':<20} {'AUC':>8} {'Acc':>8} {'F1':>8}  N")
        print("-" * 55)
        for k, v in results.items():
            auc_s = f"{v['auc']:.4f}" if v.get("auc") else "N/A"
            n_tag = k.split("_")[0]
            n = len(df[df["split"].isin([n_tag]) & df["subtype"].isin(["H1","H3","H5","H7"])])
            print(f"{k:<20} {auc_s:>8} {v.get('acc',0):>8.4f} {v.get('f1',0):>8.4f}  {n}")

        # ROC curves
        curves = {}
        test_df = df[df["split"]=="test"].copy()
        test_df["label"] = test_df[label_col].astype(int)
        test_ds = HADataset(test_df["aligned_ha_seq"].tolist(), test_df["label"].values)
        tp, tl = predict_ds(model, test_ds, device)
        if len(np.unique(tl)) > 1:
            fpr, tpr, _ = roc_curve(tl, tp)
            curves["test"] = (fpr, tpr, results["test"]["auc"])

        for ho in ["h5_holdout", "h7_holdout"]:
            ho_df = df[df["split"]==ho].copy()
            ho_df["label"] = ho_df[label_col].astype(int)
            hds = HADataset(ho_df["aligned_ha_seq"].tolist(), ho_df["label"].values)
            hp, hl = predict_ds(model, hds, device)
            if len(np.unique(hl)) > 1 and ho in results and results[ho].get("auc"):
                fpr, tpr, _ = roc_curve(hl, hp)
                curves[ho] = (fpr, tpr, results[ho]["auc"])

        if curves:
            plot_rocs({label_col: curves}, OUTPUT_DIR / f"roc_{label_col}.png")

    json.dump(all_results, open(OUTPUT_DIR / "eval_results.json", "w"), indent=2, default=str)
    print(f"\n✓ {OUTPUT_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()
