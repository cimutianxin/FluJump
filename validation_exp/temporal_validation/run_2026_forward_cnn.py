#!/usr/bin/env python3
"""2026 前向外推——CNN baseline（Borkenhagen 2024）对比

用已训练的 stage2 checkpoint（label_is_jump / label_is_jump_human）给 201 条
2026 分离株打分，与 probe / 朴素基线同协议对比（AUC + 阳性名次 + cluster 级 AUC）。
CNN 输入为 MAFFT 对齐序列 one-hot（1039×21），与训练时一致。

先用 H5 holdout 复现（AUC ~0.54 区间断言）确认 checkpoint 加载正确。

运行：/root/miniconda3/envs/borkenhagen/bin/python validation_exp/temporal_validation/run_2026_forward_cnn.py
输出：合并写入 output/forward_2026.json 的 "cnn" 节 + 排名 CSV 追加 cnn 列
"""

import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
sys.path.insert(0, "Borkenhagen/cnn_baseline")

from validation_exp.temporal_validation.config import (
    CLEAN_2026_CSV, LABEL_COLS, OUT_DIR,
)
from config import MODEL_DIR, DEVICE, ALIGNED_LENGTH          # cnn_baseline/config.py
from data_prep import HADataset, get_df, prepare_holdout_data  # cnn_baseline/data_prep.py
from model import BorkenhagenCNN

ALIGNED_2026_CSV = "data/processed_isolate_MAFFT/2026_isolates_aligned.csv"


@torch.no_grad()
def predict_logits(model, ds, device, bs=256):
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    model.eval()
    out = []
    for batch in loader:
        x = batch[0].to(device)
        out.append(model(x).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # ── 2026 数据（对齐序列，1039 等长）──
    c26 = pd.read_csv(CLEAN_2026_CSV)
    a26 = pd.read_csv(ALIGNED_2026_CSV, usecols=["accession", "subtype", "aligned_ha_seq"])
    c26 = c26.merge(a26, on=["accession", "subtype"], validate="one_to_one")
    assert len(c26) == 201
    assert (c26["aligned_ha_seq"].str.len() == ALIGNED_LENGTH).all()
    ds26 = HADataset(c26["aligned_ha_seq"].tolist(),
                     np.zeros(len(c26), dtype=np.float32))

    # ── H5 holdout（复现断言用）──
    df = get_df()

    with open(OUT_DIR / "forward_2026.json") as f:
        results = json.load(f)
    results["cnn"] = {}

    for label in LABEL_COLS:
        ckpt = MODEL_DIR / f"stage2_{label}_best.pt"
        model = BorkenhagenCNN().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        # 复现断言：H5 holdout AUC 应在 ~0.54 附近（与 evaluate.py 一致）
        h5_ds = prepare_holdout_data(df, label, "h5_holdout")
        h5_logits = predict_logits(model, h5_ds, device)
        h5_auc = roc_auc_score(h5_ds.labels.numpy(), h5_logits)
        assert 0.35 < h5_auc < 0.80, f"CNN 复现异常: H5 AUC={h5_auc}"
        print(f"[{label}] CNN H5 复现 AUC={h5_auc:.4f}")

        # ── 2026 打分 ──
        logits = predict_logits(model, ds26, device)
        y26 = c26[label].to_numpy()
        auc = roc_auc_score(y26, logits)

        c26["_cnn"] = logits
        cl = c26.groupby("cluster_id").agg(
            score=("_cnn", "mean"), label=(label, "max"))
        cl_auc = roc_auc_score(cl["label"], cl["score"])

        order = np.argsort(-logits)
        ranks = np.empty(len(c26), dtype=int)
        ranks[order] = np.arange(1, len(c26) + 1)
        pos_ranks = sorted(int(ranks[i]) for i in np.where(y26 == 1)[0])

        results["cnn"][label] = {
            "h5_repro_auc": round(float(h5_auc), 4),
            "auc_2026": round(float(auc), 4),
            "auc_2026_cluster": round(float(cl_auc), 4),
            "positive_ranks_of_201": pos_ranks,
        }
        print(f"  2026 AUC: CNN={auc:.4f} (cluster级 {cl_auc:.4f})；阳性排名: {pos_ranks}")

        if label == "label_is_jump_human":
            c26["_cnn_rank"] = ranks
            rank26 = c26[["accession", "subtype", "_cnn", "_cnn_rank"]]

    # ── 合并写回 JSON + 排名 CSV ──
    with open(OUT_DIR / "forward_2026.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = OUT_DIR / "forward_2026_ranking.csv"
    rcsv = pd.read_csv(csv_path)
    rcsv = rcsv.drop(columns=[c for c in ["_cnn", "_cnn_rank"] if c in rcsv.columns])
    rcsv = rcsv.merge(rank26, on=["accession", "subtype"], validate="one_to_one")
    rcsv = rcsv.sort_values("_rank")
    rcsv.to_csv(csv_path, index=False)

    print(f"\n→ {OUT_DIR / 'forward_2026.json'} / forward_2026_ranking.csv（已追加 CNN 列）")


if __name__ == "__main__":
    main()
