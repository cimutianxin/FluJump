"""提取 CNN 中间层表示 — §1 选择预算对齐实验

给 CNN 与 ESM 同等的表示选择预算：对冻结 checkpoint 的各中间层表示做
probing + 同一 target-val 选择协议。本脚本前向全量 11060 条对齐序列
（行序 = isolate_split.csv），提取候选表示：

  conv1–conv5：各 conv block（conv+ReLU+pool+BN）输出，length 维 mean-pool
               （stage2 freeze_conv → 三 checkpoint 共享，仅从 stage1 提取一次）
  dense_*：    classifier 的 Linear→ReLU 后 128 维隐藏（每 checkpoint 各一）
  logit_*：    最终 logit（每 checkpoint 各一；stage2 两标签的 logit 即现有
               baseline 打分，用于行序/数值回归校验）

从项目根目录运行：/root/miniconda3/envs/borkenhagen/bin/python \
    Borkenhagen/cnn_baseline/extract_intermediate_features.py
输出：output/interm_feats/*.npy（行序 = isolate_split.csv）
"""

import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, "Borkenhagen/cnn_baseline")
from config import *           # noqa: E402
from data_prep import HADataset, get_df  # noqa: E402
from model import BorkenhagenCNN  # noqa: E402

FEAT_DIR = OUTPUT_DIR / "interm_feats"

CHECKPOINTS = {
    "stage1": MODEL_DIR / "stage1_best.pt",
    "stage2_jump": MODEL_DIR / "stage2_label_is_jump_best.pt",
    "stage2_jump_human": MODEL_DIR / "stage2_label_is_jump_human_best.pt",
}


@torch.no_grad()
def forward_features(model, loader, device, want_conv):
    """返回 {name: (N, d) numpy}。want_conv=True 时含 conv1–5。"""
    model.eval()
    store = {}
    if want_conv:
        for i in range(1, 6):
            store[f"conv{i}"] = []
    store["dense"] = []
    store["logit"] = []

    for batch in loader:
        x = batch[0].to(device)
        h = x
        for i, (conv, pool, bn) in enumerate(
                zip(model.conv_blocks, model.pools, model.batch_norms), 1):
            h = bn(pool(conv(h)))
            if want_conv:
                store[f"conv{i}"].append(h.mean(dim=2).cpu().numpy())
        flat = model.classifier[0](h)
        dense = model.classifier[2](model.classifier[1](flat))
        logit = model.classifier[4](model.classifier[3](dense)).squeeze(-1)
        store["dense"].append(dense.cpu().numpy())
        store["logit"].append(logit.cpu().numpy())

    return {k: np.concatenate(v) for k, v in store.items()}


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    df = get_df()
    ds = HADataset(df["aligned_ha_seq"].tolist(),
                   np.zeros(len(df), dtype=np.float32))
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    conv_saved = False
    for name, ckpt_path in CHECKPOINTS.items():
        model = BorkenhagenCNN().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))
        feats = forward_features(model, loader, device, want_conv=not conv_saved)
        for k, v in feats.items():
            fname = k if k.startswith("conv") else f"{k}_{name}"
            np.save(FEAT_DIR / f"{fname}.npy", v)
            print(f"  {fname}.npy: {v.shape}")
        conv_saved = True

        # 回归校验：stage2 logit 复算全量 holdout AUC（对照 eval_h5h7_valtest.json）
        if name.startswith("stage2"):
            label_col = ("label_is_jump" if name == "stage2_jump"
                         else "label_is_jump_human")
            for st in ["h5", "h7"]:
                m = (df["split"] == f"{st}_holdout").values
                auc = roc_auc_score(df[label_col].values[m].astype(int),
                                    feats["logit"][m])
                print(f"  [校验] {label_col} {st} 全量 holdout AUC = {auc:.4f}")

    print(f"\n✓ 特征已保存: {FEAT_DIR}")


if __name__ == "__main__":
    main()
