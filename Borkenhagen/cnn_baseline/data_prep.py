"""数据加载 — 读取预计算 split (列拼接，避免 merge 膨胀)"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import *


class HADataset(Dataset):
    def __init__(self, sequences, labels, sample_weights=None):
        self.seq_indices = np.zeros((len(sequences), ALIGNED_LENGTH), dtype=np.int8)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                self.seq_indices[i, j] = AA_TO_IDX.get(aa, AA_TO_IDX["-"])
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.sample_weights = (
            torch.tensor(sample_weights, dtype=torch.float32)
            if sample_weights is not None else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        one_hot = np.eye(N_AMINO_ACIDS, dtype=np.float32)[self.seq_indices[idx]]
        one_hot = one_hot.T
        if self.sample_weights is not None:
            return torch.from_numpy(one_hot), self.labels[idx], self.sample_weights[idx]
        return torch.from_numpy(one_hot), self.labels[idx]


def compute_subtype_class_weights(labels, subtypes):
    groups = list(zip(labels, subtypes))
    unique = set(groups)
    n_total, n_groups = len(labels), len(unique)
    wmap = {g: n_total / (n_groups * groups.count(g)) for g in unique}
    return np.array([wmap[g] for g in groups], dtype=np.float32)


_df = None

def get_df():
    global _df
    if _df is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        d = pd.read_csv(INPUT_CSV)
        s = pd.read_csv(SPLIT_CSV)
        # 同一行顺序，直接赋值，不 merge（避免重复 accession 膨胀）
        d["split"] = s["split"].values
        _df = d
        print(f"加载: {len(_df)} isolates")
        print(f"split 分布:\n{_df['split'].value_counts().to_string()}")
    return _df


def prepare_stage1_data(df):
    """avian vs human+swine, train/val split 中 H1+H3"""
    mask = (df["split"].isin(["train", "val"]) &
            df["host_category"].isin(["avian", "human", "swine"]))
    s1 = df[mask].copy()
    s1["label"] = s1["host_category"].map(STAGE1_HOSTS)
    print(f"\nStage 1: {len(s1)} 条 (0=avian, 1=mammal)")
    print(f"  标签: {s1['label'].value_counts().to_dict()}")

    train_df = s1[s1["split"] == "train"]
    val_df = s1[s1["split"] == "val"]
    print(f"  train={len(train_df)}, val={len(val_df)}")

    w = compute_subtype_class_weights(train_df["label"].values, train_df["subtype"].values)
    print(f"  weight: [{w.min():.2f}, {w.max():.2f}]")

    td = HADataset(train_df["aligned_ha_seq"].tolist(), train_df["label"].values, sample_weights=w)
    vd = HADataset(val_df["aligned_ha_seq"].tolist(), val_df["label"].values)
    return td, vd


def prepare_stage2_data(df, label_col="label_is_jump"):
    """jump 预测, train/val/test 中的 H1+H3"""
    s2 = df[df["split"].isin(["train", "val", "test"])].copy()
    s2["label"] = s2[label_col].astype(int)
    print(f"\nStage 2 ({label_col}): {len(s2)} 条")
    print(f"  标签: {s2['label'].value_counts().to_dict()}")

    train_df = s2[s2["split"] == "train"]
    val_df = s2[s2["split"] == "val"]
    test_df = s2[s2["split"] == "test"]
    print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"  val 标签: {val_df['label'].value_counts().to_dict()}")

    w = compute_subtype_class_weights(train_df["label"].values, train_df["subtype"].values)
    print(f"  weight: [{w.min():.2f}, {w.max():.2f}]")

    td = HADataset(train_df["aligned_ha_seq"].tolist(), train_df["label"].values, sample_weights=w)
    vd = HADataset(val_df["aligned_ha_seq"].tolist(), val_df["label"].values)
    xd = HADataset(test_df["aligned_ha_seq"].tolist(), test_df["label"].values)
    return td, vd, xd


def prepare_holdout_data(df, label_col, holdout_name):
    ho = df[df["split"] == holdout_name].copy()
    ho["label"] = ho[label_col].astype(int)
    print(f"\n{holdout_name}: {len(ho)} 条, 标签: {ho['label'].value_counts().to_dict()}")
    return HADataset(ho["aligned_ha_seq"].tolist(), ho["label"].values)


def create_dataloaders(train_ds, val_ds, test_ds=None, batch_size=128):
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    if test_ds is not None:
        xl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return tl, vl, xl
    return tl, vl


if __name__ == "__main__":
    df = get_df()
    print(f"\n=== Stage 1 ===")
    td, vd = prepare_stage1_data(df)
    tl, vl = create_dataloaders(td, vd, batch_size=S1_BATCH_SIZE)
    b = next(iter(tl))
    print(f"batch: x={b[0].shape}, w={b[2].shape}")

    for ln in ["label_is_jump", "label_is_jump_human"]:
        print(f"\n=== Stage 2: {ln} ===")
        td2, vd2, xd2 = prepare_stage2_data(df, ln)
        prepare_holdout_data(df, ln, "h5_holdout")
        prepare_holdout_data(df, ln, "h7_holdout")
