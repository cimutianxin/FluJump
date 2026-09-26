"""共享数据加载模块 — interval 预测实验唯一数据入口。

2026-08-01 重构，修复两个对齐 bug：
1. split 之前用 `df["split"] = split_df["split"]` 按行位置赋值，
   但 all_isolates_clean.csv（11261 行）与 isolate_split.csv（11060 行）行序不一致
   → 改为按 (accession, subtype) inner merge。
   注意：同一 accession 可在多个亚型下出现（264 个跨亚型重复），
   单列 accession 不是唯一键，必须带 subtype。
2. embedding 之前用 `emb[df[mask].index]` 按当前 CSV 行号切片，
   但 embedding（11060 行，07-10 旧行序）与当前 clean CSV 行序不一致
   → embedding 行序与 isolate_split.csv 行序逐行一致（已验证），
   因此用 split CSV 的 (accession, subtype) → 行号建立映射切片。

标签语义（4 类，有序，风险递减）：
  0 human_first: raw_days < 0（人先于动物检出，跨物种能力已激活，最高危）
  1 <1yr:        0 ≤ raw_days < 365
  2 1-3yr:       365 ≤ raw_days < 1095
  3 3yr+:        raw_days ≥ 1095

3 类变体（y_cat3，2026-08-10 新增）：human_first 与 <1yr 合并为 <1yr
  0 <1yr:  raw_days < 365
  1 1-3yr: 365 ≤ raw_days < 1095
  2 3yr+:  raw_days ≥ 1095
raw_days 从 clean CSV 现有列重算（公式同 datascripts/build_isolate_dataset.py
的 compute_isolate_interval，但不做负值 clamp）。
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import (
    DATA_CSV, SPLIT_CSV, EMB_DIR, EMB_TYPES, SUBTYPES,
)

ANIMAL_YM_COLS = ["first_avian_ym", "first_swine_ym", "first_bovine_ym"]


def _days_between(ym1, ym2):
    """两个 YYYY-MM 之间的近似天数差 (ym2 - ym1)，与 datascripts 一致。"""
    if not ym1 or not ym2:
        return None
    try:
        y1, m1 = int(ym1[:4]), int(ym1[5:7])
        y2, m2 = int(ym2[:4]), int(ym2[5:7])
        return (y2 - y1) * 365 + (m2 - m1) * 30
    except (ValueError, IndexError):
        return None


def _raw_interval_days(row):
    """重算未 clamp 的有符号 isolate 级 interval。

    - 动物 isolate: collection_date → first_human_ym（距人类首检还有多久）
    - 人 isolate:   first_animal_ym → collection_date（动物首检后多久感染人）
    负值 = 人先于动物（human_first，最高危）。
    """
    cd = row["collection_date"]
    fh = row["first_human_ym"]
    if not cd or cd == "-1" or not fh:
        return None
    animal_yms = [row[c] for c in ANIMAL_YM_COLS if isinstance(row[c], str) and row[c]]
    if not animal_yms:
        return None
    first_animal_ym = min(animal_yms)
    if row["host_category"] == "human":
        return _days_between(first_animal_ym, cd)
    return _days_between(cd, fh)


def _load_emb_row_map(split_df):
    """(accession, subtype) → embedding 行号映射。

    embedding 行序与 isolate_split.csv 行序逐行一致（已验证 accession 序列相同），
    因此直接用 split CSV 的复合键标注 embedding 行。
    """
    return {(a, st): i for i, (a, st) in
            enumerate(zip(split_df["accession"], split_df["subtype"]))}


def load_interval_data(subtypes=None, split_csv=None, verbose=True):
    """加载 interval 实验全部数据（accession 对齐版）。

    Args:
        subtypes:  亚型过滤列表，默认 config.SUBTYPES（["H1","H3"]）
        split_csv: split 标签来源 CSV（需含 accession,subtype,split 列），
                   默认 config.SPLIT_CSV。注意无论传什么，embedding 行号映射
                   始终由 config.SPLIT_CSV（isolate_split.csv）建立 ——
                   embedding 行序与其逐行一致。

    Returns:
        dict with keys:
          X:          {emb_type: np.ndarray (N, dim)}，N 为对齐后的样本数
          y_cat:      (N,) int32，4 类分类标签（见模块 docstring）
          y_cat3:     (N,) int32，3 类分类标签（human_first 并入 <1yr）
          y_days:     (N,) float64，clamped raw_days（负值→0，回归用）
          y_days_raw: (N,) float64，有符号 raw_days
          emb_rows:   (N,) int，各样本在 isolate_split.csv / embedding 中的行号
          masks:      {"train"/"val"/"test": (N,) bool}
          subtypes:   (N,) str
          accessions: (N,) str
          cluster_ids:(N,)
          df:         对齐后的 DataFrame
    """
    subtypes = subtypes or SUBTYPES
    df = pd.read_csv(DATA_CSV, dtype=str).fillna("")
    split_df = pd.read_csv(split_csv or SPLIT_CSV, dtype=str)

    # ── 筛选：指定亚型 且 label_is_jump=1 ──
    mask = df["subtype"].isin(subtypes) & (df["label_is_jump"] == "1")
    sub = df[mask].copy()

    # ── 重算有符号 raw days（需非空）──
    sub["_raw_days"] = sub.apply(_raw_interval_days, axis=1)
    n_before = len(sub)
    sub = sub[sub["_raw_days"].notna()].copy()
    if verbose:
        print(f"  筛选 {'+'.join(subtypes)} & jump=1: {n_before} 条，"
              f"有效 interval: {len(sub)} 条")

    # ── 按 (accession, subtype) merge split（inner，丢弃无 split 的行）──
    n_before = len(sub)
    sub = sub.merge(split_df[["accession", "subtype", "split"]],
                    on=["accession", "subtype"], how="inner")
    if verbose and len(sub) < n_before:
        print(f"  ⚠ {n_before - len(sub)} 条无 split 记录，已丢弃")

    # ── 按 (accession, subtype) 对齐 embedding ──
    # 注意：embedding 行序与 isolate_split.csv（SPLIT_CSV）逐行一致，
    # 行号映射必须用它建立，与传入的 split_csv 无关。
    row_map = _load_emb_row_map(pd.read_csv(SPLIT_CSV, dtype=str))
    emb_rows = pd.Series(
        [row_map.get(k) for k in zip(sub["accession"], sub["subtype"])],
        index=sub.index,
    )
    n_before = len(sub)
    keep = emb_rows.notna()
    if verbose and (~keep).sum() > 0:
        print(f"  ⚠ {(~keep).sum()} 条无 embedding，已丢弃: "
              f"{sub.loc[~keep, 'accession'].tolist()}")
    sub = sub[keep].copy()
    emb_idx = emb_rows[keep].values.astype(int)

    X = {}
    for etype in EMB_TYPES:
        path = EMB_DIR / f"esm_emb_150M_{etype}.npy"
        X[etype] = np.load(path)[emb_idx] if path.exists() else None

    # ── 标签 ──
    raw = sub["_raw_days"].values.astype(np.float64)
    y_days_raw = raw
    y_days = np.maximum(raw, 0.0)  # clamped，回归目标
    y_cat = np.where(raw < 0, 0, np.digitize(raw, [365, 1095]) + 1).astype(np.int32)
    y_cat3 = np.digitize(raw, [365, 1095]).astype(np.int32)  # human_first 并入 <1yr

    split_arr = sub["split"].values
    data = {
        "X": X,
        "y_cat": y_cat,
        "y_cat3": y_cat3,
        "y_days": y_days,
        "y_days_raw": y_days_raw,
        "emb_rows": emb_idx,
        "masks": {s: (split_arr == s) for s in ["train", "val", "test"]},
        "subtypes": sub["subtype"].values,
        "accessions": sub["accession"].values,
        "cluster_ids": sub["cluster_id"].values,
        "df": sub,
    }
    if verbose:
        print(f"  最终样本: {len(sub)} 条 "
              f"(train={data['masks']['train'].sum()}, "
              f"val={data['masks']['val'].sum()}, "
              f"test={data['masks']['test'].sum()})")
    return data


def balanced_sample_weight(y):
    """按类别频率倒数计算 sample_weight（balanced）。"""
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([w[c] for c in y])
