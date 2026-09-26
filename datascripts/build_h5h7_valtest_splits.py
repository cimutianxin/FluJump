#!/usr/bin/env python3
"""
H5/H7 内部 5:5 val/test 切分（target-val 层选择协议）。

背景：H5/H7 原本整体 holdout（isolate_split.csv 的 h5_holdout/h7_holdout），
层选择若在 holdout 全量上看 AUC 即为"用测试标签选超参"。
本脚本把 H5/H7 内部各切出 50% val（选层）+ 50% test（只评一次）。

两种切分臂：
  - cluster 臂（主）：cluster 整体归入 val/test，防止近重复序列横跨两侧
  - isolate_random 臂（对比）：按行随机切，用于量化近重复泄露（两臂 test AUC 之差）

各 5 个种子（42-46），共 10 个文件。
输出：data/splits/h5h7_valtest/{cluster,isolate_random}_seed{42..46}.csv
每文件列：accession, subtype, cluster_id, split ∈ {h5_val, h5_test, h7_val, h7_test}
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_CSV = Path("data/processed_isolate_MAFFT/all_isolates_aligned.csv")
OUT_DIR = Path("data/splits/h5h7_valtest")
SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.5

df = pd.read_csv(DATA_CSV)
df = df[df["subtype"].isin(["H5", "H7"])].copy()
print(f"H5/H7 数据: {len(df)} isolates")
for st, sub in df.groupby("subtype"):
    print(f"  {st}: {len(sub)} isolates, {sub['cluster_id'].nunique()} clusters")

# ── 校验：cluster 内标签必须唯一（标签本为 cluster 级构造）──
for (st, cl), sub in df.groupby(["subtype", "cluster_id"]):
    for col in ["label_is_jump", "label_is_jump_human"]:
        assert sub[col].nunique() == 1, \
            f"cluster ({st}, {cl}) 内 {col} 不一致，标签并非 cluster 级唯一，终止！"

# stratify key：jump 与 jump_human 的组合（j0jh0 / j1jh0 / j1jh1）
def strat_key(frame):
    return ("j" + frame["label_is_jump"].astype(str)
            + "_jh" + frame["label_is_jump_human"].astype(str))


def audit(out, seed, arm):
    """打印一个切分文件的分布审查表"""
    print(f"\n--- {arm} seed={seed} ---")
    for sp in ["h5_val", "h5_test", "h7_val", "h7_test"]:
        sub = out[out["split"] == sp]
        jh = int(sub.merge(df[["accession", "subtype", "label_is_jump", "label_is_jump_human"]],
                           on=["accession", "subtype"])["label_is_jump_human"].sum())
        print(f"  {sp}: {len(sub):>5} isolates / {sub['cluster_id'].nunique():>4} clusters  "
              f"jh=1: {jh}")


OUT_DIR.mkdir(parents=True, exist_ok=True)

for seed in SEEDS:
    # ══ cluster 臂：cluster 级 stratified 5:5 ══
    parts = []
    for st in ["H5", "H7"]:
        sub = df[df["subtype"] == st]
        # cluster 级表：cluster_id + 标签（cluster 内唯一，上面已 assert）
        cl = sub.groupby("cluster_id")[["label_is_jump", "label_is_jump_human"]].first()
        cl_val, cl_test = train_test_split(
            cl.index, test_size=TEST_SIZE, random_state=seed,
            stratify=strat_key(cl),
        )
        val_mask = sub["cluster_id"].isin(cl_val)
        for mask, tag in [(val_mask, f"{st.lower()}_val"), (~val_mask, f"{st.lower()}_test")]:
            p = sub[mask][["accession", "subtype", "cluster_id"]].copy()
            p["split"] = tag
            parts.append(p)
    out = pd.concat(parts).sort_index()
    out.to_csv(OUT_DIR / f"cluster_seed{seed}.csv", index=False)
    audit(out, seed, "cluster")

    # ══ isolate_random 臂：按行 stratified 5:5 ══
    parts = []
    for st in ["H5", "H7"]:
        sub = df[df["subtype"] == st]
        idx_val, idx_test = train_test_split(
            sub.index, test_size=TEST_SIZE, random_state=seed,
            stratify=strat_key(sub),
        )
        for idx, tag in [(idx_val, f"{st.lower()}_val"), (idx_test, f"{st.lower()}_test")]:
            p = sub.loc[idx, ["accession", "subtype", "cluster_id"]].copy()
            p["split"] = tag
            parts.append(p)
    out = pd.concat(parts).sort_index()
    out.to_csv(OUT_DIR / f"isolate_random_seed{seed}.csv", index=False)
    audit(out, seed, "isolate_random")

print(f"\n✓ {len(SEEDS) * 2} 个切分文件已写入 {OUT_DIR}/")
