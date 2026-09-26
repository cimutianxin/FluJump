#!/usr/bin/env python3
"""全面诊断 H7 AUC 方向反转的 5 个潜在原因"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DATA_CSV = Path("/root/autodl-tmp/FluJump/data/processed_isolate_MAFFT/all_isolates_aligned.csv")
SPLIT_CSV = Path("/root/autodl-tmp/FluJump/data/splits/isolate_split.csv")
LABEL_CSV = Path("/root/autodl-tmp/FluJump/data/processed/all_subtypes_simplified.csv")
OUT_DIR = Path("/root/autodl-tmp/FluJump/ESM_clf/jump_exp/output")

# Load data
df = pd.read_csv(DATA_CSV)
split_df = pd.read_csv(SPLIT_CSV)
labels_jump = np.load(OUT_DIR / "labels_label_is_jump.npy")
labels_jump_h = np.load(OUT_DIR / "labels_label_is_jump_human.npy")
cluster_df = pd.read_csv(LABEL_CSV)

print("=" * 70)
print("【诊断 1】标签编码：正负标签是否有反转或错误")
print("=" * 70)
for name, labels in [("label_is_jump", labels_jump), ("label_is_jump_human", labels_jump_h)]:
    print(f"\n  {name}:")
    print(f"    唯一值: {np.unique(labels)}")
    print(f"    pos=1 总数: {labels.sum()} ({labels.mean()*100:.1f}%)")
    
    for st in ["H1", "H3", "H5", "H7"]:
        mask = df["subtype"].values == st
        y_st = labels[mask]
        print(f"    {st}: n={mask.sum()}, pos={y_st.sum()} ({y_st.mean()*100:.1f}%)")
    
    csv_vals = df[name]
    print(f"    CSV 中 NaN 数: {csv_vals.isna().sum()}")
    unexpected = set(csv_vals.dropna().unique()) - {0, 1, "0", "1"}
    print(f"    CSV 中非 0/1 值: {unexpected if unexpected else '无'}")
    print(f"    CSV dtype: {csv_vals.dtype}")

print("\n" + "=" * 70)
print("【诊断 2】label_is_jump 和 label_is_jump_human 的逻辑一致性")
print("=" * 70)
mask_j1 = labels_jump == 1
print(f"  jump=1 时 jump_human=1 的比例: {labels_jump_h[mask_j1].mean()*100:.1f}%")
mask_jh1 = labels_jump_h == 1
print(f"  jump_human=1 时 jump=1 的比例: {labels_jump[mask_jh1].mean()*100:.1f}%")

inconsistent = (labels_jump_h == 1) & (labels_jump == 0)
print(f"  jump_human=1 且 jump=0 的矛盾记录: {inconsistent.sum()}")
if inconsistent.sum() > 0:
    print("  → 按定义 jump_human=1 必然 jump=1，存在逻辑错误！")
else:
    print("  → 逻辑一致，无矛盾")

print("\n" + "=" * 70)
print("【诊断 3】H7 标签的生物学/定义验证")
print("=" * 70)
h7_mask = df["subtype"].values == "H7"
h7_df = df[h7_mask]
print(f"  H7 总 isolates: {len(h7_df)}")
print(f"  H7 唯一 cluster: {h7_df['cluster_id'].nunique()}")
print(f"  H7 jump=1 isolates: {labels_jump[h7_mask].sum()}")
print(f"  H7 jump_human=1 isolates: {labels_jump_h[h7_mask].sum()}")
print(f"\n  H7 host_category 分布:")
print(h7_df["host_category"].value_counts().to_string())
print(f"\n  H7 cluster_hosts 分布 (完整):")
host_dist = h7_df["cluster_hosts"].value_counts()
for k, v in host_dist.items():
    print(f"    {k}: {v}")

# 检查 cluster 级标签
print(f"\n  H7 cluster 级标签验证 (从 all_subtypes_simplified.csv):")
h7_clusters = cluster_df[cluster_df["subtype"] == "H7"]
print(f"    H7 clusters: {len(h7_clusters)}")
print(f"    jump=1: {(h7_clusters['label_is_jump']==1).sum()}/{len(h7_clusters)}")
print(f"    jump_human=1: {(h7_clusters['label_is_jump_human']==1).sum()}/{len(h7_clusters)}")
print(f"    host_categories 分布:")
for _, row in h7_clusters.iterrows():
    print(f"      cluster {row['cluster_id']}: hosts={row['host_categories']}, "
          f"jump={row['label_is_jump']}, jh={row['label_is_jump_human']}, "
          f"first_human={row['first_human_ym']}, first_avian={row['first_avian_ym']}")

print("\n" + "=" * 70)
print("【诊断 4】数据行序对齐：split_df ↔ data_df ↔ labels")
print("=" * 70)
print(f"  data_df 行数: {len(df)}")
print(f"  split_df 行数: {len(split_df)}")
print(f"  labels_jump 行数: {len(labels_jump)}")
print(f"  labels_jump_h 行数: {len(labels_jump_h)}")

acc_match = (df["accession"].values == split_df["accession"].values).all()
st_match = (df["subtype"].values == split_df["subtype"].values).all()
print(f"  df.accession == split_df.accession (逐行): {acc_match}")
print(f"  df.subtype == split_df.subtype (逐行): {st_match}")

# 检查 split 分配
for sp in ["train", "val", "test", "h5_holdout", "h7_holdout"]:
    mask = split_df["split"].values == sp
    subtypes_in_split = df["subtype"].values[mask]
    print(f"  {sp}: n={mask.sum()}, subtypes={set(subtypes_in_split)}")

print("\n" + "=" * 70)
print("【诊断 5】预测方向：模型对 H7 的预测")
print("=" * 70)

train_mask = split_df["split"].values == "train"
h7_mask_full = split_df["split"].values == "h7_holdout"

for label_name, y_all in [("label_is_jump", labels_jump), ("label_is_jump_human", labels_jump_h)]:
    print(f"\n  ── {label_name} ──")
    yt = y_all[train_mask]
    y7 = y_all[h7_mask_full]
    
    print(f"    Train: n={len(yt)}, pos_rate={yt.mean():.4f}")
    print(f"    H7:    n={len(y7)}, pos_rate={y7.mean():.4f}")
    
    for emb_name in ["L1", "L3", "L1L3"]:
        X_all = np.load(OUT_DIR / f"esm_emb_150M_{emb_name}.npy")
        Xt = X_all[train_mask]
        X7 = X_all[h7_mask_full]
        
        scl = StandardScaler()
        Xt_s = scl.fit_transform(Xt)
        lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=42)
        lr.fit(Xt_s, yt)
        
        p7 = lr.predict_proba(scl.transform(X7))
        logits7 = lr.decision_function(scl.transform(X7))
        
        auc_raw = roc_auc_score(y7, p7[:, 1])
        auc_flip_logit = roc_auc_score(y7, -logits7)
        
        pred_jump_mean = p7[y7==1, 1].mean() if (y7==1).sum() > 0 else float('nan')
        pred_nonjump_mean = p7[y7==0, 1].mean() if (y7==0).sum() > 0 else float('nan')
        
        # Correlation between prediction and label
        corr = np.corrcoef(p7[:, 1], y7)[0, 1]
        
        direction = "正向 (正确)" if pred_jump_mean > pred_nonjump_mean else "反向 (错误!)"
        
        print(f"    [{emb_name}] classes_={lr.classes_}, coef_mean={lr.coef_[0].mean():.4f}")
        print(f"      pred mean: {p7[:,1].mean():.6f}, pred std: {p7[:,1].std():.6f}")
        print(f"      jump 类平均预测: {pred_jump_mean:.6f}")
        print(f"      non-jump 类平均预测: {pred_nonjump_mean:.6f}")
        print(f"      预测-标签相关: {corr:+.4f}  ← {'✓ 正相关' if corr > 0 else '✗ 负相关 (方向反转!)'}")
        print(f"      AUC(原始): {auc_raw:.4f}  |  AUC(翻转logit): {auc_flip_logit:.4f}")
        print(f"      方向判断: {direction}")

print("\n" + "=" * 70)
print("【诊断 6】H5 对比（应该是正向的）")
print("=" * 70)

h5_mask_full = split_df["split"].values == "h5_holdout"

for label_name, y_all in [("label_is_jump", labels_jump), ("label_is_jump_human", labels_jump_h)]:
    print(f"\n  ── {label_name} ──")
    yt = y_all[train_mask]
    y5 = y_all[h5_mask_full]
    
    for emb_name in ["L3"]:  # L3 had best H5
        X_all = np.load(OUT_DIR / f"esm_emb_150M_{emb_name}.npy")
        Xt = X_all[train_mask]
        X5 = X_all[h5_mask_full]
        
        scl = StandardScaler()
        Xt_s = scl.fit_transform(Xt)
        lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=42)
        lr.fit(Xt_s, yt)
        
        p5 = lr.predict_proba(scl.transform(X5))
        corr = np.corrcoef(p5[:, 1], y5)[0, 1]
        auc5 = roc_auc_score(y5, p5[:, 1])
        
        pred_pos = p5[y5==1, 1].mean() if (y5==1).sum() > 0 else float('nan')
        pred_neg = p5[y5==0, 1].mean() if (y5==0).sum() > 0 else float('nan')
        
        print(f"    [{emb_name}] H5 jump 平均预测: {pred_pos:.6f}")
        print(f"    [{emb_name}] H5 non-jump 平均预测: {pred_neg:.6f}")
        print(f"    [{emb_name}] 预测-标签相关: {corr:+.4f}")
        print(f"    [{emb_name}] H5 AUC: {auc5:.4f}")

print("\nDone — 全部诊断完成")
