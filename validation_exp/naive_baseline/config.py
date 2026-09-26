"""朴素生物学基线（人源序列相似度排序）— 共享配置

论证目标（RESULT.md §1 待补）：检验 ESM-2 linear probe 的跨亚型迁移能力是否只是
"与已知人源分离株的全局序列相似度"的复述。在同一 H5/H7 holdout 协议下对比。
只读复用 data/ 下的 aligned 数据集与划分文件，不修改任何现有文件。
"""

from pathlib import Path

# ── 只读资产 ──
ALIGNED_CSV = Path("data/processed_isolate_MAFFT/all_isolates_aligned.csv")
SPLIT_CSV = Path("data/splits/isolate_split.csv")

# ── 打分与评估 ──
LABEL_COLS = ["label_is_jump", "label_is_jump_human"]
TOPK = 5                 # top-k identity 均值变体
N_BOOTSTRAP = 1000       # cluster 级 bootstrap 次数
RANDOM_SEED = 42

# ESM-2 linear probe 参照值（ESM_clf/jump_exp，target-val 选层协议）
ESM_REFERENCE = {
    ("h5_holdout", "label_is_jump"): 0.79,
    ("h7_holdout", "label_is_jump"): 0.892,        # -logits 翻转后（L17）
    ("h7_holdout", "label_is_jump_human"): 0.854,  # -logits 翻转后（L13）
}

OUT_DIR = Path("validation_exp/naive_baseline/output")
