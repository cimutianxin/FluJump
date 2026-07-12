"""ESM-2 150M + Ridge LR 二分类（jump / jump_human）— 共享配置"""

from pathlib import Path

# ── 数据 ──
DATA_CSV = Path("data/processed_isolate_MAFFT/all_isolates_aligned.csv")
SPLIT_CSV = Path("data/splits/isolate_split.csv")

# ── ESM-2 150M (embedding dim=640) ──
ESM_MODEL = "facebook/esm2_t30_150M_UR50D"

# ── Ridge LR ──
RIDGE_C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CV_FOLDS = 5
RANDOM_SEED = 42

# ── 两个分类任务 ──
LABEL_COLS = ["label_is_jump", "label_is_jump_human"]

# ── 输出 ──
OUT_DIR = Path("ESM_clf/jump_exp/output")
