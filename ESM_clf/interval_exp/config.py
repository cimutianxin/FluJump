"""ESM-2 150M + Linear/MLP Probe — Jump Interval 预测（回归 + 分类）共享配置"""

from pathlib import Path

# ── 数据 ──
DATA_CSV = Path("data/processed_isolate/all_isolates_clean.csv")
SPLIT_CSV = Path("data/splits/isolate_split.csv")

# ── ESM-2 150M (embedding dim=640) ──
ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
# 复用已有 embedding（全部 isolate）
EMB_DIR = Path("ESM_clf/jump_exp/output")
EMB_TYPES = ["L1", "L3", "L1L3"]  # mean-pooled from raw ha_sequence

# ── Ridge 回归 ──
RIDGE_ALPHA_VALUES = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
CV_FOLDS = 5
RANDOM_SEED = 42

# ── Interval 分类 bins（绝对天数 → 类别）──
INTERVAL_BINS = [0, 365, 1095, 1825, 999999]  # <1yr, 1-3yr, 3-5yr, 5yr+
INTERVAL_LABELS = ["<1yr", "1-3yr", "3-5yr", "5yr+"]

# ── 只使用 H1+H3 且 label_is_jump=1 的 isolate ──
SUBTYPES = ["H1", "H3"]

# ── MLP 超参数 ──
MLP_HIDDEN_CONFIGS = [
    [256, 128],   # 默认两层
    [128],        # 单层
    [512, 256],   # 更宽
]
MLP_DROPOUT = 0.3
MLP_LR = 1e-3
MLP_EPOCHS = 200
MLP_BATCH_SIZE = 64
MLP_WEIGHT_DECAY = 1e-4
MLP_EARLY_STOP_PATIENCE = 20
MLP_SEEDS = [42, 123, 456]

# ── 输出 ──
OUT_DIR = Path("ESM_clf/interval_exp/output")
