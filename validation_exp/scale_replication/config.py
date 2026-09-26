"""规模复现（ESM-2 650M / 3B 上的层方向翻转与 target-val 选择）— 共享配置

模型下载需 HF_ENDPOINT=https://hf-mirror.com（huggingface.co 本机不可达）；
已缓存模型用 HF_HUB_OFFLINE=1。
"""

from pathlib import Path

# ── 只读资产 ──
SPLIT_CSV = Path("data/splits/isolate_split.csv")
H5H7_SPLIT_DIR = Path("data/splits/h5h7_valtest")
ACCESSIONS_NPY = Path("ESM_clf/jump_exp/output/accessions.npy")
LABELS_DIR = Path("ESM_clf/jump_exp/output")
DATA_CSV = Path("data/processed_isolate_MAFFT/all_isolates_aligned.csv")

# ── 模型规模 ──
MODELS = {
    "650m": "facebook/esm2_t33_650M_UR50D",   # 34 hidden states, dim 1280
    "3b": "facebook/esm2_t36_3B_UR50D",       # 37 hidden states, dim 2560
}
EXTRACT_BATCH = {"650m": 8, "3b": 2}

LABEL_COLS = ["label_is_jump", "label_is_jump_human"]
# C 网格较 150M 缩减（3B 维度 2560，控制 GridSearchCV 耗时）
RIDGE_C_VALUES = [1e-2, 1e-1, 1.0, 10.0]
CV_FOLDS = 5
RANDOM_SEED = 42
B_BOOT = 1000
SEEDS = [42, 43, 44, 45, 46]
ARMS = ["cluster", "isolate_random"]
SUBTYPES = ["h5", "h7"]

OUT_DIR = Path("validation_exp/scale_replication/output")
