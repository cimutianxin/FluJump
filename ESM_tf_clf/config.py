"""ESM_tf_clf — 冻结 ESM-2 per-residue embedding + 可训练 Transformer 块 + FFN 分类头：共享配置"""

from pathlib import Path

# ── 数据 ──
DATA_CSV = Path("data/processed_isolate_MAFFT/all_isolates_aligned.csv")
SPLIT_CSV = Path("data/splits/isolate_split.csv")
# H5/H7 内部 5:5 val/test 切分（target-val 层选择协议）
H5H7_SPLIT_DIR = Path("data/splits/h5h7_valtest")
# 行序锚点：与 ESM_clf/jump_exp 现有 embedding/labels 完全一致
ACCESSIONS_NPY = Path("ESM_clf/jump_exp/output/accessions.npy")
LABELS_DIR = Path("ESM_clf/jump_exp/output")

# ── ESM-2 150M (embedding dim=640) ──
ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
ESM_DIM = 640

# ── 候选层（target-val 实验中 val 稳定选中的三层）──
CANDIDATE_LAYERS = [13, 17, 28]
MAX_LEN = 573          # raw ha_sequence 最大长度

# ── Transformer 头 ──
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 2
DIM_FF = 1024
DROPOUT = 0.1

# ── 训练 ──
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 10          # early stop on H1+H3 val AUC
TRAIN_SEEDS = [42, 123, 456]
RANDOM_SEED = 42

# ── 两个分类任务 ──
LABEL_COLS = ["label_is_jump", "label_is_jump_human"]

# ── 输出 ──
OUT_DIR = Path("ESM_tf_clf/output")
