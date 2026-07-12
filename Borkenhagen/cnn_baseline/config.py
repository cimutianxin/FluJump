"""共享配置"""

from pathlib import Path

# 数据路径
DATA_DIR = Path("data/processed_isolate_MAFFT")
INPUT_CSV = DATA_DIR / "all_isolates_aligned.csv"
SPLIT_CSV = Path("data/splits/isolate_split.csv")

OUTPUT_DIR = Path("Borkenhagen/cnn_baseline/output")
MODEL_DIR = OUTPUT_DIR / "models"

ALIGNED_LENGTH = 581
N_AMINO_ACIDS = 21
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

STAGE1_HOSTS = {"avian": 0, "human": 1, "swine": 1}

RANDOM_SEED = 42
DEVICE = "cuda"

# Stage 1
S1_BATCH_SIZE = 128
S1_LR = 1e-3
S1_EPOCHS = 100
S1_PATIENCE = 10

# Stage 2
S2_BATCH_SIZE = 32
S2_LR = 1e-3         # Borkenhagen grid search {0.1,0.01,0.001,0.0001}, 选 0.001
S2_EPOCHS = 100
S2_PATIENCE = 15

# 优化
CLIP_GRAD_NORM = 1.0  # 防止高样本权重导致的梯度爆炸

# CNN
CONV_FILTERS = [32, 64, 128, 256, 512]
CONV_KERNEL = 3
POOL_SIZE = 2
POOL_STRIDE = 2
DENSE_UNITS = 128
DROPOUT = 0.5
