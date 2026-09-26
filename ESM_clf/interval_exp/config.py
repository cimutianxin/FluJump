"""ESM-2 150M + Probe — Jump Interval 预测（4 类分类 + 回归）共享配置

2026-08-01 重构：
- 标签改为 4 类（human_first 独立成类，见 interval_data.py docstring）
- 删除 USE_CLAMPED_DAYS 开关（标签语义固定从有符号 raw days 派生）
- 模型选择流程规范化：train 训练 → val 选择 → test 只评估一次
"""

from pathlib import Path

# ── 数据 ──
DATA_CSV = Path("data/processed_isolate/all_isolates_clean.csv")
SPLIT_CSV = Path("data/splits/isolate_split.csv")

# ── ESM-2 150M (embedding dim=640) ──
ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
# 复用已有 embedding（全部 isolate，行序见 jump_exp/output/accessions.npy）
EMB_DIR = Path("ESM_clf/jump_exp/output")
EMB_TYPES = ["L1", "L3", "L1L3"]  # mean-pooled from raw ha_sequence

# ── 通用 ──
CV_FOLDS = 5
RANDOM_SEED = 42

# ── Interval 分类标签（4 类，有序，风险递减）──
# 0 human_first: raw_days < 0（人先于动物检出，最高危）
# 1 <1yr / 2 1-3yr / 3 3yr+: 由非负 raw_days 按 INTERVAL_BINS 分箱
INTERVAL_BINS = [0, 365, 1095, 999999]
INTERVAL_LABELS = ["human_first", "<1yr", "1-3yr", "3yr+"]

# ── 只使用 H1+H3 且 label_is_jump=1 的 isolate ──
SUBTYPES = ["H1", "H3"]

# ── GridSearchCV scoring（统一用 balanced_accuracy）──
CV_SCORING = "balanced_accuracy"

# ── 模型选择协议（P1 规范化）──
# train 上 GridSearchCV → val 上按 balanced_acc 选唯一 winner
# → train+val 重训 → test 只评估一次
N_BOOTSTRAP = 1000          # test 指标 bootstrap 95% CI 重采样次数
N_RANDOM_BASELINE = 1000    # stratified random 基线重复次数

# ── Ridge 回归 ──
RIDGE_ALPHA_VALUES = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

# ── MLP 超参数 ──
MLP_HIDDEN_CONFIGS = [
    [256, 128],   # 默认两层
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
