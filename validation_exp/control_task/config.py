"""control task（Hewitt–Liang 式打乱标签对照）— 共享配置

论证目标：linear probe 的跨亚型迁移 AUC 只能来自表示本身（容量不足以记忆），
transformer 头的同分布高分则可由 probe 容量单独产生。
只读复用 ESM_clf/jump_exp 与 ESM_tf_clf 的输出，不修改任何现有文件。
"""

from pathlib import Path

# ── 只读资产 ──
SPLIT_CSV = Path("data/splits/isolate_split.csv")
JUMP_OUT = Path("ESM_clf/jump_exp/output")
TF_OUT = Path("ESM_tf_clf/output")
ALL_LAYERS = JUMP_OUT / "esm_emb_150M_all_layers.npy"     # (31, 11060, 640)

CANDIDATE_LAYERS = [13, 17, 28]     # linear 对照用三层
TF_LAYERS = [17, 28]                # transformer 对照用两层（H7 jump/jump_human 的代表层）
LABEL_COLS = ["label_is_jump", "label_is_jump_human"]
SHUFFLE_SEEDS = [0, 1, 2, 3, 4]     # 每个 seed 在 train 内部独立置换标签

# ── linear probe（与 jump_exp 相同流程）──
RIDGE_C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CV_FOLDS = 5
RANDOM_SEED = 42

# ── transformer 头（与 ESM_tf_clf/config.py 一致）──
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
FIXED_EPOCHS = 30      # 固定 epoch 不早停：容量测量（记录逐 epoch val AUC 轨迹）
TRAIN_SEED = 42

OUT_DIR = Path("validation_exp/control_task/output")
