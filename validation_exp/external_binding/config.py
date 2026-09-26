"""外部锚定（Borkenhagen 402 条实验 binding 数据）— 共享配置

不修改 ESM_clf/binding_exp 的任何文件；embedding 与结果都写入本子目录。
"""

from pathlib import Path

# ── 只读数据 ──
BORK_CSV = Path("data/dataset_borkenhagen/borkenhagen_clean.csv")
BORK_SPLIT = Path("data/dataset_borkenhagen/borkenhagen_split.csv")
MAIN_SPLIT_CSV = Path("data/splits/isolate_split.csv")
JUMP_OUT = Path("ESM_clf/jump_exp/output")          # 主数据 all_layers + labels

ESM_MODEL = "facebook/esm2_t30_150M_UR50D"

# target-val 协议选出的层（jump: H5→L28, H7→L17；jump_human: H7→L13）
ANCHOR_LAYERS = [13, 17, 28]

LABEL_COLS = ["label_is_jump", "label_is_jump_human"]
RIDGE_C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CV_FOLDS = 5
RANDOM_SEED = 42

OUT_DIR = Path("validation_exp/external_binding/output")
