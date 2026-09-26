"""深度方向翻转机制解释（depth reversal mechanism，C9）— 共享配置

论证目标（RESULT.md §4 / TODO-9，升档关键③）：解释"中层 vs 末层编码的 jump
方向相反"（08-01-22 留下的开放问题：亚型身份全层贯穿，为何方向随深度翻转）。
自包含目录，只读复用 data/ 与其他实验目录资产，不修改任何现有文件。

预注册假设与判定规则见计划文件与 notes 实验日志：
  H1 几何夹角 | H2 方向竞争 | H3 方差结构 | H4 残差流定位 | H5 位点读出 | H6 注意力
"""

from pathlib import Path

# ── 只读资产：主数据 ──
SPLIT_CSV = Path("data/splits/isolate_split.csv")   # 行序 = 全层 npy / labels 行序
ALIGNED_CSV = Path("data/processed_isolate_MAFFT/all_isolates_aligned.csv")
ALL_LAYERS_NPY = Path("ESM_clf/jump_exp/output/esm_emb_150M_all_layers.npy")  # (31,11060,640)
LABELS_DIR = Path("ESM_clf/jump_exp/output")
LABELS = {c: LABELS_DIR / f"labels_{c}.npy"
          for c in ["label_is_jump", "label_is_jump_human"]}

# ── 只读资产：坐标映射与位点集 ──
SITE_ATTR_DIR = Path("validation_exp/site_attribution/output")
COL_TO_H3_JSON = SITE_ATTR_DIR / "col_to_h3.json"          # 1039 列 ↔ canonical H3 编号
G4_COLUMN_SCORES = Path("validation_exp/reversal_sites/output/g4_column_scores.csv")

# ── 只读资产：规模复现全层 embedding（G5）──
SCALE_NPY = {
    "650m": Path("validation_exp/scale_replication/output/esm_emb_650m_all_layers.npy"),
    "3b": Path("validation_exp/scale_replication/output/esm_emb_3b_all_layers.npy"),
}

ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
ESM_DIM = 640
N_LAYERS = 31          # hidden_states 条目数（含 embedding 层 0）
ALN_LEN = 1039
RANDOM_SEED = 42
LABEL_COLS = ["label_is_jump", "label_is_jump_human"]
RIDGE_C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CV_FOLDS = 5

# ── P0 回归校验基准（同 reversal_sites/g2：group_boundary 同层 AUC，容差含
#    09-19 标签修复位移）──
REPRO_REF = {
    (28, "label_is_jump"): (0.7980, 0.2823),
    (28, "label_is_jump_human"): (0.7936, 0.2059),
    (17, "label_is_jump"): (0.3918, 0.8937),
    (17, "label_is_jump_human"): (0.2646, 0.6277),
    (13, "label_is_jump"): (0.2483, 0.7571),
    (13, "label_is_jump_human"): (0.2523, 0.8632),
}
REPRO_TOL = 0.06

# ── 位点集（canonical H3 编号；功能集沿用 site_attribution MARKER_SITES + 已知
#    人适应位点，保证与 §2 口径一致）──
FUNCTIONAL_SITES = {
    "RBS_130_loop": [134, 135, 136, 137, 138],
    "RBS_190_helix": [186, 190, 193, 194],
    "RBS_220_loop": [221, 222, 224, 225, 226, 227, 228],
    "antigenic_glyco": [156, 159, 160],
    "known_human_adapt": [190, 225, 226, 228],
}
# 亚型诊断集：g4_column_scores.csv 中 valid 且 js_div 前 5% 的列（脚本内计算）
DIAG_JS_QUANTILE = 0.95

# ── 层段划分（预注册：中层 = H7 方向正确段，末层 = 反转段）──
MID_LAYERS = list(range(13, 23))     # 13–22
LATE_LAYERS = [28, 29, 30]

# ── H2 方向竞争 ──
N_SHUFFLE_NULL = 20                  # 每层 shuffled-label 重训次数（固定 C 单 fit）

# ── G3 / G4 GPU 参数 ──
EXTRACT_BATCH = 8
LAYER_GB = 8.1                       # 单层 per-residue fp16 (11060,573,640) 占用
RESERVE_GB = 12.0
MAX_LEN = 573
G4_SUBSET_PER_GROUP = 400            # 注意力子集每组上限
G4_BATCH = 4

OUT_DIR = Path("validation_exp/depth_reversal/output")
