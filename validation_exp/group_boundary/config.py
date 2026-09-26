"""Group 边界零样本方向检验（H10 + H4 对照）— 共享配置

论证目标（RESULT.md §3 待补）：H7 反转目前仅建立在 7 个阳性 cluster（全是
2013–2015 H7N9 单次暴发）上。纳入第二个 Group 2 亚型做零样本方向检验：
  - H10（H7 姐妹亚支，H7/H10/H15 亚支）：假说预测 raw AUC < 0.5（反转复现）
  - H4（Group 2 但属 H3 亚支）：特异性对照，预测不反转
自包含目录，只读复用 data/ 与 ESM_clf/ 资产，不修改任何现有文件。
"""

import re
from pathlib import Path

# ── 本实验目录 ──
BASE_DIR = Path("validation_exp/group_boundary")
RAW_DIR = BASE_DIR / "data/raw"
PROC_DIR = BASE_DIR / "data/processed"
OUT_DIR = BASE_DIR / "output"

SUBTYPES = ["H10", "H4"]


def _union_query(h: str) -> str:
    """亚型关键词 union：esearch 不会把 'H10N8' 分词出 'H10'，
    单用 H10[all fields] 会漏掉全部只按 H10N8 等全型标注的记录。"""
    terms = " OR ".join([f"{h}[all fields]"] +
                        [f"{h}N{i}[all fields]" for i in range(1, 10)])
    return (f"hemagglutinin[protein name] AND Influenza A virus[organism] "
            f"AND ({terms}) NOT pdb[filter]")


QUERIES = {"H10": _union_query("H10"), "H4": _union_query("H4")}

# 亚型一致性 post-filter：query 全文匹配仍会带入杂亚型（如 H5N6），
# 按 serotype / strain_name / definition 中的 "(HxNy" 标注复核
SUBTYPE_PATTERN = {st: re.compile(rf"\b{st}N\d\b|\({st}\)|\b{st}\b")
                   for st in SUBTYPES}

# ── 只读资产 ──
SPLIT_CSV = Path("data/splits/isolate_split.csv")
ALIGNED_CSV = Path("data/processed_isolate_MAFFT/all_isolates_aligned.csv")
MERGED_FAA = Path("data/processed_isolate_MAFFT/all_aligned_merged.faa")
# (31, 11060, 640)，行序 = isolate_split.csv；层下标 0..30（L1=最后一层=30）
ALL_LAYERS_NPY = Path("ESM_clf/jump_exp/output/esm_emb_150M_all_layers.npy")
LABELS_DIR = Path("ESM_clf/jump_exp/output")  # labels_{label}.npy，行序同上

ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
RIDGE_C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CV_FOLDS = 5
RANDOM_SEED = 42
LABEL_COLS = ["label_is_jump", "label_is_jump_human"]

CD_HIT = "/root/miniconda3/bin/cd-hit"
CDHIT_IDENTITY = 0.99
CDHIT_COVERAGE = 0.9

# 预注册层（全层数组下标）：历史 target-val 选层 jump→17 / jump_human→13，
# 另加历史 headline "L3（倒数第三层）"= 下标 28。不用 H10/H4 数据选层。
PREREG_LAYERS = {"label_is_jump": [17, 28], "label_is_jump_human": [13, 28]}
B_BOOT = 1000
