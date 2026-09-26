"""反转驱动位点对比（reversal sites）— 共享配置

论证目标（RESULT.md §3 / TODO-B6）：H7 反转的位点级驱动谱，H10 是否共享同一位点
（姐妹亚支机制线索），H5/H4 对照。只读复用 data/ 与其他实验目录资产。
"""

from pathlib import Path

# ── 只读资产：主数据 ──
SPLIT_CSV = Path("data/splits/isolate_split.csv")
ALIGNED_CSV = Path("data/processed_isolate_MAFFT/all_isolates_aligned.csv")
MEAN_EMB_L3 = Path("ESM_clf/jump_exp/output/esm_emb_150M_L3.npy")
ALL_LAYERS_NPY = Path("ESM_clf/jump_exp/output/esm_emb_150M_all_layers.npy")  # (31,11060,640)
RESIDUE_EMB_L28 = Path("ESM_tf_clf/output/residue_emb_L28.npy")  # (11060,573,640) fp16
RESIDUE_LENS = Path("ESM_tf_clf/output/residue_lens.npy")
LABELS_DIR = Path("ESM_clf/jump_exp/output")
LABELS = {c: LABELS_DIR / f"labels_{c}.npy"
          for c in ["label_is_jump", "label_is_jump_human"]}

# ── 只读资产：site_attribution（L28 probe 参数 + 列→H3 编号映射）──
SITE_ATTR_DIR = Path("validation_exp/site_attribution/output")
PROBE_NPZ = {c: SITE_ATTR_DIR / f"probe_{c}.npz" for c in
             ["label_is_jump", "label_is_jump_human"]}
COL_TO_H3_JSON = SITE_ATTR_DIR / "col_to_h3.json"

# ── 只读资产：group_boundary（H10/H4，同列坐标系对齐）──
GB_DIR = Path("validation_exp/group_boundary")
GB_PROC = GB_DIR / "data" / "processed"
GB_OUT = GB_DIR / "output"
GB_SUBTYPES = ["H10", "H4"]

ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
RANDOM_SEED = 42
LABEL_COLS = ["label_is_jump", "label_is_jump_human"]
RIDGE_C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CV_FOLDS = 5
ALN_LEN = 1039          # 参考对齐宽度（col_to_h3.json: aln_len）

# 层约定：ESM-2 150M hidden_states 索引 = 绝对层号（30=末层=L1，28=L3）
LAYER_IDX = {"L28": 28, "L17": 17, "L13": 13}

OUT_DIR = Path("validation_exp/reversal_sites/output")
