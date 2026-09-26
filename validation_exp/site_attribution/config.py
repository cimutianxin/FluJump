"""位点归因（site attribution）— 共享配置

论证目标（RESULT.md §2 待补）：检验 probe 方向 w 是否重现已知宿主适应标记位点
（RBS 190/225/226/228 等，H3 编号），把"几何现象"升级为"生物学陈述"。
只读复用 data/ 与 ESM_clf/ESM_tf_clf 的资产，不修改任何现有文件。
"""

from pathlib import Path

# ── 只读资产 ──
ALIGNED_CSV = Path("data/processed_isolate_MAFFT/all_isolates_aligned.csv")
MERGED_FAA = Path("data/processed_isolate_MAFFT/all_aligned_merged.faa")
ISOLATES_CSV = Path("data/processed_isolate/all_isolates.csv")
SPLIT_CSV = Path("data/splits/isolate_split.csv")
RESIDUE_EMB = Path("ESM_tf_clf/output/residue_emb_L28.npy")   # (11060,573,640)，行序=isolate_split.csv
RESIDUE_LENS = Path("ESM_tf_clf/output/residue_lens.npy")
MEAN_EMB_L3 = Path("ESM_clf/jump_exp/output/esm_emb_150M_L3.npy")  # (11060,640)
LABELS = {c: Path(f"ESM_clf/jump_exp/output/labels_{c}.npy")
          for c in ["label_is_jump", "label_is_jump_human"]}

ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
MAFFT = "/usr/bin/mafft"
RANDOM_SEED = 42
LABEL_COLS = ["label_is_jump", "label_is_jump_human"]

# ── H3 参考株（H3 编号基准）──
H3_REF_ACCESSION = "AAA43178"   # A/Aichi/2/1968, 566 aa

# ── 标记位点表（H3 编号）──
# RBS: 130-loop (134-138), 190-helix, 220-loop；抗原/糖基化相关 156/159/160
MARKER_SITES = {
    "RBS_130_loop": [134, 135, 136, 137, 138],
    "RBS_190_helix": [186, 190, 193, 194],
    "RBS_220_loop": [221, 222, 224, 225, 226, 227, 228],
    "antigenic_glyco": [156, 159, 160],
}
# 已知人适应替换（H3 编号: (野生型倾向, 人适应替换)），用于 Part B 方向性检验
KNOWN_HUMAN_SUBS = [
    (226, "Q", "L"),   # Q226L：禽 SAα2,3 → 人 SAα2,6 转换决定簇
    (228, "G", "S"),   # G228S：同上（H2/H3 人流感大流行株）
    (190, "D", "N"),   # D190N：H1/H5 人适应相关
    (190, "D", "E"),   # D190E：H1 人适应相关
    (225, "D", "G"),   # D225G：H1N1 重症/人适应相关
]

# ── ISM 规模 ──
ISM_N_BACKGROUND = 100      # H1+H3 train 背景序列（jump+/− 各 50，cluster 去重）
ISM_N_CONTROL = 20          # 对照位点数（按列熵匹配，排除功能位）
ISM_BATCH = 8

OUT_DIR = Path("validation_exp/site_attribution/output")
