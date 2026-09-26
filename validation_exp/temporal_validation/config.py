"""时间维度验证（temporal validation）— 共享配置

论证目标（RESULT.md §5 待补）：方向 w 的投影作为"距离人适应还差多少"的连续读数，
在时间维度上做两个验证：
  1) 前向：2026 年新分离株排序（天然阳性 = cluster 70 的 9 条 H5N1 暴发簇前向样本）
  2) 回溯：pre-2013 训练 → 2013–2015 H7N9 暴发簇排序
只读复用 data/ 与 ESM_clf/validation_exp 的资产，不修改任何现有文件。
"""

from pathlib import Path

# ── 只读资产 ──
CLEAN_2026_CSV = Path("data/processed_isolate/2026_isolates_clean.csv")
ALIGNED_CSV = Path("data/processed_isolate_MAFFT/all_isolates_aligned.csv")
SPLIT_CSV = Path("data/splits/isolate_split.csv")
MEAN_EMB_L3 = Path("ESM_clf/jump_exp/output/esm_emb_150M_L3.npy")   # (11060,640)，行序=isolate_split.csv
MEAN_EMB_L1 = Path("ESM_clf/jump_exp/output/esm_emb_150M_L1.npy")   # 末层（L1），层选择敏感性的对照层
ALL_ISOLATES_CSV = Path("data/processed_isolate/all_isolates.csv")  # 全量（含 2026），≤2025 标签重算用

ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
RANDOM_SEED = 42
LABEL_COLS = ["label_is_jump", "label_is_jump_human"]

# ── 2026 前向：天然阳性定义（cluster 继承标签，jump_human=1）──
FORWARD_POS_CLUSTERS = {70, 15}

# ── 回溯：年份阈值与 H7N9 阳性簇 ──
RETRO_TRAIN_YEAR_MAX = 2013          # 训练集 0 < year < 2013
RETRO_EVAL_YEAR_MIN = 2013           # 评估集 year >= 2013
H7N9_POS_CLUSTERS = {109, 149, 150, 151, 154, 155, 198}

# ── probe 训练超参（与 ESM_clf/jump_exp/train_probe.py 一致）──
RIDGE_C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CV_FOLDS = 5

# ── 前向加固（run_2026_forward_hardening.py）──
B_BOOT = 10000     # cluster bootstrap 重采样次数
B_PERM = 10000     # cluster 标签 permutation 次数

# ── 朴素基线（同 naive_baseline：与 H1+H3 人源参考的最大对齐 identity）──
NAIVE_REF_HOST = "human"

OUT_DIR = Path("validation_exp/temporal_validation/output")
