"""jev_exp — JEV 启发的"类别查询 + 原型打分"分类头：共享配置

数据路径、候选层、训练超参与评估协议全部复用 ESM_tf_clf.config（只读），
本文件只新增 jev 头的专有超参与输出目录。
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")
from ESM_tf_clf.config import *  # noqa: F401,F403  # 只读复用，不修改

# ── jev 头超参 ──
JEV_D_MODEL = 256        # 投影维度（与 ESM_tf_clf D_MODEL 对齐）
JEV_NHEAD = 8
JEV_DROPOUT = 0.1
JEV_N_CLASS = 2          # 二分类：类0/类1 各一个查询 + 原型
JEV_TAU_INIT = 0.1       # 原型相似度温度初值（CLIP 式 log 参数化）
JEV_INV_TAU_MAX = 100.0  # 1/τ 上限，防止训练初期分数爆炸

# ── 输出 ──
JEV_OUT_DIR = Path("jev_exp/output")
