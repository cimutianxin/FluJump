# 实验：650M/3B 的 participation ratio 与 AA decoding margin 曲线（S2，Major 11）

- **日期**：10-06-04
- **目的**：label-free 选层判据（PR 最小点 ≈ 最优层）目前只有 150M 证据，且依赖 "layers ≥ 2" 人为下限；检验是否跨规模成立。

## 方法
- `v1_aa_identity_decode_scale.py`：设计完全同 V1（同批 seed42 抽样序列/位置、闭式 ridge 20-way 解码、
  各向异性、PR），仅换模型——650M（esm2_t33_650M，34 states，dim 1280）与 3B（esm2_t36_3B，37 states，dim 2560）。
- gap_H7 剖面取 g5_scale_geometry.json 对应规模；对照各规模 target-val 最优层（650M L5/L3、3B L2/(L1/L2)）。

## 结果
| 规模 | PR 最小（全局 / L≥2） | rel_depth | target-val 最优 | ρ(PR, gap_H7) |
|---|---|---|---|---|
| 150M（既有） | L0(13.4) / **L17** | 0.57 | L17 | **−0.567**（p=9e-4） |
| 650M | **L15 / L15** | 0.45 | L5/L3（rd 0.15/0.09） | **+0.583**（p=3e-4） |
| 3B | **L13 / L13** | 0.36 | L2/(L1/L2)（rd 0.06/0.03–0.06） | **+0.506**（p=1.4e-3） |

- AA acc 三规模全层近饱和（≥0.998），身份内容从未丢失（跨规模成立）；margin 最小点 650M L26、3B L36。
- 650M/3B 的 PR 全局最小不在 L0/L1（L15/L13），"layers ≥ 2" 人为下限在这两个规模不影响结论。
- gap 变号边界（650M L5–6、3B L1–5）落在各自 target-val 最优层附近——最优层跟踪的是 gap 变号边界而非 PR 最小点。

## 结论
- **判据不跨规模成立（否定）**："PR 最小 ≈ 最优层"仅 150M 成立；650M/3B 的 PR 最小点（rd 0.45/0.36）
  明显深于 target-val 最优层（rd ≤0.15/≤0.06），且 ρ(PR, gap_H7) 符号跨规模翻转（150M −0.57 vs 650M/3B +0.58/+0.51）。
- 论文 Fig 5d 与 §3.4 的 150M 限定**必须保留**，并应在 Supplementary 给出本两条曲线 + 否定句；
  label-free 判据表述收缩为"仅 150M 观察"。
- 产物：`output/v1_scale_650m.json`、`output/v1_scale_3b.json`；RESULT.md §4 几何对应条已追加。下一步：论文侧如实回填。
