# 实验：fig4_boundary panel (e) 1HGG 结构映射数据 + 渲染

- **日期**：10-03-23
- **目的**：把 H7/H10 反转的共享核心驱动列映射到 1HGG（X-31 H3 HA）结构，
  导出逐列着色数据并渲染侧面 + 面向 RBS 俯视两个视角（论文 fig4 panel e）

## 方法
- 新建 `validation_exp/reversal_sites/export_structure_colors.py`：
  - 数据源：g3_column_contrib_label_is_jump_L28.csv（d_pos_H7/d_pos_H10）、
    g4_shared_driver_sets.json（core = 6 格 ≥2 共享，17 列）、
    site_attribution/output/col_to_h3.json（1HGG 验证映射；HA1 1-329 / HA2 重起 1-221）
  - col_to_h3 与 g3 CSV 自带 h3 字段交叉核验：550 列全部一致
  - region 优先级：RBS_130loop/190helix/220loop（MARKER_SITES 口径）>
    antigenic_site（Wilson & Cox 经典 H3 A-E 表）> HA2_stem > other
  - **sanity check（渲染前置）：共享核心列 0 个落 RBS ✓**（与 notes/09-19-18 一致）
  - 同脚本生成 `panele_1hgg_render.pml`（单一数据源，残基列表不手抄）
- 渲染：新建独立 `pymol` conda 环境（pymol-open-source 3.1.0，未动 env1/borkenhagen）；
  本地 PDB `validation_exp/reversal_sites/data/1HGG.pdb`（A/C/E=HA1 1-328，
  B/D/F=HA2 1-175，canonical 226=A 链 LEU 验证一致）；核心列白→红渐变
  （键 = 推负强度 −d_pos_mean），RBS 蓝色 sticks 参照；2400×2400、300 dpi、ray

## 结果
- `figdata/fig4_boundary/structure_colors.csv`（1039 行：aln_col, h3_num, chain,
  d_pos_H7, d_pos_H10, d_pos_mean, in_shared_core, region）
- 核心 17 列 region 分布：antigenic_site 4（HA1_124/129 Ag-A、196 Ag-B、312 Ag-C）、
  HA2_stem 1（HA2_83）、other 12；**0 落 RBS**
- PNG（2400×2400, 300 dpi, ray；不入 git，带外传输）：
  - `figdata/fig4_boundary/assets/panele_1hgg_side.png`（侧面：头部在上，RBS 蓝）
  - `figdata/fig4_boundary/assets/panele_1hgg_top.png`（面向 RBS 俯视：三 RBS 绕轴可见）
- 视角迭代记录：初版侧面误为轴向、俯视转向杆端且背景透明；改为
  orient→turn y 90→turn x -90（侧面）/ 再 turn x +90（俯视）+ ray_opaque_background=on

## 结论
- 达到预期：sanity check 通过，两视角渲染完成
- 下一步：PNG 由用户 scp 到本地 figs/fig4_boundary/assets/
