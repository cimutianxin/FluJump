"""fig4_boundary panel (e)：1HGG 结构映射数据导出 + PyMOL 渲染脚本生成

数据源：
  - g3_column_contrib_label_is_jump_L28.csv：d_pos_H7 / d_pos_H10（L28 逐列推负贡献）
  - g4_shared_driver_sets.json：core = 6 格（双标签 × L13/17/28）中 ≥2 格共享的驱动列
  - col_to_h3.json：aln_col → canonical H3 编号（1HGG 验证，HA1 1-329 / HA2 1-221 重起）

region 优先级：RBS_130loop/RBS_190helix/RBS_220loop（MARKER_SITES 口径，同
site_attribution）> antigenic_site（经典 Wilson & Cox H3 抗原位点 A-E）>
HA2_stem（HA2 链全部）> other。

sanity check（渲染前必须过）：共享核心列 0 个落 RBS（论文结论，notes/09-19-18）。

输出：
  figdata/fig4_boundary/structure_colors.csv
  figdata/fig4_boundary/panele_1hgg_render.pml（PyMOL 渲染脚本，PNG 不入 git）
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

G3_CSV = Path("validation_exp/reversal_sites/output/g3_column_contrib_label_is_jump_L28.csv")
G4_JSON = Path("validation_exp/reversal_sites/output/g4_shared_driver_sets.json")
COL_TO_H3 = Path("validation_exp/site_attribution/output/col_to_h3.json")
PDB_PATH = Path("validation_exp/reversal_sites/data/1HGG.pdb")
FIG4_DIR = Path("figdata/fig4_boundary")
PNG_DIR = FIG4_DIR / "assets"          # PNG 不 commit，仅带外传输

# ── RBS（MARKER_SITES 口径，canonical H3 HA1 编号）──
RBS = {
    "RBS_130loop": [134, 135, 136, 137, 138],
    "RBS_190helix": [186, 190, 193, 194],
    "RBS_220loop": [221, 222, 224, 225, 226, 227, 228],
}
# ── 经典 H3 抗原位点（Wilson & Cox 1990 表，canonical 编号）──
ANTIGENIC = {
    "A": [121, 122, 124, 126, 129, 131, 132, 133, 135, 137, 138, 140,
          142, 143, 144, 145, 146, 147],
    "B": [128, 155, 156, 157, 158, 159, 160, 163, 164, 165, 186, 187, 188,
          189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199],
    "C": [44, 45, 48, 49, 50, 51, 53, 54, 273, 275, 276, 278, 279, 290,
          291, 294, 295, 297, 305, 307, 308, 310, 311, 312],
    "D": [89, 96, 102, 104, 172, 173, 174, 175, 176, 177, 179, 182, 201,
          202, 203, 204, 205, 206, 207, 208, 209, 212, 213, 214, 215, 216,
          217, 218, 219, 220, 226, 227, 229, 230, 231, 238, 240, 242, 243,
          244, 245, 246, 248],
    "E": [57, 62, 63, 67, 75, 78, 80, 81, 82, 83, 90, 91, 92, 94, 109, 114],
}
ANTIGENIC_ALL = set().union(*[set(v) for v in ANTIGENIC.values()])


def assign_region(chain, h3_num):
    """region 优先级：RBS > antigenic_site > HA2_stem > other"""
    if chain == "HA2":
        return "HA2_stem"
    if chain != "HA1" or h3_num is None:
        return "other"
    for rbs_name, sites in RBS.items():
        if h3_num in sites:
            return rbs_name
    if h3_num in ANTIGENIC_ALL:
        return "antigenic_site"
    return "other"


def main():
    FIG4_DIR.mkdir(parents=True, exist_ok=True)

    # ── aln_col → (chain, h3_num)，并与 g3 CSV 自带 h3 字段交叉核验 ──
    m = json.load(open(COL_TO_H3))
    col_map = {}
    for c, n in m["col_to_h3_ha1"].items():
        col_map[int(c)] = ("HA1", int(n))
    for c, n in m["col_to_h3_ha2"].items():
        col_map[int(c)] = ("HA2", int(n))

    g3 = pd.read_csv(G3_CSV)
    n_checked = 0
    for _, r in g3.iterrows():
        if isinstance(r["h3"], str) and r["h3"]:
            ch, n = col_map[int(r["aln_col"])]
            assert f"{ch}_{n}" == r["h3"], \
                f"aln_col {r['aln_col']}: col_to_h3={ch}_{n} ≠ g3 列 {r['h3']}"
            n_checked += 1
    print(f"✓ col_to_h3 与 g3 CSV h3 字段交叉核验一致（{n_checked} 列）")

    core = set(json.load(open(G4_JSON))["core"])
    print(f"共享核心列 {len(core)} 个: {sorted(core)}")

    # ── 逐列组装 ──
    rows = []
    for _, r in g3.iterrows():
        c = int(r["aln_col"])
        chain, h3_num = col_map.get(c, (None, None))
        d7, d10 = r["d_pos_H7"], r["d_pos_H10"]
        d_mean = float(np.nanmean([d7, d10]))
        rows.append({
            "aln_col": c,
            "h3_num": h3_num if h3_num is not None else "",
            "chain": chain or "",
            "d_pos_H7": d7, "d_pos_H10": d10, "d_pos_mean": d_mean,
            "in_shared_core": int(c in core),
            "region": assign_region(chain, h3_num),
        })
    out = pd.DataFrame(rows)

    # ── sanity check：共享核心列 0 个落 RBS（渲染前置条件）──
    core_df = out[out["in_shared_core"] == 1]
    n_rbs = int(core_df["region"].str.startswith("RBS").sum())
    assert n_rbs == 0, f"共享核心列有 {n_rbs} 个落 RBS，与论文结论矛盾，中止渲染"
    assert core_df["h3_num"].ne("").all(), "核心列存在无 H3 编号映射的对齐列"
    print("✓ sanity check 通过：共享核心列 0 个落 RBS")
    print(core_df[["aln_col", "chain", "h3_num", "d_pos_H7", "d_pos_H10",
                   "d_pos_mean", "region"]].to_string(index=False))

    csv_path = FIG4_DIR / "structure_colors.csv"
    out.to_csv(csv_path, index=False)
    print(f"✓ {csv_path}（{len(out)} 行）")

    # ── 生成 PyMOL 渲染脚本（1HGG：A/C/E=HA1 1-328，B/D/F=HA2 1-175）──
    # 渐变键 = -d_pos_mean（推负强度：白=弱、红=强；d_pos_mean 为负值贡献）
    core_sorted = core_df.copy()
    core_sorted["strength"] = -core_sorted["d_pos_mean"]
    alter_lines = []
    for _, r in core_sorted.iterrows():
        chains = "A+C+E" if r["chain"] == "HA1" else "B+D+F"
        alter_lines.append(
            f"alter (chain {chains}) and resi {int(r['h3_num'])}, "
            f"b={r['strength']:.4f}")
    rbs_resi = "+".join(str(n) for sites in RBS.values() for n in sites)

    pml = f"""# fig4_boundary panel (e)：1HGG 共享核心驱动位点结构映射
# 由 validation_exp/reversal_sites/export_structure_colors.py 生成（勿手改）
# 白→红渐变 = 推负强度（-d_pos_mean）；蓝 = RBS 参照（130-loop/190-helix/220-loop）
# 运行: /root/miniconda3/envs/pymol/bin/pymol -cq panele_1hgg_render.pml

load {PDB_PATH}, hgg
remove hetatm
hide everything, hgg
show cartoon, hgg
set cartoon_fancy_helices, on
color grey85, hgg
bg_color white
set ray_opaque_background, on
set ray_shadows, on
set antialias, 2
set ray_trace_mode, 1

# ── RBS 参照（蓝，sticks）──
select RBS, (chain A+C+E) and (resi {rbs_resi})
show sticks, RBS
color deepblue, RBS
set stick_radius, 0.35, RBS

# ── 共享核心列（白→红渐变，sticks + CA spheres）──
{chr(10).join(alter_lines)}
select CORE, (chain A+C+E) and (resi {"+".join(str(int(r["h3_num"])) for _, r in core_sorted.iterrows() if r["chain"] == "HA1")}) or (chain B+D+F) and (resi {"+".join(str(int(r["h3_num"])) for _, r in core_sorted.iterrows() if r["chain"] == "HA2")})
show sticks, CORE
show spheres, CORE and name CA
set stick_radius, 0.35, CORE
set sphere_scale, 0.6, CORE
spectrum b, white red, CORE

# ── 视角 1：侧面（HA 主轴竖直，头部/RBS 在上）──
orient hgg
turn y, 90
turn x, -90
zoom hgg, 5
png {PNG_DIR}/panele_1hgg_side.png, width=2400, height=2400, dpi=300, ray=1

# ── 视角 2：面向 RBS 俯视（自头部沿主轴向下看；+90 转向头端）──
turn x, 90
zoom hgg, 6
png {PNG_DIR}/panele_1hgg_top.png, width=2400, height=2400, dpi=300, ray=1

quit
"""
    pml_path = FIG4_DIR / "panele_1hgg_render.pml"
    pml_path.write_text(pml)
    print(f"✓ {pml_path}")
    print(f"  渲染: /root/miniconda3/envs/pymol/bin/pymol -cq {pml_path}")


if __name__ == "__main__":
    main()
