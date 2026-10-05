# 服务器任务 S8：reversal_sites 按 S5 修复后口径重跑（2026-10-06，S5 采纳配套）

> 项目根目录 `/root/autodl-tmp/FluJump`，先 `git pull`。用 `/root/miniconda3/envs/env1/bin/python`，
> 加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。完成后写实验日志、更新 RESULT.md §3、git push。

## 背景

S5（`notes/10-06-04_h10h4_label_audit.md`）已把 H10/H4 标签切换到 09-19 词边界口径：
阳性簇 H10 13→11（H10_64/H10_263 转阴）、H4 25→18，jump_human 不变（2/0）。
但 `validation_exp/reversal_sites/` 的 H10/H4 位点剖面仍是旧口径——
`g2_extract_h10h4_contrib.py` 的回归校验基准写死"group_boundary/direction_test.json（09-11，修复前标签）"。
论文已采纳 S5 口径（§3.3 方向检验已改为 20/31、p=0.150、幅度 0.460/0.463、H4 6/31 p=8.8e-4），
位点段（ρ=0.632–0.718 / Jaccard 0.176–0.379 / JS 百分位 52.8–85.4 / H4 对照 0.369–0.421 /
"0 列落 RBS"）是目前唯一还锚在旧口径的论文数字。

## 做法

- 按依赖序重跑 `validation_exp/reversal_sites/`：
  1. `g2_extract_h10h4_contrib.py`（H10/H4 逐列贡献提取；行序对齐 group_boundary 的
     concat(H10, H4) 不变，阳性集按新标签；同时更新脚本里"修复前标签"的基准注释）
  2. `g1_h7_driver_sites.py`（H7 剖面——H7 属主数据、标签未变，理论上应不变；重跑只为口径统一，
     若输出与现有一致直接记录"不变"即可）
  3. `g3_compare_h7_h10.py`（6 个 layer×label 格的 Spearman ρ、top-20 Jaccard、H4/H5 对照）
  4. `g4_orthogonal_evidence.py`（JS 分歧百分位、subtype-identity 对照 probe）
- 报告：新旧对照表（6 格 ρ、Jaccard、共享列 RBS 命中数、JS 百分位、对照 probe ρ、
  结构聚集/糖基化结论是否变化）。
- 同步重打包 `figdata/fig4_boundary/` 的 paneld（位点驱动散点）与 panele（结构映射汇总），
  更新其 README；panelc 已是新口径（S5 已打包），不要动。

## 产物

- `validation_exp/reversal_sites/output/` 更新
- 日志 `notes/MM-DD-HH_reversal_sites_relabel.md`
- RESULT.md §3 位点条目（B6 条）数值刷新
- `figdata/fig4_boundary/` paneld/panele 重打包，git commit && push

## 论文回填位置（本地处理，不用动 tex）

§3.3 位点段（含 TODO(S8) 标记处）、fig:boundary 图注 (d)(e)；Fig 4 整图重新生成。
