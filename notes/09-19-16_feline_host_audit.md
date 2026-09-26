# 实验：host_category 子串误判排查与根因修复（B5）

- **日期**：09-19-16
- **目的**：RESULT.md TODO-B5——09-11 发现 H10_143 簇因 `/cat/` 子串误判 feline，排查主数据（H1/H3/H5/H7）同源 bug 并固化修复。

## 方法

- 定位：`download_flu.py` / `clean_data.py` / `supplement_h5.py` 的 ENGLISH/HOST_RULES 裸子串匹配（`cat`/`pig`/`cow`/`air`/`water` 误伤 furcata、flycatcher、pigeon、Moscow、Cairo、shearwater）。
- 发现今晨另一会话已 ad-hoc 修复并重跑全链（clean 13:18 → labels → isolate → aligned 13:19，留存 `host_audit_fix_diff.csv` 49 行），但脚本未固化。本实验反推其规则并脚本化：
  1. ENGLISH_RULES 改 `\b` 词边界（允许复数 `s?`）；avian 补词 waterbird/waterfowl/bluebird/flycatcher/guineafowl；属名补 tyto/cairina；strain `/cat/`→`/cats?/`、`/bluebird/`；
  2. 垃圾 species（`Influenza A virus (A/...)` 串位、"Environmental" 字面值）提取内嵌株名再推断；
  3. 覆盖逻辑 = improve-only（新推断非 unknown 才覆盖）+ 垃圾行允许降级 unknown。
- 验证：临时目录重跑四亚型 clean，与现行表逐行全列 diff；`audit_host_label_impact.py` 还原 49 行旧值重算簇标签对比现行 `all_subtypes_simplified.csv`。

## 结果

- **复现验证通过**：修复后 `clean_data.py` 重跑与现行 clean CSV **全列 diff = 0**（11,714 行），ad-hoc 修复完全脚本化。
- **主数据零 feline 残留**（四亚型 clean feline = 0；raw 中 3 条误判行 Tyto furcata ×2 / flycatcher ×1 现为 avian）。
- **标签影响**：7 个簇 `label_is_jump` 1→0（H5: 73/117/123/252，H7: 80/81/107——第二宿主均为误判的 environmental 鸟类行，修正后纯 avian）；`label_is_jump_human` **0 翻转**。涉及 isolate 41 行，全部位于 H5/H7 holdout（29/12），train/val/test 不受影响。
- **关键簇安全**：H7N9 七簇未被触及；(H5,70) 含 1 审计行但标签不变（human|avian|bovine 不依赖该行）；(H1,15) 未被触及。
- 解读：今日 B4 加固中 H5 jump 复现 0.8255（vs 08-28 的 0.798）的上移，主要来自本次修复移除的 29 条 H5 holdout 假阳性（jump_human 标签未变，复现 0.7936 vs 0.794 一致，互为印证）。
- 历史 headline（§1/§3 的 H5/H7 jump AUC）算自修复前标签：41/11,060 行（0.37%）假阳性已移除，方向性结论不受影响；RESULT.md 局限性行已披露。

## 涉及文件

- 修改：`datascripts/clean_data.py`（词边界 + 垃圾 species + improve-only）、`datascripts/download_flu.py`、`datascripts/supplement_h5.py`（同规则同步）
- 新增：`datascripts/audit_host_label_impact.py` → `data/processed_isolate/host_audit_label_impact.json`
- 文档：`datascripts/README.md`、`AGENTS.md` §4.3、`RESULT.md` TODO-5/§3 脚注

## 结论

- 达到预期：根因修复固化且精确复现现行数据；影响面量化（7 簇 jump 翻转、41 isolate 行、0 jump_human 翻转、关键簇安全）。
- 下一步：B5 完成。剩余 B6（H7/H10 位点对比）/ B7（多年份前向）。
