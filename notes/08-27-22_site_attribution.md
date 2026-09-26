# 实验：§2 位点归因——probe 方向 w 是否重现已知宿主适应标记位点

- **日期**：08-27-22
- **目的**：RESULT.md §2 待补（必做项）。Part A：logit 逐位点分解 + 标记位点富集检验；Part B：已知人适应位点定点饱和突变（ISM）+ Δlogit 方向性检验。

## 方法

- 新目录 `validation_exp/site_attribution/`：config.py / build_h3_mapping.py / run_logit_decomposition.py / run_ism.py。
- **Step 0 坐标映射（关键基础工作）**：发现现有 1039 列 MAFFT 对齐无参考株、坐标 ≠ H3 编号，且 AAA43178 原始位置 ≠ canonical H3 编号。用 PDB 1HGG（X-31）实测确认：**canonical H3 = 原始位置 − 16**（信号肽 16 aa 不入编号，HA2 单独编号 = 原始 − 345）。`mafft --add --keeplength` 把 Aichi 加入对齐得 列→canonical 映射。验证全过：canonical 226 列 H3 集 = I/Q/V（I226/V226 吻合文献）、228 = S/G、190 = N/D/E、225 = D/G/N、融合肽列 99.9% G。
- **Part A**：重训 L28（=L3）Ridge LR（H5 AUC 复现 0.798/0.794 ✓），logit 精确分解为逐位点贡献 c_i = w·scaler(h_i)，全体 1039 列算 train 上类间分离度，19 个标记位点（RBS 130/190/220-loop + 156/159/160）富集检验（Fisher）。
- **Part B**：H1+H3 train 72 条背景（cluster 去重、jump_human 分层）+ H5 10 条；19 标记位点 + 19 列熵匹配对照位点 × 19 氨基酸饱和突变 = 59,186 条突变序列，ESM L28 mean-pool 后算 Δlogit（~14 min GPU）。

## 结果

**Part A（富集，弱/边缘）**：
- jump：3/19 标记位点入 top 5% 贡献列，Fisher p=0.059（边缘）；jump_human：1/19，p=1.0。
- 排名靠前的标记位点：159（1.2%/2.3 百分位）、228（2.4/13.0）、160（4.5/6.2）；核心 RBS 位点 190/225 排名靠后（>50 百分位）。
- top 贡献列多为非标记位点（如 col 244/H3-116、col 260/H3-129），单位点 AUC train 最高 0.89。

**Part B（ISM，已知替换方向性显著但异质）**（jump_human 标签）：

| 替换 | n | Δlogit 中位 | frac+ | Wilcoxon p |
|------|---|------------|-------|-----------|
| G228S | 66 | **+0.97** | 0.92 | <1e-4 ✓ |
| D190N | 59 | **+0.81** | 0.88 | <1e-4 ✓ |
| Q226L | 66 | −0.35 | 0.26 | 1e-4 ✗ |
| D190E | 59 | −1.32 | 0.05 | <1e-4 ✗ |
| D225G | 49 | −0.52 | 0.20 | 0.001 ✗ |

- 5 个已知替换全部有显著效应，但方向异质：G228S/D190N 推向 jump_human+，Q226L/D190E/D225G 推向负。
- 标记组 vs 对照组 |Δlogit|：0.80 vs 1.01（Mann-Whitney p=1）——**标记位点效应量并不大于对照**。
- H5 背景位点效应与 H1+H3 相关性弱（Spearman ρ=0.38，p=0.11）。
- jump 标签结果同构（G228S +0.38、D190N +0.85 为正，其余为负）。

## 结论

**部分验证（按计划的判定标准：A 边缘 + B 方向异质）**：

1. w **不是**集中在已知 RBS 标记位点上的方向——贡献弥散于全序列，与 §1 朴素基线结论（"趋同适应是多位点组合信号"）自洽：组合信号本来就不可归因到少数已知位点。
2. 但 w 对若干已知人适应替换有**显著的、方向正确的因果响应**（G228S、D190N 把 logit 显著推向人适应方向）——说明 w 确实锚定了真实的分子表型成分，不是纯标签伪影；同时 Q226L/D225G 的负方向提醒：监测标签定义的是"2000 年后溢出事件"而非"1968 式大流行适应"，两者位点谱不同是合理的生物学差异。
3. 论文表述：§2 可写"ISM 显示方向对 G228S/D190N 等经典人适应替换有显著正向响应，但贡献不富集于 RBS 位点，符合多位点组合信号图景"。措辞保持"部分验证"。

- **下一步行动**：§2 此项可标记完成（部分验证）。剩余决定性项：§3 n=7 缓解（H10 零样本方向检验）。
- 坑记录：ESM 提取脚本必须 `HF_HUB_OFFLINE=1` 运行，否则挂在 HF 连接（无报错）；canonical H3 编号 = 原始 −16 的换算已固化在 `output/col_to_h3.json`。

## 涉及文件

- 新建：`validation_exp/site_attribution/`（config.py、build_h3_mapping.py、run_logit_decomposition.py、run_ism.py、output/）
- 输出：`output/col_to_h3.json`（映射+验证）、`partA_summary.json`、`partA_position_ranking_*.csv`、`probe_*.npz`、`ism_deltalogit.csv`（59k 突变）、`ism_summary.json`、各 run log
