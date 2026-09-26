# 实验：单株风险分层 workflow + 2026 前瞻集验证（TODO-8）

- **日期**：09-19-19
- **目的**：把 probe 从"批次内排序"扩展为"单株判定"产品——端到端 workflow（序列 → ESM-2 L3 → logit → train 分位数 → 高/中/低三级分层），阈值仅用 ≤2025 train 固定，2026 前瞻集（201 条，10 阳性/2 簇）做 prospective 验证。

## 方法

- **W1 `mainpipeline/risk_scorer.py`**（mainpipeline 首个主流程代码）：加载即确定性重训双 probe（不落盘），H5 复现断言（0.65–0.95）；阈值预注册：高 = train 阳性 logit 中位数，中 = train 阴性 P95；亚型域判定（H7/H10 → 边界外不适用；非 H1/H3/H5 → 未验证警告）；输出 logit + train 分位数 + 分层，主判定 = jump_human。
- **W2 `run_2026_workflow_eval.py`**：201 条走 emb 路径分层；与 `forward_2026_ranking.csv` logits 交叉核对；朴素基线（max identity to train 人源参考，自匹配剔除）同规则分层对照。

## 结果

probe 重训：jump/jump_human H5 复现 AUC 0.8255/0.7936（断言通过）；阈值 jump 高≥2.13/中≥−2.36，jump_human 高≥2.19/中≥−2.85。logits 交叉核对 ρ=1.0、Δmax=5e-05。

**2026 前瞻分层（probe）**：

| 标签 | 中及以上命中 | precision | recall | FPR | 高 |
|---|---|---|---|---|---|
| jump | 10 条（9 真阳） | 0.90 | 9/10 | 0.0052 | 1 条（真阳） |
| jump_human | 8 条（全真阳） | 1.00 | 8/10 | 0.0 | 0 条 |

**朴素基线（对照）**：阈值退化（train 阴性 P95=1.0 ≥ 阳性中位 0.998，人类季节性株自相似封顶）→ 高层 79/50 条全为阴性、0/10 阳性命中（FPR 0.41/0.26）——同规则下基线完全反向失效。

**关键发现（诚实披露）**：2026 阳性的 train 分位数中位仅 0.93（jump）/0.88（jump_human），**够不到 train 阳性中位数定义的高层**——train 上回测校准的绝对阈值对新暴发株系统性偏严（分布外漂移），高层前瞻几乎不触发；中界（阴性 P95）才是有效预警线。此现象与 §5 "批次内排序有效"的叙事自洽：绝对水平不可移植，相对排序可移植。

自测试：5 条序列 sequence↔emb 路径 logit Δ≤0.036、分层全一致（库存 emb 为 batched 提取，短序列混 EOS 的已知微差）。

## 涉及文件

- 新建 `mainpipeline/risk_scorer.py`（mainpipeline 启用）
- 新建 `validation_exp/temporal_validation/run_2026_workflow_eval.py`
- 产出 `output/workflow_2026_eval.json` + `workflow_2026_tiers.csv`

## 结论

**达到预期（附重要 caveat）**：workflow 交付且前瞻验证阳性——中及以上分层 jump_human 8/10 命中零误报、jump 9/10 命中 1 误报；朴素基线同规则下退化且反向。但"高 = train 阳性中位"层前瞻不触发，论文须表述为"分层预警（中界有效）+ 分位数读数"，不可声称绝对高风险判定。

**下一步行动**：RESULT.md §5 加确定结果并勾掉 TODO-8。开放项：高层阈值的分布漂移校正（如按 outbreak 簇重校准）留作未来工作。
