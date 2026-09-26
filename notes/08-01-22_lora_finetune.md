# 实验：LoRA 微调 ESM-2 150M——能否矫正 H7 方向

- **日期**：08-01-22
- **目的**：冻结 embedding 的所有修复（erasure/加权/训练组合/层选择）都无法同时保住 H5 和 H7，检验放开表征层（LoRA 微调 + 分类头）能否重塑出跨亚型一致的 jump 方向

## 方法

- `finetune_lora.py`：ESM-2 150M + LoRA（r=8, α=16, target q/k/v）+ mean-pool 线性头；BCEWithLogitsLoss（pos_weight 类别均衡），AdamW lr=1e-4，4 epoch，val AUC 选最佳。
- 4 配置：H1+H3 / H1+H3+H5 × jump / jump_human。报 test/H5/H7 **原始方向** AUC。
- 输出：`output/finetune_lora.json`、`output/finetune_lora.log`。

## 结果

| run | best_val | test | H5 | H7（原始方向） |
|---|---|---|---|---|
| H1+H3, jump | 0.966 | 0.968 | 0.466 | 0.453 |
| H1+H3, jump_human | 0.973 | 0.977 | 0.755 | 0.420 |
| H1+H3+H5, jump | 0.963 | 0.967 | 0.917* | 0.292 |
| H1+H3+H5, jump_human | 0.969 | 0.975 | 0.947* | **0.125** |

\* H5 在训练集内，非 holdout 迁移。

## 结论

- **未达到预期**：4 个配置的 H7 AUC 全部 ≤ 0.45，jump_human + H5 联合训练甚至反转得更彻底（0.125）。微调没有重塑出跨亚型一致的方向——模型在训练分布内越学越好（val 0.96–0.97），H7 反向越固化。
- 附带：微调后 test AUC（0.97）反而略低于冻结 probe（0.99）。
- 注意是单 seed、单 LoRA 配置的筛查结果；但结合冻结探针的全部阴性证据，可以下结论：**H7（尤其 jump_human）的反方向不是容量/架构问题，而是 H7 的 jump 信号与 H1/H3/H5 在序列-表征层面本身反向**——按预案转向"方向按目标亚型选择"的框架，而非继续堆架构。
- 下一步行动：**放弃**继续微调路线；最终方案 = 分亚型层选择（H5 用 L28，H7 jump_human 用 L13，bootstrap 已验证稳健）。
