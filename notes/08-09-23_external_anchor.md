# 实验：外部锚定 — jump 方向 vs Borkenhagen 402 条实验 binding 数据

- **日期**：08-09-23
- **目的**：回应"jump 标签是监测数据代理表型"的审稿质疑——用实验测定的 HA 唾液酸结合偏好（α2,6 人型受体）做外部验证。

## 方法

- `reviewer_exp/external_binding/`（不动 `ESM_clf/binding_exp/`）：
  - `extract_all_layers_binding.py`：402 条序列全 31 层 mean-pooled embedding（去 gap，同 binding_exp 处理）
  - `run_anchor.py` 两部分：(1) binding 全层扫描（沿用 borkenhagen_split.csv train=358/test）；(2) H1+H3 train 训好的 jump/jump_human probe（L13/17/28）直接对 402 条序列**零样本**打分，对实验 binding 标签算 AUC
- 输出：`output/external_anchor_results.json`

## 结果

### Part 1：binding 全层扫描

- 各层 AUC 平坦于 0.94–0.97（L10–L30）；anchor 层排名：L28 第 8、L13 第 17、L17 第 18 / 共 31 层
- 即：binding 表型在层间无特异性，jump 方向的层特异性**不是** binding 信号的副产品

### Part 2：jump probe → binding 零样本迁移（AUC）

| probe | L13 | L17 | L28 |
|---|---|---|---|
| jump（全部 402 条） | 0.702 | 0.708 | 0.646 |
| jump（仅 test 44 条） | 0.722 | 0.760 | 0.708 |
| jump_human（全部） | 0.731 | **0.756** | 0.700 |
| jump_human（仅 test） | 0.747 | **0.785** | 0.757 |

## 结论

1. **方向锚定成立**：监测标签训出的 jump/jump_human 方向对实验 binding 表型的零样本排序全部 >0.5（0.65–0.79），且 jump_human > jump，与"jump_human 更接近人型适应"的生物学预期一致。
2. 强度中等（0.7–0.8 而非 >0.9）是合理的：binding 只是人型适应的一个组分，jump 标签还包含其他宿主适应维度——论文中应表述为"部分验证"而非完全等价。
3. L17（H7 jump 的 val 选择层）在锚定中同样最强，交叉佐证了 val 选层的生物学合理性。
4. 下一步行动：**继续**——写入论文外部验证一节；表述注意"锚定的是方向而非数值"。

## 涉及文件

- 新建 `reviewer_exp/external_binding/`（config / extract_all_layers_binding / run_anchor）
- 输出 `reviewer_exp/external_binding/output/`（binding_emb_all_layers.npy、binding_labels.npy、external_anchor_results.json、extract.log、anchor.log）
