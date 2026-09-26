# 实验：亚型方向擦除 × cluster 加权干预（修复 H7 方向）

- **日期**：08-01-20
- **目的**：在冻结 embedding 上测试两类低成本修复——INLP 亚型方向擦除、cluster 加权——能否矫正 H7 方向

## 方法

- 新脚本 `ESM_clf/jump_exp/train_probe_interventions.py`：
  - erasure：在 H1+H3 train 上迭代（1/3/5 次）训练 H1-vs-H3 线性分类器，将 embedding 投影到其权重方向正交补后重训 probe；
  - cluster 加权：sample_weight = 1/cluster_size；
  - 条件：baseline / cluster-weight / erasure×3 档 / erasure+cw；评 test/H5/H7 原始方向 AUC。
- 输出：`output/interventions.json`。

## 结果（H7 AUC，原始方向）

| 条件 | jump (L3) | jump_human (L3) |
|---|---|---|
| baseline | 0.281 | 0.213 |
| cluster-weight | 0.319 | 0.202 |
| erasure x1 / x3 / x5 | 0.282 / 0.285 / 0.287 | 0.203 / 0.215 / 0.224 |
| erasure+cw | 0.320 | 0.238 |

- L1/L1L3 同样全部 < 0.35；无任何条件把 H7 拉到 0.5 以上。
- 附带发现：cluster 加权 + erasure 对 **H5 jump_human 有小幅提升**（0.79→0.84，L3 cluster-weight）。

## 结论

- **未达到预期**：erasure 几乎不改变 H7 AUC。原因可从机制上解释——INLP 只能擦除训练集内可见的方向（H1 vs H3 身份），而 H7 的偏移方向从未在训练中出现过，"没见过的东西擦不掉"。H7 反转不是 H1/H3 身份泄漏，而是 H7 特有的宿主特征沿 -w 偏移。
- cluster 加权同样无效，说明大谱系主导不是主因。
- 低成本修复路线中，目前**唯一有效的是 H5 联合训练**（矩阵实验：H1+H3+H5→H7 jump=0.636，但 jump_human=0.226 仍反转）。
- 下一步行动：**继续**——层扫描（全层 embedding 提取中），之后 LoRA 微调。
