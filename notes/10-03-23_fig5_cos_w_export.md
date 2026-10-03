# 实验：fig5_depth 150M cos(w_best, w_last) 导出

- **日期**：10-03-23
- **目的**：导出 150M 模型 best 层（target-val 选择：jump→L17、jump_human→L13）与
  末层 L30 的 probe 权重余弦，供论文 fig5「读出近正交旋转」证据

## 方法
- 新建 `validation_exp/depth_reversal/export_cos_w_150m.py`
- 主拟合：直接读 `output/g1_probes.npz`（g1_layer_geometry.py 所存逐层 probe）
- 多种子：seed{42..46} 按 g1 同一 fit_probe 协议（StandardScaler + GridSearchCV
  5-fold，C 网格同）在 best/last 层重训，逐 seed 算 cos(w_best, w_last)

## 结果
- label_is_jump（L17 vs L30）：主拟合 cos=+0.0483；5 seed mean=+0.0315，
  abs_max=0.0483
- label_is_jump_human（L13 vs L30）：主拟合 cos=+0.0426；5 seed mean=+0.0346，
  abs_max=0.0426
- 主拟合 cos 与 seed=42 重训值逐一相等（交叉验证协议一致 ✓）
- **超出 notes/09-26-16 预期（-0.005/+0.015，|cos|≤0.02），如实报告**：notes 记录
  的口径是 cos(w_17, w_28)；本次按任务指定 last=L30，得 |cos|≈0.04–0.05。
  量级结论不变：best 与 last 层读出方向仍近正交（cos≈0）
- 产物：`figdata/fig5_depth/cos_w_best_w_last.json`

## 结论
- 达到预期（附偏差报告）：L28 口径 |cos|≤0.02 → L30 口径 |cos|≤0.05，
  「近正交旋转」结论对 last 层定义稳健
- 下一步：继续（绘图用导出 json；正文若引用具体数值需统一 last=L30 口径）
