# 实验：ESM_tf_clf — 冻结 ESM per-residue embedding + Transformer 头 vs Linear probe

- **日期**：08-09-19
- **目的**：把 probe 头从"mean-pooled + Ridge LR"换成"per-residue embedding → 2 层 TransformerEncoder → attention pooling → FFN"，检验序列结构信息能否改善 H5/H7 跨亚型迁移。

## 方法

- 新目录 `ESM_tf_clf/`：`extract_residue_embeddings.py`（候选层 {13,17,28}，fp16 memmap 11060×573×640，行序经余弦抽查 =1.0000 验证）、`model.py`（2.19M 参数）、`train_tf_probe.py`。
- 训练：BCE + Adam（lr 1e-4），early stop 只用 H1+H3 val AUC；每 (label, layer) 3 种子 logit 集成。H1+H3 test sanity：0.983–0.991（与 linear 0.98 同量级，训练无问题）。
- 评估：与 linear 实验完全同一 target-val 协议（cluster/isolate 两臂 × 5 seeds，val argmax 选层 → test 评一次，cluster_seed42 bootstrap B=1000）。
- 首次运行踩坑：transformers 5.13 加载模型时向 HF hub 探测 `adapter_config.json`，网络不可达报错；模型缓存完整，加 `HF_HUB_OFFLINE=1` 解决。

## 结果

### cluster 臂 test AUC（5 seeds）vs linear

| 标签 | 亚型 | linear | transformer 头 | Δ |
|---|---|---|---|---|
| jump | H5 | 0.786 ± 0.040 | 0.598 ± 0.154 | **−0.19** |
| jump | H7 | 0.892 ± 0.027 | 0.768 ± 0.011 | **−0.12** |
| jump_human | H5 | 0.688 ± 0.103 | 0.617 ± 0.073 | −0.07 |
| jump_human | H7 | 0.854 ± 0.060 | 0.439 ± 0.094 | **−0.42** |

### cluster_seed42 test bootstrap（B=1000）

| 标签 | 亚型 | 层 | AUC | P(>0.5) |
|---|---|---|---|---|
| jump | H5 | L13 | 0.295 [0.234, 0.385] | 0.00（方向反） |
| jump | H7 | L17 | 0.770 [0.406, 0.852] | 0.86 |
| jump_human | H5 | L28 | 0.697 [0.585, 0.790] | 1.00 |
| jump_human | H7 | L13 | 0.507 [0.341, 0.655] | 0.51（随机水平） |

泄露差值（isolate − cluster）：+0.013 ~ +0.081，比 linear（≤0.016，除 jump_human H5）更大——复杂 head 对近重复序列过拟合更重。

## 结论

1. **Transformer 头全面不如 linear probe**，尤其 H7 jump_human 从 0.854 崩到 0.439（bootstrap P=0.51，等于随机）——同分布（H1+H3 test 0.98–0.99）学得很好，但学到的是训练分布内的信号，跨亚型不可迁移。
2. 这是"冻结 embedding 上换复杂 head"的**第 4 次失败**（MLP、elmo_mix、LoRA 之后）：证据链完整——在 ~6k 训练样本上，head 容量越大越拟合训练分布，跨亚型泛化越差。linear probe 的 L2 约束反而是优势。
3. 下一步行动：**放弃**复杂 head 路线；论文主线维持 Ridge probe，本节作为对比实验写入（head 复杂度 vs 迁移性能的消融）。

## 涉及文件

- 新建 `ESM_tf_clf/`（config / extract_residue_embeddings / model / train_tf_probe）
- 输出 `ESM_tf_clf/output/`：residue_emb_L{13,17,28}.npy、residue_lens.npy、tf_probe_results.{json,log}、extract.log
- 修改 `AGENTS.md`（结构 + 命令）
