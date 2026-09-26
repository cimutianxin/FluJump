"""JumpScorer 实验 — Ranking loss 训练，Spearman ρ 评估。

对每个 (embedding × hidden_dims × MSE权重) 训练 JumpScorer，
用 pairwise MarginRankingLoss 优化秩排序，输出 eval_jumpscore.json。

2026-08-01：数据加载改为共享 loader（accession 对齐版），
消除 split / embedding 位置错位 bug。
"""

import numpy as np
import json
import sys
import torch

sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *
from ESM_clf.interval_exp.interval_data import load_interval_data
from ESM_clf.interval_exp.jumpscore_model import train_jumpscore, evaluate_jumpscore
from sklearn.preprocessing import StandardScaler


def load_data():
    data = load_interval_data(verbose=True)
    return data["masks"], data["subtypes"], data["X"], data["y_days"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    masks, subtypes_arr, emb_dict, y_reg = load_data()

    print(f"\n{'='*60}")
    print(f"  JumpScorer — Ranking 训练")
    print(f"  Target: iso_interval_days (clamp: <0 → 0，0=最高危)")
    for s in ["train", "val", "test"]:
        m = masks[s]
        n_zero = (y_reg[m] == 0).sum()
        print(
            f"  {s}: n={m.sum():>4}  "
            f"=0:{n_zero:>4}  "
            f">0:{m.sum()-n_zero:>4}  "
            f"mean={y_reg[m].mean():.0f}"
        )
    print(f"{'='*60}")

    HIDDEN_CONFIGS = [[256, 128], [128], [512, 256]]
    SEEDS = [42, 123, 456]
    MSE_WEIGHTS = [0.0, 0.1]

    all_results = {}

    for emb_name in EMB_TYPES:
        if emb_dict[emb_name] is None:
            continue
        X = emb_dict[emb_name]
        dim = X.shape[1]
        print(f"\n{'='*40}")
        print(f"  {emb_name} (dim={dim})")
        print(f"{'='*40}")

        for hd in HIDDEN_CONFIGS:
            for mse_w in MSE_WEIGHTS:
                hd_key = "x".join(str(d) for d in hd)
                mse_tag = f"_mse{mse_w}" if mse_w > 0 else ""
                key = f"{emb_name}_h{hd_key}{mse_tag}"

                seed_results = []
                label = f"ranking{' + MSE×'+str(mse_w) if mse_w>0 else ''}"
                print(f"  [{label}  hidden={hd}]")

                for seed in SEEDS:
                    print(f"    seed={seed}  ", end="", flush=True)

                    scl = StandardScaler()
                    Xt = scl.fit_transform(X[masks["train"]])
                    Xv = scl.transform(X[masks["val"]])
                    Xx = scl.transform(X[masks["test"]])

                    model, history, best_val = train_jumpscore(
                        Xt, y_reg[masks["train"]],
                        Xv, y_reg[masks["val"]],
                        in_dim=dim, hidden_dims=hd,
                        dropout=MLP_DROPOUT, lr=MLP_LR,
                        weight_decay=MLP_WEIGHT_DECAY,
                        epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE,
                        patience=MLP_EARLY_STOP_PATIENCE,
                        use_mse_weight=mse_w, seed=seed, device=device,
                    )

                    eval_res = evaluate_jumpscore(model, Xx, y_reg[masks["test"]], device=device)
                    eval_res["epochs_run"] = len(history["train_loss"])
                    eval_res["best_val_loss"] = float(best_val)
                    eval_res["seed"] = seed
                    seed_results.append(eval_res)

                    print(
                        f"ρ={eval_res['spearman_r']:.4f}  "
                        f"τ={eval_res['kendall_tau']:.4f}  "
                        f"r={eval_res['pearson_r']:.4f}  "
                        f"ep={eval_res['epochs_run']}"
                    )

                avg = {}
                for mk in ["spearman_r", "kendall_tau", "pearson_r"]:
                    vals = [s[mk] for s in seed_results]
                    avg[mk] = float(np.mean(vals))
                    avg[f"{mk}_std"] = float(np.std(vals))
                avg["n"] = seed_results[0]["n"]

                all_results[key] = {
                    "config": {"hidden_dims": hd, "mse_weight": mse_w, "seeds": SEEDS},
                    "test_avg": avg,
                    "test_per_seed": seed_results,
                }

                print(
                    f"    => avg  ρ={avg['spearman_r']:.4f}±{avg.get('spearman_r_std',0):.4f}  "
                    f"τ={avg['kendall_tau']:.4f}±{avg.get('kendall_tau_std',0):.4f}  "
                    f"r={avg['pearson_r']:.4f}"
                )

    out_path = OUT_DIR / "eval_jumpscore.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")

    best_key = max(all_results, key=lambda k: all_results[k]["test_avg"]["spearman_r"])
    best = all_results[best_key]["test_avg"]
    print(f"\n最佳: {best_key}")
    print(f"  Spearman ρ = {best['spearman_r']:.4f}")
    print(f"  Kendall τ  = {best['kendall_tau']:.4f}")
    print(f"  Pearson r  = {best['pearson_r']:.4f}")


if __name__ == "__main__":
    main()
