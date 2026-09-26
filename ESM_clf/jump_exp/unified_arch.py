"""统一架构实验：层选择只用训练数据，H5/H7 结果才干净

动机：分亚型选层（H5→L28, H7→L13）是用测试标签选超参，论文站不住。
本实验的所有变体都在训练集内确定层的使用方式，H5/H7 只做一次性评估：

  1. avg_all：全部 31 层平均 → 640 维 probe
  2. avg_mid：中层 10–22 平均（事前按"中层可能去亚型化"的假设指定）→ probe
  3. concat_l1：31 层 concat（19840 维）+ L1 正则 probe（训练自己挑层）
  4. elmo_mix：ELMo 式可学习 softmax 层权重 + 线性头（torch，GPU），
     权重只在 train split 上学

输出：output/unified_arch.json
"""

import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from ESM_clf.jump_exp.config import *

MID_LAYERS = list(range(10, 23))  # 事前指定的中层集合
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gs_probe_auc(Xt, yt, Xe, ye, penalty="l2"):
    scl = StandardScaler()
    Xt_s = scl.fit_transform(Xt)
    if penalty == "l1":
        lr = LogisticRegression(penalty="l1", solver="liblinear",
                                max_iter=3000, random_state=RANDOM_SEED)
        c_values = [1e-3, 1e-2, 1e-1, 1.0]
    else:
        lr = LogisticRegression(penalty="l2", solver="lbfgs",
                                max_iter=5000, random_state=RANDOM_SEED)
        c_values = RIDGE_C_VALUES
    gs = GridSearchCV(lr, {"C": c_values},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(Xt_s, yt)
    if len(np.unique(ye)) < 2:
        return None, gs.best_params_["C"]
    auc = float(roc_auc_score(ye, gs.decision_function(scl.transform(Xe))))
    return auc, gs.best_params_["C"]


def elmo_mix(Xt, yt, Xe_dict, n_steps=300, lr=1e-2):
    """ELMo 式 softmax 层权重 + 线性头，全 batch 训练。Xt: (n, 31, 640)"""
    n, L, d = Xt.shape
    w_layer = torch.zeros(L, device=DEVICE, requires_grad=True)
    w_head = torch.zeros(d, device=DEVICE, requires_grad=True)
    b = torch.zeros(1, device=DEVICE, requires_grad=True)
    Xtr = torch.tensor(Xt, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(yt, dtype=torch.float32, device=DEVICE)
    pos_w = torch.tensor([(yt == 0).sum() / max((yt == 1).sum(), 1)], device=DEVICE)
    opt = torch.optim.Adam([w_layer, w_head, b], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    for step in range(n_steps):
        s = torch.softmax(w_layer, dim=0) * L
        emb = torch.einsum("l,nld->nd", s, Xtr)
        # 标准化（训练集统计）
        mu, sd = emb.mean(0, keepdim=True), emb.std(0, keepdim=True) + 1e-6
        logits = ((emb - mu) / sd) @ w_head + b
        loss = loss_fn(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        s_final = torch.softmax(w_layer, dim=0) * L
        out = {"layer_weights": s_final.cpu().numpy().tolist()}
        for name, Xe_ye in Xe_dict.items():
            Xe, ye = Xe_ye
            Xte = torch.tensor(Xe, dtype=torch.float32, device=DEVICE)
            emb_te = torch.einsum("l,nld->nd", s_final, Xte)
            logits = (((emb_te - mu) / sd) @ w_head + b).cpu().numpy()
            out[name] = float(roc_auc_score(ye, logits)) if len(np.unique(ye)) > 1 else None
    return out


def main():
    arr = np.load(OUT_DIR / "esm_emb_150M_all_layers.npy")  # (31, N, 640)
    split_df = pd.read_csv(SPLIT_CSV)
    splits = split_df["split"].values
    m_train, m_test = splits == "train", splits == "test"
    m_h5, m_h7 = splits == "h5_holdout", splits == "h7_holdout"

    results = {}
    for label_col in LABEL_COLS:
        y = np.load(OUT_DIR / f"labels_{label_col}.npy")
        print(f"\n{'='*60}\n  Label: {label_col}\n{'='*60}", flush=True)
        label_res = {}

        feats = {
            "avg_all": arr.mean(axis=0),
            "avg_mid": arr[MID_LAYERS].mean(axis=0),
            "concat_l1": arr.transpose(1, 0, 2).reshape(arr.shape[1], -1),
        }
        for name, X in feats.items():
            pen = "l1" if name == "concat_l1" else "l2"
            res = {}
            for split_name, m_ev in [("test", m_test), ("h5", m_h5), ("h7", m_h7)]:
                auc, best_c = gs_probe_auc(X[m_train], y[m_train], X[m_ev], y[m_ev], pen)
                res[split_name] = auc
                res["best_C"] = best_c
            label_res[name] = res
            print(f"  {name:>10}: test={res['test']:.3f} h5={res['h5']:.3f} "
                  f"h7={res['h7']:.3f} (C={res['best_C']})", flush=True)

        # ELMo 混合（层作为第一维输入）
        Xt = arr[:, m_train, :].transpose(1, 0, 2)
        Xe_dict = {n: (arr[:, m, :].transpose(1, 0, 2), y[m])
                   for n, m in [("test", m_test), ("h5", m_h5), ("h7", m_h7)]}
        mix_res = elmo_mix(Xt, y[m_train], Xe_dict)
        label_res["elmo_mix"] = mix_res
        top3 = np.argsort(mix_res["layer_weights"])[::-1][:3]
        print(f"  {'elmo_mix':>10}: test={mix_res['test']:.3f} h5={mix_res['h5']:.3f} "
              f"h7={mix_res['h7']:.3f}  top层={top3.tolist()}", flush=True)

        results[label_col] = label_res

    out_path = OUT_DIR / "unified_arch.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
