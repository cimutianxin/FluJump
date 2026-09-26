#!/usr/bin/env python3
"""FluJump 单株风险打分 workflow（主流程）

输入单条 HA 序列 + 亚型 → ESM-2 150M L3(=idx28) mean-pooled embedding →
Ridge LR probe logit → train 分布分位数 + 三级风险分层（高/中/低）。

分层阈值（预注册，仅用 ≤2025 train logits，前瞻数据不参与定界）：
  高 = logit ≥ train 阳性中位数；中 = logit ≥ train 阴性 P95；低 = 其余。
亚型域：H7/H10 = 方向反转区（RESULT.md §3），打分不适用；非 H1/H3/H5 附未验证警告。

probe 加载即确定性重训（不落盘权重，~1 min），H5 复现 AUC 断言作回归保护。
运行须离线模式：HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1。

用法：
  python mainpipeline/risk_scorer.py --selftest          # 序列↔emb 路径一致性自测
  python mainpipeline/risk_scorer.py --seq <HA序列> --subtype H5
  from mainpipeline.risk_scorer import RiskScorer        # 程序化调用
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

SPLIT_CSV = Path("data/splits/isolate_split.csv")
MEAN_EMB_L3 = Path("ESM_clf/jump_exp/output/esm_emb_150M_L3.npy")  # (11060,640) 行序=split
LABELS = {c: Path(f"ESM_clf/jump_exp/output/labels_{c}.npy")
          for c in ["label_is_jump", "label_is_jump_human"]}
ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
RIDGE_C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CV_FOLDS = 5
RANDOM_SEED = 42

BOUNDARY_SUBTYPES = {"H7", "H10"}       # 方向反转区（§3 失效边界）
VALID_SUBTYPES = {"H1", "H3", "H5"}     # 已验证域（train=H1H3，H5 外推验证）
NEG_Q = 0.95                            # 中风险下界 = train 阴性 P95
PRIMARY = "label_is_jump_human"         # 主判定标签（跨物种到人）


class RiskScorer:
    """加载即重训 probe；阈值/分位数全部由 train logits 预先固定。"""

    def __init__(self, load_model=True, device=None, quiet=False):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        split = pd.read_csv(SPLIT_CSV)
        X = np.load(MEAN_EMB_L3).astype(np.float32)
        assert len(X) == len(split), "embedding 行数与 isolate_split 不一致"
        tr = (split["split"] == "train").to_numpy()
        h5 = (split["split"] == "h5_holdout").to_numpy()

        self.probes, self.thresholds, self.train_logits = {}, {}, {}
        for label, lp in LABELS.items():
            y = np.load(lp)
            scl = StandardScaler().fit(X[tr])
            gs = GridSearchCV(LogisticRegression(max_iter=2000),
                              {"C": RIDGE_C_VALUES},
                              cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                                 random_state=RANDOM_SEED),
                              scoring="roc_auc", n_jobs=-1)
            gs.fit(scl.transform(X[tr]), y[tr])
            clf = gs.best_estimator_
            auc_h5 = roc_auc_score(y[h5], clf.decision_function(
                scl.transform(X[h5])))
            assert 0.65 < auc_h5 < 0.95, f"{label} H5 复现异常: {auc_h5}"
            tl = clf.decision_function(scl.transform(X[tr]))
            thr_hi = float(np.median(tl[y[tr] == 1]))
            thr_mid = float(np.quantile(tl[y[tr] == 0], NEG_Q))
            assert thr_hi > thr_mid, (f"{label} 阈值退化：阳性中位 {thr_hi:.2f} "
                                      f"<= 阴性 P95 {thr_mid:.2f}")
            self.probes[label] = (scl, clf)
            self.thresholds[label] = {"high": thr_hi, "mid": thr_mid}
            self.train_logits[label] = np.sort(tl)
            if not quiet:
                print(f"[{label}] C={gs.best_params_['C']} "
                      f"H5复现AUC={auc_h5:.4f} 阈值 高≥{thr_hi:.2f} 中≥{thr_mid:.2f}",
                      flush=True)

        self._model = self._tok = None
        if load_model:
            import torch
            from transformers import AutoTokenizer, EsmModel
            self._torch = torch
            self._device = device or ("cuda" if torch.cuda.is_available()
                                      else "cpu")
            self._tok = AutoTokenizer.from_pretrained(ESM_MODEL)
            self._model = (EsmModel.from_pretrained(
                ESM_MODEL, output_hidden_states=True)
                .to(self._device).eval())

    # ── 打分 ──

    def _tier(self, logit, label):
        t = self.thresholds[label]
        return "高" if logit >= t["high"] else ("中" if logit >= t["mid"] else "低")

    def score_embedding(self, emb, subtype):
        """emb: (640,) L3 mean-pooled embedding。返回分层结果字典。"""
        out = {"subtype": subtype,
               "domain": ("out_of_boundary" if subtype in BOUNDARY_SUBTYPES
                          else "ok" if subtype in VALID_SUBTYPES
                          else "unvalidated")}
        for label, (scl, clf) in self.probes.items():
            lg = float(clf.decision_function(scl.transform(emb[None, :]))[0])
            pct = float(np.searchsorted(self.train_logits[label], lg)
                        / len(self.train_logits[label]))
            tag = label.replace("label_is_", "")
            out[f"logit_{tag}"] = round(lg, 4)
            out[f"trainpct_{tag}"] = round(pct, 4)
            out[f"tier_{tag}"] = self._tier(lg, label)
        out["tier"] = out["tier_jump_human"]      # 主判定
        if out["domain"] == "out_of_boundary":
            out["note"] = "H7/H10 方向反转区（§3 失效边界），打分不适用"
        elif out["domain"] == "unvalidated":
            out["note"] = "未验证亚型，谨慎解读"
        return out

    def embed_sequence(self, seq):
        """单条 raw HA 序列 → (640,) L3 mean-pooled（无 padding，EOS 不混入）。"""
        assert self._model is not None, "load_model=False 时无 sequence 路径"
        seq = seq.strip().upper()
        bad = set(seq) - set("ACDEFGHIKLMNPQRSTVWY")
        assert not bad, f"非法氨基酸字符: {bad}"
        assert 300 <= len(seq) <= 700, f"HA 长度异常: {len(seq)}"
        torch = self._torch
        inp = self._tok([seq], return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inp)
        h = out.hidden_states[-3][0, 1:-1]              # L28 = 主线 L3
        mask = inp["attention_mask"][0, 1:-1].bool()
        return h[mask].mean(dim=0).cpu().numpy().astype(np.float32)

    def score_sequence(self, seq, subtype):
        return self.score_embedding(self.embed_sequence(seq), subtype)


def _selftest():
    """5 条 2026 序列：sequence 路径 vs 库存 emb 路径（logit 差 + 分层一致）。"""
    c26 = pd.read_csv("data/processed_isolate/2026_isolates_clean.csv")
    E26 = np.load("validation_exp/temporal_validation/output/emb_2026_L3.npy")
    sc = RiskScorer(load_model=True)
    for i in [0, 50, 100, 150, 200]:
        row = c26.iloc[i]
        r_seq = sc.score_sequence(row["ha_sequence"], row["subtype"])
        r_emb = sc.score_embedding(E26[i], row["subtype"])
        d = abs(r_seq["logit_jump_human"] - r_emb["logit_jump_human"])
        ok = d < 0.05 and r_seq["tier"] == r_emb["tier"]
        print(f"  [{row['accession']}] {row['subtype']} "
              f"logit_seq={r_seq['logit_jump_human']:.3f} "
              f"logit_emb={r_emb['logit_jump_human']:.3f} Δ={d:.4f} "
              f"tier={r_seq['tier']} {'OK' if ok else 'FAIL'}")
        assert ok, f"自测试失败于 {row['accession']}"
    print("selftest PASS（注：库存 emb 为 batched 提取，短序列混入 EOS，"
          "单条路径更干净，Δ<0.05 即一致）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FluJump 单株风险打分")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seq", help="单条 HA 蛋白序列")
    ap.add_argument("--subtype", help="亚型，如 H5")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.seq and args.subtype:
        import json
        sc = RiskScorer(load_model=True)
        print(json.dumps(sc.score_sequence(args.seq, args.subtype),
                         ensure_ascii=False, indent=2))
    else:
        ap.print_help()
