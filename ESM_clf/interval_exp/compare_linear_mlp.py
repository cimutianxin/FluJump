"""汇总对比 Linear (Ridge/Lasso) vs MLP 的 Interval 预测结果。

读取已有的 eval_interval_clf.json / eval_interval_reg.json
和新建的 eval_interval_mlp_clf.json / eval_interval_mlp_reg.json，
生成对比表和汇总 JSON。
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import *


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fmt_metric(val, std=None, decimals=4):
    """格式化指标 ± std。"""
    if isinstance(val, float):
        s = f"{val:.{decimals}f}"
    else:
        s = str(val)
    if std is not None and isinstance(std, (int, float)):
        s += f" ±{std:.{decimals}f}"
    return s


def print_table(title, headers, rows):
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"{'─'*80}")
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
                  for i, h in enumerate(headers)]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"  {header_line}")
    print(f"  {'-' * sum(col_widths)}")
    for row in rows:
        line = "".join(str(c).ljust(w) for c, w in zip(row, col_widths))
        print(f"  {line}")


def main():
    linear_clf = load_json(OUT_DIR / "eval_interval_clf.json")
    linear_reg = load_json(OUT_DIR / "eval_interval_reg.json")
    mlp_clf = load_json(OUT_DIR / "eval_interval_mlp_clf.json")
    mlp_reg = load_json(OUT_DIR / "eval_interval_mlp_reg.json")

    # ── 分类对比 ──
    if linear_clf and mlp_clf:
        rows = []
        for emb in EMB_TYPES:
            # Linear
            if emb in linear_clf:
                lr = linear_clf[emb]["test"]
                rows.append([f"{emb} (Linear)", fmt_metric(lr["acc"]),
                             fmt_metric(lr["balanced_acc"]),
                             fmt_metric(lr["macro_f1"])])

            # MLP: 取 best hidden_dims (按 macro_f1)
            mlp_keys = [k for k in mlp_clf if k.startswith(emb)]
            best_key = None
            best_f1 = -1
            for mk in mlp_keys:
                if "test_avg" in mlp_clf[mk]:
                    f1 = mlp_clf[mk]["test_avg"]["macro_f1"]
                    if f1 > best_f1:
                        best_f1 = f1
                        best_key = mk
            if best_key:
                mr = mlp_clf[best_key]["test_avg"]
                rows.append([f"{best_key} (MLP)",
                             fmt_metric(mr["acc"], mr.get("acc_std")),
                             fmt_metric(mr["balanced_acc"], mr.get("balanced_acc_std")),
                             fmt_metric(mr["macro_f1"], mr.get("macro_f1_std"))])

        print_table("分类对比 (Linear vs MLP)",
                    ["Model", "Test Acc", "Balanced Acc", "Macro F1"], rows)

    # ── 回归对比 ──
    if linear_reg and mlp_reg:
        rows = []
        for emb in EMB_TYPES:
            # Linear (Ridge)
            lkey = f"{emb}_Ridge"
            if lkey in linear_reg:
                lr = linear_reg[lkey]["test"]
                rows.append([f"{emb} (Ridge)", fmt_metric(lr["r2"], decimals=4),
                             fmt_metric(lr["pearson_r"], decimals=4),
                             fmt_metric(lr["mae"], decimals=0),
                             fmt_metric(lr["rmse"], decimals=0)])

            # MLP: 取 best hidden_dims (按 R²)
            mlp_keys = [k for k in mlp_reg if k.startswith(emb)]
            best_key = None
            best_r2 = -999
            for mk in mlp_keys:
                if "test_avg" in mlp_reg[mk]:
                    r2 = mlp_reg[mk]["test_avg"]["r2"]
                    if r2 > best_r2:
                        best_r2 = r2
                        best_key = mk
            if best_key:
                mr = mlp_reg[best_key]["test_avg"]
                rows.append([f"{best_key} (MLP)",
                             fmt_metric(mr["r2"], mr.get("r2_std"), decimals=4),
                             fmt_metric(mr["pearson_r"], mr.get("pearson_r_std"), decimals=4),
                             fmt_metric(mr["mae"], mr.get("mae_std"), decimals=0),
                             fmt_metric(mr["rmse"], mr.get("rmse_std"), decimals=0)])

        print_table("回归对比 (Ridge vs MLP)",
                    ["Model", "R²", "Pearson r", "MAE", "RMSE"], rows)

    # ── 保存汇总 JSON ──
    summary = {
        "classification": {
            "linear": {emb: linear_clf[emb]["test"] for emb in EMB_TYPES
                       if linear_clf and emb in linear_clf},
            "mlp_best": {},
        },
        "regression": {
            "linear_ridge": {f"{emb}_Ridge": linear_reg[f"{emb}_Ridge"]["test"]
                             for emb in EMB_TYPES
                             if linear_reg and f"{emb}_Ridge" in linear_reg},
            "mlp_best": {},
        }
    }

    if mlp_clf:
        for emb in EMB_TYPES:
            mlp_keys = [k for k in mlp_clf if k.startswith(emb) and "test_avg" in mlp_clf[k]]
            if mlp_keys:
                best_key = max(mlp_keys, key=lambda k: mlp_clf[k]["test_avg"]["macro_f1"])
                summary["classification"]["mlp_best"][best_key] = mlp_clf[best_key]

    if mlp_reg:
        for emb in EMB_TYPES:
            mlp_keys = [k for k in mlp_reg if k.startswith(emb) and "test_avg" in mlp_reg[k]]
            if mlp_keys:
                best_key = max(mlp_keys, key=lambda k: mlp_reg[k]["test_avg"]["r2"])
                summary["regression"]["mlp_best"][best_key] = mlp_reg[best_key]

    out_path = OUT_DIR / "compare_linear_vs_mlp.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n汇总已保存: {out_path}")


if __name__ == "__main__":
    main()
