"""汇总对比所有 Phase 的分类结果。

从各 eval_*.json 文件中读取结果，生成统一对比表。
以 Balanced Accuracy 为主排序指标。
输出 compare_all.json 和终端表格。
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


def fmt(val, decimals=4):
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def extract_results():
    """从所有结果文件中提取扁平化的结果列表。"""
    all_rows = []

    # Phase 2: Linear v2
    clf_v2 = load_json(OUT_DIR / "eval_interval_clf_v2.json")
    if clf_v2:
        for emb_name, emb_results in clf_v2.items():
            for cfg_label, cfg_result in emb_results.items():
                r = cfg_result["test"]
                all_rows.append({
                    "phase": "Phase2_Linear",
                    "model": f"{emb_name} {cfg_label}",
                    "acc": r["acc"],
                    "balanced_acc": r["balanced_acc"],
                    "macro_f1": r["macro_f1"],
                })

    # Phase 3: Ordinal
    ordinal = load_json(OUT_DIR / "eval_interval_ordinal.json")
    if ordinal:
        for emb_name, result in ordinal.items():
            r = result["test"]
            all_rows.append({
                "phase": "Phase3_Ordinal",
                "model": f"{emb_name} LogisticAT",
                "acc": r["acc"],
                "balanced_acc": r["balanced_acc"],
                "macro_f1": r["macro_f1"],
            })

    # Phase 4: Nonlinear
    nonlinear = load_json(OUT_DIR / "eval_interval_nonlinear.json")
    if nonlinear:
        for emb_name, emb_results in nonlinear.items():
            for model_name, result in emb_results.items():
                r = result["test"]
                all_rows.append({
                    "phase": "Phase4_Nonlinear",
                    "model": f"{emb_name} {model_name}",
                    "acc": r["acc"],
                    "balanced_acc": r["balanced_acc"],
                    "macro_f1": r["macro_f1"],
                })

    # Phase 5: Ensemble
    ensemble = load_json(OUT_DIR / "eval_interval_ensemble.json")
    if ensemble:
        for method, result in ensemble.items():
            r = result["test"]
            all_rows.append({
                "phase": "Phase5_Ensemble",
                "model": method,
                "acc": r["acc"],
                "balanced_acc": r["balanced_acc"],
                "macro_f1": r["macro_f1"],
            })

    # Baseline: majority class
    # 从 interval_data_info.json 获取类别分布
    info = load_json(OUT_DIR / "interval_data_info.json")
    if info:
        class_dist = info.get("class_dist", {})
        majority_pct = max(class_dist.values()) / info["n_test"] if class_dist else 0.0
        n_classes = len(INTERVAL_LABELS)
        all_rows.append({
            "phase": "Baseline",
            "model": "majority_class",
            "acc": round(majority_pct, 4),
            "balanced_acc": 1.0 / n_classes,
            "macro_f1": None,
        })
        all_rows.append({
            "phase": "Baseline",
            "model": "random",
            "acc": 1.0 / n_classes,
            "balanced_acc": 1.0 / n_classes,
            "macro_f1": 1.0 / n_classes,
        })

    return all_rows


def main():
    print(f"{'='*80}")
    print(f"  Interval 分类 — 全方法汇总对比（3-class, clamped days）")
    print(f"  Classes: {INTERVAL_LABELS}")
    print(f"{'='*80}")

    rows = extract_results()

    if not rows:
        print("\n  ⚠ 未找到任何结果文件，请先运行 Phase 2-5。")
        return

    # 按 balanced_acc 排序
    rows.sort(key=lambda x: x.get("balanced_acc", 0) or 0, reverse=True)

    # 打印表格
    print(f"\n  {'Phase':<18} {'Model':<35} {'Acc':>8} {'Bal Acc':>9} {'Macro F1':>9}")
    print(f"  {'-'*85}")

    for row in rows:
        phase = row["phase"]
        model = row["model"]
        acc = fmt(row["acc"])
        bal = fmt(row["balanced_acc"])
        f1 = fmt(row["macro_f1"]) if row["macro_f1"] is not None else "—"

        # 高亮最佳 balanced_acc
        marker = ""
        if rows and row["balanced_acc"] == rows[0].get("balanced_acc"):
            marker = " ★"

        print(f"  {phase:<18} {model:<35} {acc:>8} {bal:>9} {f1:>9}{marker}")

    # ── 保存 ──
    summary = {
        "task": "interval_3class_clamped",
        "classes": INTERVAL_LABELS,
        "n_classes": len(INTERVAL_LABELS),
        "label_source": "iso_interval_days (clamped, neg→0)" if USE_CLAMPED_DAYS else "jump_interval_cat",
        "results": [],
    }

    for row in rows:
        entry = {k: v for k, v in row.items()}
        summary["results"].append(entry)

    out_path = OUT_DIR / "compare_all.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
