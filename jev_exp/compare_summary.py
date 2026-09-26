"""四臂并排对比：linear probe / attention pooling / transformer 头 / JEV 头

只读加载四个结果 JSON，打印 target-val 协议下 cluster / isolate_random 两臂的
test AUC mean±std 对比表，以及 jev 头的 AUPRC/MCC 与 cluster bootstrap CI。

用法：python jev_exp/compare_summary.py
"""

import json
import sys

sys.path.insert(0, ".")
from jev_exp.config import JEV_OUT_DIR, LABEL_COLS

ARMS = ["cluster", "isolate_random"]
SUBTYPES = ["h5", "h7"]

SOURCES = {
    "linear": "ESM_clf/jump_exp/output/target_val_layer_select.json",
    "attnpool": "ESM_tf_clf/output/adaptive_pool_results.json",
    "tf": "ESM_tf_clf/output/tf_probe_results.json",
    "jev": str(JEV_OUT_DIR / "jev_results.json"),
}


def main():
    res = {name: json.load(open(path)) for name, path in SOURCES.items()}

    for arm in ARMS:
        print(f"\n{'='*86}\n{arm} 臂 test AUC（mean ± std，5 seeds，val 选层 → test 评一次）\n{'='*86}")
        head = f"  {'标签':>22} {'亚型':>4}" + "".join(f"{n:>16}" for n in SOURCES)
        print(head)
        for label_col in LABEL_COLS:
            for st in SUBTYPES:
                row = f"  {label_col:>22} {st:>4}"
                for name in SOURCES:
                    s = res[name]["summary"][arm][label_col][st]
                    row += f"{s['test_auc_mean']:>8.3f}±{s['test_auc_std']:<7.3f}"
                print(row)

    print(f"\n{'='*86}\njev 头补充指标（cluster 臂 test）\n{'='*86}")
    for label_col in LABEL_COLS:
        for st in SUBTYPES:
            s = res["jev"]["summary"]["cluster"][label_col][st]
            b = res["jev"]["bootstrap_cluster_seed42"][label_col][st]
            print(f"  {label_col:>22} {st:>4}  AUPRC={s['test_auprc_mean']:.3f}±{s['test_auprc_std']:.3f}"
                  f"  MCC={s['test_mcc_mean']:.3f}±{s['test_mcc_std']:.3f}"
                  f"  bootstrap[{b['p05']:.3f},{b['p95']:.3f}] P(>0.5)={b['p_above_0.5']:.2f}")

    print(f"\nH1+H3 test AUC（sanity，候选层 13/17/28）：")
    for name in SOURCES:
        h = res[name].get("h13_test_per_layer")
        if not h:
            continue
        for label_col in LABEL_COLS:
            if label_col in h:            # {label: {layer: auc}}（tf / jev）
                vals = "  ".join(f"L{k}={v:.3f}" for k, v in sorted(
                    h[label_col].items(), key=lambda kv: int(kv[0])))
            else:                          # {"label|Lx": auc}（adaptive_pool 全层扫描）
                vals = "  ".join(f"L{li}={h[f'{label_col}|L{li}']:.3f}"
                                 for li in (13, 17, 28)
                                 if f"{label_col}|L{li}" in h)
            print(f"  {name:>9} {label_col:>22}: {vals}")


if __name__ == "__main__":
    main()
