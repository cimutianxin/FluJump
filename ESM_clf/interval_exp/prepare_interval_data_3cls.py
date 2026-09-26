"""准备 3 类 interval 预测数据（薄封装，新建文件，不动 4 类资产）。

3 类标签（human_first 与 <1yr 合并，见 interval_data.py docstring）：
  0 <1yr (raw_days < 365) / 1 1-3yr / 2 3yr+

调用共享 loader（interval_data.load_interval_data，accession 对齐版），
保存 3 类标签与数据摘要（全部为新文件，4 类 npy/json 保持不变）：
  - labels_interval_cat3.npy      3 类分类标签
  - interval_data_info_3cls.json  数据摘要
"""

import numpy as np
import json
import sys
sys.path.insert(0, ".")
from ESM_clf.interval_exp.config import OUT_DIR
from ESM_clf.interval_exp.interval_data import load_interval_data

INTERVAL3_LABELS = ["<1yr", "1-3yr", "3yr+"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("加载数据（accession 对齐版）...")
    data = load_interval_data(verbose=True)

    y_cat3 = data["y_cat3"]
    masks = data["masks"]

    np.save(OUT_DIR / "labels_interval_cat3.npy", y_cat3)

    print(f"\n分类目标 3 类 ({INTERVAL3_LABELS}):")
    for i, l in enumerate(INTERVAL3_LABELS):
        n = (y_cat3 == i).sum()
        print(f"  class {i} ({l}): {n:>4} ({n/len(y_cat3)*100:.1f}%)")

    print(f"\nSplit × 类别分布:")
    for sp in ["train", "val", "test"]:
        m = masks[sp]
        dist = {INTERVAL3_LABELS[i]: int((y_cat3[m] == i).sum())
                for i in range(len(INTERVAL3_LABELS))}
        print(f"  {sp}: n={m.sum():>4}  {dist}")

    info = {
        "n_total": int(len(y_cat3)),
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "class_dist": {INTERVAL3_LABELS[i]: int((y_cat3 == i).sum())
                       for i in range(len(INTERVAL3_LABELS))},
        "split_class_dist": {
            sp: {INTERVAL3_LABELS[i]: int((y_cat3[masks[sp]] == i).sum())
                 for i in range(len(INTERVAL3_LABELS))}
            for sp in ["train", "val", "test"]
        },
        "interval3_labels": INTERVAL3_LABELS,
        "label_source": "raw iso_interval 重算（未 clamp），human_first 并入 <1yr",
        "alignment": "accession-based merge + accession-based embedding slicing",
    }
    with open(OUT_DIR / "interval_data_info_3cls.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nDone — 3 类数据准备完成 → {OUT_DIR}")


if __name__ == "__main__":
    main()
