"""用 Borkenhagen CNN 预测 CFG 结合偏好"""

import torch, numpy as np, pandas as pd, json, sys
sys.path.insert(0, "Borkenhagen/cnn_baseline")
from config import *
from model import BorkenhagenCNN
from sklearn.metrics import roc_auc_score

# ── 加载对齐 ──
seqs = {}
with open("data/dataset_borkenhagen/borkenhagen_aligned.faa") as f:
    name, chars = None, []
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if name: seqs[name] = "".join(chars)
            name = line[1:]; chars = []
        else: chars.append(line)
    if name: seqs[name] = "".join(chars)

h3_ref_595 = [v for k, v in seqs.items() if "YEV" in k][0]
bk = {k: v for k, v in seqs.items() if "YEV" not in k}
print(f"borkenhagen: {len(bk)} seqs")

# ── H3 参考 581 ──
ref = pd.read_csv("data/processed_isolate_MAFFT/all_isolates_aligned.csv")
h3_581 = ref[ref["subtype"] == "H3"].iloc[0]["aligned_ha_seq"]

# ── 595→581 映射 ──
i5, i8 = 0, 0
m = {}
while i5 < len(h3_ref_595) and i8 < len(h3_581):
    if h3_ref_595[i5] != "-" and h3_581[i8] != "-":
        m[i5] = i8; i5 += 1; i8 += 1
    elif h3_ref_595[i5] == "-" and h3_581[i8] == "-":
        m[i5] = i8; i5 += 1; i8 += 1
    elif h3_ref_595[i5] == "-":
        m[i5] = i8; i5 += 1
    elif h3_581[i8] == "-":
        i8 += 1
    else:
        i5 += 1; i8 += 1

# ── 标签 ──
lab = pd.read_csv("data/dataset_borkenhagen/borkenhagen_clean.csv")
name2label = dict(zip(lab["strain_name"], lab["binding_label"]))
name2subtype = dict(zip(lab["strain_name"], lab["ha_subtype"]))

# ── 构建 ──
def to581(s595):
    r = ["-"] * 581
    for i5, aa in enumerate(s595):
        if i5 in m and aa != "-":
            r[m[i5]] = aa
    return "".join(r)

X, Y, names, subtypes = [], [], [], []
for fa_name, s595 in bk.items():
    strain = fa_name.split("|")[0]
    if strain not in name2label:
        continue
    s581 = to581(s595)
    idxs = [AA_TO_IDX.get(aa, AA_TO_IDX["-"]) for aa in s581]
    oh = np.eye(N_AMINO_ACIDS, dtype=np.float32)[idxs].T
    X.append(oh); Y.append(name2label[strain])
    names.append(strain); subtypes.append(name2subtype[strain])

print(f"有效样本: {len(X)}, pos={sum(Y)} ({sum(Y)/len(Y)*100:.1f}%)")

# ── 推理 ──
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

for task in ["label_is_jump", "label_is_jump_human"]:
    ckpt = MODEL_DIR / f"stage2_{task}_best.pt"
    if not ckpt.exists():
        continue

    model = BorkenhagenCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    preds = []
    with torch.no_grad():
        for oh in X:
            x = torch.from_numpy(oh).unsqueeze(0).to(device)
            preds.append(torch.sigmoid(model(x)).item())

    p, l = np.array(preds), np.array(Y)
    if len(np.unique(l)) < 2:
        print(f"\n{task}: 标签单一, 无法计算 AUC")
        continue
    auc = roc_auc_score(l, p)
    print(f"\n{task}: AUC={auc:.4f}, pred_mean={p.mean():.3f}")

    # per subtype
    for st in ["H1", "H3", "H5", "H7"]:
        mask = np.array(subtypes) == st
        if mask.sum() >= 5 and len(np.unique(l[mask])) > 1:
            print(f"  {st} (n={mask.sum()}): AUC={roc_auc_score(l[mask], p[mask]):.4f}")

    # all subtypes combined
    st_aucs = {}
    for st in sorted(set(subtypes)):
        mask = np.array(subtypes) == st
        if mask.sum() >= 5 and len(np.unique(l[mask])) > 1:
            st_aucs[st] = roc_auc_score(l[mask], p[mask])
    print(f"  所有亚型 AUC: {st_aucs}")

    # save
    json.dump({"auc": auc, "preds": p.tolist(), "labels": l.tolist(),
               "names": names, "subtypes": subtypes},
              open(OUTPUT_DIR / f"borkenhagen_{task}.json", "w"), indent=2)

print(f"\n✓ 完成")
