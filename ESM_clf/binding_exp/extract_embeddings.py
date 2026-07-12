"""ESM-2 多层 embedding — 用 EsmModel (无 LM head) + L1+L3 concat"""
import numpy as np, pandas as pd, torch, sys
sys.path.insert(0, ".")
from ESM_clf.binding_exp.config import *
from transformers import AutoTokenizer, EsmModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

df = pd.read_csv(BORK_CSV)
df["raw_seq"] = df["ha_sequence"].str.replace("-", "", regex=False)
df["strat_key"] = df["ha_subtype"] + "_" + df["binding_label"].astype(str)
vc = df["strat_key"].value_counts()
if vc[vc < 2].any():
    df.loc[df["strat_key"].isin(vc[vc < 2].index), "strat_key"] = "RARE"

train_idx, test_idx = train_test_split(
    df.index, train_size=TRAIN_SIZE, random_state=RANDOM_SEED, stratify=df["strat_key"]
)
df["split"] = "test"; df.loc[train_idx, "split"] = "train"
df[["strain_name","ha_subtype","binding_label","split"]].to_csv(SPLIT_CSV, index=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True).to(device).eval()

L1, L3, L13 = [], [], []

for i in tqdm(range(0, len(df), 4)):
    batch = df["raw_seq"].iloc[i:i+4].tolist()
    inp = tokenizer(batch, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inp)
    for j in range(len(batch)):
        mask = inp["attention_mask"][j, 1:-1].bool()
        h1 = out.hidden_states[-1][j, 1:-1][mask].mean(dim=0)
        h3 = out.hidden_states[-3][j, 1:-1][mask].mean(dim=0)
        L1.append(h1.cpu().numpy())
        L3.append(h3.cpu().numpy())
        L13.append(torch.cat([h1, h3]).cpu().numpy())

OUT_DIR.mkdir(parents=True, exist_ok=True)
for name, arr in [("L1", L1), ("L3", L3), ("L1L3", L13)]:
    fname = OUT_DIR / f"esm_emb_150M_esmmodel_{name}.npy"
    np.save(fname, np.stack(arr))
    print(f"  {fname}: {np.stack(arr).shape}")

np.save(LABEL_FILE, df["binding_label"].values)
print("Done")
