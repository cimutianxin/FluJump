from pathlib import Path

BORK_CSV = Path("data/dataset_borkenhagen/borkenhagen_clean.csv")
SPLIT_CSV = Path("data/dataset_borkenhagen/borkenhagen_split.csv")

# ESM-2 150M (emb=640)
ESM_MODEL = "facebook/esm2_t30_150M_UR50D"

RIDGE_C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CV_FOLDS = 5
RANDOM_SEED = 42
TRAIN_SIZE = 358

OUT_DIR = Path("ESM_clf/binding_exp/output")
EMB_FILE = OUT_DIR / "esm_embeddings_150M.npy"
LABEL_FILE = OUT_DIR / "labels.npy"
