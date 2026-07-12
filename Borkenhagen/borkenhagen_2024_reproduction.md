## Borkenhagen & Runstadler (2024) — Reproduction Guide

**Paper**: Borkenhagen LK, Runstadler JA. "Examining the Influenza A Virus Sialic Acid Binding Preference Predictions of a Sequence-Based Convolutional Neural Network." *Influenza and Other Respiratory Viruses* 18 (2024): e70044. DOI: 10.1111/irv.70044

**Task**: Binary classification — predict whether an IAV HA protein prefers α2,6-linked sialic acid receptors (human-type), given the HA amino acid sequence.

**Code**: Models and code publicly available at [Bitbucket](https://bitbucket.org/borkenhagen-workspace/influenza_binding). Data S1 (sequences) and Data S2 (labels) in supplementary materials.

---

### 1. DATA

#### 1.1 Sequence Sources
| Source | Type | Count |
|---|---|---|
| In-house (Runstadler lab) | Active virus isolates | 186 |
| In-house | Inactive virus | 1 |
| In-house | Isolated HA protein | 134 |
| In-house | Pseudovirus (lentiviral) | 37 |
| CFG (Consortium for Functional Glycomics) | Glycan array binding profiles | 209 (with complete HA seq) |
| IRD | HA sequences for host pre-training | 42,416 |
| GISAID | HA sequences for host pre-training | 50,241 |

#### 1.2 Subtypes Covered
H1–H16, covering all sialic acid-binding clades (Group 1: H1/H2/H5/H6/H8/H9/H11/H12/H13/H16; Group 2: H3/H4/H7/H10/H14/H15). H17–H19 used only for novel subtype testing.

#### 1.3 Label Definition (Critical Step)
1. From CFG glycan arrays (versions 1–5.2, 200–609 glycans): identify **dominant binders** — glycans with signal ≥ 10% of max affinity for that virus/protein (Grant et al. 2016 method)
2. Pair sialylated glycans differing only in terminal sialic acid conformation (α2,3 vs α2,6)
3. Welch's t-test (p < 0.05) for each glycan pair
4. Label = 1 ("prefers α2,6") if significant difference in favor of α2,6 across **all** dominant-binder pairs; otherwise = 0
5. In-house samples: modified hemagglutination assay (see below)

#### 1.4 Dataset Split
| Set | N | Composition |
|---|---|---|
| Pre-training (host model) | 42,416 + 50,241 | IRD + GISAID, labeled as avian vs mammalian |
| Binding training | 358 | 277 α2,3-preferring / 81 α2,6-preferring |
| Binding validation (internal) | 5% of binding training | Used for early stopping |
| Independent test | 48 | H1–H16, 3 each, balanced labels (39 α2,3 / 9 α2,6) |

---

### 2. LABEL ACQUISITION (for in-house data)

#### 2.1 Modified Hemagglutination Assay
**Materials**: Turkey RBCs (Lampire), V. cholerae neuraminidase Type III, ST3Gal6, ST6Gal1, CMP-sialic acid.

**Protocol**:
1. Desialylate 62.5 μL 20% RBCs with 12.5 mU neuraminidase, 1 h, 37°C
2. Resialylate selectively with:
   - 4.4 μg ST3Gal6 + 1.5 mM CMP-SA → α2,3-RBC
   - 2.7 μg ST6Gal1 + 1.5 mM CMP-SA → α2,6-RBC
   - 1% BSA in PBS, total 75 μL, 2 h, 37°C
3. Wash, resuspend at 0.5% in PBS + 1% BSA
4. Serially dilute virus/pseudovirus (2-fold) in PBS
5. Mix equal volumes with 0.5% treated/untreated RBCs
6. Incubate RT, 30 min
7. Read hemagglutination by eye, all samples in duplicate

Controls: untreated RBC (sialylated control), desialylated-only RBC (negative control).

#### 2.2 Pseudovirus Generation (when only RNA/plasmid available)
- **Plasmid**: pcDNA3.1(+) expressing HA + HIV Gag-GFP plasmid
- **Cells**: HEK293T, DMEM + 10% FBS + 1% P/S + ≤2.5 μg/mL amphotericin B
- **Transfection**: Lipofectamine 3000 at 60% confluence
- **Harvest**: 48–72 h post-transfection, treat with 7 mU/mL neuraminidase at 24–48 h to release, ultracentrifuge at 77,000 g × 2 h with 20% sucrose cushion, resuspend in 250 μL PBS

---

### 3. DATA ENCODING & PREPROCESSING

1. **MAFFT alignment**: Heterosubtypic alignment using Burke & Smith (2014) numbering scheme — ensures all HA sequences aligned to standardized position coordinates (H1–H16 have different lengths natively)
2. **Jalview**: Manual alignment refinement
3. **One-hot encoding**: sklearn `OneHotEncoder`. Each position → binary indicator for presence/absence of each of 20 AA + gap. Result: alignment_length × 21 binary matrix per sequence

---

### 4. CNN ARCHITECTURE (adapted from Scarafoni et al. 2019)

**Framework**: Keras (TensorFlow backend)  
**Reference implementation**: Scarafoni D, Telfer BA, Ricke DO, Thornton JR, Comolli J. *Health Security* 17(6):468–476, 2019.

#### 4.1 Layer Structure
| Layer | Type | Details |
|---|---|---|
| 1–5 | Conv1D | 5 convolutional layers, ReLU activation |
| After each conv | MaxPool1D | Kernel size & stride per Table S3 |
| After each pool | BatchNorm | Standardize + rescale |
| — | Flatten | Collapse spatial dims |
| Dense 1 | Fully connected | Units per Table S3 |
| Dropout | 0.5 | Before final layer only |
| Dense 2 | Fully connected | 1 unit, sigmoid → α2,6 probability |

**Note**: Exact filter counts, kernel sizes, strides, and dense layer units are in **Table S3** of the paper (not listed in-text). The text describes architecture qualitatively only. Check supplementary Table S3 or the authors' Bitbucket repo for exact layer dimensions.

FROM SCARAFONI ORIGINAL (likely adapted): Conv filters likely increase per layer (e.g., 32→64→128→256→512), kernel size = 3 or 5, pool size = 2 with stride = 2. Verify against Table S3.

#### 4.2 Two-Stage Transfer Learning

**Stage 1 — Host Origin Pre-training**:
| Parameter | Value |
|---|---|
| Data | IRD + GISAID HA sequences (avian vs mammalian) |
| Batch size | 128 |
| Loss | Binary cross-entropy |
| Optimizer | Adam |
| Dropout | 0.5 (before final dense layer) |
| Class weighting | Yes (class + subtype weight biased) |
| Early stopping | 20% held-out validation, monitor peak performance |
| Validation AUC | 0.99 |
| Validation Accuracy | 0.99 |

**Stage 2 — Binding Specificity Fine-tuning**:
| Parameter | Value |
|---|---|
| Frozen layers | All conv + maxpool layers (weights from Stage 1) |
| Trainable layers | New fully connected layers only |
| Training data | 358 binding-labeled samples |
| Batch size | **32** (optimal from grid search {8, 16, 32, 64}) |
| Learning rate (Adam) | **0.01** (optimal from grid search {0.1, 0.01, 0.001, 0.0001}) |
| Class weighting | Yes (class + subtype) |
| Early stopping | 5% held-out validation set |
| Epochs | Until early stop triggers |

---

### 5. EVALUATION

#### 5.1 Main Results (Independent Test, n=48)
| Metric | Value |
|---|---|
| Overall Accuracy | **94%** |
| Overall AUC | **0.93** |
| False positives (α2,6 predicted but not observed) | **0** |
| False negatives (3 cases) | H2N3 (swine), H4N6 (swine), H13N2 (gull) |
| Group 1 HA accuracy | 93% (AUC 0.96) |
| Group 2 HA accuracy | 94% (AUC 0.81) |

#### 5.2 Robustness Tests
| Test | Result |
|---|---|
| Missing data tolerance | ≤40 AA missing (~7% HA) → AUC/Acc > 0.9; ≤80 AA (~14%) → still above random |
| Missing position effect | Missing at HA termini → minimal impact; missing at RBS → sharp performance drop |
| Novel subtype (H16 excluded from training) | H16 prediction dipped; also affected H3N8 and H9N2 predictions → cross-subtype feature sharing |
| H17/H18/H19 prediction | Generated (no ground truth available) |

#### 5.3 SHAP Analysis
- **Method**: SHAP values computed on independent test set, absolute mean per position
- **Key finding**: Highest SHAP at position 226 (known receptor-binding determinant)
- SHAP localized primarily to HA1 (head domain) over HA2 (stalk)
- Per-sequence SHAP used to inform site-directed mutagenesis targets for H9 and H16

#### 5.4 Experimental Validation
- **H9N2 (A/Guinea Fowl/HK/WF10/1999)**: L226Q mutation abolished α2,6 binding (reverted to α2,3). Single mutations insufficient for binding switch — suggests combinatorial requirement.
- **H16N3 (A/shorebird/Delaware/172/2006)**: K160A eliminated both α2,3 and α2,6 binding but retained sialylated RBC agglutination → possible α2,8 shift. Additive mutations: G159S + G222K + S228G (with K160A) progressively reduced binding.

---

### 6. KEY DIFFERENCES FROM YOUR WORK

| Aspect | Borkenhagen 2024 | Your Work |
|---|---|---|
| Input | HA AA seq (one-hot) | HA AA seq (ESM-2 embedding) |
| Feature learning | 5-layer CNN from scratch | Pre-trained ESM-2 (150M/650M params) |
| Training paradigm | Two-stage transfer (host → binding) | Fine-tune PLM → classification head |
| Task | Binary (α2,6 preference) | Multi-label binary (host spectrum) |
| Label type | Experimental binding assay | Database host annotation + literature |
| Alignment required | Yes (MAFFT + Burke-Smith) | No (ESM-2 accepts raw sequences) |
| Interpretability | SHAP on one-hot features | SHAP/attention on embeddings |

---

### 7. REPRODUCTION CHECKLIST

- [ ] Obtain CFG glycan array data + in-house binding assay data, or request from authors
- [ ] Download HA AA sequences from IRD + GISAID for host pre-training
- [ ] Align all sequences with MAFFT using Burke & Smith numbering scheme
- [ ] One-hot encode aligned sequences (sklearn)
- [ ] Train Stage 1 CNN (5 conv layers) on host origin task (batch=128, Adam, BCE, dropout=0.5, early stopping 20%)
- [ ] Freeze conv+maxpool weights
- [ ] Train Stage 2 with binding data (batch=32, lr=0.01, Adam, BCE, class+subtype weighting, early stopping 5%)
- [ ] Evaluate on 48-sequence test set (H1–H16, 3 each)
- [ ] Compute SHAP for amino acid importance
- [ ] (Optional) Run missing data robustness test (random gaps of 10–320 AA, 100 bootstraps)
- [ ] (Optional) Train without H16 to test novel subtype generalizability
