## Alberts et al. (2024) — Reproduction Guide

**Paper**: Alberts F, Berke O, Maboni G, Petukhova T, Poljak Z. "Utilizing machine learning and hemagglutinin sequences to identify likely hosts of influenza H3Nx viruses." *Preventive Veterinary Medicine* 233 (2024): 106351. DOI: 10.1016/j.prevetmed.2024.106351

**Task**: Single-label multi-class host identification using HA nucleotide & amino-acid sequences of H3 influenza A viruses.

---

### 1. DATA

#### 1.1 Sequence Sources
- NCBI Influenza Virus Database (IVD)
- GISAID EpiFlu
- BV-BRC (Bacterial and Viral Bioinformatics Resource Center)

#### 1.2 Hosts (7 classes, species-level)
- Canine (H3N2, H3N8)
- Chicken — *Gallus domesticus* (H3N2, H3N8)
- Equine, no donkeys (H3N8)
- Human (H3N2)
- Mallard — *Anas platyrhynchos* (H3N2, H3N6, H3N8)
- Swine (H3N1, H3N2)
- Turkey — *Meleagris gallopavo* (H3N2)

#### 1.3 Inclusion Criteria
1. Sequence length: 1600–1800 bp (full-length HA)
2. Host in strain name must match database-assigned host
3. Leading/trailing Ns trimmed; if >5% N remains → discard
4. Deduplicated across databases (exact nucleotide match → keep 1)
5. Human H3N2 and swine H3N2: if >1000 sequences → random sample 1000
6. Amino-acid equivalent must be available for download

#### 1.4 Dataset Split
- **Training**: 2693 sequences (70% of non-case-study data, random)
- **Validation**: 1150 sequences (30%)
- **Case Study 1** (independent): canine H3N2 of avian origin, n=51, stratified by year
- **Case Study 2** (independent): triple reassortant swine H3N2 USA 1998–2003, n=26
- **Case Study 3** (independent): environmental sequences, n=321
- Sequences from case studies excluded from training/validation

Full accession list in Dataset S1 (supplementary, journal website).

---

### 2. FEATURE EXTRACTION

**Software**: R 4.3.1, `Biostrings` package

Both nucleotide and amino-acid sequences were downloaded. All features extracted from **nucleotide** sequences unless noted.

| Feature | Dim | R Function / Package | Description |
|---|---|---|---|
| Nucleotide frequency | 4 | `Biostrings` | Freq of A, T, C, G in sequence |
| Amino-acid frequency | 21 | `Biostrings` | Freq of 20 AAs + stop codon |
| Dinucleotide frequency | 16 | `Biostrings` | Freq of all dinucleotides (AA, AT, …) |
| 4-mer nucleotide | 256 | `oligonucleotideFrequency()` | k-mer count, k=4 |
| 5-mer nucleotide | 1024 | `oligonucleotideFrequency()` | k-mer count, k=5 |
| Codon frequency | 64 | `oligonucleotideFrequency()` | k=3, treated as codon freq |
| Codon Pair Score (CPS) | **4096** | Eq. 1 (Babayan 2018 / Coleman 2008) | CPS for all codon pairs; accounts for both nucleotide and AA |
| Codon Pair Bias (CPB) | 1 | Average of all CPS values | Scalar summary |
| AA properties | 6 | `aminoAcidProperties()` from `alakazam` | Length, GRAVY (hydropathy index), bulkiness, pI, polar, aliphatic |

**Total initial features**: 5,488

**Note on CPS (Eq. 1)**: CPS = ln(observed codon pair count / expected codon pair count), where expected = (codon_X_freq × codon_Y_freq × AA_pair_freq / total_codons²). This encodes translational efficiency / codon usage bias signals.

---

### 3. FEATURE SELECTION

1. Train a **Random Forest** classifier on the training set (2693 sequences, all 5488 features)
2. Compute Mean Decrease in Gini Impurity for each feature
3. Select **top 10% (n = 550 features)** by Gini score
4. These 550 features used in all downstream classifiers

Selected features composition: 69.4% CPS, 19.3% 5-mer, 6% 4-mer. Only one AA property (net charge, 6th most important feature) survived. CPB was eliminated entirely.

---

### 4. CLASSIFIERS

**Software**: R 4.3.1, `caret` 6.0.94, `randomForest` 4.7.1.1, `gbm` 2.1.8.1

**Input**: 2693 sequences × 550 features  
**Train/Validation split**: 70/30 (pre-defined sets, not cross-fold split)  
**Internal resampling**: 10-fold cross-validation (used for parameter tuning only, on training set)

#### 4.1 Random Forest (RF)

| Parameter | Value |
|---|---|
| `mtry` | 5 (tuned via grid search, Table S7) |
| `ntree` | 500 (caret default) |
| Tuning method | Grid search, 10-fold CV |

#### 4.2 Gradient Boosting Machine (GBM)

| Parameter | Value |
|---|---|
| `interaction.depth` | 5 |
| `shrinkage` (learning rate) | 0.01 |
| `n.trees` | Tuned via 10-fold CV grid search |
| `n.minobsinnode` | caret default (=10) |
| `bag.fraction` | caret default (=0.5) |
| Tuning method | Grid search, 10-fold CV |

#### 4.3 k-Nearest Neighbor (kNN)

| Parameter | Value |
|---|---|
| `k` | 5 (tuned via 10-fold CV) |
| Distance metric | Euclidean (caret default) |

#### 4.4 Penalized Multinomial Ridge Logistic Regression (LR)

| Parameter | Value |
|---|---|
| Penalty type | Ridge (L2) |
| `lambda` (regularization) | Tuned via 10-fold CV grid search |
| Solver | caret default (`glmnet`) |

**Note on exact hyperparameters**: The specific tuned values (grid ranges) are in **Table S7** of the paper's supplementary materials. The paper text only states tuning was performed via caret grid search with 10-fold CV for RF, LR, and GBM. The values above for RF and GBM are defaults/care standards cited in the paper (RF: `mtry=5`; GBM: `interaction.depth=5`, `shrinkage=0.01`).

---

### 5. EVALUATION

Evaluated on the independent **validation set** (1150 sequences):

| Classifier | Accuracy | 95% CI |
|---|---|---|
| GBM | **98.00%** | 97.01–98.73 |
| RF | 97.91% | 96.91–98.66 |
| kNN | 96.52% | 95.29–97.50 |
| LR | 93.74% | 92.18–95.07 |

Baseline: No Information Rate = 27.74% (always predict largest class).

Assessment function: `confusionMatrix()` from `caret`. Metrics: accuracy, class-wise sensitivity, specificity, 95% CI, confusion matrix.

GBM class-wise sensitivity: 0.961–0.996 except chicken (0.842) and turkey (0.571).

GBM prediction time on validation set: 0.33 sec (Intel i5-8250U, 8 GB RAM).

---

### 6. KEY DIFFERENCES FROM YOUR WORK (to document in your methods)

| Aspect | Alberts 2024 | Your Work |
|---|---|---|
| Input | HA nucleotide + AA sequences | HA protein sequences |
| Features | Hand-crafted (k-mer, CPS, etc.) | ESM-2 embeddings |
| Feature count | 550 (after selection from 5488) | 640 (ESM-2 150M) or 1280 (650M) |
| Task | Single-label 7-class | Multi-label binary (4 hosts) |
| Subtype scope | H3 only | All subtypes (H1–H16) |
| Alignment | Not required (alignment-free) | Not required (PLM handles variable length) |
| Classifier | GBM | MLP head (2-layer) |
| Interpretability | Predicted probability score | SHAP / attention |

---

### 7. REPRODUCTION CHECKLIST

- [ ] Download HA nucleotide + AA sequences from NCBI IVD, GISAID, BV-BRC for H3Nx subtypes
- [ ] Filter: 1600–1800 bp, <5% N, deduplicate cross-DB, enforce host-strain-name match
- [ ] Extract features with R 4.3.1 + `Biostrings` (nt freq, AA freq, dinuc freq, 4-mer, 5-mer, codon freq)
- [ ] Compute CPS via Eq. 1 (Babayan 2018 / Coleman 2008) for all 4096 codon pairs
- [ ] Compute AA properties via `alakazam::aminoAcidProperties()`
- [ ] Train a preliminary RF on all 5488 features → select top 550 by Gini
- [ ] Train final GBM (interaction.depth=5, shrinkage=0.01) on 550 features with 10-fold CV
- [ ] Evaluate on held-out validation set (30%)
- [ ] Report accuracy, 95% CI, class-wise sensitivity/specificity, confusion matrix
