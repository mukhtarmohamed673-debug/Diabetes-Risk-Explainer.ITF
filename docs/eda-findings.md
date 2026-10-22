# EDA findings and decisions

From [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb). Figures are in `reports/figures/eda_*.png`.

These decisions are **proposed**. Confirm them (or change them) with the team and supervisor before the preprocessing notebook.

### BRFSS 2015 (253,680 rows, 21 features, 3 classes)
| Finding | Proposed decision |
|---|---|
| No missing values. All columns are whole-number codes (binary, ordinal or counts). | No imputation. Use the ordinal codes as they are. Tree models don't need scaling, so scale only for models that do (KNN, logistic regression), inside a pipeline. |
| Very imbalanced: 84.2% no diabetes, 13.9% diabetes, **1.8% prediabetes** (4,631 rows). | Use a stratified split. Handle imbalance **on the training set only** (class weights or resampling, compared). Report macro-F1 and per-class recall, not just accuracy (predicting "no diabetes" for everyone already scores 84%). |
| 23,899 duplicate rows (9.4%). These are survey answers, and two different people can give the same coded answers. 1,834 identical feature combinations have *different* labels. | Keep the duplicates (they're plausibly real people), but don't create any new duplicates before the split. Note the label conflicts as a limit on how accurate any model can be. |
| GenHlth, HighBP, BMI, DiffWalk, HighChol and Age correlate most with the target. Income, Education and PhysActivity correlate negatively. Diet and alcohol barely matter. | Expect these at the top of SHAP. Use them as the main inputs in the interface form. |
| BMI rises steadily with diabetes up to about 50, then gets noisy. 805 people have BMI > 60, and 281 have BMI ≥ 80. | Consider capping BMI (for example at the 99.5th percentile) and limiting the interface's BMI input to a realistic range. |
| Diabetes rate peaks at age group 11 (70–74) and drops slightly at 80+. | Keep Age as ordinal. A useful example for PDP/SHAP discussion. |

### Diabetes 2019 (952 rows, 17 features, binary)
| Finding | Proposed decision |
|---|---|
| **Only 321 of 952 rows are unique. 631 rows (66.3%) are exact duplicates.** In 19 cases, the same features have different labels. | **Remove the duplicates before splitting.** Otherwise most test rows are copies of training rows and the scores mean nothing. This probably helps explain the very high accuracy reported in the paper. Report results with and without deduplication. |
| Target `Diabetic`: 685 no, 266 yes, 1 missing. | Drop the row with no label. Use a stratified split. With only about 320 rows, use repeated stratified cross-validation, not a single split. |
| Typos and inconsistent labels: RegularMedicine has `'o'` (1 row), Pdiabetes has `'0'` (936) alongside `'no'` (1). | Treat `'o'` as `'no'` and `'0'` as `'no'`, and document both. |
| Missing values: Pregancies 42 (16 of them male), BMI 4, Pdiabetes 1. | Pregnancies for males → 0. For the rest, impute with the median inside the training pipeline. |
| Categorical columns with a natural order: Age, PhysicallyActive, JunkFood, Stress, BPLevel, UriationFreq. | Ordinal encoding with an explicit order (not alphabetical). One-hot encoding isn't needed. |
| **RegularMedicine:** 64% of "yes" are diabetic vs 8% of "no". **Pdiabetes** (gestational diabetes): 86% of "yes" are diabetic. | RegularMedicine probably reflects the diagnosis, not a cause, so it leaks the answer. Train with and without it. The interface model should leave it out. Keep Pdiabetes (it's a known real risk factor). |
| Clear risk patterns: age 60+, high BP, family history, no physical activity, more pregnancies, high stress. Smoking, gender and junk food make little difference. | Use as a sanity check for SHAP results. |

## Decisions confirmed and applied (22 Oct 2026)

Applied in `src/preprocess.py` and [notebooks/02_preprocessing.ipynb](../notebooks/02_preprocessing.ipynb).

- **Diabetes 2019 duplicates:** removed before splitting. 952 rows become **303 unique labelled rows** (fixing typos made a few more rows identical). With duplicates kept, **94.8%** of test rows also appear in training. After removing them, 13.1% (8 rows) still match: people with identical answers but different diagnoses.
- **New finding:** 12 male respondents had 1–2 pregnancies recorded, which is impossible. Pregnancies is now set to 0 for all men.
- **RegularMedicine:** two versions are kept, `d19` (with it) and `d19_nomed` (without it, for the interface).
- **BRFSS:** duplicates kept. 13.6% of test rows match a training row because many people give identical survey answers. Two targets: `brfss3` (3 classes, main) and `brfss_bin` (diabetes vs not, fallback).
- **Class imbalance:** compare no balancing, class weights, random oversampling and SMOTE, on training folds only, and choose by macro-F1.
- **Splits:** stratified 80/20, made before any resampling. Saved in `data/processed/`. The Diabetes 2019 test set is only 61 rows, so model selection will use repeated cross-validation on the training set.
