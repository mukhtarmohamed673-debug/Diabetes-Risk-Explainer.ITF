# Base paper notes

**Islam, M. M., Rifat, H. R., Shahid, M. S. B., Akhter, A., Uddin, M. A. and Uddin, K. M. M. (2024).**
*Explainable Machine Learning for Efficient Diabetes Prediction Using Hyperparameter Tuning,
SHAP Analysis, Partial Dependency, and LIME.* Engineering Reports, 7(1), e13080.
https://doi.org/10.1002/eng2.13080 (open access, first published 12 Dec 2024)

PDF: download from the Wiley page and save it as `references/papers/Islam2024_XAI_diabetes.pdf`.

## Datasets
| | BRFSS 2015 (Diabetes Health Indicators) | Diabetes 2019 (Tigga & Garg) |
|---|---|---|
| Rows | 253,680 | 952 (948 after removing duplicates) |
| Features | 21 | 17 |
| Target | `Diabetes_012`: 0 none, 1 prediabetes, 2 diabetes | `Diabetes`: yes / no |
| Imbalance | Class 0 = 213,703 | Class 0 ≈ 683, class 1 ≈ 265 |

> **Use of this paper:** reference only. I don't copy its pipeline settings or
> hyperparameters. I make and justify my own choices, and only compare my results with
> theirs.

## Their approach (in brief)
Preprocessing (encoding, scaling, transformation), random oversampling to balance the
classes, hyperparameter tuning with GridSearchCV, and four models (KNN, Random Forest,
Decision Tree, Extra Trees). Extra Trees performed best.

## Reported results (for comparison with mine)
| Model | BRFSS accuracy | Diabetes 2019 accuracy |
|---|---|---|
| KNN | 93.24% | 95.26% |
| RF | 96.5% | 97.45% |
| DT | 94.23% | — |
| **ET** | **97.23%** | **97.45%** |

Without oversampling or tuning, BRFSS accuracy was about 77–84%. AUC was 1.00 for RF and ET on
BRFSS and 0.99 on Diabetes 2019.

## Explainability findings
- **SHAP top features, BRFSS:** Age, HighChol, GenHlth, HighBP, BMI
- **SHAP top features, Diabetes 2019:** RegularMedicine, Age, BPLevel, Family_Diabetes, BMI
- **PDPs:** BMI and GenHlth (BRFSS); Age, RegularMedicine, Family_Diabetes, SoundSleep and
  BMI (2019)
- **LIME:** top 5 features for one test patient from each dataset

## Weaknesses (worth discussing in my report)
1. **Data leakage from oversampling before the split.** Table 5 shows the test set growing
   from 50,736 to 128,222 rows after oversampling. That means duplicated rows end up in both
   the training and test sets, which likely inflates accuracy from about 82% (earlier
   studies) to 97%, and AUC to 1.00. **My fix:** split first, oversample the training set only.
2. **RegularMedicine leaks the answer.** People who take medicine regularly probably already
   have diabetes, so this feature is a consequence of the diagnosis, not a risk factor.
3. **No interface:** the explanations are only static figures in the paper.
4. **No user evaluation:** the paper says explanations build trust but never measures it.
5. **SHAP computation time** is named as a limitation for real-time use (future work).
