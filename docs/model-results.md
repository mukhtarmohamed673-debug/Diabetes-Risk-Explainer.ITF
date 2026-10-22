# Model results

From [notebooks/03_modelling.ipynb](../notebooks/03_modelling.ipynb). Figures: `reports/figures/model_*.png`. Raw numbers: `reports/*.csv`, `reports/final_models.json`.

**1. The evaluation protocol alone explains most of the published 97%.** With the same model (Extra Trees, default settings), oversampling the whole dataset before splitting gives a macro-F1 of **0.95** on Diabetes 2019 and **0.97** on BRFSS. The leakage-free protocol gives **0.78** and **0.39**. The model doesn't change; only the order of the steps does (`model_leakage_demo.png`).

**2. Realistic performance (held-out test sets, used once):**

| Split | Model | Macro-F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|
| Diabetes 2019 | Extra Trees | 0.80 | 0.84 | 0.90 |
| Diabetes 2019, no RegularMedicine | Extra Trees + SMOTE | 0.83 | 0.85 | 0.87 |
| BRFSS 3-class | Random Forest + oversampling | 0.44 | 0.76 | 0.75 |
| BRFSS binary | Random Forest + oversampling | 0.66 | 0.79 | 0.82 |

BRFSS binary (ROC-AUC 0.82) is in line with Xie et al. (2019), who reported 0.79 on BRFSS 2014.

**3. Prediabetes can't be predicted from these survey answers.** Recall is 0.6%: prediabetes cases are split between "no diabetes" and "diabetes". This matches the EDA, where prediabetes respondents looked like a mix of both groups. The binary BRFSS model is the realistic choice for any BRFSS-based tool.

**4. Dropping RegularMedicine costs little.** In cross-validation, macro-F1 falls from 0.769 to 0.753. On the test set the version without it actually scores higher, but with only 61 test rows (18 with diabetes), one or two people change these numbers. The cross-validation standard deviation is about ±0.06, so the two versions can't be told apart.

**5. Simple models are competitive.** On Diabetes 2019, tuned Logistic Regression ties with Extra Trees in cross-validation (0.769 vs 0.769) and has a better ROC-AUC (0.865 vs 0.779 untuned). Its probabilities are also more reliable, which matters for an interface that shows a risk percentage.

**6. Practical limits.** The BRFSS Random Forest files are 123–165 MB even when compressed. Gradient Boosting scored almost the same in cross-validation (0.435 vs 0.443) and would be a fraction of the size.

**7. Imbalance handling.** Oversampling or class weights helped every model on BRFSS. The best strategy varied by model, and no single method was best everywhere.

## Interface model (notebook 04)


- **Better and far smaller.** On the test set, Gradient Boosting without oversampling scores a macro-F1 of **0.680** (Random Forest + oversampling: 0.662), accuracy **0.831** (0.788) and ROC-AUC **0.827** (0.818). The file is **0.22 MB**; the Random Forest was 123 MB.
- **Realistic probabilities.** The calibration curve sits on the diagonal (Brier score 0.097), and the average predicted risk equals the actual rate (13.9%). This is what makes it safe to show users a risk percentage.
- **Threshold.** A cut-off of **0.28** (chosen on training data only) gives the best macro-F1. At the default 0.50 the model would miss most people with diabetes (CV macro-F1 0.59). In the interface, "higher risk" means a predicted risk of 28% or more, about twice the average.
- **Still limited.** Recall for diabetes is 0.52: half of the people with diabetes are not flagged. Survey answers alone can't do better than this. The interface must say clearly that it isn't a diagnosis.
