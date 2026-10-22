# Explainability results

From [notebooks/05_explainability.ipynb](../notebooks/05_explainability.ipynb). Figures: `reports/figures/xai_*.png`.


**1. What drives the model overall (SHAP).** The five most important features are general health, high blood pressure, age, BMI and high cholesterol, matching the strongest relationships found in the EDA (the two rankings correlate at 0.67). Sex ranks 7th for SHAP but 18th by simple correlation, so the model uses it mainly in combination with other features.

**2. Feature effects (partial dependence).** On average, predicted risk rises from about 6% at BMI 20 to about 22% at BMI 45–50. It rises steadily with age up to 70–74 and falls slightly after. It rises from about 7% to 25% from "excellent" to "poor" general health. High blood pressure and high cholesterol each add about 5 percentage points on average, and higher income slightly lowers risk. The BMI curve is flat above about 55 because the pipeline caps extreme BMI values. The individual (ICE) curves show the same change matters much more for some people than others.

**3. Some effects are associations, not causes.** Heavy drinking *lowers* predicted risk, and *not* having had a cholesterol check lowers it too. These are patterns in the survey, not advice. People who have never had a cholesterol check have less contact with doctors, so are less likely to have been diagnosed. Heavy drinking may be less common among people who have already been told they have diabetes. **The interface must not present these as things to change.**

**4. SHAP and LIME mostly agree (200 test-set people).**

| Measure | Mean | Median | Minimum |
|---|---|---|---|
| Top-5 overlap | 0.70 | 0.80 | 0.40 |
| Sign agreement (shared features) | 1.00 | 1.00 | 1.00 |
| Rank correlation | 0.74 | 0.74 | 0.57 |

When both methods pick a feature, they **always** agree on whether it raises or lowers risk. They differ in emphasis: LIME names general health as the single most important feature for 154 of 200 people (SHAP: 80), and it gives more weight to yes/no features such as cholesterol check. SHAP spreads importance more evenly across age, blood pressure and BMI.

**5. Speed and stability.** SHAP takes about **2 ms** per person and is exact, so it always gives the same answer. LIME takes about **36 ms** and is random. Over 10 runs per person, the top-5 features stayed mostly the same (1–2 different sets, 80–100% overlap), but they can change.

**Decision for the interface:** use **SHAP** for the main explanation because it is exact, fast and stable. Show the top reasons in plain language, and keep LIME as an optional "second opinion" for the user study. Only suggest changes to factors a person can realistically change, such as BMI and physical activity. Label other factors as "linked to risk in the survey data".
