# Project plan

## Scope
Build my own explainable diabetes prediction models on the same two datasets as
Islam et al. (2024), then build an interface on top that explains predictions to users.
The paper is a **reference only**: I design and justify my own pipeline and settings,
and use its results as a benchmark. See [paper-notes.md](paper-notes.md).

## Part 1: Models (notebooks/ and src/)
- [ ] Download both datasets into `data/raw/`
- [ ] Explore the data (EDA) and choose my own preprocessing, justified from the data
- [ ] Split into training and test sets first, then handle class imbalance on the training set only
- [ ] Choose and train my own models, with my own hyperparameter search
- [ ] Evaluate with accuracy, precision, recall, F1, specificity, confusion matrix and ROC-AUC
- [ ] Explanations: SHAP, PDP and LIME
- [ ] Compare my results with the paper's reported results
- [ ] Save the best models to `models/` and figures to `reports/figures/`

## Part 2: Interface (app/)
A Streamlit web app:
1. Choose the dataset or model (BRFSS 3-class, Diabetes 2019 binary)
2. Input form for the user's health details
3. Prediction with class probabilities
4. Why: SHAP waterfall for this user, LIME top 5 features, plain-English summary
5. Model overview page: global SHAP summary and interactive PDPs
6. Optional: "what if" sliders (for example, how lowering BMI changes the risk)

## Part 3: Evaluation
- Model metrics compared with the paper
- Usability of the interface, for example a SUS questionnaire with a small group of users
- Optional: SHAP vs LIME agreement and how long each explanation takes to compute

## Tech stack
Python, pandas, scikit-learn, imbalanced-learn, shap, lime, matplotlib, Streamlit, joblib

## Next steps
1. Email the supervisor to confirm the scope
2. Write the project proposal
3. Download the datasets and start the replication notebook
