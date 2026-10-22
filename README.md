# Diabetes Risk Explainer
The Diabetes Risk Explainer is a web application that estimates a person’s risk of diabetes from 21 simple health and lifestyle questions. More importantly, it explains why it gave that estimate, in plain language that someone with no technical background can understand.

## About the project

Final-year project — UWL, L6 CP60046E (Sept 2026 – May 2027)
Supervisor: Dr Sama Aleshaiker

I build my own explainable diabetes prediction models on the same two datasets used by
Islam et al. (2024), then build a user-facing interface that explains each prediction with
SHAP, LIME and partial dependence plots. The paper is used as a reference and benchmark
only; I don't copy its settings.

## Folder layout

| Folder | What goes here |
|---|---|
| `docs/` | Topic brief, project plan, paper notes, meeting notes, proposal and report drafts |
| `references/` | Module topic list and saved papers (`references/papers/`) |
| `data/raw/` | Original datasets downloaded from Kaggle (do not edit) |
| `data/processed/` | Cleaned and encoded data produced by the pipeline |
| `notebooks/` | Jupyter notebooks for exploration and replicating the paper |
| `src/` | Reusable Python code: preprocessing, training, explanations |
| `models/` | Saved trained models (`.joblib`) |
| `app/` | The interface (Streamlit web app) |
| `reports/figures/` | Charts exported for the report (SHAP, PDP, LIME, confusion matrices) |

## Key documents

- [docs/topic.md](docs/topic.md) — the chosen topic and related topics from the list
- [docs/paper-notes.md](docs/paper-notes.md) — summary and critique of the base paper
- [docs/project-plan.md](docs/project-plan.md) — scope, pipeline, interface and next steps
- [docs/eda-findings.md](docs/eda-findings.md) — data findings and the preprocessing decisions
- [docs/model-results.md](docs/model-results.md) — leakage demo, model comparison, test results, interface model
- [docs/explainability-results.md](docs/explainability-results.md) — SHAP, LIME, PDP findings
- [docs/interface-design.md](docs/interface-design.md) — the app: layout, design decisions, how to run it

## Notebooks (run in order)

1. `01_eda.ipynb` — exploratory analysis
2. `02_preprocessing.ipynb` — cleaning, de-duplication, train/test splits
3. `03_modelling.ipynb` — leakage demo, 6 models × 4 imbalance methods, tuning, test results
4. `04_interface_model.ipynb` — small, well-calibrated Gradient Boosting model for the app
5. `05_explainability.ipynb` — SHAP, LIME, partial dependence, SHAP vs LIME agreement

Run the tests with `.venv/bin/python -m pytest`.

## Datasets (to download into `data/raw/`)

1. **BRFSS 2015 Diabetes Health Indicators** (Kaggle) — `diabetes_012_health_indicators_BRFSS2015.csv`
2. **Diabetes Dataset 2019** by Tigga & Garg (Kaggle) — `diabetes_dataset__2019.csv`

Backup: `cdc_diabetes_health_indicators_UCI_binary.csv` is the same BRFSS 2015 data from the
UCI repository (dataset 891), but with a binary target `Diabetes_binary` (prediabetes and
diabetes merged) and an extra `ID` column. It's not used by default.

## User study (`study/`)

- `User Study Pack.docx`: protocol, participant information sheet, consent form, task booklet, questionnaire
- `Researcher Answer Key and Analysis Plan.docx`: answers (researcher only) and how to analyse the results
- `responses_template.csv`: copy to `responses.csv` (git-ignored) and enter each participant's results
- `score_study.py`: comprehension, SUS and trust summaries (`.venv/bin/python study/score_study.py`)

## Run the interface

```bash
.venv/bin/streamlit run app/app.py
```

Then open http://localhost:8501. See [docs/interface-design.md](docs/interface-design.md).
