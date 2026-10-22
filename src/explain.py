"""SHAP and LIME explanations for the BRFSS models, plus readable feature names.

Both methods explain the fitted model step of a pipeline, working on the data
after the pipeline's preprocessing (BMI capping and imputation). These steps
don't rename or reorder columns, so the explanations map straight back to the
survey questions.
"""

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from src.preprocess import RANDOM_STATE

AGE_BANDS = {1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49", 7: "50–54",
             8: "55–59", 9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80 or older"}
GEN_HEALTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
EDUCATION = {1: "Never attended school", 2: "Primary school (grades 1–8)", 3: "Some high school",
             4: "High school graduate", 5: "Some college", 6: "College graduate"}
INCOME = {1: "Under $10,000", 2: "$10,000–14,999", 3: "$15,000–19,999", 4: "$20,000–24,999",
          5: "$25,000–34,999", 6: "$35,000–49,999", 7: "$50,000–74,999", 8: "$75,000 or more"}
YES_NO = {0: "No", 1: "Yes"}

# column -> (short label for charts, question wording, value labels or None for numbers)
BRFSS_FEATURES = {
    "HighBP": ("High blood pressure", "Have you been told you have high blood pressure?", YES_NO),
    "HighChol": ("High cholesterol", "Have you been told you have high cholesterol?", YES_NO),
    "CholCheck": ("Cholesterol check", "Have you had a cholesterol check in the past 5 years?", YES_NO),
    "BMI": ("BMI", "Body mass index (BMI)", None),
    "Smoker": ("Smoker", "Have you smoked at least 100 cigarettes in your life?", YES_NO),
    "Stroke": ("Stroke", "Have you ever had a stroke?", YES_NO),
    "HeartDiseaseorAttack": ("Heart disease", "Have you had coronary heart disease or a heart attack?", YES_NO),
    "PhysActivity": ("Physical activity", "Have you done physical activity in the past 30 days (not counting work)?", YES_NO),
    "Fruits": ("Eats fruit daily", "Do you eat fruit at least once a day?", YES_NO),
    "Veggies": ("Eats vegetables daily", "Do you eat vegetables at least once a day?", YES_NO),
    "HvyAlcoholConsump": ("Heavy drinking", "Men: more than 14 drinks a week; women: more than 7?", YES_NO),
    "AnyHealthcare": ("Has health cover", "Do you have any kind of health care cover?", YES_NO),
    "NoDocbcCost": ("Skipped doctor (cost)", "In the past year, did you skip seeing a doctor because of cost?", YES_NO),
    "GenHlth": ("General health", "In general, how would you rate your health?", GEN_HEALTH),
    "MentHlth": ("Poor mental health days", "Days of poor mental health in the past 30 days", None),
    "PhysHlth": ("Poor physical health days", "Days of poor physical health in the past 30 days", None),
    "DiffWalk": ("Difficulty walking", "Do you have serious difficulty walking or climbing stairs?", YES_NO),
    "Sex": ("Sex", "Sex", {0: "Female", 1: "Male"}),
    "Age": ("Age group", "Age group", AGE_BANDS),
    "Education": ("Education", "Highest level of education", EDUCATION),
    "Income": ("Household income", "Annual household income", INCOME),
}


def label(column: str) -> str:
    return BRFSS_FEATURES.get(column, (column,))[0]


def describe_value(column: str, value) -> str:
    """Readable version of a coded value, e.g. Age 9 -> '60–64'."""
    values = BRFSS_FEATURES.get(column, (None, None, None))[2]
    if values is None:
        return f"{value:g}" if isinstance(value, (int, float, np.number)) else str(value)
    return values.get(int(value), str(value))


def preprocess(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Apply every pipeline step except the model (and any resampler, which only acts in fit)."""
    out = X
    for name, step in pipe.steps[:-1]:
        if hasattr(step, "transform"):
            out = step.transform(out)
    return pd.DataFrame(np.asarray(out, dtype=float), columns=X.columns, index=X.index)


# ----------------------------------------------------------------------- SHAP

def shap_explainer(pipe):
    return shap.TreeExplainer(pipe.named_steps["model"])


def shap_values(pipe, X: pd.DataFrame, explainer=None) -> shap.Explanation:
    """SHAP values on the log-odds scale, for the positive class (diabetes)."""
    explainer = explainer or shap_explainer(pipe)
    Xt = preprocess(pipe, X)
    exp = explainer(Xt)
    if exp.values.ndim == 3:  # one set of values per class -> keep the positive class
        exp = exp[:, :, 1]
    exp.feature_names = [label(c) for c in X.columns]
    return exp


# ----------------------------------------------------------------------- LIME

def lime_explainer(pipe, X_background: pd.DataFrame) -> LimeTabularExplainer:
    Xt = preprocess(pipe, X_background)
    categorical = [i for i, c in enumerate(Xt.columns) if BRFSS_FEATURES.get(c, (0, 0, None))[2] is YES_NO]
    return LimeTabularExplainer(
        Xt.to_numpy(), feature_names=list(Xt.columns), class_names=["No diabetes", "Diabetes"],
        categorical_features=categorical, mode="classification", discretize_continuous=True,
        random_state=RANDOM_STATE,
    )


def lime_weights(pipe, explainer: LimeTabularExplainer, row: pd.DataFrame, num_features: int = 21,
                 num_samples: int = 2000) -> pd.Series:
    """LIME weight per original column for one person (positive = pushes towards diabetes)."""
    xt = preprocess(pipe, row).iloc[0].to_numpy()
    exp = explainer.explain_instance(xt, pipe.named_steps["model"].predict_proba, labels=(1,),
                                     num_features=num_features, num_samples=num_samples)
    weights = dict(exp.as_map()[1])
    return pd.Series([weights.get(i, 0.0) for i in range(len(row.columns))], index=row.columns)


# --------------------------------------------------------- Partial dependence

def partial_dependence(pipe, X: pd.DataFrame, feature: str, grid=None):
    """Average predicted risk when everyone's `feature` is set to each value in `grid`.

    Returns (grid, average curve, individual curves). The individual curves
    are ICE curves: one per person, shape (len(X), len(grid)).
    """
    if grid is None:
        values = np.sort(X[feature].dropna().unique())
        grid = values if len(values) <= 30 else np.unique(np.quantile(values, np.linspace(0.01, 0.99, 30)).round(1))
    ice = np.empty((len(X), len(grid)))
    for j, v in enumerate(grid):
        Xv = X.copy()
        Xv[feature] = v
        ice[:, j] = pipe.predict_proba(Xv)[:, 1]
    return np.asarray(grid), ice.mean(axis=0), ice


# ------------------------------------------------------------------ Agreement

def agreement(shap_row: pd.Series, lime_row: pd.Series, k: int = 5) -> dict:
    """How far SHAP and LIME agree for one person.

    - top_k_overlap: share of the k most important features (by absolute value) that both methods pick
    - sign_agreement: of those shared features, the share where both agree on direction
    - rank_corr: Spearman correlation of absolute importance across all features
    """
    top_s = set(shap_row.abs().nlargest(k).index)
    top_l = set(lime_row.abs().nlargest(k).index)
    shared = top_s & top_l
    signs = [np.sign(shap_row[f]) == np.sign(lime_row[f]) for f in shared]
    return {
        "top_k_overlap": len(shared) / k,
        "sign_agreement": float(np.mean(signs)) if signs else np.nan,
        "rank_corr": spearmanr(shap_row.abs(), lime_row.abs()).correlation,
    }
