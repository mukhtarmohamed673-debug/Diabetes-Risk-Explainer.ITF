"""Logic behind the Streamlit interface: risk, plain-language reasons and what-if changes.

Kept separate from the UI so it can be tested, and so the wording rules from
the explainability findings live in one place:

- SHAP is the main explanation (exact, fast and stable).
- Only factors a person can realistically change are offered as what-if changes.
- Factors that are survey associations rather than causes (e.g. heavy
  drinking, cholesterol check) are labelled as such and never given as advice.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.explain import BRFSS_FEATURES, describe_value, label, shap_explainer, shap_values

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_NAME = "interface_brfss_bin_gb"

# How each factor should be talked about.
CAN_CHANGE = {"BMI", "PhysActivity", "Fruits", "Veggies"}
MANAGE_WITH_DOCTOR = {"HighBP", "HighChol"}
# Survey patterns that are not causes and must not read as advice. MentHlth is
# here because more poor mental health days slightly *lower* predicted risk.
ASSOCIATION_ONLY = {"HvyAlcoholConsump", "CholCheck", "AnyHealthcare", "NoDocbcCost", "Income", "Education",
                    "MentHlth"}

FACTOR_TYPES = {
    "can change": "Something you can change",
    "manage": "Can be managed with a doctor",
    "association": "Linked to risk in the survey data, not a cause or advice",
    "background": "Background or health history",
}

STRENGTH = [(0.5, "strongly"), (0.2, "moderately"), (0.0, "slightly")]


def factor_type(column: str) -> str:
    if column in CAN_CHANGE:
        return "can change"
    if column in MANAGE_WITH_DOCTOR:
        return "manage"
    if column in ASSOCIATION_ONLY:
        return "association"
    return "background"


class RiskModel:
    def __init__(self, model_dir: Path = MODEL_DIR, name: str = MODEL_NAME):
        self.pipe = joblib.load(model_dir / f"{name}.joblib")
        self.meta = json.loads((model_dir / f"{name}.json").read_text())
        self.features = self.meta["features"]
        self.threshold = self.meta["threshold"]
        self.base_rate = self.meta["base_rate"]
        self.explainer = shap_explainer(self.pipe)

    def _frame(self, profile: dict) -> pd.DataFrame:
        missing = set(self.features) - set(profile)
        if missing:
            raise ValueError(f"Profile is missing {sorted(missing)}")
        return pd.DataFrame([{f: float(profile[f]) for f in self.features}])

    def risk(self, profile: dict) -> float:
        return float(self.pipe.predict_proba(self._frame(profile))[0, 1])

    def category(self, risk: float) -> tuple[str, str]:
        """(short label, sentence) comparing the risk with the average and the model's threshold."""
        times = risk / self.base_rate
        if risk < self.base_rate:
            return "Below average", (f"This is below the average of {self.base_rate:.0%} for adults in the survey.")
        if risk < self.threshold:
            return "Above average", (f"This is about {times:.1f} times the average of {self.base_rate:.0%}, "
                                     f"but below the level the model treats as higher risk ({self.threshold:.0%}).")
        return "Higher risk", (f"This is about {times:.1f} times the average of {self.base_rate:.0%}. "
                               "It may be worth talking to a doctor or pharmacist about a diabetes check.")

    def reasons(self, profile: dict) -> pd.DataFrame:
        """One row per feature, sorted by how much it moved this person's risk (SHAP, log-odds)."""
        row = self._frame(profile)
        values = shap_values(self.pipe, row, self.explainer).values[0]
        out = pd.DataFrame({
            "feature": self.features,
            "label": [label(f) for f in self.features],
            "value": [describe_value(f, profile[f]) for f in self.features],
            "shap": values,
            "type": [factor_type(f) for f in self.features],
        })
        out["direction"] = np.where(out["shap"] > 0, "raises", "lowers")
        out["strength"] = out["shap"].abs().map(lambda v: next(word for cut, word in STRENGTH if v >= cut))
        return out.reindex(out["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    def what_if(self, profile: dict, changes: dict) -> float:
        """Risk after changing only factors a person can realistically change."""
        not_allowed = set(changes) - CAN_CHANGE
        if not_allowed:
            raise ValueError(f"What-if changes are only offered for {sorted(CAN_CHANGE)}, not {sorted(not_allowed)}")
        return self.risk({**profile, **changes})


def sentence(reason: pd.Series) -> str:
    """Plain-language sentence for one reason."""
    text = f"**{reason['label']}** ({reason['value']}) {reason['strength']} {reason['direction']} your estimated risk."
    if reason["type"] == "association":
        text += " *This is a pattern in the survey data, not advice.*"
    elif reason["type"] == "manage" and reason["direction"] == "raises":
        text += " *This can often be managed with help from a doctor.*"
    return text


def summary(reasons: pd.DataFrame, n: int = 3) -> dict:
    """Top n reasons raising and lowering risk, ignoring negligible effects."""
    meaningful = reasons[reasons["shap"].abs() >= 0.05]
    return {
        "raises": [sentence(r) for _, r in meaningful[meaningful["shap"] > 0].head(n).iterrows()],
        "lowers": [sentence(r) for _, r in meaningful[meaningful["shap"] < 0].head(n).iterrows()],
    }


def default_profile() -> dict:
    """A neutral starting profile: the most common answer for each question."""
    return {
        "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 27, "Smoker": 0, "Stroke": 0,
        "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 2, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0,
        "Sex": 0, "Age": 8, "Education": 5, "Income": 7,
    }


assert set(default_profile()) == set(BRFSS_FEATURES)
