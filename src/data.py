"""Load the two raw diabetes datasets from data/raw/."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

BRFSS_FILE = "diabetes_012_health_indicators_BRFSS2015.csv"
BRFSS_TARGET = "Diabetes_012"
BRFSS_CLASSES = {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"}

D2019_FILE = "diabetes_dataset__2019.csv"
D2019_TARGET = "Diabetic"


def _read(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it from Kaggle and save it in data/raw/ (see README.md)."
        )
    return pd.read_csv(path)


def load_brfss() -> pd.DataFrame:
    """BRFSS 2015 diabetes health indicators: 21 numeric features, 3-class target."""
    df = _read(BRFSS_FILE)
    # All values are stored as floats; every column is a whole-number code.
    return df.astype(int)


def load_diabetes_2019() -> pd.DataFrame:
    """Diabetes 2019 (Tigga & Garg): mostly categorical text columns, binary target.

    Only the text values are tidied here (stray spaces, inconsistent capitals).
    Anything that needs a decision, such as typos, missing values or encoding,
    is left for the EDA notebook.
    """
    df = _read(D2019_FILE)
    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = df[text_cols].apply(lambda s: s.str.strip().str.lower())
    return df
