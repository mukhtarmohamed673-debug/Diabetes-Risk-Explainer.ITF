"""Cleaning, encoding and splitting for both datasets.

Every rule here comes from a decision recorded in docs/eda-findings.md.
Nothing is fitted on the data in this module (no scaling, imputation or
resampling): anything learned from data happens later, inside a pipeline
that only sees the training set.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import BRFSS_TARGET, D2019_TARGET

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------- BRFSS 2015

BRFSS_BINARY_TARGET = "Diabetes_binary"


def prepare_brfss(df: pd.DataFrame, binary: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Split BRFSS into features and target.

    The 3-class target is kept as it is (0 no diabetes, 1 prediabetes, 2 diabetes).
    With ``binary=True`` the target becomes diabetes (1) vs not (0), with
    prediabetes counted as "not", matching the binary version published on
    Kaggle and UCI.

    Duplicate rows are kept: in survey data, different people can give
    identical coded answers.
    """
    X = df.drop(columns=BRFSS_TARGET)
    y = df[BRFSS_TARGET]
    if binary:
        y = (y == 2).astype(int).rename(BRFSS_BINARY_TARGET)
    return X, y


# ------------------------------------------------------------- Diabetes 2019

# Categories with a natural order, encoded low -> high.
D2019_ORDINAL = {
    "Age": ["less than 40", "40-49", "50-59", "60 or older"],
    "PhysicallyActive": ["none", "less than half an hr", "more than half an hr", "one hr or more"],
    "JunkFood": ["occasionally", "often", "very often", "always"],
    "Stress": ["not at all", "sometimes", "very often", "always"],
    "BPLevel": ["low", "normal", "high"],
    "UriationFreq": ["not much", "quite often"],
}
D2019_BINARY = {
    "Gender": {"female": 0, "male": 1},
    "Family_Diabetes": {"no": 0, "yes": 1},
    "highBP": {"no": 0, "yes": 1},
    "Smoking": {"no": 0, "yes": 1},
    "Alcohol": {"no": 0, "yes": 1},
    "RegularMedicine": {"no": 0, "yes": 1},
    "Pdiabetes": {"no": 0, "yes": 1},
    D2019_TARGET: {"no": 0, "yes": 1},
}
# Typos found in the EDA, corrected before encoding.
D2019_TYPOS = {
    "RegularMedicine": {"o": "no"},
    "Pdiabetes": {"0": "no"},
}

# Probably a consequence of diagnosis rather than a cause (see EDA).
LEAKY_FEATURE = "RegularMedicine"


def _encode(series: pd.Series, mapping: dict) -> pd.Series:
    encoded = series.map(mapping)
    unknown = series.notna() & encoded.isna()
    if unknown.any():
        raise ValueError(f"{series.name}: unexpected values {sorted(series[unknown].unique())}")
    return encoded


def clean_diabetes_2019(df: pd.DataFrame) -> pd.DataFrame:
    """Fix typos, encode every column as numbers and drop rows with no target.

    Missing BMI, Pregancies and Pdiabetes values are left as NaN so they can be
    imputed inside the training pipeline. The exception is pregnancies for
    male respondents, which is set to 0: 16 were missing and 12 had an
    impossible 1 or 2 recorded.
    """
    df = df.copy()
    for col, fixes in D2019_TYPOS.items():
        df[col] = df[col].replace(fixes)

    df = df.dropna(subset=[D2019_TARGET])

    for col, order in D2019_ORDINAL.items():
        df[col] = _encode(df[col], {v: i for i, v in enumerate(order)})
    for col, mapping in D2019_BINARY.items():
        df[col] = _encode(df[col], mapping)

    df.loc[df["Gender"] == 1, "Pregancies"] = 0
    return df.reset_index(drop=True)


def prepare_diabetes_2019(
    df: pd.DataFrame, drop_leaky: bool = False, drop_duplicates: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the data, optionally remove RegularMedicine, then remove duplicates.

    Duplicates are removed after choosing the features, because dropping a
    column can make previously different rows identical.
    """
    df = clean_diabetes_2019(df)
    if drop_leaky:
        df = df.drop(columns=LEAKY_FEATURE)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=D2019_TARGET)
    y = df[D2019_TARGET].astype(int)
    return X, y


# ------------------------------------------------------------------ Splitting

def split(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 train/test split, made before any resampling.

    The test set is set aside here and only used once, for the final evaluation.
    """
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def overlap_between(train: pd.DataFrame, test: pd.DataFrame) -> int:
    """Number of test rows whose features exactly match a training row."""
    cols = list(train.columns)
    train_rows = set(map(tuple, train[cols].astype("float").fillna(-1).to_numpy()))
    return sum(tuple(r) in train_rows for r in test[cols].astype("float").fillna(-1).to_numpy())
