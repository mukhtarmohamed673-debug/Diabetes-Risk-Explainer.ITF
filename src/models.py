"""Model pipelines, class-imbalance strategies and evaluation.

Everything that learns from data (BMI capping, imputation, scaling and
resampling) sits inside an imbalanced-learn Pipeline. During cross-validation
the pipeline is re-fitted on each training fold, so the validation fold is
never seen by any of these steps, and resampled copies never reach it.
"""

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import RANDOM_STATE


class UpperClipper(BaseEstimator, TransformerMixin):
    """Cap one column at a quantile learned from the training data (e.g. extreme BMI values)."""

    def __init__(self, column="BMI", quantile=0.995):
        self.column = column
        self.quantile = quantile

    def fit(self, X, y=None):
        self.cap_ = X[self.column].quantile(self.quantile) if self.column in X else None
        return self

    def transform(self, X):
        if self.cap_ is None:
            return X
        X = X.copy()
        X[self.column] = X[self.column].clip(upper=self.cap_)
        return X


# name -> (estimator, needs scaling, supports class_weight)
MODELS = {
    "Logistic Regression": (LogisticRegression(max_iter=2000), True, True),
    "KNN": (KNeighborsClassifier(), True, False),
    "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), False, True),
    "Random Forest": (RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE), False, True),
    "Extra Trees": (ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE), False, True),
    "Gradient Boosting": (HistGradientBoostingClassifier(random_state=RANDOM_STATE), False, True),
}

IMBALANCE = ["none", "class weights", "random oversampling", "SMOTE"]

# Search spaces for tuning (prefixed for the pipeline's "model" step).
PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    "KNN": {"model__n_neighbors": [5, 11, 21, 31, 51, 101], "model__weights": ["uniform", "distance"]},
    "Decision Tree": {"model__max_depth": [3, 5, 8, 12, 20, None],
                      "model__min_samples_leaf": [1, 5, 20, 50, 100]},
    "Random Forest": {"model__n_estimators": [200, 400], "model__max_depth": [None, 8, 16, 24],
                      "model__min_samples_leaf": [1, 2, 5, 10, 20], "model__max_features": ["sqrt", 0.5]},
    "Extra Trees": {"model__n_estimators": [200, 400], "model__max_depth": [None, 8, 16, 24],
                    "model__min_samples_leaf": [1, 2, 5, 10, 20], "model__max_features": ["sqrt", 0.5]},
    "Gradient Boosting": {"model__learning_rate": [0.03, 0.06, 0.1, 0.2], "model__max_iter": [100, 200, 400],
                          "model__max_leaf_nodes": [15, 31, 63], "model__l2_regularization": [0.0, 1.0, 5.0],
                          "model__min_samples_leaf": [20, 50, 100]},
}


def build_pipeline(model_name: str, imbalance: str = "none") -> Pipeline:
    estimator, needs_scaling, supports_weights = MODELS[model_name]
    estimator = clone(estimator)
    if imbalance == "class weights":
        if not supports_weights:
            raise ValueError(f"{model_name} does not support class weights")
        estimator.set_params(class_weight="balanced")

    steps = [("cap_bmi", UpperClipper()), ("impute", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scale", StandardScaler()))
    if imbalance == "random oversampling":
        steps.append(("resample", RandomOverSampler(random_state=RANDOM_STATE)))
    elif imbalance == "SMOTE":
        steps.append(("resample", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def valid_combinations():
    for model_name, (_, _, supports_weights) in MODELS.items():
        for imbalance in IMBALANCE:
            if imbalance == "class weights" and not supports_weights:
                continue
            yield model_name, imbalance


# ------------------------------------------------------------------ Metrics

def scoring(n_classes: int) -> dict:
    s = {
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc" if n_classes == 2 else "roc_auc_ovr",
    }
    return s


def compare(X, y, cv, combos=None, n_jobs=1) -> pd.DataFrame:
    """Cross-validate every (model, imbalance strategy) pair. Returns mean and std per metric."""
    rows = []
    n_classes = y.nunique()
    for model_name, imbalance in (combos or valid_combinations()):
        start = time.time()
        res = cross_validate(build_pipeline(model_name, imbalance), X, y, cv=cv,
                             scoring=scoring(n_classes), n_jobs=n_jobs, error_score="raise")
        row = {"model": model_name, "imbalance": imbalance}
        for metric in scoring(n_classes):
            row[metric] = res[f"test_{metric}"].mean()
            row[f"{metric}_std"] = res[f"test_{metric}"].std()
        row["seconds"] = round(time.time() - start, 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def test_report(model, X_test, y_test, class_names) -> tuple[pd.Series, pd.DataFrame]:
    """Final metrics on the held-out test set, overall and per class."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    if len(class_names) == 2:
        auc = roc_auc_score(y_test, proba[:, 1])
    else:
        auc = roc_auc_score(y_test, proba, multi_class="ovr")
    overall = pd.Series({
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": auc,
    })
    labels = np.sort(y_test.unique())
    per_class = pd.DataFrame({
        "recall": recall_score(y_test, pred, labels=labels, average=None),
        "f1": f1_score(y_test, pred, labels=labels, average=None),
        "support": [int((y_test == k).sum()) for k in labels],
    }, index=[class_names[k] for k in labels])
    return overall, per_class
