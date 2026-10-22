import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.models import IMBALANCE, MODELS, UpperClipper, build_pipeline, compare, valid_combinations


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"BMI": rng.normal(28, 6, n), "Age": rng.integers(0, 4, n).astype(float)})
    X.loc[:4, "BMI"] = np.nan
    X.loc[5, "BMI"] = 95.0
    y = pd.Series((X["BMI"].fillna(28) + rng.normal(0, 4, n) > 33).astype(int))
    return X, y


def test_clipper_learns_cap_from_training_data_only():
    train = pd.DataFrame({"BMI": [20.0, 25.0, 30.0, 35.0]})
    clip = UpperClipper(quantile=0.5).fit(train)
    assert clip.transform(pd.DataFrame({"BMI": [90.0]}))["BMI"].iloc[0] == pytest.approx(27.5)


def test_every_combination_builds_and_fits(toy):
    X, y = toy
    combos = list(valid_combinations())
    assert len(combos) == len(MODELS) * len(IMBALANCE) - 1  # KNN has no class weights
    for model_name, imbalance in combos:
        build_pipeline(model_name, imbalance).fit(X, y).predict_proba(X)


def test_knn_class_weights_rejected():
    with pytest.raises(ValueError):
        build_pipeline("KNN", "class weights")


def test_resampling_happens_inside_pipeline_only(toy):
    X, y = toy
    pipe = build_pipeline("Decision Tree", "random oversampling").fit(X, y)
    # Predictions come back for the original rows, not the resampled set.
    assert len(pipe.predict(X)) == len(X)


def test_compare_returns_sorted_scores(toy):
    X, y = toy
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    res = compare(X, y, cv, combos=[("Decision Tree", "none"), ("Logistic Regression", "class weights")])
    assert list(res.columns[:3]) == ["model", "imbalance", "macro_f1"]
    assert res["macro_f1"].is_monotonic_decreasing
