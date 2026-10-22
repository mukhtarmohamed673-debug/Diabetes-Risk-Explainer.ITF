import numpy as np
import pandas as pd
import pytest

from src.explain import (BRFSS_FEATURES, agreement, describe_value, label, lime_explainer, lime_weights,
                         partial_dependence, preprocess, shap_values)
from src.models import build_pipeline


@pytest.fixture(scope="module")
def fitted():
    df = pd.read_csv("data/processed/brfss_bin_train.csv").sample(4000, random_state=0)
    X, y = df.drop(columns="Diabetes_binary"), df["Diabetes_binary"]
    return build_pipeline("Gradient Boosting", "none").fit(X, y), X


def test_every_brfss_column_has_readable_info(fitted):
    _, X = fitted
    assert set(X.columns) == set(BRFSS_FEATURES)


def test_describe_value():
    assert describe_value("Age", 9) == "60–64"
    assert describe_value("GenHlth", 5.0) == "Poor"
    assert describe_value("BMI", 31.0) == "31"
    assert label("HighBP") == "High blood pressure"


def test_preprocess_keeps_columns_and_caps_bmi(fitted):
    pipe, X = fitted
    row = X.head(1).copy()
    row["BMI"] = 98
    out = preprocess(pipe, row)
    assert list(out.columns) == list(X.columns)
    assert out["BMI"].iloc[0] < 98


def test_shap_values_add_up_to_model_output(fitted):
    pipe, X = fitted
    rows = X.head(5)
    exp = shap_values(pipe, rows)
    logit = pipe.named_steps["model"].decision_function(preprocess(pipe, rows))
    assert np.allclose(exp.values.sum(axis=1) + exp.base_values, logit, atol=1e-4)


def test_lime_weights_cover_all_columns(fitted):
    pipe, X = fitted
    w = lime_weights(pipe, lime_explainer(pipe, X), X.head(1), num_samples=500)
    assert list(w.index) == list(X.columns)
    assert (w != 0).sum() > 0


def test_partial_dependence_binary_and_numeric(fitted):
    pipe, X = fitted
    rows = X.head(50)
    grid, avg, ice = partial_dependence(pipe, rows, "HighBP")
    assert list(grid) == [0, 1] and ice.shape == (50, 2)
    assert np.allclose(avg, ice.mean(axis=0))
    assert avg[1] > avg[0]  # high blood pressure should raise predicted risk
    grid, avg, _ = partial_dependence(pipe, rows, "BMI", grid=[20, 40])
    assert avg[1] > avg[0]


def test_agreement_identical_rankings():
    s = pd.Series({"a": 3.0, "b": -2.0, "c": 1.0, "d": 0.1})
    res = agreement(s, s, k=2)
    assert res["top_k_overlap"] == 1.0 and res["sign_agreement"] == 1.0
    assert res["rank_corr"] == pytest.approx(1.0)
