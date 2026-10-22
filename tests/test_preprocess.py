import pandas as pd
import pytest

from src.data import D2019_TARGET, load_brfss, load_diabetes_2019
from src.preprocess import (
    LEAKY_FEATURE,
    clean_diabetes_2019,
    overlap_between,
    prepare_brfss,
    prepare_diabetes_2019,
    split,
)


@pytest.fixture(scope="module")
def raw_2019():
    return load_diabetes_2019()


def test_2019_cleaned_is_fully_numeric(raw_2019):
    df = clean_diabetes_2019(raw_2019)
    assert all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes)


def test_2019_typos_and_missing_target_handled(raw_2019):
    df = clean_diabetes_2019(raw_2019)
    assert df[D2019_TARGET].notna().all()
    assert set(df["RegularMedicine"].unique()) == {0, 1}
    assert set(df["Pdiabetes"].dropna().unique()) <= {0, 1}


def test_2019_males_have_zero_pregnancies(raw_2019):
    df = clean_diabetes_2019(raw_2019)
    assert df.loc[df["Gender"] == 1, "Pregancies"].eq(0).all()


def test_2019_ordinal_order_is_meaningful(raw_2019):
    df = clean_diabetes_2019(raw_2019)
    raw = raw_2019.dropna(subset=[D2019_TARGET]).reset_index(drop=True)
    assert df.loc[raw["Age"] == "less than 40", "Age"].eq(0).all()
    assert df.loc[raw["Age"] == "60 or older", "Age"].eq(3).all()


def test_2019_unknown_category_is_rejected(raw_2019):
    bad = raw_2019.copy()
    bad.loc[0, "Stress"] = "extremely"
    with pytest.raises(ValueError, match="Stress"):
        clean_diabetes_2019(bad)


def test_2019_no_duplicates_after_prepare(raw_2019):
    for drop_leaky in (False, True):
        X, y = prepare_diabetes_2019(raw_2019, drop_leaky=drop_leaky)
        assert not pd.concat([X, y], axis=1).duplicated().any()
        assert (LEAKY_FEATURE in X.columns) != drop_leaky


def test_2019_dedup_removes_almost_all_train_test_overlap(raw_2019):
    X, y = prepare_diabetes_2019(raw_2019)
    X_train, X_test, _, _ = split(X, y)
    # Only rows with identical answers but different labels can still match.
    assert overlap_between(X_train, X_test) / len(X_test) < 0.15


def test_brfss_binary_target():
    X, y = prepare_brfss(load_brfss(), binary=True)
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(y) == 253_680


def test_split_is_stratified():
    X, y = prepare_brfss(load_brfss())
    _, _, y_train, y_test = split(X, y)
    assert y_train.value_counts(normalize=True).round(3).equals(y_test.value_counts(normalize=True).round(3))
