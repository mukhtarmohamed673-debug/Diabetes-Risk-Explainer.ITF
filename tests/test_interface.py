import pytest

from src.interface import ASSOCIATION_ONLY, CAN_CHANGE, RiskModel, default_profile, sentence, summary


@pytest.fixture(scope="module")
def model():
    return RiskModel()


def test_risk_is_a_probability(model):
    r = model.risk(default_profile())
    assert 0 < r < 1


def test_missing_answer_is_rejected(model):
    profile = default_profile()
    del profile["BMI"]
    with pytest.raises(ValueError, match="BMI"):
        model.risk(profile)


def test_categories_follow_average_and_threshold(model):
    assert model.category(model.base_rate / 2)[0] == "Below average"
    assert model.category((model.base_rate + model.threshold) / 2)[0] == "Above average"
    assert model.category(model.threshold + 0.1)[0] == "Higher risk"


def test_reasons_cover_every_feature_and_are_sorted(model):
    reasons = model.reasons(default_profile())
    assert len(reasons) == 21
    assert reasons["shap"].abs().is_monotonic_decreasing
    assert set(reasons["direction"]) <= {"raises", "lowers"}


def test_worse_profile_has_higher_risk(model):
    worse = {**default_profile(), "HighBP": 1, "GenHlth": 5, "BMI": 40, "Age": 11}
    assert model.risk(worse) > model.risk(default_profile())


def test_what_if_only_allows_changeable_factors(model):
    profile = {**default_profile(), "BMI": 38}
    assert model.what_if(profile, {"BMI": 26}) < model.risk(profile)
    with pytest.raises(ValueError):
        model.what_if(profile, {"Age": 1})
    with pytest.raises(ValueError):
        model.what_if(profile, {"HvyAlcoholConsump": 1})


def test_association_factors_never_read_as_advice(model):
    reasons = model.reasons({**default_profile(), "HvyAlcoholConsump": 1, "CholCheck": 0})
    for _, r in reasons[reasons["feature"].isin(ASSOCIATION_ONLY)].iterrows():
        assert "not advice" in sentence(r)
    assert not (CAN_CHANGE & ASSOCIATION_ONLY)


def test_summary_splits_raising_and_lowering(model):
    s = summary(model.reasons({**default_profile(), "HighBP": 1, "Age": 2}))
    assert any("High blood pressure" in t and "raises" in t for t in s["raises"])
    assert any("Age group" in t and "lowers" in t for t in s["lowers"])
