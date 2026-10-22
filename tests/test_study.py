import pandas as pd
import pytest

from study.score_study import COLUMNS, COMPREHENSION_MAX, score, sus_score


def row(**overrides):
    base = {c: 3 for c in COLUMNS}
    base.update({c: top for c, top in COMPREHENSION_MAX.items()})
    base.update(participant_id="P01", minutes=12)
    base.update(overrides)
    return base


def test_template_has_every_column():
    template = pd.read_csv("study/responses_template.csv")
    assert list(template.columns) == COLUMNS


def test_sus_extremes_and_midpoint():
    best = pd.Series({f"sus_{i}": 5 if i % 2 else 1 for i in range(1, 11)})
    worst = pd.Series({f"sus_{i}": 1 if i % 2 else 5 for i in range(1, 11)})
    neutral = pd.Series({f"sus_{i}": 3 for i in range(1, 11)})
    assert sus_score(best) == 100 and sus_score(worst) == 0 and sus_score(neutral) == 50


def test_full_marks_comprehension():
    out = score(pd.DataFrame([row()]))
    assert out["comprehension"].iloc[0] == 17 and out["comprehension_pct"].iloc[0] == 100


def test_invalid_values_rejected():
    with pytest.raises(ValueError, match="c_2_1"):
        score(pd.DataFrame([row(c_2_1=3)]))
    with pytest.raises(ValueError, match="sus_4"):
        score(pd.DataFrame([row(sus_4=6)]))
