"""Score the user study: comprehension, System Usability Scale and understanding/trust items.

Enter each participant's results in study/responses.csv (copy responses_template.csv):

- c_*      comprehension points from the answer key (0 or 1; c_2_1 and c_3_1 are 0-2)
- sus_1..sus_10   SUS answers, 1 = strongly disagree ... 5 = strongly agree
- ex_11..ex_17    understanding and trust answers, same 1-5 coding
- minutes  time taken for the task booklet

Run from the project root:  .venv/bin/python study/score_study.py
"""

import sys
from pathlib import Path

import pandas as pd

COMPREHENSION_MAX = {
    "c_1_1": 1, "c_1_2": 1, "c_1_3": 1, "c_1_4": 1,
    "c_2_1": 2, "c_2_2": 1, "c_2_3": 1, "c_2_4": 1,
    "c_3_1": 2, "c_3_2": 1, "c_3_3": 1, "c_3_4": 1, "c_3_5": 1,
    "c_g1": 1, "c_g2": 1,
}
SUS_ITEMS = [f"sus_{i}" for i in range(1, 11)]
EXPLAIN_ITEMS = {
    "ex_11": "Explanations helped me understand each estimate",
    "ex_12": "Explanations were in language I understood",
    "ex_13": "Chart of reasons was easy to read",
    "ex_14": "What if? showed what could make a difference",
    "ex_15": "'Not advice' labels were clear",
    "ex_16": "Would trust estimates as a rough guide",
    "ex_17": "LIME second opinion increased confidence",
}
COLUMNS = ["participant_id", *COMPREHENSION_MAX, *SUS_ITEMS, *EXPLAIN_ITEMS, "minutes"]


def validate(df: pd.DataFrame) -> None:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for col, top in COMPREHENSION_MAX.items():
        bad = df[col].dropna()[(df[col].dropna() < 0) | (df[col].dropna() > top)]
        if len(bad):
            raise ValueError(f"{col} must be between 0 and {top}")
    for col in SUS_ITEMS + list(EXPLAIN_ITEMS):
        bad = df[col].dropna()[~df[col].dropna().isin([1, 2, 3, 4, 5])]
        if len(bad):
            raise ValueError(f"{col} must be coded 1-5")


def sus_score(row: pd.Series) -> float:
    """Brooke (1996): odd items score - 1, even items 5 - score, sum x 2.5."""
    total = sum(row[f"sus_{i}"] - 1 if i % 2 else 5 - row[f"sus_{i}"] for i in range(1, 11))
    return total * 2.5


def score(df: pd.DataFrame) -> pd.DataFrame:
    validate(df)
    out = pd.DataFrame({"participant_id": df["participant_id"]})
    out["comprehension"] = df[list(COMPREHENSION_MAX)].sum(axis=1)
    out["comprehension_pct"] = 100 * out["comprehension"] / sum(COMPREHENSION_MAX.values())
    out["sus"] = df.apply(sus_score, axis=1)
    out["minutes"] = df["minutes"]
    return out


def item_summary(df: pd.DataFrame) -> pd.DataFrame:
    comp = pd.DataFrame({
        "item": list(COMPREHENSION_MAX),
        "pct_correct": [100 * df[c].sum() / (top * len(df)) for c, top in COMPREHENSION_MAX.items()],
    })
    return comp


def explain_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "statement": list(EXPLAIN_ITEMS.values()),
        "median": [df[c].median() for c in EXPLAIN_ITEMS],
        "pct_agree": [100 * (df[c] >= 4).mean() for c in EXPLAIN_ITEMS],
    }, index=list(EXPLAIN_ITEMS))


def main(path: Path) -> None:
    df = pd.read_csv(path)
    scores = score(df)
    print(f"Participants: {len(df)}\n")
    print(scores.round(1).to_string(index=False))
    print(f"\nComprehension: mean {scores['comprehension'].mean():.1f} / {sum(COMPREHENSION_MAX.values())} "
          f"({scores['comprehension_pct'].mean():.0f}%), range {scores['comprehension'].min()}–{scores['comprehension'].max()}")
    print(f"SUS: mean {scores['sus'].mean():.1f} (SD {scores['sus'].std():.1f}); about 68 is commonly treated as average")
    print("\nPer comprehension item (% correct):")
    print(item_summary(df).round(0).to_string(index=False))
    print("\nUnderstanding and trust (1-5):")
    print(explain_summary(df).round(1).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "responses.csv")
