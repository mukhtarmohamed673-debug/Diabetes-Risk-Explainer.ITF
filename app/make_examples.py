"""Create the example profiles used in the interface and the user study.

Three anonymised people from the BRFSS test set, chosen by predicted risk:
about 3%, about the model's threshold, and about 60%. They are labelled
A, B and C (not by risk) so participants don't know their risk in advance.

Run from the project root:  .venv/bin/python app/make_examples.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interface import RiskModel  # noqa: E402

model = RiskModel()
test = pd.read_csv(ROOT / "data/processed/brfss_bin_test.csv").drop(columns="Diabetes_binary")
risk = model.pipe.predict_proba(test)[:, 1]

targets = {"Person A": model.threshold, "Person B": 0.60, "Person C": 0.03}
examples = {}
for name, target in targets.items():
    i = int(np.argmin(np.abs(risk - target)))
    examples[name] = {k: int(v) for k, v in test.iloc[i].items()}
    print(f"{name}: test row {i}, risk {risk[i]:.1%}")

(ROOT / "app/examples.json").write_text(json.dumps(examples, indent=2))
