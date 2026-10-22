# Interface design

The app lives in `app/app.py`, with the logic in `src/interface.py` (tested in `tests/test_interface.py`).

## How to run it

```bash
.venv/bin/streamlit run app/app.py
```

Then open http://localhost:8501.

## Layout

| Area | What it shows |
|---|---|
| **Disclaimer** (top) | Not a diagnosis; built from a 2015 US survey; speak to a doctor or pharmacist if worried |
| **Your answers** (left) | The 21 survey questions, in plain language, grouped into About you, Health, Lifestyle and Health care. Includes a BMI calculator and buttons for the three example people |
| **Estimated risk** | Risk %, the difference from the 14% average, a category (below average / above average / higher risk) and a scale marking the average and the 28% higher-risk level |
| **Why this estimate?** | Up to three plain-language reasons raising and three lowering risk, then a chart of the 8 biggest SHAP effects |
| **What if?** | Change BMI, physical activity, fruit or vegetables and see the new estimate |
| **Second opinion (LIME)** | On request: LIME's top 5 reasons and how far they agree with SHAP |
| **About this tool** | Data, model, accuracy, limits and the meaning of each type of factor |

## Design decisions and the evidence behind them

| Decision | Evidence (notebook) |
|---|---|
| Gradient Boosting without oversampling | Well-calibrated probabilities: mean predicted 13.9% vs actual 13.9% (04) |
| "Higher risk" from 28% | Threshold that maximised CV macro-F1 on training data (04) |
| SHAP as the main explanation | Exact, stable and about 2 ms per person; LIME varies between runs (05) |
| Plain-language sentences ("strongly / moderately / slightly") | Miller (2019): people prefer a few selective reasons; SHAP bands ≥0.5 / ≥0.2 / below 0.2 log-odds |
| What-if only for BMI, physical activity, fruit, vegetables | Avoid suggesting changes to fixed factors (age) or to non-causal associations (05) |
| "Pattern in the data, not advice" label for heavy drinking, cholesterol check, health cover, cost, income, education and mental-health days | These lower or raise predicted risk for reasons that aren't causes (05) |
| "Can be managed with a doctor" for high blood pressure and cholesterol | They raise risk and are treatable, but not by the user alone |
| Example people A, B and C | The user study uses example data, not participants' own health information (ethics, proposal §5.4). They are real test-set people at about 28%, 60% and 3% risk, labelled so participants can't guess the risk |
| LIME as an optional second opinion | Lets the user study compare one vs two explanation methods (05) |

## Checked in the browser (23 Oct 2026)

- All three example people load, and the risk, category, reasons and chart update.
- The what-if controls reset to each person's answers. BMI 33→26 for Person B: 60% → 42%. Physical activity on: 60% → 59%.
- LIME runs on request and reports agreement with SHAP.
- Works in both light and dark mode, with no browser console errors when switching between people.

## Known limitations

- Survey answers only, from the USA in 2015, so it can't account for blood tests or family history.
- About half of the people with diabetes in the test set were not flagged as higher risk.
- SHAP values are on the log-odds scale, so the chart shows relative size, not percentage points.
