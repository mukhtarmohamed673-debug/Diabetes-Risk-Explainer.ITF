# Interface design

The app lives in `app/app.py`, with the logic in `src/interface.py` (tested in `tests/test_interface.py`).
Its look is set by `.streamlit/config.toml` (theme) and `app/style.css` (page styles).

## How to run it

```bash
.venv/bin/streamlit run app/app.py
```

Then open http://localhost:8501.

## Layout

| Area | What it shows |
|---|---|
| **Header** (top) | The tool's name, a headline saying what it does, three facts (253,680 surveyed adults, 83% accuracy on unseen people, 21 questions) and the "not a medical diagnosis" notice |
| **Your answers** (left) | The 21 survey questions, in plain language, grouped into About you, Health, Lifestyle and Health care. Includes a BMI calculator, buttons for the three example people and "Reset answers" |
| **Estimated risk** | Risk %, the difference from the 14% average, a category (below average / above average / higher risk) and a drawn scale marking the average and the 28% higher-risk level |
| **Why this estimate?** | Up to three plain-language reasons raising and three lowering risk, then a bar chart of the 8 biggest SHAP effects around a zero line (right raises risk, left lowers it) |
| **What if?** | Change BMI, physical activity, fruit or vegetables and see the new estimate |
| **Second opinion (LIME)** | On request: LIME's top 5 reasons and how far they agree with SHAP |
| **About this tool** | Data, model, accuracy, limits and the meaning of each type of factor |

## Visual design

Built to the rules of a professional design system: one type family, one accent colour, one card
style, one button pair, plain wording, and small, consistent hover motion.

| Element | Choice | Why |
|---|---|---|
| Colour | Stone neutrals (`#eeecea` page, `#faf9f7` cards, `#333231` text) with one sage accent (`#5c6b54`) for buttons, toggles and focus | A calm, clinical-feeling palette; the interface stays mostly neutral so the result colours stand out |
| Result colours | Rust (`#9a4a2e`) raises risk, slate blue (`#3b5870`) lowers it; amber for "above average" | An orange/blue pair stays distinguishable for people with red-green colour blindness |
| Contrast | Body text 10.9–12.2:1; grey labels 4.8:1 or more; rust and slate 5.3:1 or more; amber (dots only, never text) 4.3:1 | Meets WCAG AA (4.5:1 text, 3:1 graphics); reading text exceeds 7:1 |
| Type | IBM Plex Sans, with IBM Plex Mono for figures | A neutral, technical typeface associated with trustworthy research software |
| Shape | Cards 20px radius, fields 12px, buttons 8px; one 1px hairline | One radius family and one card recipe throughout |
| Motion | Buttons step 1px up on hover and press 1px down; focus shows a 2px accent ring; all motion stops under "reduce motion" | Every control visibly answers the pointer; nothing moves by itself |
| Charts | The risk scale and reasons chart are drawn in HTML/CSS | Matches the page's type and colours exactly, and avoids a Streamlit chart bug that left charts blank when the answers changed |
| Layout | Two columns from 1024px; one column below. Cards adapt to their own width (container queries) | Readable from phone (375px) to desktop (1440px), with no sideways scrolling |

The labels the user study pack refers to are unchanged: "Your answers", "Person A/B/C", "Reset
answers", "What if?", "Physical activity", "Second opinion (LIME)", "Show LIME explanation",
"About this tool" and the three category names.

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

## Checked in the browser (5 Nov 2026, after the redesign)

- All three example people load, and the risk, category, reasons and chart update, with no browser
  errors across repeated switching.
- What if?: BMI 42→25 for Person A gives 8% (−20 points).
- LIME runs on request and reports agreement with SHAP.
- Checked at 1440px (desktop), 768px (tablet) and 375px (phone): no sideways scrolling, nothing cut off.
- Button hover and press motion confirmed; IBM Plex Sans loads.

## Known limitations

- Survey answers only, from the USA in 2015, so it can't account for blood tests or family history.
- About half of the people with diabetes in the test set were not flagged as higher risk.
- SHAP values are on the log-odds scale, so the chart shows relative size, not percentage points.
