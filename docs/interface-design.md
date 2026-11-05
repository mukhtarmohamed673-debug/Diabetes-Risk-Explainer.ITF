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
| **Your answers** (navy sidebar, left) | The tool's logo, then the 21 survey questions in plain language, grouped into About you, Health, Lifestyle and Health care. Person A/B/C buttons and "Reset answers" at the top; a BMI calculator. An **Update my estimate** button stays at the bottom of the sidebar. On narrow screens it folds away behind a » button |
| **Banner** | The tool's name ("Risk" in coral), what it does, and three chips: 253,680 surveyed adults, 83% accuracy on unseen people, "Not a medical diagnosis" |
| **Stat cards** | Three cards with coloured top edges and icons: Estimated risk (%), Risk level (below average / above average / higher risk) and Compared with the average (× and points) |
| **Where this estimate sits** | A drawn scale marking the 14% average and the 28% higher-risk level |
| **Why this estimate?** | Up to three plain-language reasons raising and three lowering risk, in coral and turquoise panels, then a bar chart of the 8 biggest SHAP effects around a zero line (right raises risk, left lowers it) |
| **What if?** | Change BMI, physical activity, fruit or vegetables and see the new estimate |
| **Second opinion (LIME)** | On request: LIME's top 5 reasons and how far they agree with SHAP |
| **About this tool** | Data, model, accuracy, limits and the meaning of each type of factor |
| **Notice** (bottom) | "This is not a medical diagnosis", with a link to NHS guidance on type 2 diabetes |

## Visual design

A dashboard layout with strong contrast: a deep navy frame, white cards, and colour that always
carries a meaning.

| Element | Choice | Why |
|---|---|---|
| Structure | Deep navy (`#0b1f3a`) sidebar, banner and card headers; light page (`#f2f5f9`) and white cards | Navy signals trust and gives the page a clear frame; white cards keep the results easy to read |
| Turquoise (`#14b8b0`, text `#0b7f86`) | Controls (toggles, sliders, focus) and answers that **lower** risk | One colour for "things you interact with" and for reassurance |
| Warm coral (`#ff6b57`, bars `#ec5741`, text `#c4402c`) | Answers that **raise** risk, "Higher risk", the notice banner | Warm and noticeable without the alarm of pure red |
| Emerald (`#12a170`, text `#0a7a52`) | Good outcomes: "Below average", a what-if that lowers the estimate | Green for "better" |
| Colour blindness | Raising (coral) and lowering (turquoise) never rely on red against green | An orange/blue-green pair stays distinguishable with red-green colour blindness; bars also differ by direction and sign |
| Contrast | Text 15–16.5:1 on the page and cards, 16.5:1 white on navy; grey text 5.9:1 or more; coloured text 4.8:1 or more; coral bars 3.5:1 | Meets WCAG AA (4.5:1 text, 3:1 graphics) |
| Type | Plus Jakarta Sans, 400 to 800 weight | Friendly and modern, with strong bold weights for figures and headings |
| Icons | Simple line icons (gauge, shield, bars, list, sliders, info) in each card's colour | Help people find sections at a glance; drawn as CSS masks because Streamlit removes inline SVG |
| Motion | Buttons step 1px up on hover and press 1px down; focus shows a turquoise ring; all motion stops under "reduce motion" | Every control visibly answers the pointer; nothing moves by itself |
| Charts | The risk scale and reasons chart are drawn in HTML/CSS | Matches the page's type and colours exactly, and avoids a Streamlit chart bug that left charts blank when the answers changed |
| Layout | Sidebar 380px when open; results in two columns above 1100px, one below. Cards adapt to their own width (container queries) | Readable from phone (375px) to desktop (1440px), with no sideways scrolling |

Not taken from the reference image, on purpose: health tips (the tool gives no advice, because the
model learns associations, not causes), a donut chart of percentage "contributions" (SHAP values are
not shares of the risk) and a vague "Learn more" button (replaced by a real link to NHS guidance).

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
| Changed answers only count after **Update my estimate** | Live updating left people unsure whether the results had changed. The button turns turquoise and a coral note says "You have changed some answers" until it is pressed; then it greys out and says the results match. Example people and "Reset answers" still apply straight away, so the study tasks are unchanged |
| Education in UK terms: never attended school, primary, secondary, college, university | Participants are in the UK. Each option uses one US survey code (never attended, grades 1–8, high school graduate, some college, college graduate); the example people and default answers already use these codes, so their estimates are unchanged |
| Example people A, B and C | The user study uses example data, not participants' own health information (ethics, proposal §5.4). They are real test-set people at about 28%, 60% and 3% risk, labelled so participants can't guess the risk |
| LIME as an optional second opinion | Lets the user study compare one vs two explanation methods (05) |

## Checked in the browser (5 Nov 2026, navy dashboard design)

- All three example people load, and the stat cards, scale, reasons and chart update, with no
  browser errors.
- What if?, LIME and the NHS link work.
- Changing an answer shows the "changed" note without changing the results; Update my estimate then
  updates them (BMI 27 to 38: 2% to 6%). The update bar stays at the bottom of the sidebar while it scrolls.
- Checked at 1440px, 1024px and 375px: no sideways scrolling, nothing cut off. The sidebar folds
  away and reopens with its » button on desktop and phone.
- The study flow (Persons C, A and B, Reset answers, What if?, LIME) runs without errors.

## Known limitations

- Survey answers only, from the USA in 2015, so it can't account for blood tests or family history.
- About half of the people with diabetes in the test set were not flagged as higher risk.
- SHAP values are on the log-odds scale, so the chart shows relative size, not percentage points.
