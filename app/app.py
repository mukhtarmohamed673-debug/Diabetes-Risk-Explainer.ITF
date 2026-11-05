"""Diabetes risk explainer: a Streamlit interface for the BRFSS Gradient Boosting model.

Run from the project root:
    .venv/bin/streamlit run app/app.py

Look: palette "three stones", IBM Plex Sans, soft radius. Theme in .streamlit/config.toml,
page styles in app/style.css.
"""

import html
import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explain import (AGE_BANDS, BRFSS_FEATURES, EDUCATION, GEN_HEALTH, INCOME,  # noqa: E402
                         YES_NO_FEATURES, agreement, label, lime_explainer, lime_weights)
from src.interface import (FACTOR_TYPES, WHAT_IF_TOGGLES, RiskModel, default_profile,  # noqa: E402
                           summary)

st.set_page_config(page_title="Diabetes Risk Explainer", page_icon="🩺", layout="wide")

RAISES, LOWERS = "#9a4a2e", "#3b5870"
CATEGORY_COLOUR = {"Below average": "#5c6b54", "Above average": "#8a6a1c", "Higher risk": RAISES}
EXAMPLES = json.loads((ROOT / "app/examples.json").read_text())


@st.cache_resource
def load_model() -> RiskModel:
    return RiskModel()


@st.cache_resource
def load_lime(_model: RiskModel):
    # 5,000 training rows saved by app/make_examples.py
    return lime_explainer(_model.pipe, pd.read_csv(ROOT / "app/lime_background.csv"))


model = load_model()
st.html(f"<style>{(ROOT / 'app/style.css').read_text()}</style>")


#  State

def set_profile(profile: dict):
    for f, v in profile.items():
        st.session_state[f] = bool(v) if f in YES_NO_FEATURES else int(v)


if "Age" not in st.session_state:
    set_profile(default_profile())


def current_profile() -> dict:
    return {f: int(st.session_state[f]) for f in model.features}


def bmi_from_height_weight():
    h, w = st.session_state["height_cm"], st.session_state["weight_kg"]
    st.session_state["BMI"] = int(round(w / (h / 100) ** 2))


#  HTML helpers

def md_to_html(text: str) -> str:
    """The reason sentences use **bold** and *italic*; turn them into HTML safely."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)


def heading(title: str, sub: str = ""):
    st.html(f'<h2 class="rx-h2">{html.escape(title)}</h2>' + (f'<p class="rx-sub">{html.escape(sub)}</p>' if sub else ""))


def group(title: str):
    st.html(f'<div class="rx-group">{html.escape(title)}</div>')


def points(diff: float) -> str:
    return f"{diff * 100:+.0f} points".replace("-", "−")


def reasons_chart(top: pd.DataFrame) -> str:
    """Horizontal bars of SHAP values around a zero line: right raises risk, left lowers it."""
    lo, hi = min(0.0, top["shap"].min()), max(0.0, top["shap"].max())
    span = (hi - lo) or 1.0
    zero = -lo / span * 100
    rows = []
    for _, r in top.iterrows():
        start = zero + min(0.0, r["shap"]) / span * 100
        width = abs(r["shap"]) / span * 100
        effect = "Raises risk" if r["shap"] > 0 else "Lowers risk"
        tip = f"{effect} {r['strength']} · {FACTOR_TYPES[r['type']]}"
        value = f"{r['shap']:+.2f}".replace("-", "−")
        rows.append(
            f'<div class="rx-bar" role="listitem" title="{html.escape(tip)}">'
            f'<span class="rx-bar__label">{html.escape(r["label"])} = {html.escape(r["value"])}</span>'
            f'<span class="rx-bar__track"><i style="left:{start:.2f}%;width:{max(width, .6):.2f}%;'
            f'--c:{RAISES if r["shap"] > 0 else LOWERS}"></i></span>'
            f'<span class="rx-bar__value">{value}</span></div>'
        )
    return (f'<div class="rx-bars" style="--zero:{zero:.2f}%" role="list" '
            f'aria-label="Effect of each answer on the estimate (SHAP value)">{"".join(rows)}</div>'
            f'<div class="rx-legend"><span><i style="--c:{RAISES}"></i>Raises risk</span>'
            f'<span><i style="--c:{LOWERS}"></i>Lowers risk</span>'
            f'<span class="rx-legend__unit">Bar length: effect on the estimate (SHAP value, log-odds)</span></div>')


#  Header

test = model.meta["test"]
st.html(f"""
<header class="rx-top">
  <div class="rx-brand"><span class="rx-mark" aria-hidden="true"></span>Diabetes Risk Explainer</div>
  <div class="rx-tag">Research prototype · final-year project</div>
</header>
<section class="rx-intro">
  <div>
    <h1>Diabetes risk check with the reasons explained</h1>
    <p class="rx-lead">Answer 21 health and lifestyle questions. The tool estimates the chance of diabetes
    and shows which answers raised or lowered that estimate, and by how much.</p>
  </div>
  <div class="rx-facts">
    <div class="rx-fact"><b>253,680</b><span>US adults in the 2015 health survey the model learned from</span></div>
    <div class="rx-fact"><b>{test['accuracy']:.0%}</b><span>accuracy on 50,736 people the model never saw</span></div>
    <div class="rx-fact"><b>21</b><span>questions, each one explained in every result</span></div>
  </div>
</section>
<div class="rx-note"><span><strong>This is not a medical diagnosis.</strong> It is a research tool built from
a 2015 US health survey. If you are worried about diabetes, speak to a doctor or pharmacist.</span></div>
""")

form_col, result_col = st.columns([5, 7], gap="large")

#  Inputs

with form_col, st.container(key="card-answers"):
    heading("Your answers", "Load an example person, or change any answer. The estimate updates straight away.")
    ex_cols = st.columns(len(EXAMPLES))
    for col, (name, profile) in zip(ex_cols, EXAMPLES.items()):
        col.button(name, on_click=set_profile, args=(profile,), width="stretch")
    st.button("Reset answers", on_click=set_profile, args=(default_profile(),), type="tertiary")

    group("About you")
    c1, c2 = st.columns(2)
    c1.selectbox("Age group", list(AGE_BANDS), format_func=AGE_BANDS.get, key="Age")
    c2.radio("Sex", [0, 1], format_func={0: "Female", 1: "Male"}.get, key="Sex", horizontal=True)
    st.number_input("BMI (body mass index)", min_value=12, max_value=98, step=1, key="BMI")
    with st.popover("Don't know your BMI?"):
        st.number_input("Height (cm)", 120, 220, 170, key="height_cm")
        st.number_input("Weight (kg)", 30, 250, 75, key="weight_kg")
        st.button("Use this BMI", on_click=bmi_from_height_weight)
    st.selectbox("Education", list(EDUCATION), format_func=EDUCATION.get, key="Education")
    st.selectbox("Household income", list(INCOME), format_func=INCOME.get, key="Income")

    group("Health")
    st.select_slider("In general, how would you rate your health?", options=list(GEN_HEALTH),
                     format_func=GEN_HEALTH.get, key="GenHlth")
    c1, c2 = st.columns(2)
    for i, f in enumerate(["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke", "DiffWalk"]):
        (c1 if i % 2 == 0 else c2).toggle(label(f), key=f, help=BRFSS_FEATURES[f][1])
    st.slider("Days of poor physical health in the past 30 days", 0, 30, key="PhysHlth")
    st.slider("Days of poor mental health in the past 30 days", 0, 30, key="MentHlth")

    group("Lifestyle")
    c1, c2 = st.columns(2)
    for i, f in enumerate(["PhysActivity", "Fruits", "Veggies", "Smoker", "HvyAlcoholConsump"]):
        (c1 if i % 2 == 0 else c2).toggle(label(f), key=f, help=BRFSS_FEATURES[f][1])

    group("Health care")
    for f in ["CholCheck", "AnyHealthcare", "NoDocbcCost"]:
        st.toggle(label(f), key=f, help=BRFSS_FEATURES[f][1])

profile = current_profile()
risk = model.risk(profile)
category, category_text = model.category(risk)
reasons = model.reasons(profile)

#  Results

with result_col:
    with st.container(key="card-risk"):
        colour = CATEGORY_COLOUR[category]
        x_max = 1.0 if risk > 0.55 else 0.6
        pos = {"avg": model.base_rate / x_max * 100, "thr": model.threshold / x_max * 100, "you": risk / x_max * 100}
        st.html(f"""
<div class="rx-risk">
  <div>
    <div class="rx-label">Estimated risk</div>
    <div class="rx-figure">{risk:.0%}</div>
    <div class="rx-delta">{points(risk - model.base_rate)} vs the {model.base_rate:.0%} average</div>
  </div>
  <div>
    <span class="rx-pill" style="--c:{colour}"><i></i>{html.escape(category)}</span>
    <p>{html.escape(category_text)}</p>
  </div>
</div>
<div class="rx-scale" style="--avg:{pos['avg']:.1f}%;--thr:{pos['thr']:.1f}%" role="img"
     aria-label="Estimated risk {risk:.0%}, average {model.base_rate:.0%}, higher-risk level {model.threshold:.0%}">
  <div class="rx-track"></div>
  <div class="rx-mk rx-mk--avg" style="left:{pos['avg']:.1f}%"><span>Average {model.base_rate:.0%}</span></div>
  <div class="rx-mk rx-mk--thr" style="left:{pos['thr']:.1f}%"><span>Higher-risk level {model.threshold:.0%}</span></div>
  <div class="rx-you {'rx-you--start' if pos['you'] < 8 else 'rx-you--end' if pos['you'] > 92 else ''}"
       style="left:{pos['you']:.1f}%;--c:{colour}"><span>You {risk:.0%}</span></div>
</div>
<div class="rx-axis"><span>0%</span><span>{x_max / 2:.0%}</span><span>{x_max:.0%}</span></div>
<div class="rx-scale-key">Lines: average {model.base_rate:.0%}, higher-risk level {model.threshold:.0%}</div>
""")

    with st.container(key="card-why"):
        heading("Why this estimate?", "The answers that moved this estimate most, compared with an average adult in the survey.")
        s = summary(reasons)
        raises = "".join(f"<li>{md_to_html(t)}</li>" for t in s["raises"]) or \
            "<li>Nothing in these answers noticeably raises the risk.</li>"
        lowers = "".join(f"<li>{md_to_html(t)}</li>" for t in s["lowers"]) or \
            "<li>Nothing in these answers noticeably lowers the risk.</li>"
        st.html(f"""
<div class="rx-reasons">
  <div><h3><i style="--c:{RAISES}"></i>Raising your risk</h3><ul>{raises}</ul></div>
  <div><h3><i style="--c:{LOWERS}"></i>Lowering your risk</h3><ul>{lowers}</ul></div>
</div>""")

        st.html(reasons_chart(reasons.head(8)))
        st.html('<p class="rx-caption">The 8 answers that moved the estimate most. These are patterns the model '
                'learned from survey data, not proof of cause.</p>')

    #  What-if
    with st.container(key="card-whatif"):
        heading("What if?", "Try changing things that people can realistically change. Other factors are left out, "
                            "because changing them would not be sensible advice.")
        # Widgets keep their own state, so start the what-if controls again from the
        # current answers whenever those answers change.
        if st.session_state.get("_whatif_base") != profile:
            st.session_state["_whatif_base"] = profile
            st.session_state["whatif_BMI"] = int(min(max(profile["BMI"], 15), 60))
            for f in WHAT_IF_TOGGLES:
                st.session_state[f"whatif_{f}"] = bool(profile[f])
        w1, w2 = st.columns(2, gap="large")
        changes = {"BMI": w1.slider("BMI", 15, 60, key="whatif_BMI")}
        for f in WHAT_IF_TOGGLES:
            changes[f] = int(w2.toggle(label(f), key=f"whatif_{f}"))
        changes = {k: v for k, v in changes.items() if v != profile[k]}
        if changes:
            new_risk = model.what_if(profile, changes)
            diff = new_risk - risk
            described = ", ".join(f"{label(k)} {v}" if k == "BMI" else f"{label(k)} {'yes' if v else 'no'}"
                                  for k, v in changes.items())
            tone = LOWERS if diff < 0 else RAISES if diff > 0 else "#6b6763"
            st.html(f"""
<div class="rx-whatif">
  <b>{new_risk:.0%}</b>
  <span>estimated risk with {html.escape(described)}</span>
  <span class="rx-change" style="--c:{tone}">{points(diff)}</span>
</div>""")
        else:
            st.html('<p class="rx-caption">Move the slider or switch a toggle to see how the estimate would change.</p>')

    #  LIME
    with st.expander("Second opinion (LIME)"):
        st.markdown("The explanation above uses **SHAP**. **LIME** works differently: it tests many small "
                    "variations of the answers. When the two methods agree, the explanation is more trustworthy.")
        if st.button("Show LIME explanation"):
            row = pd.DataFrame([profile])[model.features]
            lime = lime_weights(model.pipe, load_lime(model), row)
            by_feature = reasons.set_index("feature")
            agree = agreement(by_feature["shap"].reindex(model.features), lime)
            lime_top = lime.reindex(lime.abs().sort_values(ascending=False).index).head(5)
            rows = "".join(
                f"<tr><td>{i}</td><td>{html.escape(label(f))} = {html.escape(by_feature.loc[f, 'value'])}</td>"
                f"<td><span class='rx-dir' style='--c:{RAISES if v > 0 else LOWERS}'><i></i>"
                f"{'Raises risk' if v > 0 else 'Lowers risk'}</span></td>"
                f"<td>{'Pattern in the data, not advice' if by_feature.loc[f, 'type'] == 'association' else ''}</td></tr>"
                for i, (f, v) in enumerate(lime_top.items(), start=1)
            )
            st.html(f"""
<table class="rx-lime">
  <thead><tr><th></th><th>Answer</th><th>Effect</th><th>Note</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<p class="rx-caption">SHAP and LIME share <strong>{agree['top_k_overlap']:.0%}</strong> of their top-5 answers,
and agree on the direction for <strong>{agree['sign_agreement']:.0%}</strong> of the shared ones.</p>""")

    with st.expander("About this tool"):
        st.markdown(f"""
- **Data:** 253,680 adults in the 2015 US Behavioral Risk Factor Surveillance System (BRFSS) survey.
  "Diabetes" means the person had been told by a doctor that they have diabetes.
- **Model:** Gradient Boosting, tested on 50,736 people it had not seen during training.
  Accuracy {test['accuracy']:.0%}, ROC-AUC {test['roc_auc']:.2f}. Predicted risks match observed rates closely
  (average predicted {model.base_rate:.0%}).
- **Limits:** about half of the people with diabetes in the test data were not flagged as higher risk.
  The model learns patterns from survey answers, not causes, and the survey is from the USA.
- **Types of factor:** {"; ".join(f"*{v}*" for v in FACTOR_TYPES.values())}.
- **Explanations:** SHAP values (exact for this type of model). LIME is offered as a second opinion.
""")

st.html("""
<footer class="rx-foot">
  <span>Diabetes Risk Explainer · research prototype, not medical advice</span>
  <span>Data: CDC Behavioral Risk Factor Surveillance System 2015</span>
</footer>""")
