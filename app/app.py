"""Diabetes risk explainer: a Streamlit interface for the BRFSS Gradient Boosting model.

Run from the project root:
    .venv/bin/streamlit run app/app.py

Look: deep navy, turquoise, emerald and warm coral; Plus Jakarta Sans. Theme in
.streamlit/config.toml, page styles in app/style.css.
"""

import html
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explain import (AGE_BANDS, BRFSS_FEATURES, EDUCATION, EDUCATION_CHOICES,  # noqa: E402
                         GEN_HEALTH, INCOME,
                         YES_NO_FEATURES, agreement, label, lime_explainer, lime_weights)
from src.interface import (FACTOR_TYPES, WHAT_IF_TOGGLES, RiskModel, default_profile,  # noqa: E402
                           summary)

st.set_page_config(page_title="Diabetes Risk Explainer", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")

# Colour roles: coral raises risk, turquoise lowers it (a pair that stays distinct with colour
# blindness), emerald marks a good outcome, navy is structure.
CORAL, CORAL_TEXT, CORAL_SOFT = "#ec5741", "#c4402c", "#ffece8"
TURQ, TURQ_TEXT, TURQ_SOFT = "#14b8b0", "#0b7f86", "#e3f7f5"
EMERALD, EMERALD_TEXT, EMERALD_SOFT = "#12a170", "#0a7a52", "#e2f5ec"
AMBER, AMBER_SOFT = "#b26a00", "#fff3e0"
NAVY = "#0b1f3a"
CATEGORY_STYLE = {  # top edge, text colour, soft fill
    "Below average": (EMERALD, EMERALD_TEXT, EMERALD_SOFT),
    "Above average": ("#f59e0b", AMBER, AMBER_SOFT),
    "Higher risk": (CORAL, CORAL_TEXT, CORAL_SOFT),
}
NHS_GUIDANCE = "https://www.nhs.uk/conditions/type-2-diabetes/"
EXAMPLES = json.loads((ROOT / "app/examples.json").read_text())

ICONS = {
    "pulse": '<polyline points="3 12 7 12 10 5 14 19 17 12 21 12"/>',
    "gauge": '<path d="M4 15a8 8 0 1 1 16 0"/><line x1="12" y1="15" x2="16" y2="9"/><circle cx="12" cy="15" r="1.5"/>',
    "shield": '<path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z"/><polyline points="9 12 11 14 15 10"/>',
    "bars": '<line x1="6" y1="20" x2="6" y2="13"/><line x1="12" y1="20" x2="12" y2="8"/><line x1="18" y1="20" x2="18" y2="4"/>',
    "list": '<line x1="9" y1="7" x2="20" y2="7"/><line x1="9" y1="12" x2="20" y2="12"/><line x1="9" y1="17" x2="20" y2="17"/>'
            '<circle cx="5" cy="7" r="1"/><circle cx="5" cy="12" r="1"/><circle cx="5" cy="17" r="1"/>',
    "sliders": '<line x1="4" y1="7" x2="20" y2="7"/><line x1="4" y1="17" x2="20" y2="17"/>'
               '<circle cx="9" cy="7" r="2.5" fill="currentColor"/><circle cx="15" cy="17" r="2.5" fill="currentColor"/>',
    "info": '<circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/><circle cx="12" cy="8" r=".8" fill="currentColor"/>',
    "arrow": '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="13 6 19 12 13 18"/>',
}


def svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote(svg)


def icon(name: str, size: int = 24, stroke: float = 2) -> str:
    """An icon drawn in the element's text colour.

    st.html removes inline <svg>, so the icon is an SVG image used as a CSS mask over a
    block filled with currentColor.
    """
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
           f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
           f'{ICONS[name].replace("currentColor", "black")}</svg>')
    return (f'<i class="rx-ico" aria-hidden="true" style="width:{size}px;height:{size}px;'
            f'--ico:url(&quot;{svg_data_uri(svg)}&quot;)"></i>')


HERO_ART = svg_data_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 120">'
    f'<polyline points="0,70 60,70 78,70 92,28 108,104 124,44 136,70 220,70" fill="none" stroke="{TURQ}" '
    'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="124" cy="44" r="8" fill="#ff6b57"/></svg>'
)


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
    """Fill in the answers and show their estimate straight away (example people, reset)."""
    for f, v in profile.items():
        st.session_state[f] = bool(v) if f in YES_NO_FEATURES else int(v)
    apply_answers()


def current_profile() -> dict:
    """The answers as they stand in the sidebar, applied or not."""
    return {f: int(st.session_state[f]) for f in model.features}


def apply_answers():
    """Results only change when the user confirms, so they always know what the estimate is for."""
    st.session_state["applied"] = current_profile()


if "Age" not in st.session_state:
    set_profile(default_profile())


def bmi_from_height_weight():
    h, w = st.session_state["height_cm"], st.session_state["weight_kg"]
    st.session_state["BMI"] = int(round(w / (h / 100) ** 2))


#  HTML helpers

def md_to_html(text: str) -> str:
    """The reason sentences use **bold** and *italic*; turn them into HTML safely."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)


def card_head(title: str, icon_name: str):
    st.html(f'<div class="rx-cardhead">{icon(icon_name)}<h2>{html.escape(title)}</h2></div>')


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
            f'--c:{CORAL if r["shap"] > 0 else TURQ_TEXT}"></i></span>'
            f'<span class="rx-bar__value">{value}</span></div>'
        )
    return (f'<div class="rx-bars" style="--zero:{zero:.2f}%" role="list" '
            f'aria-label="Effect of each answer on the estimate (SHAP value)">{"".join(rows)}</div>'
            f'<div class="rx-legend"><span><i style="--c:{CORAL}"></i>Raises risk</span>'
            f'<span><i style="--c:{TURQ_TEXT}"></i>Lowers risk</span>'
            f'<span class="rx-legend__unit">Bar length: effect on the estimate (SHAP value, log-odds)</span></div>')


#  Sidebar: Your answers

with st.sidebar:
    st.html(f"""
<div class="rx-brand">
  <span class="rx-brand__mark" style="color:#fff">{icon("pulse", 26, 2.4)}</span>
  <span class="rx-brand__name">Diabetes <em>Risk</em> Explainer</span>
</div>
<h2 class="rx-side-h2">Your answers</h2>
<p class="rx-side-sub">Load an example person, or change any answer and press <b>Update my estimate</b>.</p>
""")
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
    st.selectbox("Education", EDUCATION_CHOICES, format_func=EDUCATION.get, key="Education",
                 help="Pick the highest level of education you have reached.")
    st.selectbox("Household income", list(INCOME), format_func=INCOME.get, key="Income")

    group("Health")
    st.select_slider("In general, how would you rate your health?", options=list(GEN_HEALTH),
                     format_func=GEN_HEALTH.get, key="GenHlth")
    for f in ["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke", "DiffWalk"]:
        st.toggle(label(f), key=f, help=BRFSS_FEATURES[f][1])
    st.slider("Days of poor physical health in the past 30 days", 0, 30, key="PhysHlth")
    st.slider("Days of poor mental health in the past 30 days", 0, 30, key="MentHlth")

    group("Lifestyle")
    for f in ["PhysActivity", "Fruits", "Veggies", "Smoker", "HvyAlcoholConsump"]:
        st.toggle(label(f), key=f, help=BRFSS_FEATURES[f][1])

    group("Health care")
    for f in ["CholCheck", "AnyHealthcare", "NoDocbcCost"]:
        st.toggle(label(f), key=f, help=BRFSS_FEATURES[f][1])

    pending = current_profile() != st.session_state["applied"]
    with st.container(key="apply-bar"):
        if pending:
            st.html('<p class="rx-pending"><span></span>You have changed some answers. '
                    'The results still show your previous answers.</p>')
        else:
            st.html('<p class="rx-applied">The results match these answers.</p>')
        st.button("Update my estimate", on_click=apply_answers, type="primary",
                  disabled=not pending, width="stretch")

profile = st.session_state["applied"]
risk = model.risk(profile)
category, category_text = model.category(risk)
reasons = model.reasons(profile)
test = model.meta["test"]

#  Banner

st.html(f"""
<section class="rx-hero">
  <div>
    <h1>Diabetes <em>Risk</em> Explainer</h1>
    <p>Answer 21 health and lifestyle questions on the left. See an estimated chance of diabetes,
    and which answers raised or lowered it.</p>
    <div class="rx-chips">
      <span class="rx-chip"><b>253,680</b> surveyed US adults</span>
      <span class="rx-chip"><b>{test['accuracy']:.0%}</b> accurate on unseen people</span>
      <span class="rx-chip rx-chip--warn">Not a medical diagnosis</span>
    </div>
  </div>
  <img class="rx-hero__art" src="{HERO_ART}" alt="">
</section>
""")

#  Stat cards

edge, text_colour, soft = CATEGORY_STYLE[category]
times = risk / model.base_rate
st.html(f"""
<div class="rx-stats">
  <div class="rx-stat" style="--c:{TURQ};--soft:{TURQ_SOFT}">
    <span class="rx-stat__icon">{icon("gauge", 26)}</span>
    <div>
      <div class="rx-stat__label">Estimated risk</div>
      <div class="rx-stat__value">{risk:.0%}</div>
      <div class="rx-stat__note">Chance of diabetes from these answers</div>
    </div>
  </div>
  <div class="rx-stat" style="--c:{edge};--soft:{soft};--vc:{text_colour}">
    <span class="rx-stat__icon">{icon("shield", 26)}</span>
    <div>
      <div class="rx-stat__label">Risk level</div>
      <div class="rx-stat__value rx-stat__value--word">{html.escape(category)}</div>
      <div class="rx-stat__note">{html.escape(category_text)}</div>
    </div>
  </div>
  <div class="rx-stat" style="--c:{NAVY};--soft:#e8edf4">
    <span class="rx-stat__icon">{icon("bars", 26)}</span>
    <div>
      <div class="rx-stat__label">Compared with the average</div>
      <div class="rx-stat__value">{times:.1f}×</div>
      <div class="rx-stat__note">{points(risk - model.base_rate)} against the {model.base_rate:.0%} average
      for surveyed adults</div>
    </div>
  </div>
</div>
""")

#  Risk scale

with st.container(key="card-scale"):
    x_max = 1.0 if risk > 0.55 else 0.6
    pos = {"avg": model.base_rate / x_max * 100, "thr": model.threshold / x_max * 100, "you": risk / x_max * 100}
    edge_cls = "rx-you--start" if pos["you"] < 8 else "rx-you--end" if pos["you"] > 92 else ""
    st.html(f"""
<div class="rx-scale-card">
  <div class="rx-scale-title">Where this estimate sits</div>
  <div class="rx-scale" style="--avg:{pos['avg']:.1f}%;--thr:{pos['thr']:.1f}%" role="img"
       aria-label="Estimated risk {risk:.0%}, average {model.base_rate:.0%}, higher-risk level {model.threshold:.0%}">
    <div class="rx-track"></div>
    <div class="rx-mk rx-mk--avg" style="left:{pos['avg']:.1f}%"><span>Average {model.base_rate:.0%}</span></div>
    <div class="rx-mk rx-mk--thr" style="left:{pos['thr']:.1f}%"><span>Higher-risk level {model.threshold:.0%}</span></div>
    <div class="rx-you {edge_cls}" style="left:{pos['you']:.1f}%;--c:{edge}"><span>This person {risk:.0%}</span></div>
  </div>
  <div class="rx-axis"><span>0%</span><span>{x_max / 2:.0%}</span><span>{x_max:.0%}</span></div>
  <div class="rx-scale-key">Lines: average {model.base_rate:.0%}, higher-risk level {model.threshold:.0%}</div>
</div>
""")

#  Reasons and what-if

why_col, whatif_col = st.columns([7, 5], gap="medium")

with why_col, st.container(key="card-why"):
    card_head("Why this estimate?", "list")
    st.html('<p class="rx-sub">The answers that moved this estimate most, compared with an average adult in the survey.</p>')
    s = summary(reasons)
    raises = "".join(f"<li>{md_to_html(t)}</li>" for t in s["raises"]) or \
        "<li>Nothing in these answers noticeably raises the risk.</li>"
    lowers = "".join(f"<li>{md_to_html(t)}</li>" for t in s["lowers"]) or \
        "<li>Nothing in these answers noticeably lowers the risk.</li>"
    st.html(f"""
<div class="rx-reasons">
  <div style="--soft:{CORAL_SOFT};--c:{CORAL};--hc:{CORAL_TEXT}"><h3><i></i>Raising your risk</h3><ul>{raises}</ul></div>
  <div style="--soft:{TURQ_SOFT};--c:{TURQ_TEXT};--hc:{TURQ_TEXT}"><h3><i></i>Lowering your risk</h3><ul>{lowers}</ul></div>
</div>""")
    st.html(reasons_chart(reasons.head(8)))
    st.html('<p class="rx-caption">The 8 answers that moved the estimate most. These are patterns the model '
            'learned from survey data, not proof of cause.</p>')

with whatif_col, st.container(key="card-whatif"):
    card_head("What if?", "sliders")
    st.html('<p class="rx-sub">Try changing things that people can realistically change. Other factors are '
            'left out, because changing them would not be sensible advice.</p>')
    # Widgets keep their own state, so start the what-if controls again from the
    # current answers whenever those answers change.
    if st.session_state.get("_whatif_base") != profile:
        st.session_state["_whatif_base"] = profile
        st.session_state["whatif_BMI"] = int(min(max(profile["BMI"], 15), 60))
        for f in WHAT_IF_TOGGLES:
            st.session_state[f"whatif_{f}"] = bool(profile[f])
    changes = {"BMI": st.slider("BMI", 15, 60, key="whatif_BMI")}
    for f in WHAT_IF_TOGGLES:
        changes[f] = int(st.toggle(label(f), key=f"whatif_{f}"))
    changes = {k: v for k, v in changes.items() if v != profile[k]}
    if changes:
        new_risk = model.what_if(profile, changes)
        diff = new_risk - risk
        described = ", ".join(f"{label(k)} {v}" if k == "BMI" else f"{label(k)} {'yes' if v else 'no'}"
                              for k, v in changes.items())
        tone = (EMERALD, EMERALD_TEXT, EMERALD_SOFT) if diff < 0 else \
            (CORAL, CORAL_TEXT, CORAL_SOFT) if diff > 0 else ("#d5dde8", "#4f5f78", "#f2f5f9")
        st.html(f"""
<div class="rx-whatif" style="--c:{tone[0]};--tc:{tone[1]};--soft:{tone[2]}">
  <b>{new_risk:.0%}</b>
  <span>estimated risk with {html.escape(described)}</span>
  <span class="rx-change">{points(diff)}</span>
</div>""")
    else:
        st.html("<p class='rx-caption'>Move the slider or switch a toggle to see how the estimate would change.</p>")

#  Second opinion and about

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
            f"<td><span class='rx-dir' style='--c:{CORAL if v > 0 else TURQ_TEXT};"
            f"--tc:{CORAL_TEXT if v > 0 else TURQ_TEXT}'><i></i>{'Raises risk' if v > 0 else 'Lowers risk'}</span></td>"
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

#  Notice and footer

st.html(f"""
<section class="rx-notice">
  <span class="rx-notice__icon">{icon("info", 26)}</span>
  <div>
    <strong>This is not a medical diagnosis.</strong>
    <span>The estimate comes from 2015 US survey answers. If you are worried about diabetes,
    speak to a doctor or pharmacist.</span>
  </div>
  <a href="{NHS_GUIDANCE}" target="_blank" rel="noopener">Read NHS guidance {icon("arrow", 18, 2.4)}</a>
</section>
<footer class="rx-foot">
  <span>Diabetes Risk Explainer · research prototype, not medical advice</span>
  <span>Data: CDC Behavioral Risk Factor Surveillance System 2015</span>
</footer>
""")
