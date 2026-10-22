"""Diabetes risk explainer: a Streamlit interface for the BRFSS Gradient Boosting model.

Run from the project root:
    .venv/bin/streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explain import (AGE_BANDS, EDUCATION, GEN_HEALTH, INCOME, YES_NO, BRFSS_FEATURES,  # noqa: E402
                         agreement, label, lime_explainer, lime_weights)
from src.interface import (CAN_CHANGE, FACTOR_TYPES, RiskModel, default_profile,  # noqa: E402
                           summary)

st.set_page_config(page_title="Diabetes Risk Explainer", page_icon="🩺", layout="wide")

RED, BLUE, GREY = "#C0504D", "#4A7FB5", "#9AA5B1"
YES_NO_FEATURES = [f for f, (_, _, values) in BRFSS_FEATURES.items() if values is YES_NO]
EXAMPLES = json.loads((ROOT / "app/examples.json").read_text())


@st.cache_resource
def load_model() -> RiskModel:
    return RiskModel()


@st.cache_resource
def load_lime(_model: RiskModel):
    train = pd.read_csv(ROOT / "data/processed/brfss_bin_train.csv").drop(columns="Diabetes_binary")
    return lime_explainer(_model.pipe, train.sample(5000, random_state=42))


model = load_model()


# ------------------------------------------------------------------ State

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


# ------------------------------------------------------------------ Header

st.title("Diabetes Risk Explainer")
st.markdown("Answer the questions to see an **estimated risk of diabetes** and, more importantly, "
            "**why** the model gave that estimate.")
st.info("**This is not a medical diagnosis.** It is a research tool built from a 2015 US health survey. "
        "If you are worried about diabetes, speak to a doctor or pharmacist.", icon="ℹ️")

form_col, result_col = st.columns([5, 7], gap="large")

# ------------------------------------------------------------------ Inputs

with form_col:
    st.subheader("Your answers")
    st.caption("Fill in the questions below, or load an example person:")
    ex_cols = st.columns(2)
    for i, (name, profile) in enumerate(EXAMPLES.items()):
        ex_cols[i % 2].button(name, on_click=set_profile, args=(profile,), use_container_width=True)
    ex_cols[1].button("Reset answers", on_click=set_profile, args=(default_profile(),), use_container_width=True)

    with st.expander("About you", expanded=True):
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

    with st.expander("Health", expanded=True):
        st.select_slider("In general, how would you rate your health?", options=list(GEN_HEALTH),
                         format_func=GEN_HEALTH.get, key="GenHlth")
        c1, c2 = st.columns(2)
        for i, f in enumerate(["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke", "DiffWalk"]):
            (c1 if i % 2 == 0 else c2).toggle(label(f), key=f, help=BRFSS_FEATURES[f][1])
        st.slider("Days of poor physical health in the past 30 days", 0, 30, key="PhysHlth")
        st.slider("Days of poor mental health in the past 30 days", 0, 30, key="MentHlth")

    with st.expander("Lifestyle", expanded=True):
        c1, c2 = st.columns(2)
        for i, f in enumerate(["PhysActivity", "Fruits", "Veggies", "Smoker", "HvyAlcoholConsump"]):
            (c1 if i % 2 == 0 else c2).toggle(label(f), key=f, help=BRFSS_FEATURES[f][1])

    with st.expander("Health care", expanded=False):
        for f in ["CholCheck", "AnyHealthcare", "NoDocbcCost"]:
            st.toggle(label(f), key=f, help=BRFSS_FEATURES[f][1])

profile = current_profile()
risk = model.risk(profile)
category, category_text = model.category(risk)
reasons = model.reasons(profile)

# ------------------------------------------------------------------ Results

with result_col:
    st.subheader("Estimated risk")
    m1, m2 = st.columns([1, 2])
    m1.metric("Estimated risk", f"{risk:.0%}",
              delta=f"{(risk - model.base_rate) * 100:+.0f} pts vs avg", delta_color="inverse")
    {"Below average": m2.success, "Above average": m2.info, "Higher risk": m2.warning}[category](
        f"**{category}.** {category_text}")

    x_max = 1.0 if risk > 0.55 else 0.6
    x = alt.X("x:Q", scale=alt.Scale(domain=[0, x_max]), axis=alt.Axis(format="%", title=None))
    marks = pd.DataFrame({"x": [model.base_rate, model.threshold, risk],
                          "name": ["Average", "Higher-risk level", "Your estimate"],
                          "colour": [GREY, GREY, RED if risk >= model.threshold else BLUE],
                          "dy": [26, 26, -26]})
    # All layers share one dataset: Streamlit can lose track of separate datasets when it redraws.
    base = alt.Chart(marks)
    ticks = base.mark_tick(thickness=3, size=34).encode(
        x=x, color=alt.Color("colour:N", scale=None), tooltip=["name", alt.Tooltip("x:Q", format=".0%")])
    # Average is labelled to the left of its tick and the threshold to the right, so they never overlap.
    labels = [
        base.transform_filter(alt.datum.name == name).mark_text(
            fontSize=11, fontWeight="bold", align=align, dx=dx).encode(
            x=x, text="name:N", color=alt.Color("colour:N", scale=None), yOffset=alt.YOffset("dy:Q", scale=None))
        for name, align, dx in (("Average", "right", -4), ("Higher-risk level", "left", 4),
                                ("Your estimate", "center", 0))
    ]
    st.altair_chart(alt.layer(ticks, *labels).properties(height=110, padding={"top": 18, "bottom": 4, "left": 8,
                                                                    "right": 8}), use_container_width=True)

    st.subheader("Why this estimate?")
    s = summary(reasons)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<span style='color:{RED}'>■</span> **Raising your risk**", unsafe_allow_html=True)
        for line in s["raises"] or ["Nothing in your answers noticeably raises your risk."]:
            st.markdown(f"- {line}")
    with c2:
        st.markdown(f"<span style='color:{BLUE}'>■</span> **Lowering your risk**", unsafe_allow_html=True)
        for line in s["lowers"] or ["Nothing in your answers noticeably lowers your risk."]:
            st.markdown(f"- {line}")

    top = reasons.head(8).copy()
    top["factor"] = top["label"] + " = " + top["value"]
    top["effect"] = top["direction"].map({"raises": "Raises risk", "lowers": "Lowers risk"})
    top["kind"] = top["type"].map(FACTOR_TYPES)
    chart = alt.Chart(top).mark_bar().encode(
        x=alt.X("shap:Q", title="Effect on the estimate (SHAP value)"),
        y=alt.Y("factor:N", sort=list(top["factor"]), title=None,
                axis=alt.Axis(labelLimit=320, labelOverlap=False, labelFontSize=12)),
        color=alt.Color("effect:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RED, BLUE]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("factor:N", title="Answer"), alt.Tooltip("effect:N", title="Effect"),
                 alt.Tooltip("strength:N", title="Strength"), alt.Tooltip("kind:N", title="Type of factor")],
    ).properties(height=44 * len(top))
    st.altair_chart(chart, use_container_width=True)
    st.caption("The 8 answers that moved your estimate most, compared with an average person in the survey. "
               "These show what the model has learned from the data. They are patterns, not proof of cause.")

    # -------------------------------------------------------------- What-if
    st.subheader("What if?")
    st.caption("Try changing things that people can realistically change. Other factors are not included, "
               "because changing them would not be sensible advice.")
    # Widgets keep their own state, so start the what-if controls again from the
    # current answers whenever those answers change.
    if st.session_state.get("_whatif_base") != profile:
        st.session_state["_whatif_base"] = profile
        st.session_state["whatif_BMI"] = int(min(max(profile["BMI"], 15), 60))
        for f in ["PhysActivity", "Fruits", "Veggies"]:
            st.session_state[f"whatif_{f}"] = bool(profile[f])
    w1, w2 = st.columns(2)
    changes = {"BMI": w1.slider("BMI", 15, 60, key="whatif_BMI")}
    for f in ["PhysActivity", "Fruits", "Veggies"]:
        changes[f] = int(w2.toggle(label(f), key=f"whatif_{f}"))
    changes = {k: v for k, v in changes.items() if v != profile[k]}
    assert set(changes) <= CAN_CHANGE
    if changes:
        new_risk = model.what_if(profile, changes)
        diff = (new_risk - risk) * 100
        described = ", ".join(f"{label(k)} {'→ ' + str(v) if k == 'BMI' else ('yes' if v else 'no')}"
                              for k, v in changes.items())
        st.metric(f"Estimated risk with {described}", f"{new_risk:.0%}", delta=f"{diff:+.0f} points",
                  delta_color="inverse")
    else:
        st.caption("Move the slider or switch a toggle to see how the estimate would change.")

    # -------------------------------------------------------------- LIME
    with st.expander("Second opinion: a different explanation method (LIME)"):
        st.markdown("The explanation above uses **SHAP**. **LIME** works differently: it tests many small "
                    "variations of your answers. When the two methods agree, the explanation is more trustworthy.")
        if st.button("Show LIME explanation"):
            row = pd.DataFrame([profile])[model.features]
            lime = lime_weights(model.pipe, load_lime(model), row)
            shap_series = reasons.set_index("feature")["shap"].reindex(model.features)
            agree = agreement(shap_series, lime)
            lime_top = lime.reindex(lime.abs().sort_values(ascending=False).index).head(5)
            by_feature = reasons.set_index("feature")
            st.dataframe(pd.DataFrame({
                "Answer": [f"{label(f)} = {by_feature.loc[f, 'value']}" for f in lime_top.index],
                "Effect": ["Raises risk" if v > 0 else "Lowers risk" for v in lime_top],
                "Note": ["Pattern in the data, not advice" if by_feature.loc[f, "type"] == "association" else ""
                         for f in lime_top.index],
            }, index=range(1, len(lime_top) + 1)), use_container_width=True)
            st.markdown(f"SHAP and LIME share **{agree['top_k_overlap']:.0%}** of their top-5 answers, and agree on "
                        f"the direction for **{agree['sign_agreement']:.0%}** of the shared ones.")

    with st.expander("About this tool"):
        test = model.meta["test"]
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
