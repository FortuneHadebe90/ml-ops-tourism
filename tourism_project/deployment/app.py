"""Streamlit front-end for the Wellness Tourism Package purchase predictor.
The form is generated from model_meta.json, so it always matches the trained model."""
import json
import os
import re

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO = os.environ.get("MODEL_REPO", "")

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳", layout="centered")


@st.cache_resource
def load_assets():
    model = joblib.load(hf_hub_download(MODEL_REPO, "model.joblib", repo_type="model"))
    meta = json.load(open(hf_hub_download(MODEL_REPO, "model_meta.json", repo_type="model")))
    return model, meta


def label(col: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", col)


st.title("🧳 Wellness Tourism Package Predictor")
st.caption("Visit with Us - estimate how likely a customer is to buy before you call them.")

if not MODEL_REPO:
    st.error("MODEL_REPO environment variable is not set on this Space.")
    st.stop()

model, meta = load_assets()
features = meta["features"]

inputs = {}
cols = st.columns(2)
for i, (name, spec) in enumerate(features.items()):
    with cols[i % 2]:
        if spec["type"] == "choice":
            inputs[name] = st.selectbox(label(name), spec["options"])
        else:
            inputs[name] = st.number_input(label(name), min_value=0.0,
                                           max_value=max(spec["max"] * 2, 1.0),
                                           value=spec["default"])

if st.button("Predict", type="primary"):
    row = pd.DataFrame([inputs])
    proba = float(model.predict_proba(row)[0, 1])
    likely = proba >= meta["threshold"]
    st.metric("Purchase probability", f"{proba:.1%}")
    if likely:
        st.success("Likely to purchase - prioritise this customer for the Wellness Package.")
    else:
        st.warning("Unlikely to purchase - consider a different offer or lower-cost outreach.")
    st.caption(f"Model: {meta['model_name']} | decision threshold: {meta['threshold']}")
