"""Research web prototype for dual-task risk estimation.

Run: streamlit run app.py -- --bundle_dir ./release_bundle
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import yaml

from predict import predict_dataframe


def parse_cli_bundle() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--bundle_dir", default="./release_bundle")
    args, _ = p.parse_known_args()
    return args.bundle_dir


bundle_dir = parse_cli_bundle()
schema_path = Path(bundle_dir) / "feature_schema.yaml"
if not schema_path.exists():
    st.error(f"Locked release bundle not found: {bundle_dir}")
    st.stop()

with open(schema_path, "r", encoding="utf-8") as f:
    schema = yaml.safe_load(f)["features"]

st.title("DP Risk Assessment")
st.caption("Research prototype for early postoperative risk assessment after distal pancreatectomy")
mode = st.radio("Input mode", ["Core variables", "All variables"], horizontal=True)
features = [x for x in schema if (x.get("core", False) or mode == "All variables")]

row = {"patient_id": "web_patient"}
for feat in features:
    label = f"{feat['description']} ({feat['stage']})"
    raw = st.text_input(label, value="", key=feat["name"], help="Leave blank if unavailable")
    if raw.strip() == "":
        row[feat["name"]] = None
    elif feat["type"] == "numeric":
        try:
            row[feat["name"]] = float(raw)
        except ValueError:
            st.warning(f"{feat['name']}: enter a numeric value or leave blank")
            row[feat["name"]] = None
    else:
        row[feat["name"]] = raw.strip()

if st.button("Estimate risk", type="primary"):
    result = predict_dataframe(pd.DataFrame([row]), bundle_dir)
    r = result.iloc[0]
    c1, c2 = st.columns(2)
    c1.metric("Pancreatic leakage probability", f"{100*r['pancreatic_leakage_probability']:.1f}%")
    c2.metric("CR-POPF probability", f"{100*r['cr_popf_probability']:.1f}%")
    st.write("Predicted category:", r["predicted_grade"])
    st.caption("This prototype is for research use and has not undergone prospective clinical utility evaluation.")
