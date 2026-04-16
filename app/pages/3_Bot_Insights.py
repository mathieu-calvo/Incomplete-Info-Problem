"""Bot Insights: show reports/latest.json if present — produced by scripts/evaluate_checkpoint.py."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.services.bot_loader import load_bot_and_meta


st.set_page_config(page_title="Bot Insights", page_icon="📈", layout="wide")
st.title("Bot Insights")

bot, meta = load_bot_and_meta()
st.subheader("Loaded checkpoint")
st.json(meta)

report_path = Path("reports/latest.json")
if not report_path.exists():
    st.info(
        "No evaluation report found. Run "
        "`python scripts/evaluate_checkpoint.py --checkpoint checkpoints/local/deepcfr.pt` "
        "to produce `reports/latest.json`."
    )
    st.stop()

report = json.loads(report_path.read_text())
st.subheader("Head-to-head vs baselines")
st.table(report.get("head_to_head", []))
st.subheader("Exploitability (LBR proxy)")
st.metric(label="LBR mbb/h (higher = more exploitable)", value=f"{report.get('lbr_mbbph', float('nan')):.2f}")
