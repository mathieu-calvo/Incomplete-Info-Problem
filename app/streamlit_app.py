"""Streamlit Cloud entry point.

Landing page for the app. Streamlit Cloud auto-discovers the files in `app/pages/` as subpages.
We keep this root page as a "how to play + bot version" overview; the actual play surface lives
in `pages/1_Play.py`.
"""

from __future__ import annotations

import streamlit as st

from app.services.bot_loader import load_bot_and_meta


st.set_page_config(page_title="Incomplete-Info-Problem — Play vs Bot", page_icon="🃏", layout="wide")

st.title("Incomplete-Info-Problem")
st.caption(
    "A heads-up Limit Texas Hold'em bot trained with Deep CFR + PPO self-play. "
    "Play against it, or browse the bot's metrics — every hand you play helps train the next version."
)

bot, meta = load_bot_and_meta()

col1, col2 = st.columns(2)
with col1:
    st.subheader("How to play")
    st.markdown(
        "- Head to the **Play** page to start a hand.\n"
        "- The bot always faces you heads-up, switching seats each hand.\n"
        "- Your actions + bot decisions are logged (anonymously) so the next training run can "
        "  adapt to human play.\n"
        "- **Bot Insights** shows exploitability + head-to-head performance vs baselines."
    )
with col2:
    st.subheader("Loaded checkpoint")
    st.write(f"**Source:** {meta.get('source', 'local')}")
    st.write(f"**Revision:** `{meta.get('revision', 'dev')}`")
    st.write(f"**Algorithm:** {meta.get('algo', 'deepcfr')}")
    st.caption(
        "Streamlit Cloud pulls the latest tagged checkpoint from Hugging Face Hub at boot. "
        "If HF creds aren't set, the app falls back to `checkpoints/local/deepcfr.pt`."
    )
