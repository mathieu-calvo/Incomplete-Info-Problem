"""Leaderboard: per-user cumulative mbb/h vs the bot, with hand count, pulled from Supabase."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from iip.io.supabase_client import HandStore  # noqa: E402

st.set_page_config(page_title="Leaderboard", page_icon="🏆", layout="wide")
st.title("Leaderboard")

store = HandStore()
if not store.is_configured:
    st.info("Supabase isn't configured — leaderboard requires persistent hand storage.")
    st.stop()

records = store.fetch_hands_since(limit=5000)
if not records:
    st.info("No hands logged yet.")
    st.stop()

df = pd.DataFrame(records)
# Aggregate per user: total mbb/h, hands, win rate.
df["mbb"] = df["payoff_hero"] * 500  # 1 chip = BB/2 ... sb=1, bb=2 in defaults → 1 chip = 500 mbb
agg = (
    df.groupby("user_id")
      .agg(hands=("mbb", "count"), mbb_sum=("mbb", "sum"), mbbph=("mbb", "mean"))
      .sort_values("mbbph", ascending=False)
)
agg["mbbph"] = agg["mbbph"].round(2)
st.dataframe(agg)
