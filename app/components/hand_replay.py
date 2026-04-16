"""Compact log of actions for the current / most recent hand."""

from __future__ import annotations

import streamlit as st

from app.services.session import PlaySession


def render_log(session: PlaySession) -> None:
    if not session.log:
        st.caption("No actions yet.")
        return
    rows = []
    for a in session.log:
        who = "you" if a.player == session.hero_seat else "bot"
        policy_str = ""
        if a.bot_policy:
            policy_str = "  ".join(f"{k}:{v:.2f}" for k, v in a.bot_policy.items())
        rows.append(f"- **{who}** @ `{a.street}` → `{a.action}` {policy_str}")
    st.markdown("\n".join(rows))
