"""Action bar — renders legal-action buttons for the hero."""

from __future__ import annotations

import streamlit as st

from iip.engine.game import HULHE, ActionType, HULHEState


def render_action_bar(game: HULHE, state: HULHEState) -> ActionType | None:
    legal = game.legal_actions(state)
    labels = {
        ActionType.FOLD: "Fold",
        ActionType.CHECK_CALL: "Check / Call",
        ActionType.BET_RAISE: "Bet / Raise",
    }
    cols = st.columns(len(legal))
    clicked: ActionType | None = None
    for col, action in zip(cols, legal, strict=True):
        with col:
            if st.button(labels[action], key=f"act-{action.name}", use_container_width=True):
                clicked = action
    return clicked
