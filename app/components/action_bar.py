"""Action bar — renders legal-action buttons for the hero with the actual bet amounts."""

from __future__ import annotations

import streamlit as st

from iip.engine.game import HULHE, ActionType, HULHEState


def render_action_bar(game: HULHE, state: HULHEState) -> ActionType | None:
    legal = game.legal_actions(state)
    to_call = game.current_bet(state)
    bet_unit = game.bet_unit(state)

    check_call_label = "Check" if to_call == 0 else f"Call {to_call}"
    raise_label = (
        f"Bet {bet_unit}"
        if state.raises_this_street == 0
        else f"Raise to {max(state.contributions) + bet_unit}"
    )

    labels = {
        ActionType.FOLD: "Fold",
        ActionType.CHECK_CALL: check_call_label,
        ActionType.BET_RAISE: raise_label,
    }
    button_types = {
        ActionType.FOLD: "secondary",
        ActionType.CHECK_CALL: "secondary",
        ActionType.BET_RAISE: "primary",
    }

    cols = st.columns(len(legal))
    clicked: ActionType | None = None
    for col, action in zip(cols, legal, strict=True):
        with col:
            if st.button(
                labels[action],
                key=f"act-{action.name}",
                use_container_width=True,
                type=button_types[action],
            ):
                clicked = action
    return clicked
