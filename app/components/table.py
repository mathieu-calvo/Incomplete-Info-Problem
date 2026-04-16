"""Minimal table renderer — shows pot, seats, stacks, community cards, hero hole cards."""

from __future__ import annotations

import streamlit as st

from iip.engine.game import HULHEState


_SUIT_SYMBOL = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}


def _card_str(card) -> str:
    rank = card.rank
    r = {11: "J", 12: "Q", 13: "K", 14: "A", 10: "T"}.get(rank, str(rank))
    return f"{r}{_SUIT_SYMBOL[card.suit]}"


def render_table(state: HULHEState, hero_seat: int, reveal_bot: bool = False) -> None:
    cols = st.columns([1, 2, 1])
    bot_seat = 1 - hero_seat
    with cols[0]:
        st.markdown("### 🤖 Bot")
        st.write(f"Stack: **{state.stacks[bot_seat]}**")
        bot_cards = "?? ??" if not reveal_bot else "  ".join(_card_str(c) for c in state.hole_cards[bot_seat])
        st.markdown(f"Cards: `{bot_cards}`")
    with cols[1]:
        st.markdown(f"### Pot: **{state.pot}**  —  {state.street.name}")
        board = "  ".join(_card_str(c) for c in state.board) or "(no community cards yet)"
        st.markdown(f"Board: `{board}`")
        st.caption(
            f"This street: hero invested {state.contributions[hero_seat]}, "
            f"bot {state.contributions[bot_seat]}. Raises this street: {state.raises_this_street}."
        )
    with cols[2]:
        st.markdown("### 🧑 You")
        st.write(f"Stack: **{state.stacks[hero_seat]}**")
        hero_cards = "  ".join(_card_str(c) for c in state.hole_cards[hero_seat])
        st.markdown(f"Cards: `{hero_cards}`")
