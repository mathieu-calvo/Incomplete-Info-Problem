"""Play page: one hand at a time vs the bot, with hand logging to Supabase."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from app.components.action_bar import render_action_bar  # noqa: E402
from app.components.hand_replay import last_bot_action_label, render_log  # noqa: E402
from app.components.table import render_table  # noqa: E402
from app.services.bot_loader import load_bot_and_meta  # noqa: E402
from app.services.hand_store import save_completed_hand  # noqa: E402
from app.services.session import get_session  # noqa: E402
from iip.io.supabase_client import HandStore  # noqa: E402

st.set_page_config(page_title="Play vs Bot", page_icon="🃏", layout="wide")
st.title("Play vs Bot")

bot, meta = load_bot_and_meta()
session = get_session(bot=bot, bot_checkpoint=meta.get("revision", "dev"))


def _render_game_over_banner() -> None:
    if session.hero_won_game():
        st.success("🏆 You won the game — bot is out of chips.")
    else:
        st.error("💀 The bot broke you. Better luck next time.")
    if st.button("New game", type="primary"):
        session.start_new_game()
        session.start_new_hand()
        st.rerun()


# Running-chip header.
left, right = st.columns(2)
left.metric("Your chips", session.hero_total_stack)
right.metric("Bot chips", session.bot_total_stack)

# No live hand: show the idle screen (either fresh start or game-over).
if session.state is None:
    if session.game_over():
        _render_game_over_banner()
    else:
        st.info("Click **New hand** to start.")
        if st.button("New hand", type="primary"):
            session.start_new_hand()
            st.rerun()
    st.stop()

reveal_bot = session.hand_over() and session.state.folded is None
render_table(
    session.state,
    session.hero_seat,
    reveal_bot=reveal_bot,
    last_bot_action=last_bot_action_label(session.game, session.state, session.hero_seat),
)

if session.hand_over():
    payoffs = session.game.payoffs(session.state)
    hero_delta = payoffs[session.hero_seat]
    if hero_delta > 0:
        st.success(f"You won {hero_delta} chips.")
    elif hero_delta < 0:
        st.error(f"You lost {-hero_delta} chips.")
    else:
        st.info("Split pot.")

    # Log the hand (no-op if Supabase isn't configured).
    store = HandStore()
    save_completed_hand(store, session)
    if store.is_configured:
        st.caption("Hand logged for future training.")

    if session.game_over():
        _render_game_over_banner()
    else:
        time.sleep(1.5 if session.state.folded is not None else 3.0)
        session.start_new_hand()
        st.rerun()
elif session.state.to_act == session.hero_seat:
    action = render_action_bar(session.game, session.state)
    if action is not None:
        session.user_action(action)
        st.rerun()

st.markdown("---")
with st.expander("Hand log", expanded=False):
    render_log(session)
