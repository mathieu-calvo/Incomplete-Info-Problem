"""Persist a completed hand to Supabase. No-ops gracefully if Supabase isn't configured."""

from __future__ import annotations

from dataclasses import asdict

from iip.io.supabase_client import HandRecord, HandStore
from app.services.session import PlaySession


def save_completed_hand(store: HandStore, session: PlaySession) -> HandRecord | None:
    s = session.state
    if s is None or not s.terminal:
        return None
    payoffs = session.game.payoffs(s)
    rec = HandRecord(
        user_id=session.user_id,
        hero_seat=session.hero_seat,
        bot_checkpoint=session.bot_checkpoint,
        hole_cards_hero=" ".join(str(c) for c in s.hole_cards[session.hero_seat]),
        hole_cards_bot=" ".join(str(c) for c in s.hole_cards[1 - session.hero_seat]),
        board=" ".join(str(c) for c in s.board),
        action_log=[asdict(a) for a in session.log],
        bot_policies=[a.bot_policy or {} for a in session.log if a.bot_policy],
        payoff_hero=int(payoffs[session.hero_seat]),
        payoff_bot=int(payoffs[1 - session.hero_seat]),
    )
    store.insert_hand(rec)
    return rec
