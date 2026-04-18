"""Hand log formatter — Hero/Villain narration with street banners and a bottom-line result.

Replays `state.history` to recover per-action bet sizes and running pot. The same replay feeds
`last_bot_action_label`, so the table's "what just happened" banner and the hand log stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import streamlit as st

from app.services.session import PlaySession
from iip.engine.cards import RANK_CHAR, Card
from iip.engine.game import HULHE, ActionType, HULHEState, Street


@dataclass
class ReplayedAction:
    player: int
    street: Street
    label: str  # e.g. "Raises to 4", "Calls", "Checks", "Bets 2", "Folds"


_SUIT_SYMBOL = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
_SUIT_COLOR = {"s": "#111", "c": "#111", "h": "#c62828", "d": "#c62828"}


def _card_html(c: Card) -> str:
    rank = RANK_CHAR[c.rank]
    symbol = _SUIT_SYMBOL[c.suit]
    color = _SUIT_COLOR[c.suit]
    return (
        f'<span style="display:inline-block;min-width:22px;padding:1px 6px;margin:0 2px;'
        f'background:#fafafa;border:1px solid #bbb;border-radius:4px;color:{color};'
        f'font-weight:700;font-family:ui-sans-serif,system-ui,sans-serif;'
        f'box-shadow:0 1px 2px rgba(0,0,0,0.25);">{rank}{symbol}</span>'
    )


def _cards_html(cards: list[Card]) -> str:
    return "".join(_card_html(c) for c in cards)


def replay(game: HULHE, state: HULHEState) -> list[ReplayedAction]:
    """Walk state.history, producing a labeled action per step with bet amounts."""
    contributions = [game.small_blind, game.big_blind]
    raises = 1  # BB counts as the preflop "open"
    current_street = Street.PREFLOP

    entries: list[ReplayedAction] = []
    for a in state.history:
        if a.street != current_street:
            current_street = a.street
            contributions = [0, 0]
            raises = 0

        actor = a.player
        if a.type is ActionType.FOLD:
            entries.append(ReplayedAction(actor, a.street, "Folds"))
            continue

        to_call = max(contributions) - contributions[actor]
        bet_unit = (
            game.small_bet
            if current_street in (Street.PREFLOP, Street.FLOP)
            else game.big_bet
        )

        if a.type is ActionType.CHECK_CALL:
            if to_call == 0:
                entries.append(ReplayedAction(actor, a.street, "Checks"))
            else:
                contributions[actor] += to_call
                entries.append(ReplayedAction(actor, a.street, "Calls"))
        else:  # BET_RAISE
            raise_cost = to_call + bet_unit
            contributions[actor] += raise_cost
            raises += 1
            new_total = contributions[actor]
            if raises == 1:
                label = f"Bets {bet_unit}"
            elif raises == 2:
                label = f"Raises to {new_total}"
            else:
                label = f"{raises}-bets to {new_total}"
            entries.append(ReplayedAction(actor, a.street, label))

    return entries


def last_bot_action_label(
    game: HULHE, state: HULHEState, hero_seat: int
) -> str | None:
    """Short label of the bot's most recent action (incl. folds and street-closers), else None."""
    if not state.history:
        return None
    bot_seat = 1 - hero_seat
    if state.history[-1].player != bot_seat:
        return None
    entries = replay(game, state)
    # Lowercase first letter for natural prefix ("Bot raises to 4").
    label = entries[-1].label
    return label[:1].lower() + label[1:]


_CAT_NAMES = {
    9: "Straight flush",
    8: "Four of a kind",
    7: "Full house",
    6: "Flush",
    5: "Straight",
    4: "Three of a kind",
    3: "Two pair",
    2: "Pair",
    1: "High card",
}


def _describe_hand(hole: list[Card], board: list[Card]) -> str:
    """Cheap human-readable description of a 7-card hand (best 5)."""
    from iip.eval.ranker import evaluate_five_card_fallback

    best_cat = 0
    best_tie: list[int] = []
    for combo in combinations(list(hole) + list(board), 5):
        cat, tie = evaluate_five_card_fallback(list(combo))
        if (cat, tuple(tie)) > (best_cat, tuple(best_tie)):
            best_cat, best_tie = cat, tie

    name = _CAT_NAMES.get(best_cat, "")
    r = RANK_CHAR
    if best_cat == 1:
        return f"{r[best_tie[0]]}-high"
    if best_cat == 2:
        return f"Pair of {r[best_tie[0]]}s"
    if best_cat == 3:
        return f"Two pair, {r[best_tie[0]]}s and {r[best_tie[1]]}s"
    if best_cat == 4:
        return f"Trips, {r[best_tie[0]]}s"
    if best_cat == 5:
        return f"Straight, {r[best_tie[0]]}-high"
    if best_cat == 6:
        return f"Flush, {r[best_tie[0]]}-high"
    if best_cat == 7:
        return f"Full house, {r[best_tie[0]]}s over {r[best_tie[1]]}s"
    if best_cat == 8:
        return f"Quads, {r[best_tie[0]]}s"
    if best_cat == 9:
        return f"Straight flush, {r[best_tie[0]]}-high"
    return name


_STREET_NAMES = {
    Street.PREFLOP: "PRE-FLOP",
    Street.FLOP: "FLOP",
    Street.TURN: "TURN",
    Street.RIVER: "RIVER",
}


def _street_banner(street: Street, pot: int, board: list[Card]) -> str:
    if street is Street.PREFLOP:
        suffix = ""
    elif street is Street.FLOP:
        suffix = f" — {_cards_html(board[:3])}"
    elif street is Street.TURN:
        suffix = f" — {_cards_html(board[:3])} &nbsp; {_cards_html([board[3]])}"
    else:  # RIVER
        suffix = f" — {_cards_html(board[:4])} &nbsp; {_cards_html([board[4]])}"
    return f"<b>{_STREET_NAMES[street]}</b> (${pot}){suffix}"


def format_hand_log(game: HULHE, state: HULHEState, hero_seat: int) -> str:
    """HTML hand log in Hero/Villain format. Villain's cards appear at hand end regardless of fold/showdown."""
    if state is None or not state.history:
        return "<i>No actions yet.</i>"

    hero = hero_seat
    villain = 1 - hero

    def who(p: int) -> str:
        return "Hero" if p == hero else "Villain"

    def pos(p: int) -> str:
        return "BTN/SB" if p == 0 else "BB"

    entries = replay(game, state)

    # Pot at start of each street (for the banner).
    SB, BB = game.small_blind, game.big_blind
    pot = SB + BB
    contributions = [SB, BB]
    current_street = Street.PREFLOP
    pot_by_street_start = {Street.PREFLOP: pot}

    for a in state.history:
        if a.street != current_street:
            current_street = a.street
            contributions = [0, 0]
            pot_by_street_start[current_street] = pot
        actor = a.player
        if a.type is ActionType.FOLD:
            continue
        to_call = max(contributions) - contributions[actor]
        bet_unit = (
            game.small_bet
            if current_street in (Street.PREFLOP, Street.FLOP)
            else game.big_bet
        )
        if a.type is ActionType.CHECK_CALL:
            contributions[actor] += to_call
            pot += to_call
        else:
            raise_cost = to_call + bet_unit
            contributions[actor] += raise_cost
            pot += raise_cost

    blocks: list[str] = []
    rendered_street: Street | None = None
    current_actions: list[str] = []

    def flush_street() -> None:
        if current_actions:
            blocks.append("<ul style='margin:4px 0 10px 18px;padding:0;'>" + "".join(current_actions) + "</ul>")
            current_actions.clear()

    for entry in entries:
        if entry.street != rendered_street:
            flush_street()
            banner_pot = pot_by_street_start.get(entry.street, 0)
            blocks.append(f"<div style='margin-top:8px;'>{_street_banner(entry.street, banner_pot, state.board)}</div>")
            rendered_street = entry.street
        label_prefix = f"{who(entry.player)} ({pos(entry.player)})" if entry.street is Street.PREFLOP else who(entry.player)
        current_actions.append(f"<li>{label_prefix}: {entry.label}</li>")
    flush_street()

    # Showdown / result block.
    if state.terminal:
        payoffs = game.payoffs(state)
        hero_delta = payoffs[hero]
        v_cards = _cards_html(state.hole_cards[villain])
        h_cards = _cards_html(state.hole_cards[hero])

        result_lines: list[str] = []
        if state.folded is not None:
            folder = state.folded
            result_lines.append("<b>RESULT</b>")
            if folder == villain:
                result_lines.append(f"<li>Villain folds. Hero wins ${state.pot}.</li>")
                result_lines.append(f"<li>Net Profit: +${hero_delta}</li>")
            else:
                result_lines.append(f"<li>Hero folds. Villain wins ${state.pot}.</li>")
                result_lines.append(f"<li>Net Profit: -${-hero_delta}</li>")
            result_lines.append(f"<li>Villain's hand: {v_cards}</li>")
            result_lines.append(f"<li>Hero's hand: {h_cards}</li>")
        else:
            v_desc = _describe_hand(state.hole_cards[villain], state.board)
            h_desc = _describe_hand(state.hole_cards[hero], state.board)
            result_lines.append(f"<b>SHOWDOWN</b> (${state.pot})")
            result_lines.append(f"<li>Villain shows: {v_cards} ({v_desc})</li>")
            result_lines.append(f"<li>Hero shows: {h_cards} ({h_desc})</li>")
            result_lines.append("<b>RESULT</b>")
            if hero_delta > 0:
                result_lines.append(f"<li>Hero wins ${state.pot}.</li>")
                result_lines.append(f"<li>Net Profit: +${hero_delta}</li>")
            elif hero_delta < 0:
                result_lines.append(f"<li>Villain wins ${state.pot}.</li>")
                result_lines.append(f"<li>Net Profit: -${-hero_delta}</li>")
            else:
                result_lines.append(f"<li>Split pot (${state.pot // 2} each).</li>")
                result_lines.append("<li>Net Profit: $0</li>")

        # Split into banner headers (<b>) and their following <li> groups.
        out: list[str] = []
        buf: list[str] = []
        for line in result_lines:
            if line.startswith("<b>"):
                if buf:
                    out.append("<ul style='margin:4px 0 10px 18px;padding:0;'>" + "".join(buf) + "</ul>")
                    buf = []
                out.append(f"<div style='margin-top:10px;'>{line}</div>")
            else:
                buf.append(line)
        if buf:
            out.append("<ul style='margin:4px 0 10px 18px;padding:0;'>" + "".join(buf) + "</ul>")
        blocks.extend(out)

    return "".join(blocks)


def _viewable_hands(
    session: PlaySession,
) -> list[tuple[HULHEState, int, int, bool]]:
    """All hands the user can review: every completed hand, plus the live in-progress hand if any.

    Returns list of (state, hero_seat, hand_index, is_live).
    """
    hands: list[tuple[HULHEState, int, int, bool]] = [
        (h.state, h.hero_seat, h.hand_index, False) for h in session.completed_hands
    ]
    live = session.state
    if live is not None and not live.terminal:
        hands.append((live, session.hero_seat, session.hand_index, True))
    return hands


def render_log(session: PlaySession) -> None:
    """Hand log with prev/next navigation across every hand played this game."""
    hands = _viewable_hands(session)
    if not hands:
        st.markdown("<i>No actions yet.</i>", unsafe_allow_html=True)
        return

    key = "hand_log_view_index"
    total = len(hands)
    # Default to the most recent hand whenever the total grows (new hand finished).
    prev_total = st.session_state.get("hand_log_total", 0)
    if key not in st.session_state or total != prev_total:
        st.session_state[key] = total - 1
    st.session_state["hand_log_total"] = total
    st.session_state[key] = max(0, min(st.session_state[key], total - 1))

    idx = st.session_state[key]
    state, hero_seat, hand_index, is_live = hands[idx]

    prev_col, label_col, next_col = st.columns([1, 3, 1])
    if prev_col.button("◀ Prev", disabled=idx == 0, key="hand_log_prev", use_container_width=True):
        st.session_state[key] = idx - 1
        st.rerun()
    label = f"Hand {hand_index} of {session.hand_index}"
    if is_live:
        label += " · live"
    label_col.markdown(
        f"<div style='text-align:center;font-weight:600;padding-top:6px;'>{label}</div>",
        unsafe_allow_html=True,
    )
    if next_col.button("Next ▶", disabled=idx >= total - 1, key="hand_log_next", use_container_width=True):
        st.session_state[key] = idx + 1
        st.rerun()

    st.markdown(format_hand_log(session.game, state, hero_seat), unsafe_allow_html=True)
