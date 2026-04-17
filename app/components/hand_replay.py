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


def _fmt_card(c: Card) -> str:
    return f"{RANK_CHAR[c.rank]}{c.suit}"


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
    """Short label of the bot's most recent action on the current street, else None."""
    if state.terminal or state.to_act != hero_seat or not state.history:
        return None
    last = state.history[-1]
    bot_seat = 1 - hero_seat
    if last.player != bot_seat or last.street != state.street:
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
        suffix = f" — [{' '.join(_fmt_card(c) for c in board[:3])}]"
    elif street is Street.TURN:
        flop = " ".join(_fmt_card(c) for c in board[:3])
        suffix = f" — [{flop}] [{_fmt_card(board[3])}]"
    else:  # RIVER
        flop_turn = " ".join(_fmt_card(c) for c in board[:4])
        suffix = f" — [{flop_turn}] [{_fmt_card(board[4])}]"
    return f"**{_STREET_NAMES[street]}** (${pot}){suffix}"


def format_hand_log(session: PlaySession) -> str:
    """Markdown hand log in Hero/Villain format. Villain's cards appear at hand end regardless of fold/showdown."""
    state = session.state
    if state is None or not state.history:
        return "No actions yet."

    game = session.game
    hero = session.hero_seat
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

    lines: list[str] = []
    rendered_street: Street | None = None
    for entry in entries:
        if entry.street != rendered_street:
            if rendered_street is not None:
                lines.append("")  # blank line between streets
            banner_pot = pot_by_street_start.get(entry.street, 0)
            lines.append(_street_banner(entry.street, banner_pot, state.board))
            rendered_street = entry.street
        label_prefix = f"{who(entry.player)} ({pos(entry.player)})" if entry.street is Street.PREFLOP else who(entry.player)
        lines.append(f"- {label_prefix}: {entry.label}")

    # Showdown / result block.
    if state.terminal:
        lines.append("")
        payoffs = game.payoffs(state)
        hero_delta = payoffs[hero]
        v_cards = " ".join(_fmt_card(c) for c in state.hole_cards[villain])
        h_cards = " ".join(_fmt_card(c) for c in state.hole_cards[hero])

        if state.folded is not None:
            folder = state.folded
            lines.append("**RESULT**")
            if folder == villain:
                lines.append(f"- Villain folds. Hero wins ${state.pot}.")
                lines.append(f"- Net Profit: +${hero_delta}")
            else:
                lines.append(f"- Hero folds. Villain wins ${state.pot}.")
                lines.append(f"- Net Profit: -${-hero_delta}")
            lines.append("")
            lines.append(f"- Villain's hand: [{v_cards}]")
            lines.append(f"- Hero's hand: [{h_cards}]")
        else:
            v_desc = _describe_hand(state.hole_cards[villain], state.board)
            h_desc = _describe_hand(state.hole_cards[hero], state.board)
            lines.append(f"**SHOWDOWN** (${state.pot})")
            lines.append(f"- Villain shows: [{v_cards}] ({v_desc})")
            lines.append(f"- Hero shows: [{h_cards}] ({h_desc})")
            lines.append("")
            lines.append("**RESULT**")
            if hero_delta > 0:
                lines.append(f"- Hero wins ${state.pot}.")
                lines.append(f"- Net Profit: +${hero_delta}")
            elif hero_delta < 0:
                lines.append(f"- Villain wins ${state.pot}.")
                lines.append(f"- Net Profit: -${-hero_delta}")
            else:
                lines.append(f"- Split pot (${state.pot // 2} each).")
                lines.append("- Net Profit: $0")

    return "\n\n".join(lines)


def render_log(session: PlaySession) -> None:
    st.markdown(format_hand_log(session))
