"""Rich table renderer — green felt, cards, chip stacks, pot, dealer button, street banner.

Rendered via `st.components.v1.html` (iframe) rather than `st.markdown`, because Streamlit's
markdown parser breaks multi-line indented HTML blocks on blank lines. The iframe is isolated
from the parent Streamlit DOM, which is fine since the table is display-only; all interaction
(action buttons) lives in separate widgets below.
"""

from __future__ import annotations

import streamlit.components.v1 as components

from iip.engine.cards import RANK_CHAR, Card
from iip.engine.game import HULHEState, Street

_SUIT_SYMBOL = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
_SUIT_COLOR_CLASS = {"s": "iip-black", "c": "iip-black", "h": "iip-red", "d": "iip-red"}


def _card_face(card: Card) -> str:
    rank = RANK_CHAR[card.rank]
    suit = _SUIT_SYMBOL[card.suit]
    cls = _SUIT_COLOR_CLASS[card.suit]
    return (
        f'<div class="iip-card {cls}">'
        f'<div class="iip-card-rank">{rank}</div>'
        f'<div class="iip-card-suit">{suit}</div>'
        f'</div>'
    )


def _card_back() -> str:
    return '<div class="iip-card iip-card-back"></div>'


def _placeholder() -> str:
    return '<div class="iip-card iip-card-placeholder"></div>'


def _cards_row(cards: list[Card], face_up: bool, pad_to: int | None = None) -> str:
    pieces = [_card_face(c) if face_up else _card_back() for c in cards]
    if pad_to is not None:
        while len(pieces) < pad_to:
            pieces.append(_placeholder())
    return "".join(pieces) or _placeholder()


def _bet_chip(amount: int) -> str:
    if amount <= 0:
        return ""
    return f'<div class="iip-bet"><span class="iip-chip-dot"></span>{amount}</div>'


def _dealer_button() -> str:
    return '<div class="iip-dealer" title="Dealer / Button">D</div>'


def _turn_banner(state: HULHEState, hero_seat: int) -> str:
    if state.terminal:
        return ""
    if state.to_act == hero_seat:
        return '<div class="iip-turn iip-turn-hero">Your turn</div>'
    return '<div class="iip-turn iip-turn-bot">Bot is thinking…</div>'


_CSS = """
<style>
body { margin: 0; background: transparent; }
.iip-table {
    background: radial-gradient(ellipse at center, #1e7a48 0%, #0c4a28 100%);
    border: 8px solid #4a2e1a;
    border-radius: 180px / 110px;
    padding: 22px 36px;
    color: #f4f4f4;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 18px;
    box-shadow: inset 0 0 70px rgba(0,0,0,0.4), 0 6px 22px rgba(0,0,0,0.45);
    max-width: 640px;
    margin: 10px auto 18px auto;
}
.iip-seat {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    min-width: 220px;
}
.iip-seat-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.05em;
    font-weight: 600;
}
.iip-seat-active .iip-seat-name {
    color: #ffe57a;
    text-shadow: 0 0 6px rgba(255, 229, 122, 0.5);
}
.iip-dealer {
    width: 26px; height: 26px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, #fff2a8 0%, #e8b83a 70%, #b88620 100%);
    color: #3a2a00;
    font-weight: 800;
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.5), inset 0 1px 1px rgba(255,255,255,0.6);
    border: 1px solid #7a5a0f;
}
.iip-cards {
    display: flex;
    gap: 6px;
    justify-content: center;
}
.iip-card {
    width: 46px;
    height: 64px;
    background: #fafafa;
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    box-shadow: 0 2px 5px rgba(0,0,0,0.5);
    padding: 2px;
}
.iip-card-rank { font-size: 1.15em; line-height: 1; }
.iip-card-suit { font-size: 1.5em; line-height: 1; margin-top: 2px; }
.iip-red { color: #c62828; }
.iip-black { color: #111; }
.iip-card-back {
    background:
        repeating-linear-gradient(45deg, #163a74 0, #163a74 5px, #214f9a 5px, #214f9a 10px);
    border: 2px solid #0b244d;
}
.iip-card-placeholder {
    background: rgba(255,255,255,0.05);
    border: 1px dashed rgba(255,255,255,0.25);
    box-shadow: none;
}
.iip-stack {
    font-size: 0.85em;
    background: rgba(0,0,0,0.3);
    padding: 3px 12px;
    border-radius: 12px;
    letter-spacing: 0.3px;
}
.iip-stack b { color: #ffd96a; }
.iip-bet {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(180deg, #ffd86b 0%, #c38c18 100%);
    color: #2a1a00;
    font-weight: 700;
    padding: 3px 12px 3px 8px;
    border-radius: 14px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.45);
    font-size: 0.85em;
}
.iip-chip-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, #ffefb3 0%, #a37516 100%);
    display: inline-block;
    box-shadow: inset 0 0 0 1.5px #7a5500;
}
.iip-middle {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 18px;
    background: rgba(0,0,0,0.12);
    width: 100%;
    max-width: 480px;
}
.iip-street {
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-size: 0.72em;
    opacity: 0.75;
}
.iip-pot {
    text-align: center;
    font-size: 0.8em;
    background: rgba(0,0,0,0.25);
    padding: 5px 18px;
    border-radius: 16px;
    line-height: 1.15;
}
.iip-pot-num {
    display: block;
    font-size: 1.7em;
    font-weight: 800;
    color: #ffd96a;
    letter-spacing: 0.5px;
}
.iip-board {
    display: flex;
    gap: 8px;
    min-height: 64px;
    align-items: center;
    justify-content: center;
}
.iip-empty-board {
    opacity: 0.5;
    font-style: italic;
    font-size: 0.85em;
    letter-spacing: 1px;
}
.iip-turn {
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 12px;
    font-size: 0.85em;
    letter-spacing: 0.4px;
}
.iip-turn-hero {
    background: #2e7d32;
    color: #fff;
    animation: iip-pulse 1.6s ease-in-out infinite;
}
.iip-turn-bot {
    background: rgba(255,255,255,0.12);
    color: #e0e0e0;
    font-style: italic;
}
@keyframes iip-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(80, 200, 120, 0.55); }
    50%      { box-shadow: 0 0 0 8px rgba(80, 200, 120, 0); }
}
.iip-last-action {
    font-size: 0.8em;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 10px;
    background: rgba(255, 229, 122, 0.18);
    color: #ffe57a;
    border: 1px solid rgba(255, 229, 122, 0.45);
    letter-spacing: 0.3px;
}
</style>
"""


def render_table(
    state: HULHEState,
    hero_seat: int,
    reveal_bot: bool = False,
    last_bot_action: str | None = None,
) -> None:
    bot_seat = 1 - hero_seat
    hero_has_button = hero_seat == 0  # Dealer is always seat 0 in this engine.

    bot_active = state.to_act == bot_seat and not state.terminal
    hero_active = state.to_act == hero_seat and not state.terminal

    bot_cards_html = _cards_row(
        state.hole_cards[bot_seat], face_up=reveal_bot, pad_to=2
    )
    hero_cards_html = _cards_row(state.hole_cards[hero_seat], face_up=True, pad_to=2)

    # Community cards expand across streets — pad to 5 placeholders so spacing is stable.
    board_pieces = [_card_face(c) for c in state.board]
    while len(board_pieces) < 5:
        board_pieces.append(_placeholder())
    board_html = "".join(board_pieces)

    bot_bet_html = _bet_chip(state.contributions[bot_seat])
    hero_bet_html = _bet_chip(state.contributions[hero_seat])
    dealer_bot_html = "" if hero_has_button else _dealer_button()
    dealer_hero_html = _dealer_button() if hero_has_button else ""

    street_name = state.street.name if state.street != Street.SHOWDOWN else "SHOWDOWN"

    bot_active_cls = "iip-seat-active" if bot_active else ""
    hero_active_cls = "iip-seat-active" if hero_active else ""
    turn_html = _turn_banner(state, hero_seat)
    last_action_html = (
        f'<div class="iip-last-action">Bot {last_bot_action}</div>'
        if last_bot_action
        else ""
    )

    body = (
        f'<div class="iip-table">'
        f'<div class="iip-seat {bot_active_cls}">'
        f'<div class="iip-seat-header"><span class="iip-seat-name">🤖 Bot</span>{dealer_bot_html}</div>'
        f'<div class="iip-cards">{bot_cards_html}</div>'
        f'<div class="iip-stack">Stack: <b>{state.stacks[bot_seat]}</b></div>'
        f'{bot_bet_html}'
        f'{last_action_html}'
        f'</div>'
        f'<div class="iip-middle">'
        f'<div class="iip-street">{street_name}</div>'
        f'<div class="iip-pot">Pot<span class="iip-pot-num">{state.pot}</span></div>'
        f'<div class="iip-board">{board_html}</div>'
        f'{turn_html}'
        f'</div>'
        f'<div class="iip-seat {hero_active_cls}">'
        f'{hero_bet_html}'
        f'<div class="iip-stack">Stack: <b>{state.stacks[hero_seat]}</b></div>'
        f'<div class="iip-cards">{hero_cards_html}</div>'
        f'<div class="iip-seat-header"><span class="iip-seat-name">🧑 You</span>{dealer_hero_html}</div>'
        f'</div>'
        f'</div>'
    )
    components.html(_CSS + body, height=680, scrolling=False)
