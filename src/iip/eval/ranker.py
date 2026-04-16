"""Hand ranking.

Primary: `treys.Evaluator` (fast C-like lookup). Used in the hot path (Deep CFR / PPO rollouts).
Fallback: pure-Python 5-card evaluator ported from the legacy `pokerbot/hand_evaluation/hand.py`,
kept so unit tests can cross-check treys output and the code still runs if treys is unavailable.

Lower rank number from treys == stronger hand (1 = royal flush, 7462 = worst). We negate that so
`rank_seven_card` returns a larger-is-stronger score.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from iip.engine.cards import Card

try:
    from treys import Card as TCard
    from treys import Evaluator as TEvaluator

    _TREYS = TEvaluator()

    def _to_treys(card: Card) -> int:
        return TCard.new(f"{card.rank if card.rank <= 9 else 'TJQKA'[card.rank - 10]}{card.suit}")

    def rank_seven_card(cards: list[Card]) -> int:
        if len(cards) != 7:
            raise ValueError(f"rank_seven_card requires 7 cards, got {len(cards)}")
        tboard = [_to_treys(c) for c in cards[2:]]   # last 5 treated as board
        thole = [_to_treys(c) for c in cards[:2]]
        # treys: lower = better; we return a larger-is-better score.
        return -_TREYS.evaluate(tboard, thole)

    _HAVE_TREYS = True
except Exception:  # pragma: no cover — treys missing falls back to pure Python
    _HAVE_TREYS = False

    def rank_seven_card(cards: list[Card]) -> int:
        return _seven_card_score_fallback(cards)


# ---------- pure-Python fallback (and ground truth for tests) ----------

_STRAIGHTS: list[set[int]] = [
    {14, 2, 3, 4, 5},
    *({r + i for i in range(5)} for r in range(2, 11)),
]


def evaluate_five_card_fallback(hand: list[Card]) -> tuple[int, list[int]]:
    """Return (category, tiebreaker) for a 5-card hand. Higher category = stronger."""
    ranks = [c.rank for c in hand]
    suits = [c.suit for c in hand]
    rc = Counter(ranks)
    is_flush = len(set(suits)) == 1
    rset = set(ranks)
    is_straight = rset in _STRAIGHTS
    is_wheel = rset == {14, 2, 3, 4, 5}

    if is_flush and is_straight:
        return 9, [5] if is_wheel else [max(ranks)]
    if 4 in rc.values():
        return 8, sorted(rc, key=lambda r: (rc[r], r), reverse=True)
    if sorted(rc.values()) == [2, 3]:
        return 7, sorted(rc, key=lambda r: (rc[r], r), reverse=True)
    if is_flush:
        return 6, sorted(ranks, reverse=True)
    if is_straight:
        return 5, [5] if is_wheel else [max(ranks)]
    if sorted(rc.values()) == [1, 1, 3]:
        return 4, sorted(rc, key=lambda r: (rc[r], r), reverse=True)
    if sorted(rc.values()) == [1, 2, 2]:
        return 3, sorted(rc, key=lambda r: (rc[r], r), reverse=True)
    if sorted(rc.values()) == [1, 1, 1, 2]:
        return 2, sorted(rc, key=lambda r: (rc[r], r), reverse=True)
    return 1, sorted(ranks, reverse=True)


def _seven_card_score_fallback(cards: list[Card]) -> int:
    """Score a 7-card hand by picking the best 5-card combination. Packs (category, tiebreaker) into an int."""
    best = (0, [0, 0, 0, 0, 0])
    for combo in combinations(cards, 5):
        cat, tie = evaluate_five_card_fallback(list(combo))
        key = (cat, tuple(tie + [0] * (5 - len(tie))))
        if key > (best[0], tuple(best[1] + [0] * (5 - len(best[1])))):
            best = (cat, tie)
    tie = best[1] + [0] * (5 - len(best[1]))
    score = best[0] * 15**5
    for t in tie:
        score = score * 15 + t
    return score


def compare_seven_card_hands(hand_a: list[Card], hand_b: list[Card]) -> int:
    """Return +1 if hand_a is stronger, -1 if weaker, 0 if tied. Each input is (2 hole + 5 board) = 7 cards."""
    ra = rank_seven_card(hand_a)
    rb = rank_seven_card(hand_b)
    if ra > rb:
        return 1
    if ra < rb:
        return -1
    return 0
