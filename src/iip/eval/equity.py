"""Monte-Carlo equity + preflop bucketing.

`monte_carlo_equity` estimates hero's win probability against a uniform random opponent range,
given known hole cards and any subset of community cards already dealt. It also doubles as a
ground truth for training data generation.

`preflop_bucket_id` maps the 169 distinct starting hands to a stable integer id — used by the
infoset encoder so Deep CFR sees a compact hand representation instead of raw cards.
"""

from __future__ import annotations

import random
from functools import lru_cache

from iip.engine.cards import Card, Deck
from iip.eval.ranker import compare_seven_card_hands


def monte_carlo_equity(
    hole_cards: list[Card],
    board: list[Card] | None = None,
    n_samples: int = 1000,
    rng: random.Random | None = None,
) -> float:
    """Win rate for `hole_cards` vs a random villain hand, averaged over `n_samples` rollouts.

    Ties contribute 0.5. Board may be partial (0/3/4 cards); remaining cards are dealt uniformly.
    """
    rng = rng or random.Random()
    board = board or []
    known = set(hole_cards + board)
    remaining_board = 5 - len(board)

    wins = 0.0
    for _ in range(n_samples):
        deck = Deck(rng=rng)
        deck.remove(known)
        pool = deck.remaining()
        rng.shuffle(pool)
        # Villain 2 cards + remaining board
        villain = pool[:2]
        extra_board = pool[2 : 2 + remaining_board]
        full_board = board + extra_board
        cmp = compare_seven_card_hands(hole_cards + full_board, villain + full_board)
        if cmp > 0:
            wins += 1.0
        elif cmp == 0:
            wins += 0.5
    return wins / n_samples


# ---------- preflop bucketing ----------

def starting_hand_bucket(hole_cards: list[Card]) -> str:
    """Return a canonical 169-entry key like "AKs", "QJo", "77"."""
    c1, c2 = sorted(hole_cards, key=lambda c: -c.rank)
    r1 = _rank_char(c1.rank)
    r2 = _rank_char(c2.rank)
    if c1.rank == c2.rank:
        return r1 + r2
    return r1 + r2 + ("s" if c1.suit == c2.suit else "o")


def _rank_char(r: int) -> str:
    return {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}.get(r, str(r))


@lru_cache(maxsize=1)
def _preflop_id_table() -> dict[str, int]:
    """Deterministic 0..168 ordering: pocket pairs first, then suited, then offsuit."""
    ranks = "23456789TJQKA"
    entries: list[str] = []
    # pairs
    for r in reversed(ranks):
        entries.append(r + r)
    # suited
    for i in range(len(ranks) - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            entries.append(ranks[i] + ranks[j] + "s")
    # offsuit
    for i in range(len(ranks) - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            entries.append(ranks[i] + ranks[j] + "o")
    return {e: idx for idx, e in enumerate(entries)}


def preflop_bucket_id(hole_cards: list[Card]) -> int:
    return _preflop_id_table()[starting_hand_bucket(hole_cards)]


N_PREFLOP_BUCKETS: int = 169
