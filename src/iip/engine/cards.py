"""Card + Deck primitives. Immutable, hashable, cheap to copy.

A Card is a frozen dataclass; rank ∈ [2..14] (Ace high), suit ∈ {"s", "h", "d", "c"}.
The integer encoding is `(rank - 2) * 4 + suit_index` with suit order s<h<d<c, giving
ids 0..51 that line up with treys when its own encoding is consulted through ranker.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

SUITS: tuple[str, ...] = ("s", "h", "d", "c")
RANKS: tuple[int, ...] = tuple(range(2, 15))
RANK_CHAR: dict[int, str] = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}
CHAR_RANK: dict[str, int] = {v: k for k, v in RANK_CHAR.items()}
SUIT_INDEX: dict[str, int] = {s: i for i, s in enumerate(SUITS)}


@dataclass(frozen=True, slots=True)
class Card:
    rank: int
    suit: str

    def __post_init__(self) -> None:
        if self.rank not in CHAR_RANK.values():
            raise ValueError(f"Invalid rank {self.rank}")
        if self.suit not in SUIT_INDEX:
            raise ValueError(f"Invalid suit {self.suit!r}")

    @property
    def id(self) -> int:
        return (self.rank - 2) * 4 + SUIT_INDEX[self.suit]

    def __str__(self) -> str:
        return f"{RANK_CHAR[self.rank]}{self.suit}"

    def __repr__(self) -> str:
        return str(self)


def card_from_str(s: str) -> Card:
    """Parse "As", "Td", "7h" into a Card."""
    if len(s) != 2:
        raise ValueError(f"Card string must be 2 chars: {s!r}")
    return Card(rank=CHAR_RANK[s[0].upper()], suit=s[1].lower())


def card_from_id(card_id: int) -> Card:
    if not 0 <= card_id < 52:
        raise ValueError(f"Card id out of range: {card_id}")
    rank = card_id // 4 + 2
    suit = SUITS[card_id % 4]
    return Card(rank=rank, suit=suit)


class Deck:
    """Shuffled 52-card deck. Deterministic when seeded."""

    __slots__ = ("_cards", "_rng")

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self._cards: list[Card] = [Card(r, s) for r in RANKS for s in SUITS]
        self._rng.shuffle(self._cards)

    def deal(self, n: int) -> list[Card]:
        if n > len(self._cards):
            raise ValueError(f"Cannot deal {n} cards, {len(self._cards)} remain")
        dealt, self._cards = self._cards[:n], self._cards[n:]
        return dealt

    def remaining(self) -> list[Card]:
        return list(self._cards)

    def remove(self, cards: Iterable[Card]) -> None:
        """Remove known cards from the deck (useful for equity MC given dealt community cards)."""
        cs = set(cards)
        self._cards = [c for c in self._cards if c not in cs]
