"""Card, Deck, and id round-trip checks."""

from __future__ import annotations

import random

from iip.engine.cards import Card, Deck, card_from_id, card_from_str


def test_card_str_roundtrip():
    c = Card(14, "s")
    assert str(c) == "As"
    assert card_from_str("As") == c
    assert card_from_str("td") == Card(10, "d")


def test_card_id_unique_and_complete():
    ids = {Card(r, s).id for r in range(2, 15) for s in ("s", "h", "d", "c")}
    assert ids == set(range(52))


def test_card_from_id_inverse():
    for i in range(52):
        assert card_from_id(i).id == i


def test_deck_deal_is_deterministic():
    d1 = Deck(rng=random.Random(7))
    d2 = Deck(rng=random.Random(7))
    assert d1.deal(5) == d2.deal(5)


def test_deck_remove():
    d = Deck(rng=random.Random(0))
    first = d.deal(2)
    rest = d.remaining()
    assert len(rest) == 50
    assert not set(first) & set(rest)
