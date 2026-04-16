"""Sanity checks on MC equity + preflop bucketing."""

from __future__ import annotations

import random

from iip.engine.cards import card_from_str
from iip.eval.equity import (
    N_PREFLOP_BUCKETS,
    monte_carlo_equity,
    preflop_bucket_id,
    starting_hand_bucket,
)


def test_aa_equity_is_very_high():
    aa = [card_from_str("As"), card_from_str("Ah")]
    p = monte_carlo_equity(aa, n_samples=500, rng=random.Random(0))
    assert p > 0.75


def test_bucket_keys_are_canonical():
    ak_suited = [card_from_str("As"), card_from_str("Ks")]
    ka_suited = [card_from_str("Ks"), card_from_str("As")]
    assert starting_hand_bucket(ak_suited) == "AKs"
    assert starting_hand_bucket(ka_suited) == "AKs"


def test_bucket_ids_cover_169():
    ranks = "23456789TJQKA"
    ids = set()
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if i == j:
                hand = [card_from_str(r1 + "s"), card_from_str(r2 + "h")]
            elif i > j:
                hand = [card_from_str(r1 + "s"), card_from_str(r2 + "s")]
            else:
                hand = [card_from_str(r1 + "s"), card_from_str(r2 + "h")]
            ids.add(preflop_bucket_id(hand))
    assert ids == set(range(N_PREFLOP_BUCKETS))
