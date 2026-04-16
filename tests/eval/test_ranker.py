"""Cross-check: fallback 5-card evaluator matches expected rankings, and
7-card treys output preserves ordering when cards form known hands."""

from __future__ import annotations

from iip.engine.cards import card_from_str
from iip.eval.ranker import (
    compare_seven_card_hands,
    evaluate_five_card_fallback,
    rank_seven_card,
)


def test_fallback_recognises_categories():
    straight_flush = [card_from_str(x) for x in ["As", "Ks", "Qs", "Js", "Ts"]]
    assert evaluate_five_card_fallback(straight_flush)[0] == 9

    quads = [card_from_str(x) for x in ["As", "Ah", "Ad", "Ac", "Kh"]]
    assert evaluate_five_card_fallback(quads)[0] == 8

    full_house = [card_from_str(x) for x in ["Kh", "Kd", "Kc", "2s", "2h"]]
    assert evaluate_five_card_fallback(full_house)[0] == 7

    flush = [card_from_str(x) for x in ["As", "Js", "9s", "6s", "2s"]]
    assert evaluate_five_card_fallback(flush)[0] == 6

    wheel = [card_from_str(x) for x in ["As", "2h", "3d", "4c", "5h"]]
    assert evaluate_five_card_fallback(wheel) == (5, [5])

    high_card = [card_from_str(x) for x in ["As", "Jh", "9d", "6c", "2h"]]
    assert evaluate_five_card_fallback(high_card)[0] == 1


def test_seven_card_quads_beats_flush():
    quads = [card_from_str(x) for x in ["Ad", "Ah", "As", "Ac", "3s", "4s", "5s"]]
    flush = [card_from_str(x) for x in ["Ks", "Qs", "Js", "9s", "3s", "4s", "5s"]]
    assert compare_seven_card_hands(quads, flush) == 1


def test_seven_card_tie_yields_zero():
    same_a = [card_from_str(x) for x in ["As", "Kh", "Qd", "Jc", "Ts", "2h", "3d"]]
    same_b = [card_from_str(x) for x in ["Ad", "Kc", "Qs", "Jh", "Tc", "2s", "3h"]]
    # Both hit A-high straight. Tie expected.
    assert compare_seven_card_hands(same_a, same_b) == 0


def test_rank_seven_card_is_monotonic_with_category():
    high_card = [card_from_str(x) for x in ["2h", "4d", "6s", "8c", "Ts", "Jh", "3d"]]
    one_pair = [card_from_str(x) for x in ["2h", "2d", "6s", "8c", "Ts", "Jh", "3d"]]
    assert rank_seven_card(one_pair) > rank_seven_card(high_card)
