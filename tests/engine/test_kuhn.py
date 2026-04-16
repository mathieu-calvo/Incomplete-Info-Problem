"""Kuhn poker rules: exhaustive trajectory checks."""

from __future__ import annotations

import random

from iip.engine.kuhn import Kuhn, KuhnAction


def test_both_pass_showdown():
    g = Kuhn()
    s = g.new_hand(rng=random.Random(0))
    s.cards = [3, 1]  # force player 0 ace, player 1 low
    g.step(s, KuhnAction.PASS)
    g.step(s, KuhnAction.PASS)
    assert s.terminal
    assert g.payoffs(s) == [1, -1]


def test_bet_fold_gives_2_to_bettor_minus_ante_etc():
    g = Kuhn()
    s = g.new_hand(rng=random.Random(0))
    s.cards = [1, 3]  # doesn't matter for fold
    g.step(s, KuhnAction.BET)
    g.step(s, KuhnAction.PASS)
    assert s.terminal
    # Player 0 bet (invested 2), won the 3-chip pot -> net +1; player 1 anted 1 and folded -> -1.
    assert g.payoffs(s) == [1, -1]


def test_pass_bet_call_showdown_4_chips():
    g = Kuhn()
    s = g.new_hand(rng=random.Random(0))
    s.cards = [3, 2]
    g.step(s, KuhnAction.PASS)
    g.step(s, KuhnAction.BET)
    g.step(s, KuhnAction.BET)  # call
    assert s.terminal
    # Both invested 2; pot = 4; higher card wins.
    assert g.payoffs(s) == [2, -2]
