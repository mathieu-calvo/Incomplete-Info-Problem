"""HULHE engine: blinds, street transitions, fold/showdown payoffs, legal actions."""

from __future__ import annotations

import random

from iip.agents.random_agent import RandomAgent
from iip.engine.game import HULHE, ActionType, Street


def test_new_hand_posts_blinds():
    g = HULHE()
    s = g.new_hand(rng=random.Random(0))
    assert s.contributions == [1, 2] or s.contributions == [2, 1]
    assert s.pot == 3
    assert s.to_act == 0
    assert s.street is Street.PREFLOP
    assert s.raises_this_street == 1


def test_fold_settles_pot_to_opponent():
    g = HULHE()
    s = g.new_hand(rng=random.Random(0))
    g.step(s, ActionType.FOLD)
    assert s.terminal
    # Player 0 folded; player 1 wins the 3-chip pot.
    payoffs = g.payoffs(s)
    assert sum(payoffs) == 0
    assert payoffs[0] < 0 and payoffs[1] > 0


def test_legal_actions_cap_on_raises():
    g = HULHE(max_raises_per_round=4)
    s = g.new_hand(rng=random.Random(0))
    # Max out raises
    for _ in range(3):
        if ActionType.BET_RAISE in g.legal_actions(s):
            g.step(s, ActionType.BET_RAISE)
    # After cap hit, only FOLD and CHECK_CALL should remain.
    legal = g.legal_actions(s)
    assert ActionType.BET_RAISE not in legal or s.raises_this_street < 4


def test_full_random_game_terminates():
    g = HULHE()
    a = RandomAgent(rng=random.Random(1))
    for seed in range(25):
        s = g.new_hand(rng=random.Random(seed))
        depth = 0
        while not s.terminal and depth < 200:
            action = a.act(g, s, s.to_act)
            g.step(s, action, rng=random.Random(seed + depth))
            depth += 1
        assert s.terminal
        # Zero-sum.
        p = g.payoffs(s)
        assert sum(p) == 0
