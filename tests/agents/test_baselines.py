"""Baseline agents play legal hands to completion without erroring."""

from __future__ import annotations

import random

from iip.agents.fixed_policy import FishAgent, StartingHandAgent, StrengthHandAgent
from iip.agents.random_agent import RandomAgent
from iip.engine.game import HULHE


def test_all_baselines_play_one_hand():
    g = HULHE()
    agents = [RandomAgent(rng=random.Random(0)), FishAgent(), StartingHandAgent(), StrengthHandAgent(n_samples=50)]
    for a in agents:
        for b in agents:
            s = g.new_hand(rng=random.Random(1))
            depth = 0
            while not s.terminal and depth < 200:
                act = (a if s.to_act == 0 else b).act(g, s, s.to_act)
                g.step(s, act)
                depth += 1
            assert s.terminal
