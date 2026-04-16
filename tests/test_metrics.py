"""Smoke tests for mbb/h and LBR wrappers."""

from __future__ import annotations

import random

from iip.agents.fixed_policy import FishAgent
from iip.agents.random_agent import RandomAgent
from iip.engine.game import HULHE
from iip.metrics.exploitability import local_best_response
from iip.metrics.mbb import head_to_head_mbb


def test_head_to_head_returns_result():
    g = HULHE()
    r = head_to_head_mbb(g, FishAgent(), RandomAgent(rng=random.Random(0)), n_hands=20, seed=0)
    assert r.hands == 20
    assert r.hero_wins + r.villain_wins + r.ties == 20


def test_lbr_smoke():
    g = HULHE()
    score = local_best_response(g, FishAgent(), n_hands=10, rollouts_per_decision=2, seed=0)
    # Fish is obviously exploitable — LBR should return a non-trivial number (sign not asserted).
    assert isinstance(score, float)
