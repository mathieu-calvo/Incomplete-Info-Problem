"""Milli-big-blinds per hand — the canonical HU poker metric.

Plays `n_hands` between two agents and reports hero's win rate in mbb/h with a 95% CI
derived from the per-hand sample standard deviation. Seats alternate every hand so that
position advantage cancels out.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from iip.agents.base import Agent
from iip.engine.game import HULHE


@dataclass
class HeadToHeadResult:
    hero_name: str
    villain_name: str
    hands: int
    mbbph: float        # hero's average milli-big-blinds / hand
    ci95: float         # ± mbb/h at 95% confidence
    hero_wins: int
    villain_wins: int
    ties: int

    def __str__(self) -> str:
        return (
            f"{self.hero_name} vs {self.villain_name}: "
            f"{self.mbbph:+.2f} ± {self.ci95:.2f} mbb/h over {self.hands} hands "
            f"({self.hero_wins}W/{self.villain_wins}L/{self.ties}T)"
        )


def head_to_head_mbb(
    game: HULHE,
    hero: Agent,
    villain: Agent,
    n_hands: int,
    seed: int = 0,
) -> HeadToHeadResult:
    rng = random.Random(seed)
    per_hand_mbb: list[float] = []
    hero_wins = villain_wins = ties = 0
    bb = max(game.big_blind, 1)

    for h in range(n_hands):
        hero_seat = h % 2
        state = game.new_hand(rng=random.Random(rng.random()))
        while not state.terminal:
            cur = state.to_act
            agent = hero if cur == hero_seat else villain
            action = agent.act(game, state, cur)
            game.step(state, action, rng=random.Random(rng.random()))
        payoff = game.payoffs(state)[hero_seat]
        per_hand_mbb.append(payoff * 1000.0 / bb)
        if payoff > 0:
            hero_wins += 1
        elif payoff < 0:
            villain_wins += 1
        else:
            ties += 1

    arr = np.array(per_hand_mbb, dtype=np.float64)
    mean = float(arr.mean())
    ci = 1.96 * float(arr.std(ddof=1)) / math.sqrt(max(len(arr), 1)) if len(arr) > 1 else 0.0
    return HeadToHeadResult(
        hero_name=getattr(hero, "name", "Hero"),
        villain_name=getattr(villain, "name", "Villain"),
        hands=n_hands,
        mbbph=mean,
        ci95=ci,
        hero_wins=hero_wins,
        villain_wins=villain_wins,
        ties=ties,
    )
