"""Agent that picks uniformly from legal actions."""

from __future__ import annotations

import random

from iip.agents.base import ActionDist
from iip.engine.game import HULHE, ActionType, HULHEState


class RandomAgent:
    name = "Random"

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def policy(self, game: HULHE, state: HULHEState, player: int) -> ActionDist:
        legal = game.legal_actions(state)
        p = 1.0 / len(legal)
        return {a: p for a in legal}

    def act(self, game: HULHE, state: HULHEState, player: int) -> ActionType:
        return self.rng.choice(game.legal_actions(state))

    def observe(self, game: HULHE, state: HULHEState, player: int) -> None:
        pass
