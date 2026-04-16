"""Adapters that let the Deep CFR and PPO trainers speak a single interface to any game.

The trainers only need:
- `num_actions`: int
- `new_hand(rng)`: returns a state
- `is_terminal(state)`: bool
- `payoffs(state)`: list[int] (length 2)
- `to_act(state)`: int
- `legal_mask(state)`: np.ndarray of shape (num_actions,), 1 for legal, 0 else
- `encode(state, player)`: np.ndarray feature vector
- `step(state, action_idx)`: mutates/returns a new state
- `feature_dim`: int

The Kuhn adapter is used by the sanity test; the HULHE adapter is used for the real bot.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from iip.engine.game import HULHE, ActionType, HULHEState
from iip.engine.kuhn import Kuhn, KuhnAction, KuhnState
from iip.features.infoset import (
    INFOSET_DIM_HULHE,
    INFOSET_DIM_KUHN,
    HULHEInfosetEncoder,
    KuhnInfosetEncoder,
    legal_action_mask_hulhe,
    legal_action_mask_kuhn,
)


class GameAdapter(Protocol):
    num_actions: int
    feature_dim: int
    name: str

    def new_hand(self, rng: random.Random | None = None): ...
    def is_terminal(self, state) -> bool: ...
    def payoffs(self, state) -> list[int]: ...
    def to_act(self, state) -> int: ...
    def legal_mask(self, state) -> np.ndarray: ...
    def encode(self, state, player: int) -> np.ndarray: ...
    def step(self, state, action_idx: int, rng: random.Random | None = None): ...


@dataclass
class HULHEAdapter:
    game: HULHE
    name: str = "HULHE"
    num_actions: int = 3
    feature_dim: int = INFOSET_DIM_HULHE

    def __post_init__(self) -> None:
        self._enc = HULHEInfosetEncoder(game=self.game)

    def new_hand(self, rng: random.Random | None = None) -> HULHEState:
        return self.game.new_hand(rng=rng)

    def is_terminal(self, state: HULHEState) -> bool:
        return state.terminal

    def payoffs(self, state: HULHEState) -> list[int]:
        return self.game.payoffs(state)

    def to_act(self, state: HULHEState) -> int:
        return state.to_act

    def legal_mask(self, state: HULHEState) -> np.ndarray:
        return legal_action_mask_hulhe(self.game, state)

    def encode(self, state: HULHEState, player: int) -> np.ndarray:
        return self._enc.encode(state, player)

    def step(self, state: HULHEState, action_idx: int, rng: random.Random | None = None) -> HULHEState:
        return self.game.step(state, ActionType(action_idx), rng=rng)


@dataclass
class KuhnAdapter:
    game: Kuhn
    name: str = "Kuhn"
    num_actions: int = 2
    feature_dim: int = INFOSET_DIM_KUHN

    def __post_init__(self) -> None:
        self._enc = KuhnInfosetEncoder()

    def new_hand(self, rng: random.Random | None = None) -> KuhnState:
        return self.game.new_hand(rng=rng)

    def is_terminal(self, state: KuhnState) -> bool:
        return state.terminal

    def payoffs(self, state: KuhnState) -> list[int]:
        return self.game.payoffs(state)

    def to_act(self, state: KuhnState) -> int:
        return state.to_act

    def legal_mask(self, state: KuhnState) -> np.ndarray:
        return legal_action_mask_kuhn(state)

    def encode(self, state: KuhnState, player: int) -> np.ndarray:
        return self._enc.encode(state, player)

    def step(self, state: KuhnState, action_idx: int, rng: random.Random | None = None) -> KuhnState:
        return self.game.step(state, KuhnAction(action_idx))
