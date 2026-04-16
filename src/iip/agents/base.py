"""Agent protocol used by the game engine and training loops.

Every agent exposes:
- `act(state, player) -> ActionType`: sample an action from the agent's policy.
- `policy(state, player) -> dict[ActionType, float]`: the full action distribution,
  over *legal* actions only. Training loops that need counterfactual values (CFR) use this.
- Optional `observe(state, player)` hook for stateful bots (kept no-op by default).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from iip.engine.game import HULHE, ActionType, HULHEState

ActionDist = dict[ActionType, float]


@runtime_checkable
class Agent(Protocol):
    name: str

    def act(self, game: HULHE, state: HULHEState, player: int) -> ActionType: ...
    def policy(self, game: HULHE, state: HULHEState, player: int) -> ActionDist: ...
    def observe(self, game: HULHE, state: HULHEState, player: int) -> None: ...
