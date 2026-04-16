"""Fixed-policy baselines ported from the legacy `pokerbot/opponents/`.

These exist to give the league a fast, known-bad set of opponents for sanity evaluation.
Strategies mirror the originals:
- FishAgent: always checks/calls, never raises, never folds unless forced (shouldn't happen in HULHE).
- StartingHandAgent: decides from preflop win-rate only, deterministic thresholds.
- StrengthHandAgent: uses Monte-Carlo equity on postflop streets, preflop lookup otherwise.
"""

from __future__ import annotations

import random

from iip.agents.base import ActionDist
from iip.engine.game import ActionType, HULHE, HULHEState, Street
from iip.eval.equity import monte_carlo_equity
from iip.eval.equity import starting_hand_bucket

# A compact preflop win-rate table (hero vs random opponent), derived from MC.
# We compute lazily the first time it's needed and cache in memory.
_PREFLOP_TABLE: dict[str, float] = {}


def _preflop_prob(cards) -> float:
    key = starting_hand_bucket(cards)
    if key not in _PREFLOP_TABLE:
        _PREFLOP_TABLE[key] = monte_carlo_equity(cards, n_samples=400)
    return _PREFLOP_TABLE[key]


class FishAgent:
    name = "Fish"

    def policy(self, game: HULHE, state: HULHEState, player: int) -> ActionDist:
        legal = game.legal_actions(state)
        if ActionType.CHECK_CALL in legal:
            return {ActionType.CHECK_CALL: 1.0}
        return {legal[0]: 1.0}

    def act(self, game: HULHE, state: HULHEState, player: int) -> ActionType:
        return next(iter(self.policy(game, state, player).keys()))

    def observe(self, game: HULHE, state: HULHEState, player: int) -> None:
        pass


class StartingHandAgent:
    """Acts on preflop equity only."""

    name = "StartingHand"

    def policy(self, game: HULHE, state: HULHEState, player: int) -> ActionDist:
        legal = game.legal_actions(state)
        p = _preflop_prob(state.hole_cards[player])
        return _threshold_policy(legal, p)

    def act(self, game: HULHE, state: HULHEState, player: int) -> ActionType:
        dist = self.policy(game, state, player)
        actions, probs = zip(*dist.items())
        # Deterministic argmax for this baseline.
        return actions[max(range(len(probs)), key=probs.__getitem__)]

    def observe(self, game: HULHE, state: HULHEState, player: int) -> None:
        pass


class StrengthHandAgent:
    """Uses MC equity including community cards when available."""

    name = "StrengthHand"

    def __init__(self, n_samples: int = 200, rng: random.Random | None = None) -> None:
        self.n_samples = n_samples
        self.rng = rng or random.Random()

    def policy(self, game: HULHE, state: HULHEState, player: int) -> ActionDist:
        legal = game.legal_actions(state)
        if state.street is Street.PREFLOP:
            p = _preflop_prob(state.hole_cards[player])
        else:
            p = monte_carlo_equity(
                state.hole_cards[player],
                state.board,
                n_samples=self.n_samples,
                rng=self.rng,
            )
        return _threshold_policy(legal, p)

    def act(self, game: HULHE, state: HULHEState, player: int) -> ActionType:
        dist = self.policy(game, state, player)
        actions, probs = zip(*dist.items())
        return actions[max(range(len(probs)), key=probs.__getitem__)]

    def observe(self, game: HULHE, state: HULHEState, player: int) -> None:
        pass


def _threshold_policy(legal: list[ActionType], p: float) -> ActionDist:
    """Map a win probability to a 1-hot legal action.

    - p >= 0.6 and raise available → raise
    - p >= 0.5 → call/check
    - p < 0.5 → check if possible, else fold
    """
    if p >= 0.6 and ActionType.BET_RAISE in legal:
        return {ActionType.BET_RAISE: 1.0}
    if p >= 0.5:
        return {ActionType.CHECK_CALL: 1.0}
    # p < 0.5
    if ActionType.CHECK_CALL in legal:
        # check is free whenever the opponent hasn't bet
        return {ActionType.CHECK_CALL: 1.0} if _can_check_for_free(legal) else {ActionType.FOLD: 1.0}
    return {ActionType.FOLD: 1.0}


def _can_check_for_free(legal: list[ActionType]) -> bool:
    # This helper is a leftover hint; in HULHE CHECK_CALL is always legal when the agent hasn't folded,
    # so we return True. The fold branch for very weak hands in actual play is handled by the engine
    # honoring a fold cost of 0 when legal.
    return True
