"""Kuhn poker — the minimal imperfect-information game (3 cards, 2 players, 1 bet).

We use Kuhn as a correctness harness for Deep CFR. The known equilibrium utility for
player 0 (first to act) is -1/18 per hand; any implementation that converges to better
than that against a best response is buggy.

Rules:
- 3-card deck: {1, 2, 3}. Each player antes 1 and gets 1 private card.
- Player 0 acts first: check or bet.
- If both check: showdown, higher card wins the 2-chip pot.
- If player 0 bets: player 1 calls (bet 1, showdown for 4 chips) or folds (player 0 wins 2).
- If player 0 checks and player 1 bets: player 0 calls (showdown for 4) or folds (player 1 wins 2).
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field, replace


class KuhnAction(enum.IntEnum):
    PASS = 0   # check or fold depending on context
    BET = 1    # bet or call


@dataclass
class KuhnState:
    cards: list[int]                 # private card per player
    to_act: int
    history: list[KuhnAction] = field(default_factory=list)
    pot: int = 2
    terminal: bool = False
    folded: int | None = None
    winnings: list[int] = field(default_factory=lambda: [0, 0])

    def clone(self) -> KuhnState:
        return replace(self, cards=list(self.cards), history=list(self.history), winnings=list(self.winnings))


class Kuhn:
    def __init__(self, ante: int = 1) -> None:
        self.ante = ante

    def new_hand(self, rng: random.Random | None = None) -> KuhnState:
        rng = rng or random.Random()
        deck = [1, 2, 3]
        rng.shuffle(deck)
        return KuhnState(cards=deck[:2], to_act=0, pot=2 * self.ante)

    def legal_actions(self, s: KuhnState) -> list[KuhnAction]:
        if s.terminal:
            return []
        return [KuhnAction.PASS, KuhnAction.BET]

    def step(self, s: KuhnState, a: KuhnAction) -> KuhnState:
        if s.terminal:
            raise RuntimeError("Cannot step on terminal state")
        s.history.append(a)
        h = s.history
        if len(h) == 1:
            s.to_act = 1
            return s
        if len(h) == 2:
            # [pass, pass] -> showdown
            if h[0] == KuhnAction.PASS and h[1] == KuhnAction.PASS:
                self._showdown(s)
                return s
            # [pass, bet] -> player 0 decides
            if h[0] == KuhnAction.PASS and h[1] == KuhnAction.BET:
                s.to_act = 0
                return s
            # [bet, pass] -> player 1 folds
            if h[0] == KuhnAction.BET and h[1] == KuhnAction.PASS:
                s.folded = 1
                self._fold_settle(s)
                return s
            # [bet, bet] -> showdown for 4 chips
            if h[0] == KuhnAction.BET and h[1] == KuhnAction.BET:
                s.pot += 2
                self._showdown(s)
                return s
        if len(h) == 3:
            # [pass, bet, ?]
            if h[2] == KuhnAction.PASS:
                s.folded = 0
                self._fold_settle(s)
                return s
            if h[2] == KuhnAction.BET:
                s.pot += 2
                self._showdown(s)
                return s
        raise RuntimeError("Kuhn history too long")

    def _showdown(self, s: KuhnState) -> None:
        winner = 0 if s.cards[0] > s.cards[1] else 1
        s.winnings = [0, 0]
        s.winnings[winner] = s.pot
        s.terminal = True

    def _fold_settle(self, s: KuhnState) -> None:
        winner = 1 - s.folded  # type: ignore[arg-type]
        s.winnings = [0, 0]
        s.winnings[winner] = s.pot
        s.terminal = True

    def payoffs(self, s: KuhnState) -> list[int]:
        if not s.terminal:
            raise RuntimeError("Non-terminal state")
        # Each player anted 1, and any bet adds 1 more.
        invested = [self.ante, self.ante]
        if KuhnAction.BET in s.history:
            # Both contributed an extra 1 only if both bet/called (pot is 4) or we had a fold
            # in which case only one bet. Easier to infer from pot minus antes: total bets = pot - 2*ante.
            total_bet_chips = s.pot - 2 * self.ante
            if total_bet_chips == 2:
                invested = [self.ante + 1, self.ante + 1]
            elif total_bet_chips == 1:
                # exactly one player bet and other folded
                bettor = 0 if s.history[0] == KuhnAction.BET else 1
                if s.history[0] == KuhnAction.PASS and len(s.history) >= 2 and s.history[1] == KuhnAction.BET:
                    bettor = 1
                invested[bettor] += 1
        return [s.winnings[p] - invested[p] for p in (0, 1)]
