"""Heads-Up Limit Texas Hold'em engine.

Clean state-machine replacing the old `pokerbot/flow_control/{hugame,headsupgame,handplayed,hdplayed}`.
One class (`HULHE`), one immutable-ish state (`HULHEState`), three actions (FOLD, CHECK_CALL, BET_RAISE).

Rules encoded:
- 2 players, dealer posts small blind, non-dealer posts big blind.
- Preflop first-to-act: dealer (SB). Postflop first-to-act: non-dealer (BB).
- Small bet on preflop + flop; big bet on turn + river.
- At most `max_raises_per_round` aggressive actions per street (initial bet counts as one).
- On fold, opponent wins the pot. On showdown, hand ranker decides; ties split.

The engine is deterministic for a given `random.Random`, so Deep CFR / PPO rollouts can be reproduced.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field, replace

from iip.engine.cards import Card, Deck


class ActionType(enum.IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2


class Street(enum.IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


@dataclass(frozen=True, slots=True)
class Action:
    player: int
    type: ActionType
    street: Street

    def __str__(self) -> str:
        return f"P{self.player}:{self.type.name}@{self.street.name}"


@dataclass
class HULHEState:
    """Mutable state for a single HULHE hand. Players are 0 (dealer/SB) and 1 (BB)."""

    hole_cards: list[list[Card]]
    board: list[Card]
    pot: int
    contributions: list[int]         # this-street contributions per player
    total_invested: list[int]        # total invested in this hand per player
    stacks: list[int]
    to_act: int
    street: Street
    raises_this_street: int
    last_aggressor: int | None
    folded: int | None               # player index who folded, else None
    history: list[Action] = field(default_factory=list)
    terminal: bool = False
    winnings: list[int] = field(default_factory=lambda: [0, 0])

    def clone(self) -> "HULHEState":
        return replace(
            self,
            hole_cards=[list(h) for h in self.hole_cards],
            board=list(self.board),
            contributions=list(self.contributions),
            total_invested=list(self.total_invested),
            stacks=list(self.stacks),
            history=list(self.history),
            winnings=list(self.winnings),
        )


class HULHE:
    """Heads-Up Limit Texas Hold'em. One instance encodes the rules; each `new_hand` yields state."""

    def __init__(
        self,
        small_blind: int = 1,
        big_blind: int = 2,
        small_bet: int = 2,
        big_bet: int = 4,
        max_raises_per_round: int = 4,
        starting_stack: int = 200,
    ) -> None:
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.small_bet = small_bet
        self.big_bet = big_bet
        self.max_raises_per_round = max_raises_per_round
        self.starting_stack = starting_stack

    # ---------- lifecycle ----------

    def new_hand(self, rng: random.Random | None = None, dealer: int = 0) -> HULHEState:
        """Start a new hand. `dealer` is player 0 by default (SB/button in HU)."""
        deck = Deck(rng=rng)
        # Deal two hole cards to each player, in dealing order (dealer first, then BB, alternating).
        hole_cards: list[list[Card]] = [[], []]
        for _ in range(2):
            for p in (dealer, 1 - dealer):
                hole_cards[p].append(deck.deal(1)[0])
        # Blinds.
        stacks = [self.starting_stack, self.starting_stack]
        contributions = [0, 0]
        total_invested = [0, 0]
        # dealer posts SB, non-dealer posts BB.
        stacks[dealer] -= self.small_blind
        contributions[dealer] += self.small_blind
        total_invested[dealer] += self.small_blind
        stacks[1 - dealer] -= self.big_blind
        contributions[1 - dealer] += self.big_blind
        total_invested[1 - dealer] += self.big_blind
        pot = self.small_blind + self.big_blind

        return HULHEState(
            hole_cards=hole_cards,
            board=[],
            pot=pot,
            contributions=contributions,
            total_invested=total_invested,
            stacks=stacks,
            to_act=dealer,           # SB acts first preflop
            street=Street.PREFLOP,
            raises_this_street=1,    # the big blind counts as an "open" that SB must complete/raise
            last_aggressor=1 - dealer,
            folded=None,
        )

    # ---------- queries ----------

    def legal_actions(self, s: HULHEState) -> list[ActionType]:
        if s.terminal:
            return []
        out: list[ActionType] = [ActionType.FOLD, ActionType.CHECK_CALL]
        if s.raises_this_street < self.max_raises_per_round:
            out.append(ActionType.BET_RAISE)
        return out

    def current_bet(self, s: HULHEState) -> int:
        return max(s.contributions) - s.contributions[s.to_act]

    def bet_unit(self, s: HULHEState) -> int:
        return self.small_bet if s.street in (Street.PREFLOP, Street.FLOP) else self.big_bet

    # ---------- transitions ----------

    def step(self, s: HULHEState, action: ActionType, rng: random.Random | None = None) -> HULHEState:
        """Apply an action, returning the resulting state. Mutates `s` in place and also returns it."""
        if s.terminal:
            raise RuntimeError("Cannot step on a terminal state")
        if action not in self.legal_actions(s):
            raise ValueError(f"Illegal action {action.name} in state {s.history}")

        player = s.to_act
        to_call = self.current_bet(s)
        s.history.append(Action(player=player, type=action, street=s.street))

        if action is ActionType.FOLD:
            s.folded = player
            self._settle_on_fold(s)
            return s

        if action is ActionType.CHECK_CALL:
            if to_call > 0:
                pay = min(to_call, s.stacks[player])
                s.stacks[player] -= pay
                s.contributions[player] += pay
                s.total_invested[player] += pay
                s.pot += pay
            # Check/call can close a street if the round is complete.
            if self._round_is_closed_after_call(s):
                self._advance_street(s, rng=rng)
            else:
                s.to_act = 1 - player
            return s

        # BET_RAISE
        unit = self.bet_unit(s)
        # Amount the raiser puts in beyond current call, equal to one betting unit.
        raise_cost = to_call + unit
        pay = min(raise_cost, s.stacks[player])
        s.stacks[player] -= pay
        s.contributions[player] += pay
        s.total_invested[player] += pay
        s.pot += pay
        s.raises_this_street += 1
        s.last_aggressor = player
        s.to_act = 1 - player
        return s

    # ---------- helpers ----------

    def _round_is_closed_after_call(self, s: HULHEState) -> bool:
        """True when the street's betting has equalized and both have acted."""
        if s.contributions[0] != s.contributions[1]:
            return False
        # Special case: preflop, if SB just called, BB still has option to raise (the "BB option").
        # Encoded below: BB option is granted iff we are preflop, no raise occurred (raises_this_street == 1),
        # and the player who just called is the SB/dealer (player whose first action it was).
        if s.street is Street.PREFLOP and s.raises_this_street == 1:
            # The action that just happened was a CHECK_CALL by dealer; BB still gets option.
            last = s.history[-1]
            if last.type is ActionType.CHECK_CALL and last.player == _dealer_of(s):
                return False
        return True

    def _advance_street(self, s: HULHEState, rng: random.Random | None) -> None:
        # Reset street-local counters.
        s.contributions = [0, 0]
        s.raises_this_street = 0
        s.last_aggressor = None
        # Postflop: non-dealer (BB) acts first.
        s.to_act = 1 - _dealer_of(s)

        if s.street is Street.PREFLOP:
            s.street = Street.FLOP
            s.board.extend(_deal_known(s, 3, rng=rng))
        elif s.street is Street.FLOP:
            s.street = Street.TURN
            s.board.extend(_deal_known(s, 1, rng=rng))
        elif s.street is Street.TURN:
            s.street = Street.RIVER
            s.board.extend(_deal_known(s, 1, rng=rng))
        elif s.street is Street.RIVER:
            s.street = Street.SHOWDOWN
            self._settle_showdown(s)

    def _settle_on_fold(self, s: HULHEState) -> None:
        winner = 1 - s.folded  # type: ignore[arg-type]
        s.winnings = [0, 0]
        s.winnings[winner] = s.pot
        s.stacks[winner] += s.pot
        s.terminal = True

    def _settle_showdown(self, s: HULHEState) -> None:
        from iip.eval.ranker import compare_seven_card_hands

        cmp = compare_seven_card_hands(
            s.hole_cards[0] + s.board,
            s.hole_cards[1] + s.board,
        )
        s.winnings = [0, 0]
        if cmp > 0:
            s.winnings[0] = s.pot
            s.stacks[0] += s.pot
        elif cmp < 0:
            s.winnings[1] = s.pot
            s.stacks[1] += s.pot
        else:
            half = s.pot // 2
            leftover = s.pot - 2 * half
            s.winnings[0] = half + leftover  # odd-chip goes to player 0 by convention
            s.winnings[1] = half
            s.stacks[0] += s.winnings[0]
            s.stacks[1] += s.winnings[1]
        s.terminal = True

    def payoffs(self, s: HULHEState) -> list[int]:
        """Net chip change this hand, excluding the starting stack (i.e. +pot to winner minus invested)."""
        if not s.terminal:
            raise RuntimeError("Non-terminal state has no payoff")
        return [s.winnings[p] - s.total_invested[p] for p in (0, 1)]


def _dealer_of(s: HULHEState) -> int:
    """Infer which seat is dealer from the first blind postings visible in pot structure.

    Convention: dealer is the seat that posted the small blind — the seat with smaller initial
    street-0 contribution in the preflop street. For simplicity we store dealer as player 0 here.
    """
    # We keep dealer as player 0 by construction in `new_hand`, so this is a constant.
    # When we later add alternating dealer across hands, the caller rotates hands (see league loop).
    return 0


def _deal_known(s: HULHEState, n: int, rng: random.Random | None) -> list[Card]:
    """Deal `n` cards that aren't already in any player's hole cards or board."""
    deck = Deck(rng=rng)
    known = set(s.hole_cards[0] + s.hole_cards[1] + s.board)
    deck.remove(known)
    return deck.deal(n)
