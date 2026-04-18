"""Wrap the HULHE engine in a Streamlit-friendly session object.

`PlaySession` keeps one game's state in `st.session_state`:
- stacks persist across hands (hero/bot each start at `starting_stack`, evolve with each hand),
- when either player can't post the big blind, the session is `game_over()` and the user
  can `start_new_game()` to reset chips.

Also enforces a table-talk rule the trained bot can occasionally violate: the bot never
folds when it can check for free. We resample its policy excluding FOLD in that case.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import streamlit as st

from app.services.session_tracker import get_session_id
from iip.agents.base import Agent
from iip.engine.game import HULHE, ActionType, HULHEState

_SESSION_KEY = "iip_session"


@dataclass
class LoggedAction:
    player: int
    action: str
    street: str
    bot_policy: dict[str, float] | None = None


@dataclass
class CompletedHand:
    """Snapshot of a finished hand, kept so the user can browse past hand logs."""
    state: HULHEState
    hero_seat: int
    hand_index: int


@dataclass
class PlaySession:
    user_id: str
    game: HULHE
    bot: Agent
    bot_checkpoint: str
    hero_seat: int = 0
    state: HULHEState | None = None
    log: list[LoggedAction] = field(default_factory=list)
    hand_index: int = 0
    # Persistent stacks by physical player (hero / bot), carried across hands.
    hero_total_stack: int = 0
    bot_total_stack: int = 0
    # Terminal states of every hand played this game, in order. Used by the hand-log navigator.
    completed_hands: list[CompletedHand] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.hero_total_stack == 0:
            self.hero_total_stack = self.game.starting_stack
        if self.bot_total_stack == 0:
            self.bot_total_stack = self.game.starting_stack

    def start_new_game(self) -> None:
        """Reset chips and hand counter. Does not start a hand — caller follows with start_new_hand."""
        self.hero_total_stack = self.game.starting_stack
        self.bot_total_stack = self.game.starting_stack
        self.hand_index = 0
        self.state = None
        self.log = []
        self.completed_hands = []

    def game_over(self) -> bool:
        """True when either player can't post the big blind."""
        bb = self.game.big_blind
        return self.hero_total_stack < bb or self.bot_total_stack < bb

    def hero_won_game(self) -> bool | None:
        """True if hero busted bot, False if hero busted, None while game still in progress."""
        if not self.game_over():
            return None
        bb = self.game.big_blind
        hero_bust = self.hero_total_stack < bb
        bot_bust = self.bot_total_stack < bb
        if hero_bust and not bot_bust:
            return False
        if bot_bust and not hero_bust:
            return True
        # Edge case: both short. Declare by chip count.
        return self.hero_total_stack >= self.bot_total_stack

    def start_new_hand(self, seed: int | None = None) -> None:
        if self.game_over():
            return
        rng = random.Random(seed) if seed is not None else random.Random()
        self.hero_seat = self.hand_index % 2
        # Place per-player stacks into seat indices: player 0 is the dealer/SB.
        stacks = [0, 0]
        stacks[self.hero_seat] = self.hero_total_stack
        stacks[1 - self.hero_seat] = self.bot_total_stack
        self.state = self.game.new_hand(rng=rng, stacks=stacks)
        self.log = []
        self.hand_index += 1
        self._advance_bot_until_hero_or_terminal()

    def user_action(self, action: ActionType) -> None:
        assert self.state is not None
        assert self.state.to_act == self.hero_seat
        self.log.append(
            LoggedAction(
                player=self.hero_seat,
                action=action.name,
                street=self.state.street.name,
            )
        )
        self.game.step(self.state, action)
        self._advance_bot_until_hero_or_terminal()
        if self.state is not None and self.state.terminal:
            self._settle_hand_stacks()

    def hand_over(self) -> bool:
        return self.state is not None and self.state.terminal

    def _settle_hand_stacks(self) -> None:
        """Copy seat-indexed stacks from the terminal state back into persistent per-player totals."""
        assert self.state is not None and self.state.terminal
        self.hero_total_stack = self.state.stacks[self.hero_seat]
        self.bot_total_stack = self.state.stacks[1 - self.hero_seat]
        if not self.completed_hands or self.completed_hands[-1].hand_index != self.hand_index:
            self.completed_hands.append(
                CompletedHand(
                    state=self.state.clone(),
                    hero_seat=self.hero_seat,
                    hand_index=self.hand_index,
                )
            )

    def _sample_bot_action(self) -> tuple[ActionType, dict[str, float]]:
        """Bot's action sample, filtering FOLD when checking is free. Returns (action, full_policy)."""
        cur = self.state.to_act  # type: ignore[union-attr]
        dist = self.bot.policy(self.game, self.state, cur)  # type: ignore[arg-type]
        policy_report = {k.name: float(v) for k, v in dist.items()}
        if self.game.current_bet(self.state) == 0:  # type: ignore[arg-type]
            dist.pop(ActionType.FOLD, None)
        total = sum(dist.values())
        actions = list(dist.keys())
        if total <= 0 or not actions:
            return (actions[0] if actions else ActionType.CHECK_CALL), policy_report
        r = random.random() * total
        cum = 0.0
        chosen = actions[-1]
        for a, p in dist.items():
            cum += p
            if r <= cum:
                chosen = a
                break
        return chosen, policy_report

    def _advance_bot_until_hero_or_terminal(self) -> None:
        while (
            self.state is not None
            and not self.state.terminal
            and self.state.to_act != self.hero_seat
        ):
            cur = self.state.to_act
            action, policy = self._sample_bot_action()
            self.log.append(
                LoggedAction(
                    player=cur,
                    action=action.name,
                    street=self.state.street.name,
                    bot_policy=policy,
                )
            )
            self.game.step(self.state, action)
        if self.state is not None and self.state.terminal:
            self._settle_hand_stacks()


def get_session(bot: Agent, bot_checkpoint: str) -> PlaySession:
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = PlaySession(
            user_id=get_session_id(),
            game=HULHE(),
            bot=bot,
            bot_checkpoint=bot_checkpoint,
        )
    return st.session_state[_SESSION_KEY]
