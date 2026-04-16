"""Wrap the HULHE engine in a Streamlit-friendly session object.

`PlaySession` keeps one hand of state in `st.session_state`. It drives the bot forward after
each user action until control returns to the user (or the hand ends). All interactions are
sync; each Streamlit script re-run reads from the session.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import streamlit as st

from app.services.session_tracker import get_session_id
from iip.agents.base import Agent
from iip.engine.game import ActionType, HULHE, HULHEState

_SESSION_KEY = "iip_session"


@dataclass
class LoggedAction:
    player: int
    action: str
    street: str
    bot_policy: dict[str, float] | None = None


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

    def start_new_hand(self, seed: int | None = None) -> None:
        rng = random.Random(seed) if seed is not None else random.Random()
        self.state = self.game.new_hand(rng=rng)
        self.log = []
        self.hero_seat = self.hand_index % 2
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

    def hand_over(self) -> bool:
        return self.state is not None and self.state.terminal

    def _advance_bot_until_hero_or_terminal(self) -> None:
        while self.state is not None and not self.state.terminal and self.state.to_act != self.hero_seat:
            cur = self.state.to_act
            dist = self.bot.policy(self.game, self.state, cur)
            policy = {k.name: float(v) for k, v in dist.items()}
            action = self.bot.act(self.game, self.state, cur)
            self.log.append(
                LoggedAction(
                    player=cur,
                    action=action.name,
                    street=self.state.street.name,
                    bot_policy=policy,
                )
            )
            self.game.step(self.state, action)


def get_session(bot: Agent, bot_checkpoint: str) -> PlaySession:
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = PlaySession(
            user_id=get_session_id(),
            game=HULHE(),
            bot=bot,
            bot_checkpoint=bot_checkpoint,
        )
    return st.session_state[_SESSION_KEY]
