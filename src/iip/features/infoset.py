"""Information-set encoders — take a game state + acting player, return a fixed-size float vector.

The encoder is what the Deep CFR advantage/strategy nets and the PPO actor/critic consume.
It must be deterministic, bounded, and (for Deep CFR soundness) a function only of the acting
player's information set — i.e. it must not look at the opponent's hidden cards.

HULHE encoding (`INFOSET_DIM_HULHE`):
- One-hot preflop bucket id (169)  — compact hand id from eval.equity
- Board card multi-hot by rank     (13)
- Board suit summary (max of each suit count / 5, for flush potential) (4)
- Street one-hot                   (4)
- Pot / starting_stack (scalar, clamped)                   (1)
- Stack_self / starting_stack                              (1)
- Stack_opp / starting_stack                               (1)
- Contribution_self / big_bet                              (1)
- Contribution_opp / big_bet                               (1)
- Raises_this_street / max_raises                          (1)
- Position (is_dealer)                                     (1)
- Legal action mask (3)                                    (3)

Total: 169 + 13 + 4 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 3 = 200.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iip.engine.cards import SUIT_INDEX
from iip.engine.game import HULHE, HULHEState, Street
from iip.engine.kuhn import KuhnState
from iip.eval.equity import N_PREFLOP_BUCKETS, preflop_bucket_id

INFOSET_DIM_HULHE: int = N_PREFLOP_BUCKETS + 13 + 4 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 3
INFOSET_DIM_KUHN: int = 3 + 2 + 4  # card one-hot + position + action history padded


@dataclass
class HULHEInfosetEncoder:
    game: HULHE

    def encode(self, state: HULHEState, player: int) -> np.ndarray:
        x = np.zeros(INFOSET_DIM_HULHE, dtype=np.float32)
        off = 0

        # Preflop bucket.
        x[off + preflop_bucket_id(state.hole_cards[player])] = 1.0
        off += N_PREFLOP_BUCKETS

        # Board: rank multi-hot.
        for c in state.board:
            x[off + (c.rank - 2)] = 1.0
        off += 13

        # Suit summary: count per suit / 5.
        suit_counts = np.zeros(4, dtype=np.float32)
        for c in state.board:
            suit_counts[SUIT_INDEX[c.suit]] += 1.0
        x[off : off + 4] = suit_counts / 5.0
        off += 4

        # Street one-hot.
        x[off + int(state.street)] = 1.0
        off += 4

        # Pot & stacks (normalised).
        stack_norm = max(self.game.starting_stack, 1)
        x[off] = min(state.pot / stack_norm, 2.0)
        off += 1
        x[off] = state.stacks[player] / stack_norm
        off += 1
        x[off] = state.stacks[1 - player] / stack_norm
        off += 1

        # Contributions (this street, normalised by big bet).
        big_bet = max(self.game.big_bet, 1)
        x[off] = state.contributions[player] / big_bet
        off += 1
        x[off] = state.contributions[1 - player] / big_bet
        off += 1

        # Raises this street / cap.
        x[off] = state.raises_this_street / max(self.game.max_raises_per_round, 1)
        off += 1

        # Position (is dealer / player 0).
        x[off] = 1.0 if player == 0 else 0.0
        off += 1

        # Legal action mask.
        for a in self.game.legal_actions(state):
            x[off + int(a)] = 1.0
        off += 3

        assert off == INFOSET_DIM_HULHE
        return x


def _street_from_state(state: HULHEState) -> Street:
    return state.street


@dataclass
class KuhnInfosetEncoder:
    def encode(self, state: KuhnState, player: int) -> np.ndarray:
        x = np.zeros(INFOSET_DIM_KUHN, dtype=np.float32)
        card = state.cards[player]
        x[card - 1] = 1.0
        x[3 + player] = 1.0
        # Action history padded to 4 slots: -1 = empty, 0 = PASS, 1 = BET.
        for i, a in enumerate(state.history[:4]):
            x[5 + i] = float(int(a) + 1) / 2.0
        return x


def legal_action_mask_hulhe(game: HULHE, state: HULHEState) -> np.ndarray:
    mask = np.zeros(3, dtype=np.float32)
    for a in game.legal_actions(state):
        mask[int(a)] = 1.0
    return mask


def legal_action_mask_kuhn(state: KuhnState) -> np.ndarray:
    if state.terminal:
        return np.zeros(2, dtype=np.float32)
    return np.array([1.0, 1.0], dtype=np.float32)
