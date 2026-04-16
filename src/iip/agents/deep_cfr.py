"""Deep CFR agent.

Holds two neural nets per learning player:
- Advantage net `V_p(I)` — predicts counterfactual regret for each action at infoset I.
- Strategy net `π̄(I)`   — predicts the *average* strategy (the approximate Nash).

At inference time we serve `π̄` (with legal-action masking). Regret-matching is used
only during training to generate sampled strategies for traversals.

This module only defines the agent + regret-matching helpers. The outer training loop
(traversals, reservoir buffers, optimiser steps) lives in `iip.train.deepcfr_trainer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from iip.agents.networks import MLP
from iip.engine.game import HULHE, ActionType, HULHEState


@dataclass
class DeepCFRConfig:
    hidden_sizes: list[int]
    dropout: float = 0.0
    learning_rate: float = 3e-4
    grad_clip: float = 5.0


def regret_matching(advantages: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Turn action advantages into a probability distribution over *legal* actions.

    r_plus = max(0, a) on legal actions only; if all zero, play uniformly over legal actions.
    """
    a = advantages.copy()
    a[legal_mask == 0] = 0.0
    a = np.maximum(a, 0.0)
    s = a.sum()
    if s > 0:
        return a / s
    legal = legal_mask.astype(np.float32)
    return legal / legal.sum()


class DeepCFRAgent:
    """Inference-side wrapper around the trained strategy net.

    For training use, see `iip.train.deepcfr_trainer.DeepCFRTrainer`.
    """

    name = "DeepCFR"

    def __init__(
        self,
        feature_dim: int,
        num_actions: int,
        hidden_sizes: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.strategy_net = MLP(feature_dim, num_actions, hidden_sizes or [256, 256]).to(self.device)
        self.strategy_net.eval()

    @torch.no_grad()
    def strategy(self, features: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        logits = self.strategy_net(x).squeeze(0).cpu().numpy()
        masked = np.where(legal_mask > 0, logits, -1e9)
        m = masked.max()
        e = np.exp(masked - m)
        e = e * legal_mask
        s = e.sum()
        if s == 0:
            return legal_mask / legal_mask.sum()
        return e / s

    # ---------- HULHE convenience wrappers for the Agent protocol ----------

    def policy(self, game: HULHE, state: HULHEState, player: int) -> dict[ActionType, float]:
        from iip.features.infoset import HULHEInfosetEncoder, legal_action_mask_hulhe

        enc = HULHEInfosetEncoder(game=game)
        x = enc.encode(state, player)
        mask = legal_action_mask_hulhe(game, state)
        probs = self.strategy(x, mask)
        return {ActionType(i): float(probs[i]) for i in range(len(probs)) if mask[i] > 0}

    def act(self, game: HULHE, state: HULHEState, player: int) -> ActionType:
        dist = self.policy(game, state, player)
        actions = list(dist.keys())
        probs = np.array(list(dist.values()), dtype=np.float64)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
        idx = int(np.random.choice(len(actions), p=probs))
        return actions[idx]

    def observe(self, game: HULHE, state: HULHEState, player: int) -> None:
        pass

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "strategy_state_dict": self.strategy_net.state_dict(),
                "feature_dim": self.feature_dim,
                "num_actions": self.num_actions,
                "hidden_sizes": [m.out_features for m in self.strategy_net.net if hasattr(m, "out_features")][:-1],
            },
            str(p),
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> DeepCFRAgent:
        blob = torch.load(str(path), map_location=device)
        agent = cls(
            feature_dim=blob["feature_dim"],
            num_actions=blob["num_actions"],
            hidden_sizes=blob["hidden_sizes"],
            device=device,
        )
        agent.strategy_net.load_state_dict(blob["strategy_state_dict"])
        agent.strategy_net.eval()
        return agent
