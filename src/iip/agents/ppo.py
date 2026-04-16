"""PPO actor-critic for self-play fine-tuning.

Actor outputs logits over the 3 HULHE actions (FOLD, CHECK_CALL, BET_RAISE); we mask illegal
actions before softmax. Critic outputs a scalar value. Both share the feature encoder
(stateless numpy `HULHEInfosetEncoder`).

The agent is *inference-only* here — rollout and update live in `iip.train.ppo_trainer`.
A PPO actor can be warm-started from a Deep CFR strategy net via `warm_start_from_deepcfr`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from iip.agents.networks import MLP
from iip.engine.game import HULHE, ActionType, HULHEState


@dataclass
class PPOConfig:
    actor_hidden: list[int]
    critic_hidden: list[int]
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    grad_clip: float = 1.0
    epochs: int = 4
    batch_size: int = 512
    device: str = "cpu"


class PPOAgent:
    name = "PPO"

    def __init__(
        self,
        feature_dim: int,
        num_actions: int,
        actor_hidden: list[int] | None = None,
        critic_hidden: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.actor = MLP(feature_dim, num_actions, actor_hidden or [256, 256]).to(self.device)
        self.critic = MLP(feature_dim, 1, critic_hidden or [256, 256]).to(self.device)
        self.actor.eval()
        self.critic.eval()

    def warm_start_from_deepcfr(self, dcfr_state_dict: dict) -> None:
        """Copy the Deep CFR strategy-net weights into the PPO actor (same shape)."""
        self.actor.load_state_dict(dcfr_state_dict)

    @torch.no_grad()
    def action_distribution(self, features: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        logits = self.actor(x).squeeze(0).cpu().numpy()
        masked = np.where(legal_mask > 0, logits, -1e9)
        m = masked.max()
        e = np.exp(masked - m)
        e = e * legal_mask
        s = e.sum()
        if s == 0:
            return legal_mask / legal_mask.sum()
        return e / s

    # ---------- Agent protocol ----------

    def policy(self, game: HULHE, state: HULHEState, player: int) -> dict[ActionType, float]:
        from iip.features.infoset import HULHEInfosetEncoder, legal_action_mask_hulhe

        enc = HULHEInfosetEncoder(game=game)
        x = enc.encode(state, player)
        mask = legal_action_mask_hulhe(game, state)
        probs = self.action_distribution(x, mask)
        return {ActionType(i): float(probs[i]) for i in range(len(probs)) if mask[i] > 0}

    def act(self, game: HULHE, state: HULHEState, player: int) -> ActionType:
        dist = self.policy(game, state, player)
        actions = list(dist.keys())
        probs = np.array(list(dist.values()), dtype=np.float64)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
        return actions[int(np.random.choice(len(actions), p=probs))]

    def observe(self, game: HULHE, state: HULHEState, player: int) -> None:
        pass

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "feature_dim": self.feature_dim,
                "num_actions": self.num_actions,
            },
            str(p),
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> PPOAgent:
        blob = torch.load(str(path), map_location=device)
        agent = cls(
            feature_dim=blob["feature_dim"],
            num_actions=blob["num_actions"],
            device=device,
        )
        agent.actor.load_state_dict(blob["actor_state_dict"])
        agent.critic.load_state_dict(blob["critic_state_dict"])
        agent.actor.eval()
        agent.critic.eval()
        return agent
