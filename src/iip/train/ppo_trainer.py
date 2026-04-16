"""PPO self-play trainer for HULHE.

Key choices:
- Rollouts are generated one hand at a time. The learner plays seat 0 in half the hands and
  seat 1 in the other half; the opponent is sampled from the league.
- Rewards are per-hand net chip change in milli-big-blinds (mbb/hand). We assign the terminal
  reward to the last transition of the learner.
- Advantage estimation uses GAE with gamma close to 1 (hands are short so gamma matters less).
- Standard PPO clip + entropy + value loss. Illegal actions are masked *before* softmax.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from iip.agents.base import Agent
from iip.agents.ppo import PPOAgent, PPOConfig
from iip.train.game_adapter import HULHEAdapter

log = logging.getLogger(__name__)


@dataclass
class Transition:
    features: np.ndarray
    legal_mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False


@dataclass
class RolloutBatch:
    features: np.ndarray
    legal_masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class PPOTrainer:
    def __init__(
        self,
        adapter: HULHEAdapter,
        agent: PPOAgent,
        config: PPOConfig,
        opponents: list[Agent],
        seed: int = 0,
    ) -> None:
        self.adapter = adapter
        self.agent = agent
        self.cfg = config
        self.opponents = opponents
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.optimizer = optim.Adam(
            list(agent.actor.parameters()) + list(agent.critic.parameters()),
            lr=config.learning_rate,
        )

    def train_round(self, n_hands: int) -> dict[str, float]:
        batch = self._collect_rollout(n_hands)
        stats = self._update(batch)
        return stats

    # ---------- rollout ----------

    def _collect_rollout(self, n_hands: int) -> RolloutBatch:
        transitions: list[Transition] = []
        for h in range(n_hands):
            opp = self.rng.choice(self.opponents) if self.opponents else self.agent
            learner_seat = 0 if h % 2 == 0 else 1
            rng = random.Random(self.rng.random())
            state = self.adapter.new_hand(rng=rng)
            learner_transitions: list[Transition] = []
            while not self.adapter.is_terminal(state):
                cur = self.adapter.to_act(state)
                mask = self.adapter.legal_mask(state)
                feat = self.adapter.encode(state, cur)
                if cur == learner_seat:
                    action, log_prob, value = self._sample_action(feat, mask)
                    learner_transitions.append(
                        Transition(
                            features=feat,
                            legal_mask=mask,
                            action=action,
                            log_prob=log_prob,
                            value=value,
                        )
                    )
                else:
                    dist = opp.policy(self.adapter.game, state, cur)
                    actions = list(dist.keys())
                    probs = np.array(list(dist.values()), dtype=np.float64)
                    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
                    action = int(actions[np.random.choice(len(actions), p=probs)])
                self.adapter.step(state, action, rng=random.Random(rng.random()))
            # attribute terminal reward to last learner transition
            if learner_transitions:
                payoffs = self.adapter.payoffs(state)
                reward_mbb = payoffs[learner_seat] * 1000.0 / max(self.adapter.game.big_blind, 1)
                learner_transitions[-1].reward = reward_mbb
                learner_transitions[-1].done = True
            transitions.extend(learner_transitions)

        return self._compute_gae(transitions)

    def _sample_action(self, feat: np.ndarray, mask: np.ndarray) -> tuple[int, float, float]:
        x = torch.from_numpy(feat).float().unsqueeze(0).to(self.agent.device)
        with torch.no_grad():
            logits = self.agent.actor(x).squeeze(0)
            value = float(self.agent.critic(x).squeeze(0).item())
        mask_t = torch.from_numpy(mask).float().to(self.agent.device)
        masked = logits.masked_fill(mask_t == 0, -1e9)
        probs = torch.softmax(masked, dim=0)
        dist = torch.distributions.Categorical(probs=probs)
        a = int(dist.sample().item())
        log_prob = float(torch.log(probs[a] + 1e-12).item())
        return a, log_prob, value

    def _compute_gae(self, transitions: list[Transition]) -> RolloutBatch:
        if not transitions:
            empty = np.zeros((0, self.adapter.feature_dim), dtype=np.float32)
            return RolloutBatch(empty, np.zeros((0, self.adapter.num_actions), dtype=np.float32),
                                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32),
                                np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        values = np.array([t.value for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for i in reversed(range(len(transitions))):
            next_v = 0.0 if (i == len(transitions) - 1 or dones[i]) else values[i + 1]
            delta = rewards[i] + self.cfg.gamma * next_v * (1 - dones[i]) - values[i]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones[i]) * last_gae
            advantages[i] = last_gae
        returns = advantages + values
        # Normalise advantages.
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return RolloutBatch(
            features=np.stack([t.features for t in transitions]),
            legal_masks=np.stack([t.legal_mask for t in transitions]),
            actions=np.array([t.action for t in transitions], dtype=np.int64),
            log_probs=np.array([t.log_prob for t in transitions], dtype=np.float32),
            advantages=advantages.astype(np.float32),
            returns=returns.astype(np.float32),
        )

    # ---------- update ----------

    def _update(self, batch: RolloutBatch) -> dict[str, float]:
        if len(batch.actions) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        self.agent.actor.train()
        self.agent.critic.train()

        device = self.agent.device
        feats = torch.from_numpy(batch.features).to(device)
        masks = torch.from_numpy(batch.legal_masks).to(device)
        actions = torch.from_numpy(batch.actions).to(device)
        old_log_probs = torch.from_numpy(batch.log_probs).to(device)
        advantages = torch.from_numpy(batch.advantages).to(device)
        returns = torch.from_numpy(batch.returns).to(device)

        stats: dict[str, float] = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        idxs = np.arange(len(actions))
        for _ in range(self.cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), self.cfg.batch_size):
                mb = idxs[start : start + self.cfg.batch_size]
                if len(mb) == 0:
                    continue
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)
                mb_feats = feats[mb_t]
                mb_masks = masks[mb_t]
                mb_actions = actions[mb_t]
                mb_old_log_probs = old_log_probs[mb_t]
                mb_advantages = advantages[mb_t]
                mb_returns = returns[mb_t]

                logits = self.agent.actor(mb_feats)
                masked = logits.masked_fill(mb_masks == 0, -1e9)
                log_probs_all = torch.log_softmax(masked, dim=1)
                log_probs = log_probs_all.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                entropy = -(torch.softmax(masked, dim=1) * log_probs_all).sum(dim=1).mean()

                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.agent.critic(mb_feats).squeeze(1)
                value_loss = ((values - mb_returns) ** 2).mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.agent.actor.parameters()) + list(self.agent.critic.parameters()),
                    self.cfg.grad_clip,
                )
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())

        self.agent.actor.eval()
        self.agent.critic.eval()
        n = self.cfg.epochs * max(1, (len(actions) + self.cfg.batch_size - 1) // self.cfg.batch_size)
        return {k: v / n for k, v in stats.items()}
