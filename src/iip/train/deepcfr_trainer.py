"""Deep CFR trainer — external sampling MCCFR driving neural advantage + strategy nets.

Outline (Brown et al. 2019 Alg. 1):
    for iter t = 1..T:
        for traverser p in (0, 1):
            for k = 1..K traversals:
                traverse(root, p, t, reach_p=1, reach_opp=1)
            train advantage_net[p] on advantage_memory[p] from scratch
        # every few iterations, optionally train the strategy net
    train strategy_net on strategy_memory

`traverse`:
    if terminal: return payoff_p
    if chance: sample chance node, recurse
    current_player = state.to_act
    if current_player == traverser:
        sigma = regret_match(advantage_net[p](I), legal_mask)
        v = sum_a sigma[a] * traverse_child(a)
        # compute counterfactual values
        for a in legal:
            v_a = traverse_child(a)
        advantages = v_a - v  (over legal actions)
        store (I, t, advantages)
        return v
    else:
        sigma = regret_match(advantage_net[other](I), legal_mask)
        store (I, t, sigma) in strategy_memory[other]
        a ~ sigma
        return traverse_child(a)

We implement chance by resetting the game to a fresh hand per traversal — i.e. we sample
a single chance event (card deal) per traversal root. This is external sampling.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from iip.agents.deep_cfr import DeepCFRAgent, regret_matching
from iip.agents.networks import MLP
from iip.train.game_adapter import GameAdapter
from iip.train.replay import ReservoirBuffer

log = logging.getLogger(__name__)


@dataclass
class DeepCFRTrainerConfig:
    hidden_sizes_advantage: list[int] = field(default_factory=lambda: [256, 256])
    hidden_sizes_strategy: list[int] = field(default_factory=lambda: [256, 256])
    advantage_buffer_size: int = 1_000_000
    strategy_buffer_size: int = 4_000_000
    traversals_per_iteration: int = 20_000
    advantage_train_steps: int = 4_000
    strategy_train_steps: int = 4_000
    batch_size: int = 1024
    learning_rate: float = 3e-4
    grad_clip: float = 5.0
    device: str = "cpu"


class DeepCFRTrainer:
    def __init__(self, adapter: GameAdapter, config: DeepCFRTrainerConfig, seed: int = 0) -> None:
        self.adapter = adapter
        self.cfg = config
        self.device = torch.device(config.device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.adv_nets = [
            MLP(adapter.feature_dim, adapter.num_actions, config.hidden_sizes_advantage).to(self.device)
            for _ in range(2)
        ]
        self.strategy_net = MLP(adapter.feature_dim, adapter.num_actions, config.hidden_sizes_strategy).to(self.device)

        self.adv_buffers = [
            ReservoirBuffer(capacity=config.advantage_buffer_size, rng=random.Random(seed + 1))
            for _ in range(2)
        ]
        self.strategy_buffer = ReservoirBuffer(capacity=config.strategy_buffer_size, rng=random.Random(seed + 2))

    # ---------- public ----------

    def iterate(self, n_iters: int = 1) -> None:
        for t in range(1, n_iters + 1):
            for traverser in (0, 1):
                for _ in range(self.cfg.traversals_per_iteration):
                    rng = random.Random(self.rng.random())
                    root = self.adapter.new_hand(rng=rng)
                    self._traverse(root, traverser, t, rng)
                self._train_advantage(traverser)
            log.info("Deep CFR iter %d done — adv buffers: %s, strat buffer: %d", t,
                     [len(b) for b in self.adv_buffers], len(self.strategy_buffer))

    def finalize_strategy(self) -> DeepCFRAgent:
        self._train_strategy()
        agent = DeepCFRAgent(
            feature_dim=self.adapter.feature_dim,
            num_actions=self.adapter.num_actions,
            hidden_sizes=self.cfg.hidden_sizes_strategy,
            device=self.cfg.device,
        )
        agent.strategy_net.load_state_dict(self.strategy_net.state_dict())
        agent.strategy_net.eval()
        return agent

    # ---------- traversal ----------

    def _traverse(self, state, traverser: int, iter_t: int, rng: random.Random) -> float:
        if self.adapter.is_terminal(state):
            return float(self.adapter.payoffs(state)[traverser])
        cur = self.adapter.to_act(state)
        mask = self.adapter.legal_mask(state)
        feat = self.adapter.encode(state, cur)

        with torch.no_grad():
            x = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)
            advantages = self.adv_nets[cur](x).squeeze(0).cpu().numpy()
        sigma = regret_matching(advantages, mask)

        if cur == traverser:
            legal_idx = [i for i, m in enumerate(mask) if m > 0]
            child_values = np.zeros(self.adapter.num_actions, dtype=np.float32)
            for a in legal_idx:
                child_state = copy.deepcopy(state)
                self.adapter.step(child_state, a, rng=random.Random(rng.random()))
                child_values[a] = self._traverse(child_state, traverser, iter_t, rng)
            v = float(np.sum(sigma * child_values))
            adv = (child_values - v) * mask
            self.adv_buffers[traverser].add((feat.copy(), iter_t, adv.copy(), mask.copy()))
            return v
        else:
            # store strategy sample (linear CFR weighting by iter_t)
            self.strategy_buffer.add((feat.copy(), iter_t, sigma.copy(), mask.copy()))
            # sample action according to sigma over legal
            legal_idx = [i for i, m in enumerate(mask) if m > 0]
            probs = np.array([sigma[i] for i in legal_idx], dtype=np.float64)
            probs = np.ones_like(probs) / len(probs) if probs.sum() == 0 else probs / probs.sum()
            a = int(np.random.choice(legal_idx, p=probs))
            child_state = copy.deepcopy(state)
            self.adapter.step(child_state, a, rng=random.Random(rng.random()))
            return self._traverse(child_state, traverser, iter_t, rng)

    # ---------- training ----------

    def _train_advantage(self, player: int) -> None:
        buf = self.adv_buffers[player]
        if len(buf) < self.cfg.batch_size:
            return
        # Re-initialise the net from scratch each iteration (Deep CFR recipe).
        net = MLP(self.adapter.feature_dim, self.adapter.num_actions, self.cfg.hidden_sizes_advantage).to(self.device)
        opt = optim.Adam(net.parameters(), lr=self.cfg.learning_rate)
        loss_fn = nn.SmoothL1Loss(reduction="none")

        for _ in range(self.cfg.advantage_train_steps):
            batch = buf.sample(self.cfg.batch_size)
            if not batch:
                break
            feats = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(self.device)
            iters = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            targets = torch.from_numpy(np.stack([b[2] for b in batch])).float().to(self.device)
            masks = torch.from_numpy(np.stack([b[3] for b in batch])).float().to(self.device)
            preds = net(feats)
            losses = loss_fn(preds, targets) * masks
            # Linear-CFR weighting: weight each sample by its iter_t.
            loss = (losses.mean(dim=1) * iters.squeeze(1)).mean() / max(iters.mean().item(), 1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
            opt.step()
        self.adv_nets[player] = net

    def _train_strategy(self) -> None:
        buf = self.strategy_buffer
        if len(buf) < self.cfg.batch_size:
            log.warning("Not enough strategy samples (%d) — strategy net will be under-trained.", len(buf))
        opt = optim.Adam(self.strategy_net.parameters(), lr=self.cfg.learning_rate)

        for _ in range(self.cfg.strategy_train_steps):
            batch = buf.sample(self.cfg.batch_size)
            if not batch:
                break
            feats = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(self.device)
            iters = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            targets = torch.from_numpy(np.stack([b[2] for b in batch])).float().to(self.device)
            masks = torch.from_numpy(np.stack([b[3] for b in batch])).float().to(self.device)
            preds = self.strategy_net(feats)
            # Mask logits for illegal actions to -inf then softmax cross-entropy against the target prob.
            masked_logits = preds.masked_fill(masks == 0, -1e9)
            log_probs = torch.log_softmax(masked_logits, dim=1)
            losses = -(targets * log_probs).sum(dim=1)
            loss = (losses * iters.squeeze(1)).mean() / max(iters.mean().item(), 1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.strategy_net.parameters(), self.cfg.grad_clip)
            opt.step()
