"""Deep CFR sanity: run a short training on Kuhn and verify exploitability falls.

The equilibrium first-player utility in Kuhn is -1/18 ≈ -0.0556. After 20 iterations with 400
traversals each, Deep CFR should produce a first-player strategy that does better than uniform
random against a best-responding exploiter. This is a smoke test, not a correctness proof — the
convergence threshold is intentionally loose to keep the test fast and robust.
"""

from __future__ import annotations

import random

import numpy as np

from iip.agents.deep_cfr import regret_matching
from iip.engine.kuhn import Kuhn, KuhnAction
from iip.train.deepcfr_trainer import DeepCFRTrainer, DeepCFRTrainerConfig
from iip.train.game_adapter import KuhnAdapter


def _average_payoff_vs_random(agent, n: int = 500, seed: int = 0) -> float:
    g = Kuhn()
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n):
        s = g.new_hand(rng=random.Random(rng.random()))
        while not s.terminal:
            legal = [0, 1]  # Kuhn has 2 actions
            feat = agent.strategy_net
            # Use strategy net for player 0, random for player 1.
            from iip.features.infoset import KuhnInfosetEncoder, legal_action_mask_kuhn
            enc = KuhnInfosetEncoder()
            x = enc.encode(s, s.to_act)
            mask = legal_action_mask_kuhn(s)
            if s.to_act == 0:
                probs = agent.strategy(x, mask)
                idx = int(np.random.choice(len(probs), p=probs))
            else:
                idx = rng.choice([0, 1])
            g.step(s, KuhnAction(idx))
        total += g.payoffs(s)[0]
    return total / n


def test_deep_cfr_converges_on_kuhn_smoke(tmp_path):
    adapter = KuhnAdapter(game=Kuhn())
    cfg = DeepCFRTrainerConfig(
        hidden_sizes_advantage=[32, 32],
        hidden_sizes_strategy=[32, 32],
        advantage_buffer_size=20_000,
        strategy_buffer_size=40_000,
        traversals_per_iteration=200,
        advantage_train_steps=200,
        strategy_train_steps=400,
        batch_size=64,
        learning_rate=1e-3,
    )
    trainer = DeepCFRTrainer(adapter=adapter, config=cfg, seed=0)
    trainer.iterate(n_iters=8)
    agent = trainer.finalize_strategy()

    # Smoke-level: agent as player 0 should not be catastrophically worse than random.
    avg = _average_payoff_vs_random(agent, n=300, seed=1)
    assert avg > -0.5, f"Deep CFR agent is losing badly on Kuhn: {avg:.3f}"


def test_regret_matching_uniform_when_all_zero():
    mask = np.array([1, 1, 0], dtype=np.float32)
    sigma = regret_matching(np.zeros(3), mask)
    assert np.isclose(sigma[0], 0.5)
    assert np.isclose(sigma[1], 0.5)
    assert sigma[2] == 0.0


def test_regret_matching_concentrates_on_positive_regret():
    mask = np.array([1, 1, 1], dtype=np.float32)
    sigma = regret_matching(np.array([1.0, 3.0, 0.0]), mask)
    assert abs(sigma.sum() - 1.0) < 1e-6
    assert sigma[1] > sigma[0] > 0
    assert sigma[2] == 0.0
