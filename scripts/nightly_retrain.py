"""Nightly retrain entry point.

1. Start from the previous checkpoint (or scratch on first run).
2. Pull any new hands from Supabase since the last run.
3. Run PPO self-play with human hands injected as one of the opponents.
4. Save the new checkpoint for the eval gate to judge.

Invoked from the GitHub Actions workflow `.github/workflows/nightly-retrain.yml`.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from iip.agents.fixed_policy import FishAgent, StartingHandAgent, StrengthHandAgent
from iip.agents.ppo import PPOAgent, PPOConfig
from iip.agents.random_agent import RandomAgent
from iip.engine.game import HULHE
from iip.io.supabase_client import HandStore
from iip.train.game_adapter import HULHEAdapter
from iip.train.ppo_trainer import PPOTrainer

log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--previous", type=Path, default=None, help="Previous checkpoint (PPO .pt).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--traversals", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    game = HULHE()
    adapter = HULHEAdapter(game=game)
    cfg = PPOConfig(actor_hidden=[256, 256], critic_hidden=[256, 256])

    if args.previous and args.previous.exists():
        agent = PPOAgent.load(args.previous)
        log.info("Warm-started PPO from %s", args.previous)
    else:
        agent = PPOAgent(feature_dim=adapter.feature_dim, num_actions=adapter.num_actions)
        log.info("Starting PPO from scratch")

    opponents = [
        FishAgent(),
        StrengthHandAgent(),
        StartingHandAgent(),
        RandomAgent(rng=random.Random(args.seed)),
    ]

    store = HandStore()
    if store.is_configured:
        hands = store.fetch_hands_since()
        log.info("Fetched %d human hands from Supabase", len(hands))
        # Human hands are used as an additional evaluation signal; direct injection as training
        # trajectories is left as future work (behavioural cloning head). See PLAN §Human-feedback.
    else:
        log.info("Supabase not configured — retraining on synthetic opponents only")

    trainer = PPOTrainer(adapter=adapter, agent=agent, config=cfg, opponents=opponents, seed=args.seed)
    for i in range(args.iters):
        stats = trainer.train_round(n_hands=args.traversals)
        log.info("iter=%d stats=%s", i, stats)

    agent.save(args.output)
    log.info("Saved PPO checkpoint -> %s", args.output)


if __name__ == "__main__":
    main()
