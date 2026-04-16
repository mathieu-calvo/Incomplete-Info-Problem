"""Evaluation gate + report generator.

Runs head-to-head matches vs baselines and (if a previous checkpoint is supplied) vs the
previous bot. Also computes the LBR exploitability proxy. Writes `reports/latest.json` and
exits non-zero if the new checkpoint regresses against the previous one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from iip.agents.deep_cfr import DeepCFRAgent
from iip.agents.fixed_policy import FishAgent, StartingHandAgent, StrengthHandAgent
from iip.agents.random_agent import RandomAgent
from iip.agents.ppo import PPOAgent
from iip.engine.game import HULHE
from iip.metrics.exploitability import local_best_response
from iip.metrics.mbb import head_to_head_mbb

log = logging.getLogger(__name__)


def _load_any(path: Path):
    """Try PPO first, then Deep CFR. Matches on file layout."""
    try:
        return PPOAgent.load(path)
    except Exception:
        return DeepCFRAgent.load(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--previous", type=Path, default=None)
    p.add_argument("--hands", type=int, default=20_000)
    p.add_argument("--lbr-hands", type=int, default=500)
    p.add_argument("--output", type=Path, default=Path("reports/latest.json"))
    p.add_argument("--regression-margin-mbb", type=float, default=-30.0,
                   help="New must not be worse than previous by more than this (in mbb/h, negative).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    game = HULHE()
    hero = _load_any(args.checkpoint)

    baselines = {
        "fish": FishAgent(),
        "strength": StrengthHandAgent(),
        "starting": StartingHandAgent(),
        "random": RandomAgent(),
    }
    head_to_head = []
    for name, opp in baselines.items():
        r = head_to_head_mbb(game, hero, opp, n_hands=args.hands, seed=0)
        head_to_head.append({"opponent": name, **asdict(r)})
        log.info(str(r))

    lbr = local_best_response(game, hero, n_hands=args.lbr_hands, seed=0)
    log.info("LBR exploitability proxy: %.2f mbb/h", lbr)

    regressed = False
    prev_delta = None
    if args.previous and args.previous.exists():
        prev = _load_any(args.previous)
        r = head_to_head_mbb(game, hero, prev, n_hands=args.hands, seed=1)
        prev_delta = r.mbbph
        log.info("Head-to-head vs previous: %s", r)
        if r.mbbph < args.regression_margin_mbb:
            regressed = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.checkpoint),
        "previous": str(args.previous) if args.previous else None,
        "head_to_head": head_to_head,
        "vs_previous_mbbph": prev_delta,
        "lbr_mbbph": lbr,
        "regressed": regressed,
    }
    args.output.write_text(json.dumps(report, indent=2))
    log.info("Wrote report -> %s", args.output)

    if regressed:
        log.error("Eval gate failed: regression vs previous checkpoint.")
        sys.exit(1)


if __name__ == "__main__":
    main()
