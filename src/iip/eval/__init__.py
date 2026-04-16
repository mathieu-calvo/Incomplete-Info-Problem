"""Hand evaluation: treys-backed 7-card ranker + Monte-Carlo equity + preflop bucket helpers."""

from iip.eval.equity import monte_carlo_equity, preflop_bucket_id, starting_hand_bucket
from iip.eval.ranker import (
    compare_seven_card_hands,
    evaluate_five_card_fallback,
    rank_seven_card,
)

__all__ = [
    "compare_seven_card_hands",
    "evaluate_five_card_fallback",
    "monte_carlo_equity",
    "preflop_bucket_id",
    "rank_seven_card",
    "starting_hand_bucket",
]
