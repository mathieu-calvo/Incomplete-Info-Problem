
import logging

from collections import Counter
from itertools import combinations

from ..globals import STRAIGHTS, HUMAN_READABLE_RANKINGS

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


def evaluate_hand_rank(hand):

    # organize information from hand
    suits_counter = Counter(map(lambda x: x.suit, hand))
    ranks_counter = Counter(map(lambda x: x.rank, hand))

    # infer information about force of hand
    is_flush = (len(suits_counter.keys()) == 1)
    is_straight = (set([card.rank for card in hand]) in STRAIGHTS)
    if is_straight:
        is_wheel = (set([card.rank for card in hand]) == {14, 2, 3, 4, 5})
    else:
        is_wheel = False

    # classify hand
    if is_flush and is_straight:
        # tie breaker is the card with highest rank, accounting for the wheel
        if is_wheel:
            tiebreaker = [5]
        else:
            tiebreaker = [max(ranks_counter.keys())]
        return 9, tiebreaker  # , "Straight flush"
    elif 4 in ranks_counter.values():
        # tie breaker is the rank that makes quads + kicker
        tiebreaker = sorted(ranks_counter, key=ranks_counter.get,
                            reverse=True)
        return 8, tiebreaker  # , "Four of a kind"
    elif {2, 3} == set(ranks_counter.values()):
        # tie breaker is the rank of trips + rank of pair
        tiebreaker = sorted(ranks_counter, key=ranks_counter.get,
                            reverse=True)
        return 7, tiebreaker  # , "Full house"
    elif is_flush:
        # tie breaker is ranks of cards, in reverse order
        tiebreaker = sorted(ranks_counter, reverse=True)
        return 6, tiebreaker  # , "Flush"
    elif is_straight:
        # tie breaker is the card with highest rank, accounting for the wheel
        if is_wheel:
            tiebreaker = [5]
        else:
            tiebreaker = [max(ranks_counter.keys())]
        return 5, tiebreaker  # , "Straight"
    elif [1, 1, 3] == sorted(list(ranks_counter.values())):
        # tie breaker is the rank of trips + kickers
        tiebreaker = [item[0] for item in
                      sorted(ranks_counter.most_common(),
                             key=lambda x: (-x[1], -x[0]))]
        return 4, tiebreaker  # , "Three of a kind"
    elif [1, 2, 2] == sorted(list(ranks_counter.values())):
        # tie breaker is the rank of top pair + second pair + kicker
        tiebreaker = [item[0] for item in
                      sorted(ranks_counter.most_common(),
                             key=lambda x: (-x[1], -x[0]))]
        return 3, tiebreaker  # , "Two pairs"
    elif [1, 1, 1, 2] == sorted(list(ranks_counter.values())):
        # tie breaker is the rank of pair + kickers
        tiebreaker = [item[0] for item in
                      sorted(ranks_counter.most_common(),
                             key=lambda x: (-x[1], -x[0]))]
        return 2, tiebreaker  # , "One pair"
    else:
        # tie breaker is the rank of cards, in reverse order
        tiebreaker = sorted(ranks_counter, reverse=True)
        return 1, tiebreaker  # , "High card"


def compare_two_hands(hand1, hand2):
    if hand1.best_rank > hand2.best_rank:
        return "hand1"
    elif hand1.best_rank < hand2.best_rank:
        return "hand2"
    else:
        i = 0
        while i < len(hand1.best_tiebreaker):
            if hand1.best_tiebreaker[i] > hand2.best_tiebreaker[i]:
                return "hand1"
            elif hand1.best_tiebreaker[i] < hand2.best_tiebreaker[i]:
                return "hand2"
            else:
                i += 1
        return "draw"


def tie_breaking(hands, tiebreakers):
    logging.debug("{} hands going to tiebreak".format(len(hands)))
    best_hands = hands
    i = 0
    while i < len(tiebreakers[0]) and len(best_hands) > 1:
        # find highest i card
        highest_card = max([tiebreaker[i] for tiebreaker in tiebreakers])
        # keep hands with highest tiebreakers
        best_idx = [idx for idx, tiebreaker in enumerate(tiebreakers)
                    if tiebreaker[i] == highest_card]
        # replace variables
        best_hands = [hands[idx] for idx in best_idx]
        hands = [hands[idx] for idx in best_idx]
        tiebreakers = [tiebreakers[idx] for idx in best_idx]
        i += 1
    # if more than one hand with equivalent strength, keep first arbitrarily
    if len(best_hands) > 1:
        return [best_hands[0]], [tiebreakers[0]]
    else:
        return best_hands, tiebreakers


class Hand:

    def __init__(self, private_cards):
        self.private_cards = private_cards
        self.public_cards = []
        self.best_rank = []
        self.best_combination = []
        self.best_tiebreaker = []

    def add_public_cards(self, public_cards):
        self.public_cards = self.public_cards + public_cards

    def update_best_combination(self):

        assert self.public_cards  # check list is not empty

        logging.debug('Evaluating best combination')
        # number of combinations
        hands = list(combinations(self.private_cards + self.public_cards, 5))

        # evaluate possible hands
        hand_evaluations = [evaluate_hand_rank(hand) for hand in hands]
        hand_rankings = [hand_evaluation[0]
                         for hand_evaluation in hand_evaluations]

        # select hands with best rankings
        best_rank = max(hand_rankings)
        best_hands_index = [idx for idx, rank in enumerate(hand_rankings)
                            if rank == best_rank]
        best_hands = [hands[idx] for idx in best_hands_index]
        best_hands_tiebreakers = [hand_evaluations[idx][1]
                                  for idx in best_hands_index]

        # tiebreak only if necessary
        if len(best_hands) == 1:
            self.best_rank = best_rank
            self.best_combination = best_hands
            self.best_tiebreaker = best_hands_tiebreakers
        else:
            logging.debug('Tiebreaking best combination')
            best_hand, best_tiebreaker = \
                tie_breaking(best_hands, best_hands_tiebreakers)
            self.best_rank = best_rank
            self.best_combination = best_hand
            self.best_tiebreaker = best_tiebreaker

    def pretty_str_best_combination(self):
        return [str(card.pretty_rank) + card.pretty_suit
                for card in self.best_combination[0]]

    def human_readable_rank(self):
        return HUMAN_READABLE_RANKINGS[self.best_rank]

    def __repr__(self):
        return str([str(card.pretty_rank) + card.pretty_suit
                    for card in self.private_cards + self.public_cards])
