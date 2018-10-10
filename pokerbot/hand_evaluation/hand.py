
import logging

from collections import Counter
from itertools import combinations

from ..globals import STRAIGHTS, HUMAN_READABLE_RANKINGS

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


def evaluate_hand_ranking(hand):
    """
    Evaluate ranking of poker hand on a scale from 1 to 9 and provide
    information useful for tiebreaking hands of the same ranking

    Args:
        hand (tuple): tuple of 5 Card objects

    Returns:
        (int):  ranking of poker hand on a scale from 1 to 9
        (list): list of integers representing, in decreasing order of
        importance, ranking of cards relevant for tiebreaking. Logic
        is dependent on the ranking.
    """

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
    """
    Compares two class.Hand objects based on their resp. best combination
    and tiebreaker

    Args:
        hand1 (class.Hand): class object Hand
        hand2 (class.Hand): class object Hand

    Returns:
        (str): describing outcome: ["hand1","hand2","draw"]
    """
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
    """
    Compares combinations with same ranking based on their resp. tiebreaker

    Args:
        hands (list): list containing tuples of 5 Card objects
        tiebreakers (list): list containing lists of integers

    Returns:
        best_hands (list): list containing one tuple of 5 Card objects
        tiebreakers (list): list containing one list of integers
    """
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


class Hand(object):
    """
    Hand object for a given player that integrate private and communal cards
    when available. Include methods to assess strength of the hand.

    Attributes:
        private_cards (list): list of two Card objects given pre flop
        public_cards (list): list of Card objects given post flop
        best_rank (list): rank of best combination at the moment of evaluation
        best_combination (list): best combination at the moment of evaluation
        best_tiebreaker (list): tiebreaker for best combination at the
        moment of evaluation
    """

    def __init__(self, private_cards):
        """
        Instantiate a hand object based on private cards
        e.g. Hand(Card(14,"C"),Card(13,"C"))
        """
        self.private_cards = private_cards
        self.public_cards = []
        self.best_rank = []
        self.best_combination = []
        self.best_tiebreaker = []

    def add_public_cards(self, public_cards):
        """
        Method to ingest public cards

        Args:
            public_cards (list): list containing one or several Card objects
        """
        self.public_cards = self.public_cards + public_cards

    def update_best_combination(self):
        """
        Method to evaluate best combination and update corresponding
        attributes based on private and public cards collected so far
        """
        assert self.public_cards  # check list is not empty

        logging.debug('Evaluating best combination, based on cards')
        # number of combinations
        hands = list(combinations(self.private_cards + self.public_cards, 5))
        logging.debug('{} combinations to evaluate'.format(len(hands)))

        # evaluate possible hands
        hand_evaluations = [evaluate_hand_ranking(hand) for hand in hands]
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
            logging.debug('Tiebreaking best combination, among {} candidates'
                          .format(len(best_hands)))
            best_hand, best_tiebreaker = \
                tie_breaking(best_hands, best_hands_tiebreakers)
            self.best_rank = best_rank
            self.best_combination = best_hand
            self.best_tiebreaker = best_tiebreaker

    def pretty_str_best_combination(self):
        """ Returns pretty card representation for the best_combination """
        return [card.pretty_rank + card.pretty_suit
                for card in self.best_combination[0]]

    def human_readable_rank(self):
        """ Converts numerical ranking of hand into human-readable version """
        return HUMAN_READABLE_RANKINGS[self.best_rank]

    def __repr__(self):
        """ Pretty card representation of all cards in hand """
        return str([card.pretty_rank + card.pretty_suit
                    for card in self.private_cards + self.public_cards])
