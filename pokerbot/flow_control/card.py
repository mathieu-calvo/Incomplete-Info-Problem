
import logging

from ..globals import FACE_CARDS_RANK_DICT, \
    PRETTY_SUIT_DICT, NUMERICAL_SUIT_DICT

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


def prettify_rank(numeric_rank):
    """
    Transform rank for face cards from numeric to str character

    Args:
        numeric_rank (int): explicit

    Returns:
        str: rank in prettified str format
    """
    # check that value is acceptable range
    if not 2 <= numeric_rank <= 14:
        raise ValueError('Invalid card, numeric rank: {}'
                         .format(numeric_rank))
    # transform rank to pretty format for face cards
    if numeric_rank in FACE_CARDS_RANK_DICT.keys():
        return FACE_CARDS_RANK_DICT[numeric_rank]
    # otherwise keep numeric value
    return str(numeric_rank)


def prettify_suit(suit_str):
    """
    Transform suit from letter to symbol

    Args:
        suit_str (int): possible entries: C,D,H or S

    Returns:
        str: suit in prettified symbol
    """
    # check that value is within acceptable range
    if suit_str not in PRETTY_SUIT_DICT.keys():
        raise ValueError('Invalid card, suit: {}'.format(suit_str))
    else:
        return PRETTY_SUIT_DICT[suit_str]


class Card(object):
    """
    Instantiating the object returns a card object from a 52-card deck

    Attributes:
        rank (int): rank of the card from 2 to 14
        suit (str): suit of the card, possible entries: C,D,H,S
        pretty_rank (str): rank in prettified str format for face cards
        pretty_suit (str): suit in prettified symbol
        numerical_id (int): unique numerical id for the card, [1-52]
    """

    def __init__(self, numeric_rank, suit):
        """
        Instantiating the object using a numeric rank and a one letter suit
        e.g. Card(14,"C")
        """
        self.rank = numeric_rank
        self.suit = suit
        self.pretty_rank = prettify_rank(self.rank)
        self.pretty_suit = prettify_suit(self.suit)
        self.numerical_id = 4 * (self.rank - 2) + NUMERICAL_SUIT_DICT[
            self.suit]

    def __str__(self):
        return str(self.rank) + self.suit

    def __repr__(self):
        return self.pretty_rank + self.pretty_suit

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
