
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


def prettify_rank(numeric_rank):
    # check that value is acceptable range
    if not 2 <= numeric_rank <= 14:
        raise ValueError('Invalid card, numeric rank: {}'.format(
            numeric_rank))
    # transform to pretty format
    face_cards_dict = {14: 'A', 11: 'J', 12: 'Q', 13: 'K'}
    if numeric_rank in face_cards_dict.keys():
        return face_cards_dict[numeric_rank]
    else:
        return numeric_rank


def prettify_suit(suit_str):
    pretty_suit_dict = {'C': '\u2667', 'D': '\u2662', 'H': '\u2661',
                        'S': '\u2664'}
    # check that value is within acceptable range
    if suit_str not in pretty_suit_dict.keys():
        raise ValueError('Invalid card, suit: {}'.format(suit_str))
    else:
        return pretty_suit_dict[suit_str]


class Card:
    
    def __init__(self, numeric_rank, suit):
        self.rank = numeric_rank
        self.suit = suit
        self.pretty_rank = prettify_rank(self.rank)
        self.pretty_suit = prettify_suit(self.suit)

    def __str__(self):
        return str(self.rank) + self.suit

    def __repr__(self):
        return str(self.pretty_rank) + self.pretty_suit
