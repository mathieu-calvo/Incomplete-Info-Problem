
import random
import logging

from .card import Card

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class Deck:

    def __init__(self):
        # initialize deck
        self.cards = [Card(numeric_rank, suit)
                      for numeric_rank in range(2, 15)
                      for suit in ['S', 'C', 'D', 'H']]

    def deal_cards(self, number_cards):
        # Defining drawing cards (5 common cards + 2 times number of player):
        return random.sample(self.cards, number_cards)
