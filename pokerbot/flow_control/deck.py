
import random
import logging

from .card import Card

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class Deck(object):
    """
    Instantiating the object returns a list of 52 flow_control.card.Card
    objects, corresponding to the 52 cards present in a normal deck of cards

    Attributes:
        cards (list): list of our 52 unique Card objects
    """

    def __init__(self):
        """
        Instantiating the object does not need any argument
        It creates all the cards needed in a 52-card deck
        """
        # initialize deck
        self.cards = [Card(numeric_rank, suit)
                      for numeric_rank in range(2, 15)
                      for suit in ['S', 'C', 'D', 'H']]

    def deal_cards(self, number_cards):
        """
        Drawn any number of cards randomly and without replacement from the
        deck

        Args:
            number_cards (int): explicit

        Returns:
            list: list containing drawn cards, as Card objects
        """
        # make sure number makes sense
        assert number_cards in range(1, 53), \
            "Incorrect number of cards to draw from deck, {} was passed on " \
            .format(number_cards)
        # use the random library to sample from deck
        return random.sample(self.cards, number_cards)
