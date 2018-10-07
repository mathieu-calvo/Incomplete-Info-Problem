
import random
import logging

from .handplayed import HandPlayed
from .deck import Deck
from .player import Player

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class HeadsUpGame:
    """
    Heads-up game object managing flow control for game being played being two
    poker players, based on the parameters of the game

    Attributes:
        max_nb_hands (int): maximum number of hands, game stops once reached
        starting_stack (int): initial number of chips per player
        big_blind (int): initial compulsory stake
        position (bool): randomly selected initial position
        hand_number (int): index to keep track of number of hands played
        player1 (class.Player): first player
        player2 (class.Player): second player
        deck (class.Deck): 52-card deck
    """

    def __init__(self, max_nb_hands, starting_stack, big_blind,
                 player1, player2):
        """
        Instantiate a game object based on parameters and players' names
        e.g. HeadsUpGame(100, 1000, 10, "Joe", "Mike")
        """
        # parameters of the game
        self.max_nb_hands = max_nb_hands
        self.starting_stack = starting_stack
        self.big_blind = big_blind
        # initialising variables
        self.position = random.choice([True, False])
        self.hand_number = 1
        self.player1 = Player(starting_stack, player1)
        self.player2 = Player(starting_stack, player2)
        self.deck = Deck()

    def start_game(self):
        """
        Method to handle flow control during game, updating variables of the
        game while hands are being played and checking whether one of the end
        of game conditions has been met
        """
        all_players_have_chips = True
        # play while game is not over and max nb of hands not reached
        while self.hand_number <= self.max_nb_hands and all_players_have_chips:
            # start playing hands
            logging.info('Hand #{} starts'.format(self.hand_number))
            # draw nine cards randomly from deck, 5 common + 2 per player
            drawn_cards = self.deck.deal_cards(9)
            # current position determines which player plays big blind
            if self.position:
                current_hand = HandPlayed(self.player1, self.player2,
                                          self.big_blind, drawn_cards)
            else:
                current_hand = HandPlayed(self.player2, self.player1,
                                          self.big_blind, drawn_cards)
            # start playing the hand given current attributes
            current_hand.play()
            # check if one player is out of chips
            if self.player1.stack == 0:
                logging.info('{} won the game'.format(self.player2.name))
                all_players_have_chips = False
            elif self.player2.stack == 0:
                logging.info('{} won the game'.format(self.player1.name))
                all_players_have_chips = False
            # update attributes when hand is over
            self.hand_number += 1
            self.position = not self.position
            # TODO: potentially store evolution of stacks and perf metrics here
            # print white line for readability
            logging.info("")
        # TODO: display end of game metrics: winner, winnings, etc...
        if all_players_have_chips:
            logging.info("End of the game, all the hands have been played:")
            logging.info("{} final stack: {} $".format(self.player1.name,
                                                       self.player1.stack))
            logging.info("{} final stack: {} $".format(self.player2.name,
                                                       self.player2.stack))
