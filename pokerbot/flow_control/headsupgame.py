
import random
import logging

from .handplayed import HandPlayed
from .deck import Deck

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class HeadsUpGame(object):
    """
    Heads-up game object managing flow control for game being played being two
    poker players, based on the parameters of the game. One of the player is
    our Hero, while his opponent is the Villain

    Attributes:
        max_nb_hands (int): maximum number of hands, game stops once reached
        big_blind (int): initial compulsory stake
        is_fixed_limit (bool): fixed limit game if True, no-limit if False
        hero_is_big_blind (bool): randomly selected initial position
        hand_number (int): index to keep track of number of hands played
        player_hero (subclass.Player): first player, our Hero
        player_villain (subclass.Player): second player, our Villain
        deck (class.Deck): 52-card deck
        hero_game_history (list): game history object from the point of view of
        our hero player
    """

    def __init__(self, max_nb_hands, big_blind,
                 player_hero, player_villain, is_fixed_limit):
        """
        Instantiate a game object based on parameters and players' object
        e.g. HeadsUpGame(100, 10, HumanPlayer(100,"Joe"), FishPlayer(100,
        "Mike"), True)
        """
        # parameters of the game
        self.max_nb_hands = max_nb_hands
        self.big_blind = big_blind
        self.is_fixed_limit = is_fixed_limit
        # initialising variables
        self.hero_is_big_blind = random.choice([True, False])
        self.hand_number = 1
        self.player_hero = player_hero
        self.player_villain = player_villain
        self.deck = Deck()
        self.hero_game_history = []

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
            logging.debug('Hand #{} starts'.format(self.hand_number))
            # draw nine cards randomly from deck, 5 common + 2 per player
            drawn_cards = self.deck.deal_cards(9)
            # current position determines which player plays big blind
            if self.hero_is_big_blind:
                current_hand = HandPlayed(self.player_hero,
                                          self.player_villain,
                                          self.big_blind,
                                          self.is_fixed_limit,
                                          drawn_cards,
                                          self.hand_number)
                # start playing the hand given current attributes
                current_hand.play()
                logging.debug("{}".format(current_hand.hand_history_BB))
                self.hero_game_history.append(current_hand.hand_history_BB)
            else:
                current_hand = HandPlayed(self.player_villain,
                                          self.player_hero,
                                          self.big_blind,
                                          self.is_fixed_limit,
                                          drawn_cards,
                                          self.hand_number)
                # start playing the hand given current attributes
                current_hand.play()
                logging.debug("{}".format(current_hand.hand_history_SB))
                self.hero_game_history.append(current_hand.hand_history_SB)
            # check if one player is out of chips
            if self.player_hero.stack == 0:
                logging.info('{} won the game'
                             .format(self.player_villain.name))
                all_players_have_chips = False
            elif self.player_villain.stack == 0:
                logging.info('{} won the game'
                             .format(self.player_hero.name))
                all_players_have_chips = False
            # update attributes when hand is over
            self.hand_number += 1
            self.hero_is_big_blind = not self.hero_is_big_blind
            # TODO: potentially store evolution of stacks and perf metrics here
        # TODO: display end of game metrics: winner, winnings, etc...
        if all_players_have_chips:
            logging.info("End of the game, all the hands have been played:")
            logging.info("{} final stack: {} $"
                         .format(self.player_hero.name,
                                 self.player_hero.stack))
            logging.info("{} final stack: {} $"
                         .format(self.player_villain.name,
                                 self.player_villain.stack))
