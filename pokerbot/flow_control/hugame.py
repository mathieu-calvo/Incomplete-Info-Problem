
import random
import logging

from .hdplayed import HdPlayed
from .deck import Deck

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class HuGame(object):
    """
    Heads-up game object managing flow control for game being played being two
    poker players, based on the parameters of the game. One of the player is
    our Hero, while his opponent is the Villain

    Attributes:
        max_nb_hands (int): maximum number of hands, game stops once reached
        big_blind (int): initial compulsory stake
        is_fixed_limit (bool): fixed limit game if True, no-limit if False
        hero_is_big_blind (bool): randomly selected initial position
        current_hand (HdPlayed): hand being played
        hand_number (int): index to keep track of number of hands played
        player_hero (subclass.Player): first player, our Hero
        player_villain (subclass.Player): second player, our Villain
        deck (class.Deck): 52-card deck
        game_over (bool): boolean indicating if game is over
        hero_game_history (list): game history object from the point of view of
        our hero player
    """

    def __init__(self, max_nb_hands, big_blind,
                 player_hero, player_villain, is_fixed_limit):
        """
        Instantiate a game object based on parameters and players' object
        e.g. HuGame(100, 10, RandomPlayer(100,"Joe"), FishPlayer(100,
        "Mike"), True)
        """
        # parameters of the game
        self.max_nb_hands = max_nb_hands
        self.big_blind = big_blind
        self.is_fixed_limit = is_fixed_limit
        # initialising variables
        self.hero_is_big_blind = random.choice([True, False])
        self.hand_number = 0
        self.player_hero = player_hero
        self.player_villain = player_villain
        self.deck = Deck()
        self.game_over = False
        self.hero_game_history = []
        self.current_hand = None

    def _deal_hand(self):
        """
        Method to draw nine cards randomly from deck, 5 common + 2 per player,
        and create a HdPlayed object

        Returns:
             HdPlayed object
        """
        # update the hand count and change positions
        self.hand_number += 1
        self.hero_is_big_blind = not self.hero_is_big_blind
        # draw nine cards randomly from deck, 5 common + 2 per player
        # current position determines which player plays big blind
        return HdPlayed(self.hero_is_big_blind,
                        self.player_hero,
                        self.player_villain,
                        self.big_blind,
                        self.is_fixed_limit,
                        self.deck.deal_cards(9),
                        self.hand_number)

    def _is_game_over(self):
        """
        Method to check whether conditions for the end of the game have been
        met.

        Returns:
            (bool) indicating is game is over
        """
        # check if all hands have been played
        if self.hand_number >= self.max_nb_hands:
            return True
        # check if one player is out of chips
        if self.player_hero.stack == 0 or self.player_villain.stack == 0:
            return True
        return False

    def reset(self):
        """
        Method to re-initialise attributes
        """
        self.hero_is_big_blind = random.choice([True, False])
        self.hand_number = 1
        self.player_hero.reset_stack()
        self.player_villain.reset_stack()
        self.game_over = False
        self.hero_game_history = []
        self.current_hand = None

    def step(self, action):
        """
        Method to take a step into the game for the agent

        Args:
            action (str): action selected by agent

        Returns:
            next_state (array): numerical representation of environment
            reward (int): numerical reward
            game_over (bool): indicating if game is over
            hand_done (bool): indicating if hand is done
            info (object): hand histories for debugging purposes
        """
        next_state, reward, hand_done, info = self.current_hand.step(action)
        if hand_done:
            # check if game is over
            self.game_over = self._is_game_over()
        return next_state, reward, self.game_over, hand_done, info

    def initial_step(self):
        """
        Method to take the initial step into the game for the agent
        Deals a hand

        Returns:
            state (array): numerical representation of environment
        """
        # deal new hand
        self.current_hand = self._deal_hand()
        # get initial state from it
        state = self.current_hand.initial_step()
        return state



