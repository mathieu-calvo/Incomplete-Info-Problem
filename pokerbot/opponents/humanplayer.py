
import logging

from pokerbot.flow_control.player import Player
from ..utils import action_input, amount_input

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class HumanPlayer(Player):
    """
    Poker player object capable of playing games
    Takes action and decide about amount to be bet by prompting human user

    Inherits from the Player class
    Only methods to take action and choose amount have been added
    """

    def take_action(self, actions, hand_hist=None):
        """
        Getting action from player by prompting user for answers

        Args:
            actions (list): set of str action the player can choose from
            hand_hist (dict): json format hand history, default None

        Returns:
            choice (str): the action taken
        """
        logging.debug('Action is on {}'.format(self.name))
        logging.debug('{} has a stack of {}$'.format(self.name, self.stack))
        choice = action_input("Action?", actions)
        logging.debug('{}\'s choice is: {}'.format(self.name, choice))
        return choice

    def choose_amount(self, minimum=None, maximum=None, pot_size=None):
        """
        Getting amount to bet from player by prompting user for answers

        Args:
            minimum (int): minimum amount that is required, default None
            maximum (int): maximum amount that is required, default None
            pot_size (int): pot size at the moment of decision, default None

        Returns:
            bet_size (int): the amount to bet
        """
        bet_size = amount_input("Amount?",
                                minimum=minimum,
                                maximum=maximum,
                                pot_size=pot_size)
        self.bet_amount(bet_size)
        return bet_size
