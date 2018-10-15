
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

    def take_action(self, imbalance_size):
        """
        Getting action from player by prompting user for answers

        Args:
            imbalance_size (int): pre action imbalance size, i.e.
            positive if one player has put more into the pot than the other

        Returns:
            choice (str): the action taken
        """
        logging.debug('Action is on {}'.format(self.name))
        logging.debug('{} has a stack of {}$'.format(self.name, self.stack))

        if imbalance_size > 0:
            if imbalance_size >= self.stack:
                actions = ['all-in', 'fold']
            else:
                actions = ['call', 'raise', 'fold', 'all-in']
        else:
            actions = ['check', 'bet', 'all-in']

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
