
import random
import logging

from ..flow_control.player import Player

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class RandomPlayer(Player):
    """
    Poker player object capable of playing games
    Takes action and decide about amount to be bet randomly

    Inherits from the Player class
    Only methods to take action and choose amount have been added
    """

    def take_action(self, actions, hand_hist=None):
        """
        Getting action from player by randomly selecting an option

        Args:
            actions (list): set of str action the player can choose from
            hand_hist (dict): json format hand history, default None

        Returns:
            choice (str): the action taken
        """
        logging.debug('Action is on {}'.format(self.name))
        logging.debug('{} has a stack of {}$'.format(self.name, self.stack))
        choice = random.choice(actions)
        logging.debug('{}\'s choice is: {}'.format(self.name, choice))
        return choice
