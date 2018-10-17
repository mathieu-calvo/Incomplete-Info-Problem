
import random
import logging
import numpy as np

from pokerbot.flow_control.player import Player

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


def truncated_normal(mean, stddev, minval, maxval):
    """
    Method to draw a random number from a truncated normal distribution

    Args:
        mean (int): mean of distribution
        stddev (float): standard deviation of distribution
        minval (int): lower boundary
        maxval (int): upper boundary

    Returns:
        (int): the number drawn
    """
    return int(np.clip(np.random.normal(mean, stddev), minval, maxval))


class RandomPlayer(Player):
    """
    Poker player object capable of playing games
    Takes action and decide about amount to be bet randomly

    Inherits from the Player class
    Only methods to take action and choose amount have been added
    """

    def take_action(self, actions):
        """
        Getting action from player by randomly selecting an option

        Args:
            actions (list): set of str action the player can choose from

        Returns:
            choice (str): the action taken
        """
        logging.debug('Action is on {}'.format(self.name))
        logging.debug('{} has a stack of {}$'.format(self.name, self.stack))
        choice = random.choice(actions)
        logging.debug('{}\'s choice is: {}'.format(self.name, choice))
        return choice

    def choose_amount(self, minimum=None, maximum=None,
                      pot_size=None, std_dev=50):
        """
        Getting amount to bet from player by selecting an amount randomly
        by drawing a number from a truncated normal distribution centered
        around the pot size

        Args:
            minimum (int): minimum amount that is required, default None
            maximum (int): maximum amount that is required, default None
            pot_size (int): amount in pot at the moment of decision, used as
            the mean of the distribution , default None
            std_dev (float): standard deviation of the distribution,
            default is set arbitrarily to 50

        Returns:
            bet_size (int): the amount to bet
        """
        if self.stack <= minimum:
            bet_size = self.stack
        elif maximum:
            bet_size = truncated_normal(pot_size, std_dev, minimum, maximum)
        else:
            bet_size = truncated_normal(pot_size, std_dev, minimum, maximum)
        self.bet_amount(bet_size)
        return bet_size
