
import random
import logging
import numpy as np

from pokerbot.flow_control.player import Player
from ..globals import PRE_FLOP_WINNING_PROB

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


class FixedPolicyPlayer(Player):
    """
    Poker player object capable of playing games
    Takes action and decide about amount to be bet based on a fixed policy

    Inherits from the Player class
    Only methods to take action and choose amount have been added
    """

    def take_action(self, actions, hand_hist=None):
        """
        Getting action from player by assessing strength of starting hand

        Args:
            actions (list): set of str action the player can choose from
            hand_hist (dict): json format hand history, default None

        Returns:
            choice (str): the action taken
        """
        logging.debug('Action is on {}'.format(self.name))
        logging.debug('{} has a stack of {}$'.format(self.name, self.stack))
        if hand_hist:
            p = PRE_FLOP_WINNING_PROB[hand_hist['preflop']['simp_rep']]
            if p < 0.5:
                if 'check' in actions:
                    choice = 'check'
                else:
                    choice = 'fold'
            else:
                if 'raise' in actions:
                    choice = np.random.choice(['raise', 'call'], 1,
                                              p=[p, 1-p])[0]
                elif 'bet' in actions:
                    choice = np.random.choice(['bet', 'check'], 1,
                                              p=[p, 1-p])[0]
                else:
                    choice = 'all-in'
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
