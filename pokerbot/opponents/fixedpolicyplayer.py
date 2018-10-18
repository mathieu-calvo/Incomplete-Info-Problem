
import logging
import numpy as np

from pokerbot.flow_control.player import Player
from ..globals import PRE_FLOP_WINNING_PROB

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


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
            simp_pre_flop_hand = hand_hist['preflop']['simp_rep']
            p = PRE_FLOP_WINNING_PROB[simp_pre_flop_hand]
            logging.debug('{} has {}'.format(self.name, simp_pre_flop_hand))
            logging.debug('p = {}'.format(p))
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
                elif ['call', 'fold'] == actions:
                    choice = 'call'
                else:
                    choice = 'all-in'
            logging.debug('{}\'s choice is: {}'.format(self.name, choice))
            return choice
