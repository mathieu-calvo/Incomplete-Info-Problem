
import logging
import numpy as np

from ..flow_control.player import Player
from ..hand_evaluation.hand_potential import estimate_win_rate
from ..globals import PRE_FLOP_WINNING_PROB

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class FishPlayer(Player):
    """
    Poker player object capable of playing games
    Dumb player who takes always the decision to check or call and never bets

    Inherits from the Player class
    Only method to take action has been added
    """

    def take_action(self, actions, hand_hist=None):
        """
        Getting action from player by always selecting the option to check
        or call

        Args:
            actions (list): set of str action the player can choose from
            hand_hist (dict): json format hand history, default None

        Returns:
            choice (str): the action taken
        """
        del hand_hist

        logging.debug('Action is on {}'.format(self.name))
        logging.debug('{} has a stack of {}$'.format(self.name, self.stack))

        if 'check' in actions:
            choice = 'check'
        elif 'call' in actions:
            choice = 'call'
        else:
            choice = 'all-in'

        logging.debug('{}\'s choice is: {}'.format(self.name, choice))
        return choice


class StartingHandPlayer(Player):
    """
    Poker player object capable of playing games
    Takes action and decide about amount to be bet based on a fixed policy
    Namely by assessing strength of starting hand

    Inherits from the Player class
    Only method to take action has been added
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
            # # select actions based on win rate at the beginning
            # if p < 0.5:

            #     if 'check' in actions:
            #         choice = 'check'
            #     else:
            #         choice = 'fold'
            # else:
            #     if 'raise' in actions:
            #         choice = np.random.choice(['raise', 'call'], 1,
            #                                   p=[p, 1-p])[0]
            #     elif 'bet' in actions:
            #         choice = np.random.choice(['bet', 'check'], 1,
            #                                   p=[p, 1-p])[0]
            #     elif ['call', 'fold'] == actions:
            #         choice = 'call'
            #     else:
            #         choice = 'all-in'
            # select actions based on win rate at the beginning
            if p < 0.5:
                if 'check' in actions:
                    choice = 'check'
                else:
                    choice = 'fold'
            elif p < 0.6:
                if 'call' in actions:
                    choice = 'call'
                elif 'bet' in actions:
                    choice = 'bet'
                elif ['call', 'fold'] == actions:
                    choice = 'call'
                else:
                    choice = 'all-in'
            else:
                if 'raise' in actions:
                    choice = 'raise'
                elif 'bet' in actions:
                    choice = 'bet'
                elif ['call', 'fold'] == actions:
                    choice = 'call'
                else:
                    choice = 'all-in'
            logging.debug('{}\'s choice is: {}'.format(self.name, choice))
            return choice


class StrengthHandPlayer(Player):
    """
    Poker player object capable of playing games
    Takes action and decide about amount to be bet based on a fixed policy
    Namely by assessing strength of hand given all visible cards

    Inherits from the Player class
    Only method to take action has been added
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
            # get relevant info from hand history
            simp_pre_flop_hand = hand_hist['preflop']['simp_rep']
            hole_cards = hand_hist['preflop']['hole_cards']
            community_cards = hand_hist['community_cards']
            nb_simulations = 1000
            # if stage is pre-flop, agent rely on lookup table for win rate
            if community_cards:
                p = estimate_win_rate(nb_simulations,
                                      hole_cards,
                                      community_cards=community_cards)
            else:
                p = PRE_FLOP_WINNING_PROB[simp_pre_flop_hand]
            logging.debug('{} has {}'.format(self.name, simp_pre_flop_hand))
            logging.debug('p = {}'.format(p))
            # select actions based on win rate
            if p < 0.5:
                if 'check' in actions:
                    choice = 'check'
                else:
                    choice = 'fold'
            elif p < 0.8:
                if 'call' in actions:
                    choice = 'call'
                elif 'bet' in actions:
                    choice = 'bet'
                elif ['call', 'fold'] == actions:
                    choice = 'call'
                else:
                    choice = 'all-in'
            else:
                if 'raise' in actions:
                    choice = 'raise'
                elif 'bet' in actions:
                    choice = 'bet'
                elif ['call', 'fold'] == actions:
                    choice = 'call'
                else:
                    choice = 'all-in'
            logging.debug('{}\'s choice is: {}'.format(self.name, choice))
            return choice
