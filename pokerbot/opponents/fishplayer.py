
import logging

from pokerbot.flow_control.player import Player

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
