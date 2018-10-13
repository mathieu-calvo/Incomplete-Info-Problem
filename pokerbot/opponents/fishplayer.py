
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

    def take_action(self, imbalance_size):
        """
        Getting action from player by always selecting the option to check
        or call

        Args:
            imbalance_size (int): pre action imbalance size, i.e.
            positive if one player has put more into the pot than the other

        Returns:
            choice (str): the action taken
        """
        logging.info('Action is on {}'.format(self.name))
        logging.info('{} has a stack of {}$'.format(self.name, self.stack))

        if imbalance_size > 0:
            if imbalance_size >= self.stack:
                actions = ['all-in', 'fold']
            else:
                actions = ['call', 'raise', 'fold']
        else:
            actions = ['check', 'bet']

        if 'check' in actions:
            choice = 'check'
        elif 'call' in actions:
            choice = 'call'
        else:
            choice = 'all-in'

        logging.info('{}\'s choice is: {}'.format(self.name, choice))
        return choice
