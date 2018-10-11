
import random
import logging

from pokerbot.flow_control.player import Player

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class RandomPolicyOpponent(Player):
    """
    Poker player object capable of playing games
    Takes action and decide about amount to be bet randomly

    Inherits from the Player class
    Only methods to take action and choose amount have been replaced
    """

    def take_action(self, imbalance_size):
        """
        Getting action from player by randomly selecting an option

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
                actions = ['call', 'raise', 'fold', 'all-in']
        else:
            actions = ['check', 'bet', 'fold', 'all-in']

        choice = random.choice(actions)
        logging.info('{}\'s choice is: {}'.format(self.name, choice))

        return choice

    def choose_amount(self, minimum=None, maximum=None):
        """
        Getting amount to bet from player by prompting user for answers

        Args:
            minimum (int): minimum amount that is required, default None
            maximum (int): maximum amount that is required, default None

        Returns:
            bet_size (int): the amount to bet
        """
        if maximum:
            bet_size = random.choice(range(minimum, maximum, 1))
        else:
            bet_size = random.choice(range(minimum, self.stack, 1))
        self.bet_amount(bet_size)
        return bet_size
