
import logging

from ..utils import action_input, amount_input

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class Player(object):
    """
    Poker player object capable of playing games
    Takes action and decide about amount to be bet by prompting user

    Attributes:
        stack (int): amount the chips he owns at any given time
        name (str): name of the player
    """

    def __init__(self, stack, name):
        """
        Instantiating the object using a numeric stack and a name
        e.g. Player(100,"Joe")
        """
        self.stack = stack
        self.name = name

    def bet_amount(self, bet_size):
        """
        Adjust stack based on amount player is betting
        Player is all-in if bets more than stack

        Args:
            bet_size (int): any number
        """
        if bet_size >= self.stack:
            logging.info('{} bets {}$ into the pot and is ALL-IN'
                         .format(self.name, self.stack))
            self.stack = 0
        else:
            self.stack = self.stack - bet_size
            logging.info('{} bets {}$ into the pot'
                         .format(self.name, bet_size))

    def win_pot(self, pot_size):
        """
        Adjust stack based on amount player is winning

        Args:
            pot_size (int): any number
        """
        self.stack = self.stack + pot_size
        logging.info('{} wins the pot: +{}$'
                     .format(self.name, pot_size))

    def split_pot(self, pot_size):
        """
        Adjust stack based on amount player is getting from a split pot

        Args:
            pot_size (int): any number
        """
        self.stack = self.stack + int(pot_size/2)
        logging.info('{} splits the pot: +{}$'
                     .format(self.name, int(pot_size/2)))

    def get_back_from_pot(self, amount):
        """
        Adjust stack based on amount player is getting back from a pot where he
        has put in more in the pot than the other player has chips,
        and the other player end up going all in

        Args:
            amount (int): any number
        """
        self.stack = self.stack + amount
        logging.info('{} gets +{}$ back from the pot'
                     .format(self.name, amount))

    def take_action(self, imbalance_size):
        """
        Getting action from player by prompting user for answers

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

        choice = action_input("Action?", actions)
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
        bet_size = amount_input("Amount?", minimum=minimum, maximum=maximum)
        self.bet_amount(bet_size)
        return bet_size
