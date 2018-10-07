
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class Player:
    """
    Poker player object capable of playing games

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
