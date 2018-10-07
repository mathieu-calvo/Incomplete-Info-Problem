
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class Player:

    def __init__(self, stack, name):
        self.stack = stack
        self.name = name

    def bet_amount(self, bet_size):
        if bet_size >= self.stack:
            logging.info('{} bets {}$ into the pot and is ALL-IN'
                         .format(self.name, self.stack))
            self.stack = 0
        else:
            self.stack = self.stack - bet_size
            logging.info('{} bets {}$ into the pot'
                         .format(self.name, bet_size))

    def win_pot(self, pot_size):
        self.stack = self.stack + pot_size
        logging.info('{} wins the pot: +{}$'
                     .format(self.name, pot_size))

    def split_pot(self, pot_size):
        self.stack = self.stack + int(pot_size/2)
        logging.info('{} splits the pot: +{}$'
                     .format(self.name, int(pot_size/2)))