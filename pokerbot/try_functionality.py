
import logging

from .flow_control.headsupgame import HeadsUpGame
from .opponents.humanplayer import HumanPlayer
from .opponents.randomplayer import RandomPlayer
from .opponents.fishplayer import FishPlayer

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.DEBUG)


def run():
    player_one = HumanPlayer(500, "Luci")
    player_two = RandomPlayer(500, "Cybi")
    player_three = FishPlayer(500, "Nemo")
    hu = HeadsUpGame(5, 1000, 20, player_three, player_two)
    hu.start_game()
